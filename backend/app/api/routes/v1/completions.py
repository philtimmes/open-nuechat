"""
OpenAI-compatible /v1/chat/completions endpoint

Supports:
- Streaming and non-streaming responses
- Base models and Custom GPTs
- Knowledge base RAG
- Filter chain execution
- Tool/function calling
- Token billing
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.models import User, APIKey, CustomAssistant, KnowledgeStore
from app.api.routes.api_keys import get_api_key_user, require_scope
from app.services.settings_service import SettingsService
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.billing import BillingService
from app.filters.executor import ChainExecutor

logger = logging.getLogger(__name__)

router = APIRouter()


# === Schemas (OpenAI-compatible) ===

class ChatMessage(BaseModel):
    """A message in the conversation"""
    role: str = Field(..., description="Role: system, user, assistant, or tool")
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    """Function definition for tool calling"""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDefinition(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    n: Optional[int] = Field(default=1, ge=1, le=10)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    user: Optional[str] = None
    # Tool calling
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None
    # NueChat extensions
    knowledge_store_ids: Optional[List[str]] = None  # Override knowledge stores


class ChatCompletionChoice(BaseModel):
    """A completion choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class ChatCompletionChunkChoice(BaseModel):
    """A chunk choice in streaming"""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# === Helper Functions ===

def assistant_name_to_model_id(name: str) -> str:
    """
    Convert assistant name to a clean model ID.
    
    - Replace spaces with underscores
    - Remove any characters that aren't alphanumeric, dash, or underscore
    - Lowercase for consistency
    
    Example: "My Cool Assistant!" -> "my_cool_assistant"
    """
    import re
    # Replace spaces with underscores
    clean = name.replace(" ", "_")
    # Remove any non-alphanumeric characters except - and _
    clean = re.sub(r'[^a-zA-Z0-9_-]', '', clean)
    # Lowercase
    clean = clean.lower()
    # Remove consecutive underscores/dashes
    clean = re.sub(r'[-_]+', '_', clean)
    # Strip leading/trailing underscores
    clean = clean.strip('_-')
    return clean


async def resolve_model(
    model_id: str,
    user: User,
    api_key: APIKey,
    db: AsyncSession,
) -> tuple[str, Optional[CustomAssistant]]:
    """
    Resolve model ID to actual model and optional assistant.
    
    Supports:
    - Base model names (e.g., "llama3.2", "gpt-4")
    - Cleaned assistant names (e.g., "my_research_assistant")  
    - Legacy gpt: prefix format (e.g., "gpt:uuid")
    
    Returns: (model_name, assistant_or_none)
    """
    from sqlalchemy import or_
    
    # Legacy gpt: format (still supported)
    if model_id.startswith("gpt:"):
        assistant_id = model_id[4:]
        
        # Check API key restrictions
        if api_key.allowed_assistants and assistant_id not in api_key.allowed_assistants:
            raise HTTPException(status_code=403, detail="Access to this model is not allowed")
        
        result = await db.execute(
            select(CustomAssistant).where(
                CustomAssistant.id == assistant_id,
            )
        )
        assistant = result.scalar_one_or_none()
        
        if not assistant:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Check access
        is_owner = assistant.user_id == user.id
        is_public = assistant.is_public
        
        prefs = user.preferences or {}
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
        subscribed_ids = prefs.get("subscribed_assistants", [])
        is_subscribed = assistant.id in subscribed_ids
        
        if not (is_owner or is_public or is_subscribed):
            raise HTTPException(status_code=403, detail="Access denied to this model")
        
        return assistant.model, assistant
    
    # Try to find assistant by cleaned name
    result = await db.execute(
        select(CustomAssistant).where(
            or_(
                CustomAssistant.user_id == user.id,
                CustomAssistant.is_public == True,
            )
        )
    )
    all_assistants = result.scalars().all()
    
    # Find assistant matching the model_id
    for assistant in all_assistants:
        if assistant_name_to_model_id(assistant.name) == model_id:
            # Check API key restrictions
            if api_key.allowed_assistants and assistant.id not in api_key.allowed_assistants:
                raise HTTPException(status_code=403, detail="Access to this model is not allowed")
            
            # Check access
            is_owner = assistant.user_id == user.id
            is_public = assistant.is_public
            
            prefs = user.preferences or {}
            if isinstance(prefs, str):
                prefs = json.loads(prefs)
            subscribed_ids = prefs.get("subscribed_assistants", [])
            is_subscribed = assistant.id in subscribed_ids
            
            if not (is_owner or is_public or is_subscribed):
                raise HTTPException(status_code=403, detail="Access denied to this model")
            
            return assistant.model, assistant
    
    # Base model - no assistant
    return model_id, None


async def get_knowledge_context(
    assistant: Optional[CustomAssistant],
    messages: List[ChatMessage],
    override_kb_ids: Optional[List[str]],
    user: User,
    db: AsyncSession,
) -> str:
    """Get RAG context from knowledge bases."""
    context_parts = []
    
    # Determine which knowledge stores to use
    kb_ids = []
    
    if override_kb_ids:
        kb_ids = override_kb_ids
    elif assistant:
        # Get knowledge stores attached to assistant
        # This requires loading the relationship
        from app.models.assistant import assistant_knowledge_stores
        result = await db.execute(
            select(assistant_knowledge_stores.c.knowledge_store_id).where(
                assistant_knowledge_stores.c.assistant_id == assistant.id
            )
        )
        kb_ids = [row[0] for row in result.fetchall()]
    
    if not kb_ids:
        return ""
    
    # Get the last user message for RAG query
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        return ""
    
    query = user_messages[-1].content or ""
    if not query:
        return ""
    
    # Search knowledge stores
    rag_service = RAGService()
    
    try:
        # Use search_knowledge_stores with bypass for assistant access
        results = await rag_service.search_knowledge_stores(
            db=db,
            user=user,
            query=query,
            knowledge_store_ids=kb_ids,
            top_k=5,
            bypass_access_check=bool(assistant),  # Bypass if using assistant
        )
        if results:
            context_parts.append("--- Knowledge Base Results ---")
            for i, result in enumerate(results, 1):
                context_parts.append(f"[{i}] {result.get('content', '')[:1000]}")
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
    
    return "\n".join(context_parts)


def format_sse_message(data: dict) -> str:
    """Format data as SSE message."""
    return f"data: {json.dumps(data)}\n\n"


# === Main Completion Logic ===

async def create_completion(
    request: ChatCompletionRequest,
    user: User,
    api_key: APIKey,
    db: AsyncSession,
    client_request: Request,
) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
    """Create a chat completion."""
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now(timezone.utc).timestamp())
    
    # Resolve model
    model_name, assistant = await resolve_model(request.model, user, api_key, db)
    
    # Build messages list
    messages = []
    
    # Add system prompt (from assistant or default)
    system_prompt = None
    if assistant and assistant.system_prompt:
        system_prompt = assistant.system_prompt
    else:
        system_prompt = await SettingsService.get(db, "default_system_prompt")
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Get RAG context
    rag_context = await get_knowledge_context(
        assistant,
        request.messages,
        request.knowledge_store_ids,
        user,
        db,
    )
    
    if rag_context:
        # Inject RAG context into system prompt or as separate message
        rag_msg = f"\n\n--- Relevant Context ---\n{rag_context}\n--- End Context ---"
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += rag_msg
        else:
            messages.insert(0, {"role": "system", "content": rag_msg})
    
    # Add user messages
    for msg in request.messages:
        messages.append({
            "role": msg.role,
            "content": msg.content,
            **({"name": msg.name} if msg.name else {}),
            **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
            **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
        })
    
    # Get model parameters
    temperature = request.temperature
    if temperature is None and assistant:
        temperature = assistant.temperature
    if temperature is None:
        temperature = 0.7
    
    max_tokens = request.max_tokens
    if max_tokens is None and assistant:
        max_tokens = assistant.max_tokens
    if max_tokens is None:
        max_tokens = 4096
    
    # Prepare LLM service from database settings
    llm_service = await LLMService.from_database(db)
    
    # Build tools if provided
    tools = None
    if request.tools:
        tools = [t.model_dump() for t in request.tools]
    
    # Execute filters if assistant has a filter chain
    filter_chain_id = None
    if assistant:
        filter_chain_id = getattr(assistant, 'filter_chain_id', None)
    
    if filter_chain_id:
        try:
            executor = ChainExecutor(db)
            # Run pre-LLM filters
            filter_result = await executor.execute_chain(
                chain_id=filter_chain_id,
                content=messages[-1].get("content", "") if messages else "",
                context={
                    "user_id": user.id,
                    "assistant_id": assistant.id if assistant else None,
                    "api_key_id": api_key.id,
                    "source": "v1_api",
                },
                direction="to_llm",
            )
            if filter_result.get("blocked"):
                raise HTTPException(
                    status_code=400,
                    detail=filter_result.get("reason", "Request blocked by filter")
                )
            # Update content if modified
            if filter_result.get("content"):
                messages[-1]["content"] = filter_result["content"]
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Filter execution failed: {e}")
    
    # Create streaming or non-streaming response
    if request.stream:
        return _create_streaming_response(
            completion_id=completion_id,
            created=created,
            model=request.model,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=request.tool_choice,
            stop=request.stop,
            user=user,
            db=db,
            llm_service=llm_service,
        )
    else:
        return await _create_non_streaming_response(
            completion_id=completion_id,
            created=created,
            model=request.model,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=request.tool_choice,
            stop=request.stop,
            user=user,
            db=db,
            llm_service=llm_service,
        )


async def _create_non_streaming_response(
    completion_id: str,
    created: int,
    model: str,
    model_name: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    tools: Optional[List[dict]],
    tool_choice: Optional[Union[str, dict]],
    stop: Optional[Union[str, List[str]]],
    user: User,
    db: AsyncSession,
    llm_service: LLMService,
) -> ChatCompletionResponse:
    """Create a non-streaming completion response."""
    
    # Call LLM
    response_content = ""
    tool_calls = None
    finish_reason = "stop"
    
    try:
        # Use the LLM service's completion method
        result = await llm_service.complete(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=False,
        )
        
        response_content = result.get("content", "")
        tool_calls = result.get("tool_calls")
        finish_reason = result.get("finish_reason", "stop")
        
    except Exception as e:
        logger.error(f"LLM completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")
    
    # Estimate token usage (simplified)
    prompt_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)
    completion_tokens = len(response_content.split()) * 1.3
    
    # Bill tokens
    billing = BillingService()
    await billing.record_usage(
        db=db,
        user_id=user.id,
        input_tokens=int(prompt_tokens),
        output_tokens=int(completion_tokens),
        model=model_name,
        source="v1_api",
    )
    
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens),
        ),
    )


def _create_streaming_response(
    completion_id: str,
    created: int,
    model: str,
    model_name: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    tools: Optional[List[dict]],
    tool_choice: Optional[Union[str, dict]],
    stop: Optional[Union[str, List[str]]],
    user: User,
    db: AsyncSession,
    llm_service: LLMService,
) -> AsyncGenerator[str, None]:
    """Create a streaming completion response."""
    
    async def generate():
        completion_tokens = 0
        
        try:
            # First chunk with role
            yield format_sse_message(
                ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                ).model_dump()
            )
            
            # Stream content
            async for chunk in llm_service.stream_complete(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                stop=stop,
            ):
                content = chunk.get("content", "")
                if content:
                    completion_tokens += len(content.split()) * 1.3
                    yield format_sse_message(
                        ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=content),
                                    finish_reason=None,
                                )
                            ],
                        ).model_dump()
                    )
                
                # Handle tool calls in stream
                if chunk.get("tool_calls"):
                    yield format_sse_message(
                        ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(
                                        tool_calls=chunk["tool_calls"]
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        ).model_dump()
                    )
            
            # Final chunk with finish reason
            yield format_sse_message(
                ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                ).model_dump()
            )
            
            yield "data: [DONE]\n\n"
            
            # Bill tokens after streaming
            prompt_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)
            billing = BillingService()
            await billing.record_usage(
                db=db,
                user_id=user.id,
                input_tokens=int(prompt_tokens),
                output_tokens=int(completion_tokens),
                model=model_name,
                source="v1_api",
            )
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield format_sse_message({"error": str(e)})
    
    return generate()


# === Endpoints ===

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    client_request: Request,
    auth: tuple[User, APIKey] = Depends(require_scope("completions")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a chat completion.
    
    Compatible with OpenAI's /v1/chat/completions endpoint.
    
    **Model Selection:**
    - Use base model name directly: `"llama3.2"`, `"gpt-4"`, etc.
    - Use Custom GPT: `"gpt:<assistant-id>"` or `"gpt:<assistant-name>"`
    
    **Features:**
    - Streaming responses (`stream: true`)
    - Tool/function calling
    - Custom GPT system prompts and knowledge bases
    - Filter chain execution
    - Token billing
    """
    user, api_key = auth
    
    # Rate limiting check could go here
    # TODO: Implement per-key rate limiting
    
    result = await create_completion(request, user, api_key, db, client_request)
    
    if request.stream:
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    return result


# Legacy /v1/completions endpoint (text completion API)
# Some clients like SillyTavern use this instead of /v1/chat/completions
class LegacyCompletionRequest(BaseModel):
    """Legacy OpenAI text completion request - accepts extra fields for SillyTavern compatibility"""
    model: Optional[str] = None  # SillyTavern doesn't always send this
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_new_tokens: Optional[int] = None  # SillyTavern uses this
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    top_k: Optional[int] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    stopping_strings: Optional[List[str]] = None  # SillyTavern uses this
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    repetition_penalty: Optional[float] = None  # SillyTavern
    rep_pen: Optional[float] = None  # SillyTavern alias
    user: Optional[str] = None
    
    # Allow any extra fields SillyTavern sends (typical_p, min_p, mirostat, etc.)
    model_config = {"extra": "allow"}


class LegacyCompletionChoice(BaseModel):
    """Legacy completion choice"""
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None


class LegacyCompletionResponse(BaseModel):
    """Legacy completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[LegacyCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


def _create_legacy_streaming_response(
    completion_id: str,
    created: int,
    model: str,
    model_name: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    stop: Optional[Union[str, List[str]]],
    user: User,
    db: AsyncSession,
    llm_service: LLMService,
) -> AsyncGenerator[str, None]:
    """Create a legacy streaming completion response."""
    
    async def generate():
        completion_tokens = 0
        
        try:
            # Stream content in legacy format
            async for chunk in llm_service.stream_complete(
                messages=messages,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            ):
                content = chunk.get("content", "")
                if content:
                    completion_tokens += len(content.split()) * 1.3
                    # Legacy format uses "text" instead of "delta.content"
                    yield format_sse_message({
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "text": content,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }]
                    })
            
            # Final chunk with finish reason
            yield format_sse_message({
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }]
            })
            
            yield "data: [DONE]\n\n"
            
            # Bill tokens after streaming
            prompt_tokens = sum(len(str(m.get("content", "")).split()) * 1.3 for m in messages)
            billing = BillingService()
            await billing.record_usage(
                db=db,
                user_id=user.id,
                input_tokens=int(prompt_tokens),
                output_tokens=int(completion_tokens),
                model=model_name,
                source="v1_api",
            )
            
        except Exception as e:
            logger.error(f"Legacy streaming error: {e}")
            yield format_sse_message({"error": str(e)})
    
    return generate()


@router.post("/completions")
async def create_legacy_completion(
    request: LegacyCompletionRequest,
    client_request: Request,
    auth: tuple[User, APIKey] = Depends(require_scope("completions")),
    db: AsyncSession = Depends(get_db),
):
    """
    Legacy text completion endpoint.
    
    Converts to chat completion format internally.
    For compatibility with older clients like SillyTavern.
    """
    user, api_key = auth
    
    # Convert prompt to messages format
    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)
    
    # Resolve model - use default if not provided (SillyTavern doesn't always send model)
    model_id = request.model or "default"
    model_name, assistant = await resolve_model(model_id, user, api_key, db)
    
    # Build messages
    messages = [{"role": "user", "content": prompt}]
    
    # Add assistant system prompt if applicable
    if assistant and assistant.system_prompt:
        messages.insert(0, {"role": "system", "content": assistant.system_prompt})
    
    completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now(timezone.utc).timestamp())
    
    temperature = request.temperature if request.temperature is not None else 0.7
    # Use max_tokens or max_new_tokens (SillyTavern preference)
    max_tokens = request.max_tokens or request.max_new_tokens or 1000
    # Combine stop sequences from both fields
    stop = request.stop or request.stopping_strings
    
    llm_service = LLMService()
    
    if request.stream:
        # Use legacy streaming format
        result = _create_legacy_streaming_response(
            completion_id=completion_id,
            created=created,
            model=model_id,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            user=user,
            db=db,
            llm_service=llm_service,
        )
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    # Non-streaming response
    try:
        result = await llm_service.complete(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=False,
        )
        
        response_content = result.get("content", "")
        finish_reason = result.get("finish_reason", "stop")
        
    except Exception as e:
        logger.error(f"Legacy completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")
    
    # Estimate token usage
    prompt_tokens = int(len(prompt.split()) * 1.3)
    completion_tokens = int(len(response_content.split()) * 1.3)
    
    # Bill tokens
    billing = BillingService()
    await billing.record_usage(
        db=db,
        user_id=user.id,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        model=model_name,
        source="v1_api",
    )
    
    return LegacyCompletionResponse(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[
            LegacyCompletionChoice(
                text=response_content,
                index=0,
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
