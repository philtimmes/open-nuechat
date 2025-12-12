"""
LLM Service with streaming, tool calling, token tracking, and bidirectional filtering

Supports OpenAI-compatible APIs including:
- Ollama (http://localhost:11434/v1)
- LM Studio (http://localhost:1234/v1)
- vLLM (http://localhost:8000/v1)
- LocalAI (http://localhost:8080/v1)
- text-generation-webui (with --api flag)
- Any OpenAI-compatible endpoint

Filter Flow:
    User Input -> [OverrideToLLM filters in priority order] -> LLM
    LLM Output -> [OverrideFromLLM filters in priority order] -> User
"""
from typing import AsyncGenerator, Optional, List, Dict, Any, Callable
from datetime import datetime, timezone
import json
import asyncio
import time
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.logging import get_logger, log_llm_request, log_duration
from app.models.models import Chat, Message, MessageRole, ContentType, TokenUsage, User
from app.services.billing import BillingService
from app.filters import (
    get_chat_filters,
    FilterContext,
    FilterResult,
    StreamChunk,
    ChatFilterManager,
)

# Structured logger for LLM service
llm_logger = get_logger("llm_service")


class LLMService:
    """Handle LLM interactions with OpenAI-compatible APIs and bidirectional filtering"""
    
    _resolved_default_model: Optional[str] = None  # Cache the resolved default model
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        self.base_url = base_url or settings.LLM_API_BASE_URL
        self.api_key = api_key or settings.LLM_API_KEY
        self.timeout = timeout or settings.LLM_TIMEOUT
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.temperature = temperature or settings.LLM_TEMPERATURE
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        self.billing_service = BillingService()
    
    @classmethod
    async def from_database(cls, db: AsyncSession) -> "LLMService":
        """Create an LLMService using settings from the database."""
        from app.services.settings_service import SettingsService
        
        llm_settings = await SettingsService.get_llm_settings(db)
        
        return cls(
            base_url=llm_settings["base_url"],
            api_key=llm_settings["api_key"],
            timeout=llm_settings["timeout"],
            max_tokens=llm_settings["max_tokens"],
            temperature=llm_settings["temperature"],
        )
    
    async def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
    ) -> str:
        """
        Simple non-streaming completion for filter chain decisions.
        Returns just the text response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        model = await self._get_effective_model()
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"simple_completion error: {e}")
            return ""
    
    async def _get_effective_model(self, requested_model: Optional[str] = None) -> str:
        """Get the effective model to use, resolving 'default' if needed"""
        model = requested_model or settings.LLM_MODEL
        
        # If model is explicitly set to something other than "default", use it
        if model and model != "default":
            return model
        
        # Check if we have a cached resolved default
        if LLMService._resolved_default_model:
            return LLMService._resolved_default_model
        
        # Fetch available models and use the first one
        try:
            models = await self.list_models()
            for m in models:
                if "error" not in m and "id" in m:
                    LLMService._resolved_default_model = m["id"]
                    return m["id"]
        except Exception:
            pass
        
        # Fallback - return the requested model even if it's "default"
        return model or "default"
    
    def _get_filter_manager(self, chat: Chat) -> ChatFilterManager:
        """Get the filter manager for a chat."""
        return get_chat_filters(str(chat.id))
    
    def _create_filter_context(
        self,
        user: User,
        chat: Chat,
        message_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> FilterContext:
        """Create a filter context for processing."""
        metadata = {
            "user_tier": user.tier.value if user.tier else "free",
            "user_email": user.email,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return FilterContext(
            user_id=str(user.id),
            chat_id=str(chat.id),
            message_id=message_id,
            model=chat.model or settings.LLM_MODEL,
            metadata=metadata,
        )
    
    async def create_message(
        self,
        db: AsyncSession,
        user: User,
        chat: Chat,
        user_message: str,
        attachments: Optional[List[Dict]] = None,
        tools: Optional[List[Dict]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        parent_id: Optional[str] = None,  # For conversation branching
    ) -> Message:
        """Create a non-streaming message with optional tool use and filtering"""
        from app.models.models import CustomAssistant
        
        # Resolve the model to use
        # Handle "gpt:assistant_id" format - look up the assistant's actual model
        chat_model = chat.model
        if chat_model and chat_model.startswith("gpt:"):
            assistant_id = chat_model[4:]
            result = await db.execute(
                select(CustomAssistant).where(CustomAssistant.id == assistant_id)
            )
            assistant = result.scalar_one_or_none()
            if assistant:
                chat_model = assistant.model
        
        effective_model = await self._get_effective_model(chat_model)
        
        # Check if safety filters are enabled (admin setting)
        from app.services.settings_service import SettingsService
        safety_filters_enabled = await SettingsService.get_bool(db, "enable_safety_filters")
        
        # Get filter manager for this chat
        filter_manager = self._get_filter_manager(chat)
        extra_metadata = {"safety_filters_disabled": True} if not safety_filters_enabled else None
        context = self._create_filter_context(user, chat, extra_metadata=extra_metadata)
        
        # === OverrideToLLM: Filter user input before sending to LLM ===
        to_llm_result = await filter_manager.process_to_llm(user_message, context)
        
        if to_llm_result.blocked:
            # Create error message for blocked content
            error_message = Message(
                chat_id=chat.id,
                role=MessageRole.ASSISTANT,
                content=f"I'm sorry, but I cannot process that request. Reason: {to_llm_result.block_reason}",
                content_type=ContentType.TEXT,
                is_error=True,
                parent_id=parent_id,
            )
            db.add(error_message)
            await db.flush()
            return error_message
        
        filtered_user_message = to_llm_result.content
        
        # Build messages from chat history (using tree structure)
        messages = await self._build_messages(db, chat, filtered_user_message, attachments, tools, parent_id)
        
        # Prepare API call parameters
        api_params = {
            "model": effective_model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Add tools if provided and supported
        if tools:
            api_params["tools"] = self._convert_tools_to_openai_format(tools)
        
        # Make API call
        response = await self.client.chat.completions.create(**api_params)
        
        # Track tokens
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        await self._track_usage(
            db=db,
            user=user,
            chat=chat,
            model=effective_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        # Handle response content
        choice = response.choices[0]
        text_content = choice.message.content or ""
        tool_calls = []
        
        # Handle tool calls
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                })
        
        # === OverrideFromLLM: Filter LLM output before sending to user ===
        if text_content:
            from_llm_result = await filter_manager.process_from_llm(text_content, context)
            text_content = from_llm_result.content
        
        # Create assistant message
        assistant_message = Message(
            chat_id=chat.id,
            role=MessageRole.ASSISTANT,
            content=text_content,
            content_type=ContentType.TEXT if not tool_calls else ContentType.TOOL_CALL,
            tool_calls=tool_calls if tool_calls else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=effective_model,
        )
        db.add(assistant_message)
        
        # Handle tool calls if present and handlers provided
        if tool_calls and tool_handlers:
            for tool_call in tool_calls:
                if tool_call["name"] in tool_handlers:
                    result = await tool_handlers[tool_call["name"]](tool_call["arguments"])
                    
                    # Create tool result message
                    tool_result_message = Message(
                        chat_id=chat.id,
                        role=MessageRole.TOOL,
                        content=json.dumps(result),
                        content_type=ContentType.TOOL_RESULT,
                        tool_call_id=tool_call["id"],
                    )
                    db.add(tool_result_message)
        
        # Update chat stats
        chat.total_input_tokens += input_tokens
        chat.total_output_tokens += output_tokens
        chat.updated_at = datetime.now(timezone.utc)
        
        await db.flush()
        return assistant_message
    
    async def stream_message(
        self,
        db: AsyncSession,
        user: User,
        chat: Chat,
        user_message: str,
        attachments: Optional[List[Dict]] = None,
        tools: Optional[List[Dict]] = None,
        tool_executor: Optional[Callable] = None,
        parent_id: Optional[str] = None,  # For conversation branching - parent of the assistant message
        cancel_check: Optional[Callable[[], bool]] = None,  # Returns True if generation should be cancelled
        stream_setter: Optional[Callable[[Any], None]] = None,  # Callback to set active stream for cancellation
        is_file_content: bool = False,  # True when content is user-uploaded file (skip injection filters)
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a message response with prompt-based tool orchestration.
        
        Since the LLM may not support native function calling, we orchestrate tools via prompts:
        1. Ask LLM if it needs current/external information
        2. If yes, ask LLM to generate a search query
        3. Execute the tool ourselves
        4. Pass results back to LLM for summarization
        
        Args:
            parent_id: For new messages, this is the user_message.id.
                       For regeneration, this is the original user message ID.
            cancel_check: Optional callback that returns True if generation should stop.
            stream_setter: Optional callback to pass the stream object for external cancellation.
        """
        import logging
        from app.models.models import CustomAssistant
        logger = logging.getLogger(__name__)
        
        # Track timing for structured logging
        stream_start_time = time.time()
        tool_calls_count = 0
        filters_applied = []
        
        # Resolve the model to use
        # Handle "gpt:assistant_id" format - look up the assistant's actual model
        chat_model = chat.model
        if chat_model and chat_model.startswith("gpt:"):
            assistant_id = chat_model[4:]
            logger.debug(f"Resolving assistant model for {assistant_id}")
            result = await db.execute(
                select(CustomAssistant).where(CustomAssistant.id == assistant_id)
            )
            assistant = result.scalar_one_or_none()
            if assistant:
                chat_model = assistant.model
                logger.debug(f"Using assistant's model: {chat_model}")
            else:
                logger.warning(f"Assistant {assistant_id} not found, using default model")
                chat_model = None
        
        effective_model = await self._get_effective_model(chat_model)
        
        # Check if safety filters are enabled (admin setting)
        from app.services.settings_service import SettingsService
        safety_filters_enabled = await SettingsService.get_bool(db, "enable_safety_filters")
        
        # Get filter manager for this chat
        filter_manager = self._get_filter_manager(chat)
        # Pass is_file_content and safety_filters_enabled to control filter behavior
        extra_metadata = {}
        if is_file_content:
            extra_metadata["is_file_content"] = True
        if not safety_filters_enabled:
            extra_metadata["safety_filters_disabled"] = True
        
        context = self._create_filter_context(
            user, chat, 
            extra_metadata=extra_metadata if extra_metadata else None
        )
        
        # === OverrideToLLM: Filter user input before sending to LLM ===
        to_llm_result = await filter_manager.process_to_llm(user_message, context)
        
        if to_llm_result.blocked:
            yield {
                "type": "error",
                "error": f"Request blocked: {to_llm_result.block_reason}",
                "blocked": True,
            }
            return
        
        filtered_user_message = to_llm_result.content
        
        # Build messages from chat history (using tree structure)
        messages = await self._build_messages(db, chat, filtered_user_message, attachments, tools, parent_id)
        
        # Create placeholder assistant message
        assistant_message = Message(
            chat_id=chat.id,
            role=MessageRole.ASSISTANT,
            content="",
            content_type=ContentType.TEXT,
            is_streaming=True,
            model=effective_model,
            parent_id=parent_id,  # Link to user message for conversation branching
        )
        db.add(assistant_message)
        # CRITICAL: Commit immediately to ensure message exists before any child messages
        # are created. Without this, if the user sends another message before streaming
        # completes, the child would reference a non-existent parent (orphaned message).
        await db.commit()
        await db.refresh(assistant_message)
        
        yield {
            "type": "message_start",
            "message_id": str(assistant_message.id),
        }
        
        # Create stream context for filters
        stream_context = FilterContext(
            user_id=str(user.id),
            chat_id=str(chat.id),
            message_id=str(assistant_message.id),
            model=effective_model,
            metadata={
                "user_tier": user.tier.value if user.tier else "free",
            }
        )
        
        try:
            total_input_tokens = 0
            total_output_tokens = 0
            filtered_content = ""
            tool_results = []
            
            # Automatic search disabled - use filter chains for web search orchestration
            # search_tools = [t for t in (tools or []) if 'search' in t.get('name', '').lower() or 'fetch' in t.get('name', '').lower()]
            # has_search = bool(search_tools) and tool_executor is not None
            has_search = False  # Disabled - filter chains handle search orchestration
            
            if has_search:
                logger.debug(f"Search tools available: {[t.get('name') for t in search_tools]}")
                
                # Step 1: Ask LLM if it needs to search
                needs_search = await self._check_needs_search(effective_model, messages, filtered_user_message)
                logger.debug(f"LLM needs search: {needs_search}")
                
                if needs_search:
                    # Step 2: Get search query from LLM
                    search_query = await self._get_search_query(effective_model, messages, filtered_user_message)
                    logger.debug(f"Generated search query: {search_query}")
                    
                    if search_query:
                        # Step 3: Execute search tool
                        # Prefer tavily_search, then fetch_webpage
                        search_tool = None
                        for t in search_tools:
                            if 'tavily' in t.get('name', '').lower() and 'search' in t.get('name', '').lower():
                                search_tool = t
                                break
                        if not search_tool:
                            search_tool = search_tools[0]
                        
                        tool_name = search_tool.get('name')
                        logger.debug(f"Executing tool: {tool_name} with query: {search_query}")
                        
                        try:
                            # Build tool arguments
                            if 'tavily' in tool_name.lower():
                                tool_args = {"query": search_query}
                            elif 'fetch' in tool_name.lower():
                                tool_args = {"url": search_query}
                            else:
                                tool_args = {"query": search_query}
                            
                            result = await tool_executor(tool_name, tool_args)
                            
                            # Truncate result if too long
                            result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                            if len(result_str) > 15000:
                                result_str = result_str[:15000] + "... [truncated]"
                            
                            tool_results.append({
                                "tool": tool_name,
                                "query": search_query,
                                "result": result_str,
                            })
                            logger.debug(f"Tool result length: {len(result_str)}")
                            
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            tool_results.append({
                                "tool": tool_name,
                                "query": search_query,
                                "error": str(e),
                            })
            
            # Step 4: Generate final response (with or without tool results)
            if tool_results:
                # Add tool results to context
                tool_context = "\n\n---\nSEARCH RESULTS (use this information to answer the user's question):\n"
                for tr in tool_results:
                    if "error" in tr:
                        tool_context += f"\n[Search for '{tr['query']}' failed: {tr['error']}]\n"
                    else:
                        tool_context += f"\n[Search results for '{tr['query']}']:\n{tr['result']}\n"
                tool_context += "\n---\n\nBased on the search results above, provide a helpful, well-organized answer. Summarize the key points clearly."
                
                # Add to messages
                messages.append({
                    "role": "user",
                    "content": tool_context,
                })
            
            # Stream the final response
            api_params = {
                "model": effective_model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }
            
            stream = await self.client.chat.completions.create(**api_params)
            
            # Pass stream to handler for direct cancellation capability
            if stream_setter:
                stream_setter(stream)
                logger.debug(f"Stream set for cancellation: {type(stream)}")
            
            cancelled = False
            try:
                async for chunk in stream:
                    # Check for cancellation at start of each iteration
                    if cancel_check and cancel_check():
                        logger.info("Stream cancelled by user request - breaking loop")
                        cancelled = True
                        # Try to close the stream
                        try:
                            await stream.close()
                        except Exception:
                            pass
                        break
                    
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    if delta.content:
                        filtered_content += delta.content
                        yield {
                            "type": "text_delta",
                            "text": delta.content,
                        }
                    
                    if hasattr(chunk, 'usage') and chunk.usage:
                        total_input_tokens = chunk.usage.prompt_tokens or 0
                        total_output_tokens = chunk.usage.completion_tokens or 0
                    
                    if chunk.choices[0].finish_reason:
                        break
            except asyncio.CancelledError:
                # Task was cancelled - this is expected when stop is requested
                logger.info("Stream task cancelled (CancelledError)")
                cancelled = True
            except Exception as e:
                # Stream was forcibly closed (e.g., httpx.ResponseClosed)
                if cancel_check and cancel_check():
                    logger.info(f"Stream interrupted by user request: {type(e).__name__}")
                    cancelled = True
                else:
                    # Re-raise if not a cancellation
                    raise
            
            # If cancelled, still save partial content and notify
            if cancelled:
                # Use direct SQL update for robustness
                from sqlalchemy import update
                message_id = str(assistant_message.id)
                try:
                    await db.execute(
                        update(Message)
                        .where(Message.id == message_id)
                        .values(
                            content=filtered_content,
                            is_streaming=False,
                        )
                    )
                except Exception as update_err:
                    logger.warning(f"Failed to save cancelled message (may have been deleted): {update_err}")
                yield {
                    "type": "stream_cancelled",
                    "content": filtered_content,
                    "parent_id": parent_id,
                }
                return
            
            # Estimate tokens if not provided
            if total_input_tokens == 0:
                total_input_tokens = self._estimate_tokens(messages)
            if total_output_tokens == 0:
                total_output_tokens = self._estimate_tokens(filtered_content)
            
            # Use direct SQL update instead of ORM to avoid stale object issues
            # This is more robust during long streaming sessions
            from sqlalchemy import update
            message_id = str(assistant_message.id)
            chat_id = str(chat.id)
            
            try:
                # Update message with final content using direct SQL
                await db.execute(
                    update(Message)
                    .where(Message.id == message_id)
                    .values(
                        content=filtered_content,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        is_streaming=False,
                    )
                )
                
                # Update chat stats using direct SQL
                await db.execute(
                    update(Chat)
                    .where(Chat.id == chat_id)
                    .values(
                        total_input_tokens=Chat.total_input_tokens + total_input_tokens,
                        total_output_tokens=Chat.total_output_tokens + total_output_tokens,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                
                # Track usage (uses separate queries, should be safe)
                await self._track_usage(
                    db=db,
                    user=user,
                    chat=chat,
                    model=effective_model,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    message_id=message_id,
                )
                
            except Exception as update_err:
                logger.warning(f"Failed to update message (may have been deleted): {update_err}")
                # Don't rollback here - let the caller handle it
            
            # Log LLM request metrics
            duration_ms = (time.time() - stream_start_time) * 1000
            total_tokens = total_input_tokens + total_output_tokens
            log_llm_request(
                model=effective_model,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                duration_ms=duration_ms,
                user_id=str(user.id),
                chat_id=str(chat.id),
                filters_applied=filters_applied if filters_applied else None,
                tool_calls=tool_calls_count if tool_calls_count > 0 else None,
            )
            
            yield {
                "type": "message_end",
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "parent_id": parent_id,  # For conversation tree tracking
            }
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            # Use direct SQL for error update too
            message_id = str(assistant_message.id) if assistant_message else None
            if message_id:
                try:
                    await db.execute(
                        update(Message)
                        .where(Message.id == message_id)
                        .values(
                            content=f"Error: {str(e)}",
                            is_error=True,
                            is_streaming=False,
                        )
                    )
                except Exception as update_err:
                    logger.warning(f"Failed to update error message: {update_err}")
            
            yield {
                "type": "error",
                "error": str(e),
            }
    
    async def _check_needs_search(self, model: str, messages: List[Dict], user_query: str) -> bool:
        """Ask LLM if it needs to search for current information."""
        check_prompt = f"""You are deciding whether to search the web for information.

User's question: {user_query}

Can you answer this question accurately using only your training knowledge, or do you need current/real-time information from the web?

Reply with ONLY one word:
- "SEARCH" if you need to search the web for current information (news, prices, weather, recent events, etc.)
- "ANSWER" if you can answer accurately from your existing knowledge

Your response (one word only):"""

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": check_prompt}],
                max_tokens=10,
                temperature=0,
            )
            result = response.choices[0].message.content.strip().upper()
            return "SEARCH" in result
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Check needs search error: {e}")
            return False
    
    async def _get_search_query(self, model: str, messages: List[Dict], user_query: str) -> Optional[str]:
        """Ask LLM to generate a search query."""
        query_prompt = f"""Generate a concise web search query to find information for this question:

User's question: {user_query}

Write ONLY the search query, nothing else. Keep it short and focused (under 10 words).
Do not include quotes or special characters.

Search query:"""

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query_prompt}],
                max_tokens=50,
                temperature=0,
            )
            query = response.choices[0].message.content.strip()
            # Clean up the query
            query = query.strip('"\'`')
            query = query.replace('\n', ' ').strip()
            return query if query else None
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Get search query error: {e}")
            return None
    
    def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert tools to OpenAI function calling format"""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                }
            })
        return openai_tools
    
    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content"""
        if isinstance(content, str):
            # Rough estimate: ~4 chars per token
            return len(content) // 4
        elif isinstance(content, list):
            total = 0
            for msg in content:
                if isinstance(msg, dict):
                    content_val = msg.get("content", "")
                    if isinstance(content_val, str):
                        total += len(content_val) // 4
                    elif isinstance(content_val, list):
                        for item in content_val:
                            if isinstance(item, dict) and "text" in item:
                                total += len(item["text"]) // 4
            return total
        return 0
    
    async def _build_messages(
        self,
        db: AsyncSession,
        chat: Chat,
        user_message: str,
        attachments: Optional[List[Dict]] = None,
        tools: Optional[List[Dict]] = None,
        parent_id: Optional[str] = None,  # For tree-based conversation building
    ) -> List[Dict]:
        """
        Build message list from chat history in OpenAI format.
        
        Uses tree structure: if parent_id is provided, traces back from that message
        to root to build the conversation path. Otherwise falls back to chronological.
        """
        
        # Get ALL messages for this chat (we'll filter by tree structure)
        result = await db.execute(
            select(Message)
            .where(Message.chat_id == chat.id)
            .order_by(Message.created_at)
        )
        all_messages = {str(m.id): m for m in result.scalars().all()}
        
        # Build the conversation path
        if parent_id and parent_id in all_messages:
            # Trace back from parent_id to root
            path_ids = []
            current_id = parent_id
            while current_id:
                path_ids.append(current_id)
                msg = all_messages.get(current_id)
                current_id = msg.parent_id if msg else None
            
            # Reverse to get root-to-current order
            path_ids.reverse()
            history = [all_messages[mid] for mid in path_ids if mid in all_messages]
        else:
            # Fallback: use chronological order with selected_versions
            # This handles legacy chats without parent_id
            selected = chat.selected_versions or {}
            history = []
            
            # Group messages by parent_id
            children_by_parent = {}
            for msg in all_messages.values():
                pid = msg.parent_id or "root"
                if pid not in children_by_parent:
                    children_by_parent[pid] = []
                children_by_parent[pid].append(msg)
            
            # Sort children by created_at
            for children in children_by_parent.values():
                children.sort(key=lambda m: m.created_at)
            
            # Walk the tree from root
            def walk_tree(parent_id_key):
                children = children_by_parent.get(parent_id_key, [])
                if not children:
                    return
                
                # If there are multiple children (siblings), pick selected or newest
                if len(children) > 1:
                    selected_child_id = selected.get(parent_id_key)
                    if selected_child_id:
                        child = next((c for c in children if str(c.id) == selected_child_id), children[-1])
                    else:
                        child = children[-1]  # Default to newest
                else:
                    child = children[0]
                
                history.append(child)
                walk_tree(str(child.id))
            
            walk_tree("root")
        
        messages = []
        
        # Add system message
        system_prompt = chat.system_prompt or "You are a helpful AI assistant."
        
        # If tools are available, add guidance for when search results are provided
        if tools:
            tools_instruction = """

When you receive search results or external data in the conversation, follow these rules:
1. Summarize the information in natural, conversational prose
2. Do NOT show raw data, JSON, or code blocks to the user
3. Extract the key points and present them clearly
4. If multiple sources are provided, synthesize them into a coherent answer
5. Present information as your own knowledge - do not mention "search results" or "according to the data"
6. Be concise but comprehensive"""
            
            system_prompt = system_prompt + tools_instruction
        
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
        
        for msg in history:
            if msg.role == MessageRole.SYSTEM:
                continue  # Already handled above
            
            # Handle tool results
            if msg.content_type == ContentType.TOOL_RESULT:
                messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
                continue
            
            # Build message content
            if msg.role == MessageRole.USER:
                content = self._build_user_content(msg.content, msg.attachments)
                messages.append({
                    "role": "user",
                    "content": content,
                })
            
            elif msg.role == MessageRole.ASSISTANT:
                message_dict = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                
                # Add tool calls if present
                if msg.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                
                messages.append(message_dict)
        
        # Add current user message
        content = self._build_user_content(user_message, attachments)
        messages.append({
            "role": "user",
            "content": content,
        })
        
        return messages
    
    def _build_user_content(
        self,
        text: str,
        attachments: Optional[List[Dict]] = None,
    ) -> Any:
        """Build user message content, handling text and images"""
        
        if not attachments:
            return text
        
        # Multi-modal content
        content = []
        
        for attachment in attachments:
            if attachment.get("type") == "image":
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{attachment.get('mime_type', 'image/jpeg')};base64,{attachment.get('data')}",
                    }
                })
        
        content.append({
            "type": "text",
            "text": text,
        })
        
        return content
    
    async def _track_usage(
        self,
        db: AsyncSession,
        user: User,
        chat: Chat,
        model: str,
        input_tokens: int,
        output_tokens: int,
        message_id: Optional[str] = None,
    ):
        """Track token usage for billing"""
        now = datetime.now(timezone.utc)
        
        # Calculate costs (can be 0 for local models)
        input_cost = (input_tokens / 1_000_000) * settings.INPUT_TOKEN_PRICE
        output_cost = (output_tokens / 1_000_000) * settings.OUTPUT_TOKEN_PRICE
        total_cost = input_cost + output_cost
        
        usage = TokenUsage(
            user_id=user.id,
            chat_id=chat.id,
            message_id=message_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            year=now.year,
            month=now.month,
        )
        db.add(usage)
        
        # Update user's monthly usage
        user.tokens_used_this_month += input_tokens + output_tokens
        
        await db.flush()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the API"""
        try:
            models = await self.client.models.list()
            return [
                {
                    "id": model.id,
                    "owned_by": getattr(model, "owned_by", "unknown"),
                    "created": getattr(model, "created", None),
                }
                for model in models.data
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM API is accessible"""
        try:
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "api_base": settings.LLM_API_BASE_URL,
                "model_count": len(models.data),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_base": settings.LLM_API_BASE_URL,
                "error": str(e),
            }
