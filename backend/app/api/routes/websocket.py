"""
WebSocket endpoint for real-time features
"""
import asyncio
import uuid
from datetime import datetime, timezone, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, text
from sqlalchemy.orm import selectinload
import json
import logging
from typing import Optional

from app.db.database import async_session_maker
from app.services.auth import AuthService
from app.services.websocket import ws_manager, StreamingHandler
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.billing import BillingService
from app.services.procedural_memory import ProceduralMemoryService, get_procedural_context
from app.services.tool_service import ToolService
from app.services.image_gen import detect_image_request, detect_image_request_async
from app.tools.registry import tool_registry, store_session_file
from app.models.models import User, Chat, Message, MessageRole, ContentType, AssistantConversation, CustomAssistant
from app.core.config import settings
from app.core.logging import log_websocket_event
from app.api.ws_types import (
    create_stream_start, create_stream_chunk, create_stream_end,
    create_stream_error, create_error,
    WSStreamStart, WSStreamChunk, WSStreamEnd, WSStreamError,
    WSMessageSaved, WSToolCall, WSToolResult, WSImageGeneration,
)

logger = logging.getLogger(__name__)


router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for real-time features.
    
    Message types:
    - subscribe: Subscribe to a chat room
    - unsubscribe: Unsubscribe from a chat room
    - chat_message: Send a message to AI
    - client_message: Send client-to-client message
    - ping: Keep-alive ping
    """
    
    # Accept the connection first so we can send proper close codes
    await websocket.accept()
    
    # Authenticate after accepting
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    payload = AuthService.decode_token(token)
    if not payload or payload.get("type") != "access":
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    user_id = payload.get("sub")
    
    # Get user from database
    async with async_session_maker() as db:
        user = await AuthService.get_user_by_id(db, user_id)
        if not user or not user.is_active:
            await websocket.close(code=4001, reason="User not found or inactive")
            return
    
    # Register connection with manager (already accepted)
    connection = await ws_manager.connect(websocket, user_id, already_accepted=True)
    streaming_handler = StreamingHandler(ws_manager, connection)
    
    # Log connection
    log_websocket_event("connect", user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data[:100]}")
                await ws_manager.send_to_connection(connection, 
                    create_error("Invalid JSON")
                )
                continue
            
            msg_type = message.get("type")
            payload = message.get("payload", {})
            
            # Only log important message types (not ping)
            if msg_type != "ping":
                logger.debug(f"WS message: type={msg_type}")
            
            if msg_type == "ping":
                await ws_manager.send_to_connection(connection, {"type": "pong"})
            
            elif msg_type == "subscribe":
                chat_id = payload.get("chat_id")
                if chat_id:
                    await ws_manager.subscribe_to_chat(connection, chat_id)
                    await ws_manager.send_to_connection(connection, {
                        "type": "subscribed",
                        "payload": {"chat_id": chat_id},
                    })
            
            elif msg_type == "unsubscribe":
                chat_id = payload.get("chat_id")
                if chat_id:
                    await ws_manager.unsubscribe_from_chat(connection, chat_id)
                    await ws_manager.send_to_connection(connection, {
                        "type": "unsubscribed",
                        "payload": {"chat_id": chat_id},
                    })
            
            elif msg_type == "chat_message":
                save_msg = payload.get("save_user_message", True)
                content_len = len(payload.get("content", ""))
                logger.info(f"chat_message: chat={payload.get('chat_id')}, content_len={content_len}, save={save_msg}")
                
                # Debug: log first and last 100 chars to verify content integrity
                content = payload.get("content", "")
                if content_len > 200:
                    logger.debug(f"chat_message content head: {content[:100]}")
                    logger.debug(f"chat_message content tail: {content[-100:]}")
                
                await handle_chat_message(connection, streaming_handler, user_id, payload)
            
            elif msg_type == "regenerate_message":
                await handle_regenerate_message(connection, streaming_handler, user_id, payload)
            
            elif msg_type == "stop_generation":
                chat_id = payload.get("chat_id")
                logger.info(f"Stop generation requested for chat: {chat_id}")
                if chat_id:
                    # Set cancellation flag and close the stream
                    await streaming_handler.request_stop()
                    await ws_manager.send_to_connection(connection, {
                        "type": "stream_stopped",
                        "payload": {"chat_id": chat_id, "message": "Generation stopped by user"},
                    })
            
            elif msg_type == "client_message":
                await handle_client_message(connection, user_id, payload)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await ws_manager.send_to_connection(connection, {
                    "type": "error",
                    "payload": {"error": f"Unknown message type: {msg_type}"},
                })
    
    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for user {user_id}")
        log_websocket_event("disconnect", user_id)
    except Exception as e:
        logger.exception(f"WebSocket error for user {user_id}: {e}")
        log_websocket_event("error", user_id, error=str(e))
        await ws_manager.send_to_connection(connection, 
            create_error(str(e))
        )
    finally:
        await ws_manager.disconnect(connection)


async def handle_chat_message(
    connection,
    streaming_handler: StreamingHandler,
    user_id: str,
    payload: dict,
    save_user_message: bool = True,
):
    """Handle AI chat message with streaming"""
    
    chat_id = payload.get("chat_id")
    content = payload.get("content", "")
    enable_tools = payload.get("enable_tools", True)
    enable_rag = payload.get("enable_rag", False)
    document_ids = payload.get("document_ids")
    attachments = payload.get("attachments", [])
    parent_id = payload.get("parent_id")  # For conversation branching
    client_message_id = payload.get("message_id")  # Client-generated UUID for the user message
    
    # Log attachment info
    if attachments:
        logger.info(f"[ATTACHMENTS] Received {len(attachments)} attachments")
        for i, att in enumerate(attachments):
            att_type = att.get("type")
            filename = att.get("filename", "unknown")
            content_len = len(att.get("content", "")) if att.get("content") else 0
            data_len = len(att.get("data", "")) if att.get("data") else 0
            logger.info(f"[ATTACHMENTS] [{i}] type={att_type}, filename={filename}, content_len={content_len}, data_len={data_len}")
    
    # Store file attachments for tool access (partial viewing)
    if attachments:
        for att in attachments:
            if att.get("type") == "file" and att.get("filename") and att.get("content"):
                store_session_file(chat_id, att["filename"], att["content"])
                logger.info(f"[ATTACHMENTS] Stored for tools: {att['filename']} ({len(att['content'])} chars)")
    
    # Allow payload to override save_user_message (for file content continuation)
    if "save_user_message" in payload:
        save_user_message = payload.get("save_user_message", True)
    
    if not chat_id or not content:
        logger.warning(f"Missing chat_id or content")
        await ws_manager.send_to_connection(connection, {
            "type": "error",
            "payload": {"error": "Missing chat_id or content"},
        })
        return
    
    async with async_session_maker() as db:
        # Get user and chat
        user = await AuthService.get_user_by_id(db, user_id)
        
        result = await db.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        
        if not chat or (chat.owner_id != user_id):
            logger.warning(f"Chat not found or access denied")
            await ws_manager.send_to_connection(connection, {
                "type": "error",
                "payload": {"error": "Chat not found"},
            })
            return
        
        # Check usage limits
        billing = BillingService()
        limit_check = await billing.check_usage_limit(db, user, billing.estimate_tokens(content))
        
        if not limit_check["can_proceed"]:
            await ws_manager.send_to_connection(connection, {
                "type": "error",
                "payload": {"error": "Token limit exceeded. Please upgrade your plan."},
            })
            return
        
        # Check if this is an image generation request (with LLM confirmation)
        is_image_request, image_prompt = await detect_image_request_async(content, db)
        if is_image_request:
            logger.debug(f"Image generation request detected: {content[:100]}...")
            
            # Import queue service
            from app.services.image_queue import get_image_queue, ensure_queue_started, TaskStatus
            from app.services.image_gen import extract_size_from_text, extract_aspect_ratio_from_text
            
            queue = get_image_queue()
            
            # Check queue capacity
            if queue.is_full:
                await ws_manager.send_to_connection(connection, {
                    "type": "error",
                    "payload": {"error": "Image generation queue is full. Please wait for current tasks to complete."},
                })
                return
            
            # Extract size from prompt
            width, height = extract_size_from_text(content)
            if not width:
                width, height = extract_aspect_ratio_from_text(content)
            if not width:
                width, height = 1024, 1024
            
            # Save user message first (use client ID if provided)
            img_msg_kwargs = {
                "chat_id": chat_id,
                "sender_id": user_id,
                "role": MessageRole.USER,
                "content": content,
                "content_type": ContentType.TEXT,
                "attachments": attachments if attachments else None,
                "parent_id": parent_id,
            }
            if client_message_id:
                img_msg_kwargs["id"] = client_message_id
            
            user_message = Message(**img_msg_kwargs)
            db.add(user_message)
            await db.flush()
            
            await ws_manager.send_to_connection(connection, {
                "type": "message_saved",
                "payload": {
                    "message_id": user_message.id,
                    "parent_id": parent_id,
                },
            })
            
            # Create placeholder assistant message
            assistant_message = Message(
                chat_id=chat_id,
                sender_id=None,
                role=MessageRole.ASSISTANT,
                content="Generating image...",
                content_type=ContentType.TEXT,
                parent_id=user_message.id,
                message_metadata={
                    "image_generation": {
                        "status": "pending",
                        "prompt": image_prompt,
                        "width": width,
                        "height": height,
                        "queue_position": queue.get_pending_count() + 1,
                    }
                },
            )
            db.add(assistant_message)
            await db.commit()
            await db.refresh(assistant_message)
            
            message_id = assistant_message.id
            
            # Send stream_start to show placeholder (using typed model)
            await ws_manager.send_to_connection(connection, 
                create_stream_start(message_id, chat_id)
            )
            
            await ws_manager.send_to_connection(connection, 
                create_stream_chunk(message_id, "Generating image...", chat_id)
            )
            
            # Send generation started notification with queue info
            await ws_manager.send_to_connection(connection, {
                "type": "image_generation_started",
                "payload": {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "prompt": image_prompt,
                    "width": width,
                    "height": height,
                    "queue_position": queue.get_pending_count() + 1,
                },
            })
            
            await ws_manager.send_to_connection(connection, 
                create_stream_end(
                    message_id, chat_id, 
                    input_tokens=0, output_tokens=0, 
                    parent_id=user_message.id
                )
            )
            
            # Create callback to notify frontend when complete
            async def notify_completion():
                task = queue.get_task(task_id)
                if not task:
                    return
                
                try:
                    async with async_session_maker() as notify_db:
                        # First check if message still exists
                        result = await notify_db.execute(
                            text("SELECT id FROM messages WHERE id = :id"),
                            {"id": message_id}
                        )
                        if not result.fetchone():
                            logger.warning(f"Message {message_id} no longer exists, skipping image update")
                            return
                        
                        if task.status == TaskStatus.COMPLETED:
                            # Build metadata JSON
                            new_metadata = json.dumps({
                                "image_generation": {
                                    "status": "completed",
                                    "prompt": task.prompt,
                                    "width": task.width,
                                    "height": task.height,
                                    "seed": task.seed,
                                    "generation_time": task.generation_time,
                                    "job_id": task.job_id,
                                },
                                "generated_image": {
                                    "url": task.image_url,
                                    "width": task.width,
                                    "height": task.height,
                                    "seed": task.seed,
                                    "prompt": task.prompt,
                                    "job_id": task.job_id,
                                }
                            })
                            
                            # Use raw SQL to avoid SQLAlchemy metadata naming conflict
                            await notify_db.execute(
                                text("UPDATE messages SET content = :content, message_metadata = :metadata WHERE id = :id"),
                                {
                                    "content": "Here's the image I generated based on your request:",
                                    "metadata": new_metadata,
                                    "id": message_id,
                                }
                            )
                            
                            # Also update chat timestamp
                            await notify_db.execute(
                                text("UPDATE chats SET updated_at = :now WHERE id = :id"),
                                {"now": datetime.now(timezone.utc), "id": chat_id}
                            )
                            await notify_db.commit()
                            
                            # Notify frontend
                            image_data = {
                                "url": task.image_url,
                                "width": task.width,
                                "height": task.height,
                                "seed": task.seed,
                                "prompt": task.prompt,
                                "job_id": task.job_id,
                            }
                            
                            await ws_manager.send_to_user(user_id, {
                                "type": "image_generated",
                                "payload": {
                                    "chat_id": chat_id,
                                    "message_id": message_id,
                                    "image": image_data,
                                },
                            })
                        
                        else:
                            # Build error metadata JSON
                            error_metadata = json.dumps({
                                "image_generation": {
                                    "status": "failed",
                                    "error": task.error,
                                    "prompt": task.prompt,
                                }
                            })
                            
                            await notify_db.execute(
                                text("UPDATE messages SET content = :content, message_metadata = :metadata WHERE id = :id"),
                                {
                                    "content": f"I wasn't able to generate the image: {task.error or 'Unknown error'}",
                                    "metadata": error_metadata,
                                    "id": message_id,
                                }
                            )
                            await notify_db.commit()
                            
                            await ws_manager.send_to_user(user_id, {
                                "type": "image_generation_failed",
                                "payload": {
                                    "chat_id": chat_id,
                                    "message_id": message_id,
                                    "error": task.error,
                                },
                            })
                
                except Exception as e:
                    logger.error(f"Error in image notify_completion: {e}")
            
            # Ensure queue is started
            await ensure_queue_started()
            
            # Add task to queue
            task_id = await queue.add_task(
                prompt=image_prompt,
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id,
                width=width,
                height=height,
                notify_callback=notify_completion,
            )
            
            logger.debug(f"Added image task {task_id} to queue for message {message_id}")
            
            return  # Image request queued, don't continue to LLM
        
        # Determine the parent for the assistant message
        assistant_parent_id = parent_id  # For regenerate: parent is the user message
        
        # Validate parent_id belongs to this chat (prevent cross-chat references)
        if parent_id:
            parent_check = await db.execute(
                select(Message).where(Message.id == parent_id, Message.chat_id == chat_id)
            )
            if not parent_check.scalar_one_or_none():
                logger.warning(f"parent_id {parent_id} does not belong to chat {chat_id}, ignoring")
                parent_id = None
                assistant_parent_id = None
        
        # Create user message only if save_user_message is True
        if save_user_message:
            # Build message kwargs, using client ID if provided
            msg_kwargs = {
                "chat_id": chat_id,
                "sender_id": user_id,
                "role": MessageRole.USER,
                "content": content,
                "content_type": ContentType.TEXT,
                "attachments": attachments if attachments else None,
                "parent_id": parent_id,  # User message follows the previously shown assistant message
            }
            if client_message_id:
                msg_kwargs["id"] = client_message_id
            
            user_message = Message(**msg_kwargs)
            db.add(user_message)
            await db.flush()
            logger.debug(f"User message saved: {user_message.id} with parent_id={parent_id}")
            
            # Assistant message will be child of this user message
            assistant_parent_id = user_message.id
            
            # Send confirmation with parent_id for frontend tree tracking
            await ws_manager.send_to_connection(connection, {
                "type": "message_saved",
                "payload": {
                    "message_id": user_message.id,
                    "parent_id": parent_id,
                },
            })
        else:
            # File content continuation - don't create user message
            logger.info(f"File content continuation mode (save_user_message=False), content_len={len(content)}, assistant_parent_id={assistant_parent_id}")
            # Keep assistant_parent_id as-is for tree structure
        
        # ==== FILTER CHAIN EXECUTION ====
        # Run enabled filter chains on the user message before LLM processing
        from app.filters.manager import get_chain_manager
        from app.filters.executor import ChainExecutor
        
        chain_manager = get_chain_manager()
        enabled_chains = chain_manager.get_enabled_chains()
        
        # Track if any chain modified the content
        original_content = content
        filter_context_items = []
        
        if enabled_chains:
            # Pre-load MCP/OpenAPI tools for filter chains
            mcp_tools_cache = {}
            
            try:
                # Load enabled tools from database
                from app.models.models import Tool
                query = select(Tool).where(Tool.is_enabled == True)
                # For filter chains (admin-defined), use all enabled tools
                # Non-admins would only see public tools in normal chat
                if not user.is_admin:
                    query = query.where(Tool.is_public == True)
                tools_result = await db.execute(query)
                for tool in tools_result.scalars().all():
                    mcp_tools_cache[tool.name] = tool
                    logger.debug(f"Loaded MCP tool for filter chains: {tool.name}")
            except Exception as e:
                logger.debug(f"No MCP tools loaded for filter chains: {e}")
            
            if mcp_tools_cache:
                logger.debug(f"MCP tools cache has {len(mcp_tools_cache)} tools: {list(mcp_tools_cache.keys())}")
            
            # Create tool service for MCP tool execution
            tool_service = ToolService()
            
            # Create executor with LLM and tool execution functions
            async def filter_llm_func(prompt: str, system: str = None) -> str:
                """Simple LLM call for filter chain decisions."""
                llm_svc = await LLMService.from_database(db)
                result = await llm_svc.simple_completion(prompt, system)
                return result
            
            async def filter_tool_func(tool_name: str, params: dict) -> str:
                """Tool execution for filter chains (supports built-in and MCP/OpenAPI tools)."""
                logger.debug(f"filter_tool_func called with tool_name='{tool_name}', params={params}")
                
                # Check if it's an MCP/OpenAPI tool (format: "ToolName:operation" or "ToolName")
                if ":" in tool_name:
                    parts = tool_name.split(":", 1)
                    mcp_tool_name = parts[0]
                    operation = parts[1] if len(parts) > 1 else None
                    logger.debug(f"Parsed as MCP tool: name='{mcp_tool_name}', operation='{operation}'")
                    
                    if mcp_tool_name in mcp_tools_cache:
                        tool_db = mcp_tools_cache[mcp_tool_name]
                        logger.debug(f"Found MCP tool in cache: {tool_db.name} (type={tool_db.tool_type})")
                        try:
                            result = await tool_service.execute_tool(
                                db=db,
                                tool=tool_db,
                                tool_name=operation or tool_name,
                                params=params,
                                user_id=user.id,
                                chat_id=chat_id,
                                message_id=None,
                            )
                            logger.debug(f"MCP tool result: {str(result)[:200]}")
                            return str(result) if result else ""
                        except Exception as e:
                            logger.error(f"MCP tool execution error: {e}")
                            return f"Error: {e}"
                    else:
                        logger.warning(f"MCP tool '{mcp_tool_name}' not found in cache. Available: {list(mcp_tools_cache.keys())}")
                        return f"Error: MCP tool '{mcp_tool_name}' not found"
                
                # Check if it's an MCP tool without operation (just tool name)
                if tool_name in mcp_tools_cache:
                    tool_db = mcp_tools_cache[tool_name]
                    logger.debug(f"Found MCP tool (no op) in cache: {tool_db.name}")
                    try:
                        result = await tool_service.execute_tool(
                            db=db,
                            tool=tool_db,
                            tool_name=tool_name,
                            params=params,
                            user_id=user.id,
                            chat_id=chat_id,
                            message_id=None,
                        )
                        return str(result) if result else ""
                    except Exception as e:
                        logger.error(f"MCP tool execution error: {e}")
                        return f"Error: {e}"
                
                # Fall back to built-in tool registry
                logger.debug(f"Trying built-in tool registry for: {tool_name}")
                try:
                    result = await tool_registry.execute(
                        tool_name,
                        params,
                        {"db": db, "user": user, "chat_id": chat_id},
                    )
                    return str(result) if result else ""
                except Exception as e:
                    logger.warning(f"Built-in tool '{tool_name}' not found: {e}")
                    return f"Error: Unknown tool '{tool_name}'"
            
            executor = ChainExecutor(
                llm_func=filter_llm_func,
                tool_func=filter_tool_func,
            )
            
            for chain in enabled_chains:
                try:
                    result = await executor.execute(
                        chain_def=chain,
                        query=content,
                        user_id=user_id,
                        chat_id=chat_id,
                    )
                    
                    # Update content if modified
                    if result.modified and result.content:
                        content = result.content
                    
                    # Collect context items
                    if result.context and result.context.context_items:
                        filter_context_items.extend(result.context.context_items)
                    
                    # Check if we should proceed to LLM
                    if not result.proceed_to_llm:
                        # Chain decided not to proceed to LLM
                        # Send the result directly to the user
                        if result.content:
                            # Create assistant message with chain result
                            assistant_msg = Message(
                                chat_id=chat_id,
                                sender_id=None,
                                role=MessageRole.ASSISTANT,
                                content=result.content,
                                content_type=ContentType.TEXT,
                                parent_id=assistant_parent_id,
                            )
                            db.add(assistant_msg)
                            await db.commit()
                            
                            # Send to frontend using typed models
                            await ws_manager.send_to_connection(connection, 
                                create_stream_start(assistant_msg.id, chat_id)
                            )
                            await ws_manager.send_to_connection(connection, 
                                create_stream_chunk(assistant_msg.id, result.content, chat_id)
                            )
                            await ws_manager.send_to_connection(connection, {
                                "type": "stream_end",
                                "payload": {
                                    "chat_id": chat_id,
                                    "message_id": assistant_msg.id,
                                    "from_filter_chain": chain.get("name", "unknown"),
                                },
                            })
                        
                        return  # Don't proceed to LLM
                    
                except Exception as e:
                    logger.error(f"Filter chain '{chain.get('name', 'unknown')}' error: {e}")
                    # Continue to LLM on error
        
        # Get system prompt from chat, admin settings, or config default
        from app.api.routes.admin import get_system_setting
        default_prompt = await get_system_setting(db, "default_system_prompt")
        system_prompt = chat.system_prompt or default_prompt
        
        # Check if this chat is associated with an assistant (for RAG auto-enable)
        assistant_result = await db.execute(
            select(AssistantConversation)
            .where(AssistantConversation.chat_id == chat_id)
        )
        assistant_conv = assistant_result.scalar_one_or_none()
        assistant_ks_ids = []
        
        if assistant_conv:
            # Get the assistant's knowledge stores
            assistant_result = await db.execute(
                select(CustomAssistant)
                .where(CustomAssistant.id == assistant_conv.assistant_id)
                .options(selectinload(CustomAssistant.knowledge_stores))
            )
            assistant = assistant_result.scalar_one_or_none()
            
            if assistant and assistant.knowledge_stores:
                assistant_ks_ids = [str(ks.id) for ks in assistant.knowledge_stores]
                # Auto-enable RAG for assistants with knowledge stores
                enable_rag = True
                logger.info(f"Auto-enabled RAG for assistant with knowledge stores: {assistant_ks_ids}")
        
        # Get RAG context if enabled (or auto-enabled for assistants)
        if enable_rag:
            rag_service = RAGService()
            
            if assistant_ks_ids:
                # Use assistant's knowledge stores (allows private KB access through public GPT)
                context = await rag_service.get_knowledge_store_context(
                    db=db,
                    user=user,
                    query=content,
                    knowledge_store_ids=assistant_ks_ids,
                    bypass_access_check=True,  # Allow access through assistant
                )
                logger.debug(f"Using assistant knowledge stores for chat {chat_id}: {assistant_ks_ids}")
            else:
                # Regular user documents
                context = await rag_service.get_context_for_query(
                    db=db,
                    user=user,
                    query=content,
                    document_ids=document_ids,
                )
            
            if context:
                rag_prompt = await get_system_setting(db, "rag_context_prompt")
                system_prompt = f"{system_prompt}\n\n{rag_prompt}\n\n{context}"
        
        # Get Procedural Memory context (learned skills)
        enable_procedural_memory = payload.get("enable_procedural_memory", settings.PROCEDURAL_MEMORY_ENABLED)
        if enable_procedural_memory:
            try:
                procedural_context = await get_procedural_context(str(user.id), content)
                if procedural_context:
                    system_prompt = f"{system_prompt}\n\n{procedural_context}"
                    logger.debug(f"Injected procedural memory context for chat {chat_id}")
            except Exception as e:
                logger.warning(f"Failed to get procedural memory context: {e}")
        
        # Inject zip manifest if provided in payload (preferred) or from DB (fallback)
        zip_context = payload.get("zip_context")
        if zip_context:
            system_prompt = f"{system_prompt}\n\n{zip_context}"
            logger.debug(f"Injected zip context from payload for chat {chat_id} ({len(zip_context)} chars)")
        else:
            # Fallback to database
            from app.api.routes.chats import get_zip_manifest_from_db
            zip_manifest = await get_zip_manifest_from_db(db, chat_id)
            if zip_manifest:
                system_prompt = f"{system_prompt}\n\n{zip_manifest}"
                logger.debug(f"Injected zip manifest from DB for chat {chat_id} ({len(zip_manifest)} chars)")
            else:
                logger.debug(f"No zip context found for chat {chat_id}")
        
        # Update chat system prompt temporarily in memory for LLM call
        # We don't want to persist this change, just use it for the LLM request
        chat.system_prompt = system_prompt
        
        # Expunge chat from session so SQLAlchemy doesn't track/flush our temporary change
        # We'll use direct SQL for any actual updates later
        db.expunge(chat)
        
        # Get built-in tools
        tools = tool_registry.get_tool_definitions() if enable_tools else []
        
        # Load MCP/OpenAPI tools from database
        mcp_tool_map = {}  # Maps tool name to (tool_db_record, operation_name)
        if enable_tools:
            try:
                tool_service = ToolService()
                mcp_tools, mcp_map = await tool_service.get_tools_for_llm(db, user)
                tools = (tools or []) + mcp_tools
                mcp_tool_map = mcp_map
                if mcp_tools:
                    logger.debug(f"Loaded {len(mcp_tools)} MCP/OpenAPI tools")
            except Exception as e:
                logger.debug(f"No MCP tools available: {e}")
        
        logger.debug(f"Tools enabled: {enable_tools}, tool_count: {len(tools) if tools else 0}")
        
        # Stream response (using database settings)
        llm = await LLMService.from_database(db)
        logger.debug(f"Starting LLM stream for chat {chat_id}")
        
        # Create tool executor function
        async def execute_tool(tool_name: str, arguments: dict):
            """Execute a tool and return result"""
            # Check if it's an MCP/OpenAPI tool
            if tool_name in mcp_tool_map:
                tool_db, operation = mcp_tool_map[tool_name]
                tool_service = ToolService()
                result = await tool_service.execute_tool(
                    db=db,
                    tool=tool_db,
                    tool_name=operation or tool_name,
                    params=arguments,
                    user_id=user.id,
                    chat_id=chat_id,
                    message_id=streaming_handler._current_message_id,
                )
            else:
                # Built-in tool
                result = await tool_registry.execute(
                    tool_name,
                    arguments,
                    {"db": db, "user": user, "chat_id": chat_id},
                )
            
            logger.debug(f"Tool executed, result length: {len(str(result))}")
            return result
        
        try:
            # Log content being sent to LLM
            logger.info(f"[LLM_CALL] Sending to LLM: content_len={len(content)}, chat={chat_id}")
            if len(content) > 200:
                logger.debug(f"[LLM_CALL] Content head: {content[:100]}")
                logger.debug(f"[LLM_CALL] Content tail: {content[-100:]}")
            
            async for event in llm.stream_message(
                db=db,
                user=user,
                chat=chat,
                user_message=content,
                attachments=attachments,
                tools=tools,
                tool_executor=execute_tool if tools else None,
                parent_id=assistant_parent_id,  # For conversation branching
                cancel_check=streaming_handler.is_stop_requested,  # Allow LLM to check cancellation
                stream_setter=streaming_handler.set_active_stream,  # Allow direct stream cancellation
                is_file_content=not save_user_message,  # Skip injection filter for user-uploaded files
            ):
                # Check if stop was requested (backup check)
                if streaming_handler.is_stop_requested():
                    logger.debug(f"Stopping generation for chat {chat_id}")
                    # Send stream_stopped with partial content
                    await ws_manager.send_to_connection(connection, {
                        "type": "stream_stopped",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "message": "Generation stopped by user",
                        },
                    })
                    streaming_handler.reset_stop()
                    break
                
                event_type = event.get("type")
                logger.debug(f"LLM event: {event_type}")
                
                if event_type == "message_start":
                    await streaming_handler.start_stream(event["message_id"], chat_id)
                
                elif event_type == "text_delta":
                    await streaming_handler.send_chunk(event["text"])
                
                elif event_type == "message_end":
                    logger.debug(f"LLM stream complete: input={event.get('input_tokens', 0)}, output={event.get('output_tokens', 0)}")
                    
                    # CRITICAL: Commit the assistant message BEFORE sending stream_end
                    # This ensures the message is visible to any continuation requests
                    # (e.g., file content requests that immediately follow)
                    try:
                        await db.commit()
                        logger.debug(f"Committed assistant message to DB")
                    except Exception as commit_err:
                        logger.warning(f"Failed to commit before stream_end: {commit_err}")
                        # Don't fail - the message may still be usable
                    
                    await streaming_handler.end_stream(
                        event.get("input_tokens", 0),
                        event.get("output_tokens", 0),
                        event.get("parent_id"),  # For conversation tree tracking
                    )
                
                elif event_type == "stream_cancelled":
                    logger.debug(f"LLM stream cancelled for chat {chat_id}")
                    # Commit any partial content before notifying frontend
                    try:
                        await db.commit()
                    except Exception:
                        pass
                    # Send stopped notification (LLM already saved partial content)
                    await ws_manager.send_to_connection(connection, {
                        "type": "stream_stopped",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "message": "Generation stopped by user",
                            "parent_id": event.get("parent_id"),
                        },
                    })
                    streaming_handler.reset_stop()
                
                elif event_type == "error":
                    logger.error(f"LLM error: {event['error']}")
                    await streaming_handler.send_error(event["error"])
        except asyncio.CancelledError:
            logger.debug(f"Streaming task cancelled for chat {chat_id}")
            await ws_manager.send_to_connection(connection, {
                "type": "stream_stopped",
                "payload": {
                    "chat_id": chat_id,
                    "message_id": streaming_handler._current_message_id,
                    "message": "Generation stopped by user",
                },
            })
            streaming_handler.reset_stop()
        except Exception as e:
            logger.exception(f"Exception during LLM streaming: {e}")
            await streaming_handler.send_error(str(e))
            # Rollback to clear any pending transaction issues
            try:
                await db.rollback()
            except Exception:
                pass
        
        # Note: chat was expunged from session, so we use direct SQL for any updates
        
        # Learn skill from successful interaction (async, non-blocking)
        if enable_procedural_memory:
            try:
                # Get the assistant message content
                msg_result = await db.execute(
                    select(Message.content, Message.id)
                    .where(Message.chat_id == chat_id)
                    .where(Message.role == MessageRole.ASSISTANT)
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
                msg_row = msg_result.first()
                
                if msg_row and msg_row.content and len(msg_row.content) > 50:
                    # Only learn from substantial responses
                    model_used = await llm._get_effective_model() if llm else None
                    asyncio.create_task(
                        ProceduralMemoryService.learn_skill_from_interaction(
                            user_id=str(user.id),
                            input_text=content,
                            output_text=msg_row.content,
                            chat_id=chat_id,
                            message_id=str(msg_row.id),
                            model_used=model_used,
                            quality_score=0.8,  # Default quality, updated by feedback
                        )
                    )
                    logger.debug(f"Queued skill learning from chat {chat_id}")
            except Exception as skill_err:
                logger.debug(f"Skill learning skipped: {skill_err}")
        
        # Auto-generate title using LLM
        # Check current title with direct SQL
        try:
            result = await db.execute(select(Chat.title).where(Chat.id == chat_id))
            current_title = result.scalar_one_or_none()
        except Exception as e:
            logger.warning(f"Failed to check chat title (session may have been rolled back): {e}")
            current_title = None
        
        if current_title == "New Chat":
            try:
                generated_title = await generate_chat_title(llm, content, db)
                await db.execute(
                    update(Chat).where(Chat.id == chat_id).values(title=generated_title)
                )
                logger.debug(f"Generated title for chat {chat_id}: {generated_title}")
                
                # Notify frontend about title update
                await ws_manager.send_to_connection(connection, {
                    "type": "chat_updated",
                    "payload": {
                        "chat_id": chat_id,
                        "title": generated_title,
                    }
                })
            except Exception as e:
                logger.warning(f"Failed to generate title: {e}")
                fallback_title = content[:50] + ("..." if len(content) > 50 else "")
                await db.execute(
                    update(Chat).where(Chat.id == chat_id).values(title=fallback_title)
                )
                # Still notify with fallback title
                await ws_manager.send_to_connection(connection, {
                    "type": "chat_updated",
                    "payload": {
                        "chat_id": chat_id,
                        "title": fallback_title,
                    }
                })
        
        try:
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to commit at end of handle_chat_message: {e}")
            await db.rollback()


async def handle_regenerate_message(
    connection,
    streaming_handler: StreamingHandler,
    user_id: str,
    payload: dict,
):
    """Handle regenerating an AI response without saving a new user message"""
    logger.debug(f"handle_regenerate_message: chat_id={payload.get('chat_id')}")
    # Reuse handle_chat_message but skip saving user message
    await handle_chat_message(
        connection,
        streaming_handler,
        user_id,
        payload,
        save_user_message=False,
    )


async def generate_chat_title(llm: LLMService, first_message: str, db: AsyncSession) -> str:
    """Generate a chat title using the LLM."""
    from app.core.config import settings
    from app.api.routes.admin import get_system_setting
    import re
    
    try:
        # Get title prompt from admin settings or use default
        title_prompt = await get_system_setting(db, "title_generation_prompt")
        
        # Get effective model (resolves "default" to actual model)
        effective_model = await llm._get_effective_model()
        
        response = await llm.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": title_prompt},
                {"role": "user", "content": first_message[:500]},  # Limit input
            ],
            max_tokens=30,
            temperature=0.7,
        )
        
        title = response.choices[0].message.content.strip()
        
        # Clean up the title
        title = title.strip('"\'`')  # Remove quotes and backticks
        title = re.sub(r'^(Title:|Chat:|Topic:)\s*', '', title, flags=re.IGNORECASE)  # Remove common prefixes
        title = re.sub(r'\*+', '', title)  # Remove markdown bold/italic
        title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
        title = title[:60]  # Limit length
        
        # Don't return empty or very short titles
        if len(title) < 3:
            return first_message[:50] + ("..." if len(first_message) > 50 else "")
            
        return title
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return first_message[:50] + ("..." if len(first_message) > 50 else "")


async def handle_client_message(
    connection,
    user_id: str,
    payload: dict,
):
    """Handle client-to-client message"""
    
    chat_id = payload.get("chat_id")
    content = payload.get("content", "")
    attachments = payload.get("attachments", [])
    
    if not chat_id or not content:
        await ws_manager.send_to_connection(connection, {
            "type": "error",
            "payload": {"error": "Missing chat_id or content"},
        })
        return
    
    async with async_session_maker() as db:
        # Verify user has access to chat
        result = await db.execute(select(Chat).where(Chat.id == chat_id))
        chat = result.scalar_one_or_none()
        
        if not chat or not chat.is_shared:
            await ws_manager.send_to_connection(connection, {
                "type": "error",
                "payload": {"error": "Chat not found or not a shared chat"},
            })
            return
        
        # Create message
        msg = Message(
            chat_id=chat_id,
            sender_id=user_id,
            role=MessageRole.USER,
            content=content,
            content_type=ContentType.TEXT,
            attachments=attachments if attachments else None,
        )
        db.add(msg)
        await db.commit()
        await db.refresh(msg)
        
        # Broadcast to other participants
        await ws_manager.broadcast_to_chat(
            chat_id,
            {
                "type": "client_message",
                "payload": {
                    "message_id": msg.id,
                    "chat_id": chat_id,
                    "sender_id": user_id,
                    "content": content,
                    "attachments": attachments,
                    "created_at": msg.created_at.isoformat(),
                },
            },
            exclude_user=user_id,
        )
        
        # Confirm to sender
        await ws_manager.send_to_connection(connection, {
            "type": "message_sent",
            "payload": {"message_id": msg.id},
        })
