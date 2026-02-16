"""
WebSocket endpoint for real-time features
"""
import asyncio
import uuid
import re
from datetime import datetime, timezone, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, text
from sqlalchemy.orm import selectinload
import json
import logging
from typing import Optional, List, Tuple

from app.db.database import async_session_maker
from app.services.auth import AuthService
from app.services.websocket import ws_manager, StreamingHandler
from app.services.llm import LLMService
from app.services.rag import RAGService
from app.services.billing import BillingService
from app.services.procedural_memory import ProceduralMemoryService, get_procedural_context
from app.services.tool_service import ToolService
from app.services.image_gen import detect_image_request, detect_image_request_async
from app.services.task_queue import TaskQueueService
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

# Browser proxy: pending fetch requests awaiting browser response
_browser_fetch_pending: dict[str, asyncio.Future] = {}


router = APIRouter(tags=["WebSocket"])


async def browser_fetch(connection, url: str, timeout: float = 30.0) -> dict:
    """Request the user's browser to fetch a URL and return the content.
    Used as a proxy for sites that block server-side requests (e.g. YouTube).
    """
    req_id = str(uuid.uuid4())[:8]
    future = asyncio.get_event_loop().create_future()
    _browser_fetch_pending[req_id] = future
    
    try:
        await ws_manager.send_to_connection(connection, {
            "type": "browser_fetch_request",
            "payload": {"request_id": req_id, "url": url},
        })
        result = await asyncio.wait_for(future, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        return {"error": "Browser fetch timed out"}
    finally:
        _browser_fetch_pending.pop(req_id, None)


# YouTube URL patterns for detection
YOUTUBE_URL_PATTERNS = [
    r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
    r'https?://youtu\.be/([a-zA-Z0-9_-]+)',
    r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
    r'https?://(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)',
]


def extract_youtube_urls(text: str) -> List[Tuple[str, str]]:
    """Extract YouTube URLs and their video IDs from text.
    
    Returns list of tuples: (full_url, video_id)
    """
    results = []
    seen_ids = set()
    
    for pattern in YOUTUBE_URL_PATTERNS:
        for match in re.finditer(pattern, text):
            video_id = match.group(1)
            if video_id not in seen_ids:
                seen_ids.add(video_id)
                results.append((match.group(0), video_id))
    
    return results


async def fetch_youtube_context(video_ids: List[str], connection=None, chat_id: str = None, db=None) -> str:
    """Fetch YouTube video subtitles via youtube-transcript-api with optional proxy."""
    if not video_ids:
        return ""
    
    context_parts = []
    context_dict = {}
    if chat_id:
        context_dict["chat_id"] = chat_id
    if db:
        context_dict["db"] = db
    
    for video_id in video_ids[:3]:
        try:
            result = await tool_registry._fetch_youtube_subtitles(video_id, 'en', context_dict)
            if result.get("success") and result.get("transcript"):
                transcript = result["transcript"]
                if len(transcript) > 15000:
                    transcript = transcript[:15000] + "... [truncated]"
                context_parts.append(f"[YouTube Video: {video_id}]\nhttps://www.youtube.com/watch?v={video_id}\nTranscript:\n{transcript}\n")
                logger.info(f"[YOUTUBE] Transcript for {video_id}: {len(transcript)} chars")
            else:
                error = result.get("error", "Unknown error")
                logger.info(f"[YOUTUBE] No transcript for {video_id}: {error}")
        except Exception as e:
            logger.warning(f"[YOUTUBE] Error for {video_id}: {e}")
    
    return "\n---\n".join(context_parts) if context_parts else ""


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
            
            elif msg_type == "browser_fetch_response":
                # Browser completed a fetch request on our behalf
                req_id = payload.get("request_id")
                if req_id and req_id in _browser_fetch_pending:
                    _browser_fetch_pending[req_id].set_result(payload)
            
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
                # Debug: log if save_user_message was explicitly in payload
                has_save_key = "save_user_message" in payload
                logger.info(f"chat_message: chat={payload.get('chat_id')}, content_len={content_len}, save={save_msg}, explicit_in_payload={has_save_key}")
                
                # Debug: log first and last 100 chars to verify content integrity
                content = payload.get("content", "")
                if content_len > 200:
                    logger.debug(f"chat_message content head: {content[:100]}")
                    logger.debug(f"chat_message content tail: {content[-100:]}")
                
                # Stop any existing stream before starting new one
                # This prevents content from two concurrent requests mixing together
                if streaming_handler.is_streaming():
                    logger.warning("Stopping existing stream before starting new chat message")
                    await streaming_handler.request_stop()
                
                # Run in background task so we can still receive stop_generation
                async def run_chat_message():
                    try:
                        await handle_chat_message(connection, streaming_handler, user_id, payload)
                    except asyncio.CancelledError:
                        logger.info("Chat message task cancelled by user")
                    except Exception as e:
                        logger.exception(f"Error in chat message handler: {e}")
                        try:
                            await ws_manager.send_to_connection(connection, {
                                "type": "stream_error",
                                "payload": {"error": str(e)},
                            })
                        except Exception:
                            pass
                
                task = asyncio.create_task(run_chat_message())
                streaming_handler.set_streaming_task(task)
            
            elif msg_type == "regenerate_message":
                # Run in background task so we can still receive stop_generation
                async def run_regenerate():
                    try:
                        await handle_regenerate_message(connection, streaming_handler, user_id, payload)
                    except asyncio.CancelledError:
                        logger.info("Regenerate task cancelled by user")
                    except Exception as e:
                        logger.exception(f"Error in regenerate handler: {e}")
                        try:
                            await ws_manager.send_to_connection(connection, {
                                "type": "stream_error",
                                "payload": {"error": str(e)},
                            })
                        except Exception:
                            pass
                
                task = asyncio.create_task(run_regenerate())
                streaming_handler.set_streaming_task(task)
            
            elif msg_type == "stop_generation":
                chat_id = payload.get("chat_id")
                logger.info(f"Stop generation requested for chat: {chat_id}")
                if chat_id:
                    # Set cancellation flag and close the stream IMMEDIATELY
                    await streaming_handler.request_stop()
                    
                    # Send confirmation to client
                    await ws_manager.send_to_connection(connection, {
                        "type": "stream_stopped",
                        "payload": {"chat_id": chat_id, "message": "Generation stopped by user"},
                    })
            
            elif msg_type == "client_message":
                await handle_client_message(connection, user_id, payload)
            
            elif msg_type == "mermaid_error":
                await handle_mermaid_error(connection, streaming_handler, user_id, payload)
            
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
    
    # Debug: log payload keys and save_user_message sources
    logger.info(f"[PAYLOAD_DEBUG] keys={list(payload.keys())}, 'save_user_message' in payload={'save_user_message' in payload}, func_param={save_user_message}")
    
    logger.info(f"[CONTENT_TRACE] handle_chat_message entry: content='{content[:50] if content else 'EMPTY'}...' len={len(content)}, save_user_message={save_user_message}")
    
    enable_tools = payload.get("enable_tools", True)
    enable_rag = payload.get("enable_rag", False)
    document_ids = payload.get("document_ids")
    attachments = payload.get("attachments", [])
    parent_id = payload.get("parent_id")  # For conversation branching
    client_message_id = payload.get("message_id")  # Client-generated UUID for the user message
    user_timezone = payload.get("timezone")  # NC-0.8.0.13: Browser timezone (e.g. "America/Chicago")
    
    # NC-0.8.0.8: Reorganized tool categories into logical sections
    # Maps frontend category toggles to backend tool names
    # 
    # Built-in tool categories:
    #   - utilities: Basic helper tools (calculator, time, etc.)
    #   - task_mgr: Task queue management
    #   - web: Web fetching tools
    #   - knowledge_bases: Document/RAG search
    #   - chat_history: Agent memory for past conversations
    #   - file_manager: File operations
    #   - code_exec: Python execution
    #   - image_gen: Image generation
    #   - mcp_install: Temporary MCP installation
    #
    # External tool categories (dynamic, loaded from database):
    #   - extern_tools_mcp: External MCP server tools
    #   - extern_tools_openapi: External OpenAPI tools
    
    TOOL_CATEGORY_MAP = {
        # Utilities - Basic helper tools
        "utilities": [
            "calculator", "get_current_time", "format_json", "analyze_text",
        ],
        
        # TaskMgr - Task queue management tools
        "task_mgr": [
            "add_task", "add_tasks_batch", "complete_task", "fail_task", 
            "skip_task", "get_task_queue", "clear_task_queue", 
            "pause_task_queue", "resume_task_queue",
        ],
        
        # Web - Web fetching and search tools
        "web": [
            "fetch_webpage", "fetch_urls", "web_search", "web_extract",
        ],
        
        # KnowledgeBases - Document/RAG search tools
        "knowledge_bases": [
            "search_documents",
        ],
        
        # ChatHistory - Agent memory tools for past conversations
        "chat_history": [
            "memory_search", "memory_read",
            "agent_search", "agent_read",  # Legacy aliases
        ],
        
        # FileManager - File operations tools
        "file_manager": [
            "view_file_lines", "search_in_file", "list_uploaded_files", 
            "view_signature", "request_file", "search_replace",
            "grep_files", "sed_files",
        ],
        
        # CodeExec - Code execution tools
        "code_exec": [
            "execute_python",
        ],
        
        # ImageGen - Image generation tools
        "image_gen": [
            "generate_image",
        ],
        
        # MCPInstall - Temporary MCP server installation
        "mcp_install": [
            "install_mcp_server", "uninstall_mcp_server", "list_mcp_servers",
        ],
        
        # ExternToolsMCP - External MCP tools (populated dynamically)
        # These are tools from MCP servers configured in Admin > Tools
        "extern_tools_mcp": [],  # Populated at runtime from database
        
        # ExternToolsOpenAPI - External OpenAPI tools (populated dynamically)
        # These are tools from OpenAPI specs configured in Admin > Tools
        "extern_tools_openapi": [],  # Populated at runtime from database
        
        # Legacy category mappings (for backward compatibility with existing UI)
        "web_search": ["fetch_webpage", "fetch_urls", "web_search", "web_extract"],
        "kb_search": ["search_documents"],
        "file_ops": ["view_file_lines", "search_in_file", "list_uploaded_files", "view_signature", "request_file", "search_replace", "grep_files", "sed_files"],
        "user_chats_kb": ["memory_search", "memory_read", "agent_search", "agent_read"],  # Legacy alias for chat_history
    }
    
    # Tools that are always available when tools are enabled
    # These are essential utilities that should always be accessible
    UTILITY_TOOLS = [
        "calculator", "get_current_time", "format_json", "analyze_text",
    ]
    
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
        logger.info(f"[SAVE_USER_MSG] Override from payload: {save_user_message}")
    
    if not chat_id or (not content and not attachments):
        logger.warning(f"Missing chat_id or content/attachments")
        await ws_manager.send_to_connection(connection, {
            "type": "error",
            "payload": {"error": "Missing chat_id or content"},
        })
        return
    
    # NC-0.8.0.21: If user uploaded files without text, set minimal content
    if not content and attachments:
        content = "Refer to the following materials."
    
    # NC-0.8.0.21: Build upload context for attached files so LLM knows what was uploaded
    _upload_context = None
    if attachments:
        _file_list = []
        for att in attachments:
            _fn = att.get("filename", "unknown")
            _type = att.get("type", "file")
            _size = len(att.get("content", "")) if att.get("content") else len(att.get("data", "")) if att.get("data") else 0
            if _type == "youtube":
                _file_list.append(f"  - {_fn} (YouTube transcript, {_size:,} chars)")
            elif _type == "image":
                _file_list.append(f"  - {_fn} (image)")
            else:
                _ext = _fn.rsplit('.', 1)[-1].lower() if '.' in _fn else 'unknown'
                _file_list.append(f"  - {_fn} (.{_ext}, {_size:,} chars)")
        if _file_list:
            _upload_context = "[UPLOADED_FILES]\nThe user has uploaded the following files with this message:\n" + "\n".join(_file_list) + "\n\nFile contents are included as attachments. Use view_file_lines or request_file tools to access full content if truncated.\n[/UPLOADED_FILES]"
    
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
        
        # NC-0.8.0.7: Check if image request filter is disabled (let LLM use tool instead)
        from app.services.settings_service import SettingsService
        disable_image_filter = await SettingsService.get_bool(db, "disable_image_request_filter", default=False)
        
        # Check if this is an image generation request (with LLM confirmation)
        # Skip this check if the filter is disabled - let the LLM use image gen tool
        is_image_request = False
        image_prompt = ""
        if not disable_image_filter:
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
                # NC-0.8.0.9: Use admin settings for default dimensions
                try:
                    from app.services.settings_service import SettingsService
                    from app.core.settings_keys import SK
                    width = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_WIDTH)
                    height = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_HEIGHT)
                    logger.debug(f"[IMAGE_AUTO] Using admin defaults: {width}x{height}")
                except Exception as e:
                    logger.warning(f"[IMAGE_AUTO] Could not load admin settings, using 1024x1024: {e}")
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
            
            # Update chat's updated_at - this is the user's temporal relation to the chat
            chat.updated_at = datetime.now(timezone.utc)
            
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
            await db.flush()
            await db.commit()
            
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
        
        # ==== PROCESS LARGE FILE ATTACHMENTS (BEFORE saving message) ====
        # Store large files as artifacts with truncated placeholders
        attachment_manifest = None
        if attachments and save_user_message:
            try:
                from app.services.attachment_processor import process_large_attachments
                from app.services.settings_service import SettingsService
                
                model_context = int(await SettingsService.get(db, "model_context_size") or "128000")
                
                # Process attachments - large files get stored as artifacts
                # The returned 'attachments' will have content truncated for large files
                attachments, attachment_manifest, was_processed = await process_large_attachments(
                    db=db,
                    chat_id=chat_id,
                    user_id=user_id,
                    attachments=attachments,
                    model_context_size=model_context,
                )
                
                if was_processed:
                    logger.info(f"[ATTACH_PROC_EARLY] Processed large attachments before save")
                    await db.commit()  # Commit the stored files
                    
            except Exception as e:
                logger.error(f"[ATTACH_PROC_EARLY] Failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Create user message only if save_user_message is True
        logger.info(f"[SAVE_USER_MSG_CHECK] save_user_message={save_user_message}, chat_id={chat_id}, content_len={len(content)}")
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
            logger.info(f"[USER_MSG_SAVED] id={user_message.id}, parent_id={parent_id}, content_len={len(content) if content else 0}, content_preview='{content[:50] if content else 'NONE'}...'")
            
            # Update chat's updated_at - this is the user's temporal relation to the chat
            chat.updated_at = datetime.now(timezone.utc)
            
            # IMPORTANT: Commit user message immediately to ensure it's not lost
            # if filter chain processing fails or times out
            try:
                await db.commit()
                logger.info(f"[USER_MSG_COMMITTED] id={user_message.id}")
            except Exception as commit_err:
                logger.error(f"[USER_MSG_COMMIT_FAILED] {commit_err}")
                # Continue anyway - the message is at least flushed
            
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
        
        # ==== SYSTEM PROMPT SETUP ====
        from app.api.routes.admin import get_system_setting
        default_prompt = await get_system_setting(db, "default_system_prompt")
        system_prompt = chat.system_prompt or default_prompt
        
        # ==== RAG SEARCHES ====
        # Track if any RAG source had results (for filter chain skip logic)
        rag_had_results = False
        global_kb_had_results = False  # NC-0.8.0.7: Track Global KB separately
        import time
        
        # NC-0.8.0.7: Check if pre-emptive RAG is enabled (default: True for backward compatibility)
        from app.services.settings_service import SettingsService
        preemptive_rag_enabled = await SettingsService.get_bool(db, "enable_preemptive_rag", default=True)
        
        # Skip RAG only for actual tool result continuations (system-generated responses)
        # NOT for regenerations - those are real user queries that need knowledge context
        # Tool results are identified by their content prefix, not by save_user_message flag
        is_tool_result = content.startswith("[SYSTEM TOOL RESULT") or content.startswith("[SYSTEM NOTIFICATION")
        
        # 1. Global Knowledge Base search (ALWAYS runs, independent of preemptive_rag setting)
        # Skip for tool results to avoid unnecessary searches and potential GPU issues
        if is_tool_result:
            logger.debug("[GLOBAL_RAG_DEBUG] Skipping RAG search for tool result continuation")
        else:
            try:
                from app.models.models import KnowledgeStore
                global_store_check = await db.execute(
                    select(KnowledgeStore.id, KnowledgeStore.name).where(KnowledgeStore.is_global == True)
                )
                global_stores_found = global_store_check.fetchall()
                has_global_stores = len(global_stores_found) > 0
                
                if has_global_stores:
                    logger.info(f"[GLOBAL_RAG_DEBUG] Found {len(global_stores_found)} global stores: {[(str(s.id), s.name) for s in global_stores_found]}")
                    global_kb_start = time.time()
                    rag_service = RAGService()
                    global_results, global_store_names = await rag_service.search_global_stores(db, content, chat_id=chat_id)
                    global_kb_search_time = time.time() - global_kb_start
                    
                    logger.info(f"[GLOBAL_RAG_DEBUG] Search completed in {global_kb_search_time:.3f}s - Found {len(global_results)} results from stores: {global_store_names}")
                
                    if global_results:
                        global_kb_had_results = True  # NC-0.8.0.7: Track separately
                        # NOTE: Don't set rag_had_results here - Global KB shouldn't block filter chains
                        # Format global context
                        global_context_parts = []
                        for i, result in enumerate(global_results):
                            doc_name = result.get("document_name", "Unknown")
                            chunk_content = result.get("content", "")
                            score = result.get("similarity", 0)
                            chunk_id = result.get("chunk_id", "N/A")
                            
                            logger.debug(f"[GLOBAL_RAG_DEBUG] Result {i+1}: doc='{doc_name}', chunk_id={chunk_id}, score={score:.4f}, content_len={len(chunk_content)}")
                            
                            global_context_parts.append(
                                f"[Source: {doc_name} | Confidence: {score:.0%}]\n{chunk_content}"
                            )
                        
                        global_context = "\n\n---\n\n".join(global_context_parts)
                        global_stores_list = ", ".join(global_store_names)
                        
                        # Get custom prompt or use default
                        custom_global_prompt = await get_system_setting(db, "rag_prompt_global_kb")
                        if custom_global_prompt:
                            # Use custom prompt with {context} and {sources} placeholders
                            knowledge_addendum = custom_global_prompt.replace("{context}", global_context).replace("{sources}", global_stores_list)
                        else:
                            # Default authoritative prompt
                            knowledge_addendum = f"""

## AUTHORITATIVE KNOWLEDGE BASE

<trusted_knowledge source="{global_stores_list}">
IMPORTANT: The following information comes from the organization's verified global knowledge base. This content is DEFINITIVE and TRUSTED - treat it as the authoritative source of truth for the topics it covers. When this knowledge conflicts with your general training, defer to this information.

{global_context}
</trusted_knowledge>

When answering questions related to the above topics, you MUST use this authoritative information as your primary source. Cite it naturally in your responses when relevant."""

                        system_prompt = f"{system_prompt}{knowledge_addendum}"
                        logger.info(f"[GLOBAL_RAG_DEBUG] Injected {len(global_results)} results from global KB")
                else:
                    logger.info("[GLOBAL_RAG_DEBUG] No global knowledge stores exist (is_global=True), skipping search")
            except Exception as e:
                logger.warning(f"[GLOBAL_RAG_DEBUG] Failed to search global stores: {e}", exc_info=True)
        
        # 2. User's Chat History Knowledge Base search (skip for tool results OR if preemptive RAG disabled)
        # NC-0.8.0.7: Only run if preemptive_rag_enabled
        if not is_tool_result and preemptive_rag_enabled:
            try:
                chat_kb_enabled = getattr(user, 'all_chats_knowledge_enabled', False)
                chat_kb_store = getattr(user, 'chat_knowledge_store_id', None)
                if chat_kb_enabled and chat_kb_store:
                    chat_kb_start = time.time()
                    rag_service = RAGService()
                    chat_kb_results = await rag_service.get_knowledge_store_context(
                        db=db,
                        user=user,
                        query=content,
                        knowledge_store_ids=[chat_kb_store],
                        bypass_access_check=True,  # User owns this KB
                        chat_id=chat_id,  # For context-aware query enhancement
                    )
                    chat_kb_time = time.time() - chat_kb_start
                    
                    if chat_kb_results:
                        rag_had_results = True
                        
                        # Get custom prompt or use default
                        custom_chat_prompt = await get_system_setting(db, "rag_prompt_chat_history")
                        if custom_chat_prompt:
                            chat_kb_addendum = custom_chat_prompt.replace("{context}", chat_kb_results)
                        else:
                            chat_kb_addendum = f"""

## YOUR CHAT HISTORY KNOWLEDGE

<chat_history_context>
The following is relevant information from your previous conversations:

{chat_kb_results}
</chat_history_context>
"""
                        system_prompt = f"{system_prompt}{chat_kb_addendum}"
                        logger.info(f"[CHAT_KB_DEBUG] Found chat history context in {chat_kb_time:.3f}s")
            except Exception as e:
                logger.warning(f"[CHAT_KB_DEBUG] Failed to search chat history KB: {e}")
        elif not preemptive_rag_enabled and not is_tool_result:
            logger.debug("[CHAT_KB_DEBUG] Skipping chat history KB (preemptive RAG disabled)")
        
        # 3. Assistant KB and User RAG search (skip for tool results OR if preemptive RAG disabled)
        # NC-0.8.0.7: Only run if preemptive_rag_enabled
        if not is_tool_result and preemptive_rag_enabled:
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
                    enable_rag = True
                    logger.info(f"Auto-enabled RAG for assistant with knowledge stores: {assistant_ks_ids}")
            
            if enable_rag:
                rag_service = RAGService()
                
                if assistant_ks_ids:
                    # Use assistant's knowledge stores
                    context = await rag_service.get_knowledge_store_context(
                        db=db,
                        user=user,
                        query=content,
                        knowledge_store_ids=assistant_ks_ids,
                        bypass_access_check=True,
                        chat_id=chat_id,  # For context-aware query enhancement
                    )
                    logger.debug(f"Using assistant knowledge stores for chat {chat_id}: {assistant_ks_ids}")
                    
                    if context:
                        rag_had_results = True
                        # Get custom GPT KB prompt or fall back to legacy
                        custom_gpt_prompt = await get_system_setting(db, "rag_prompt_gpt_kb")
                        if custom_gpt_prompt:
                            rag_addendum = custom_gpt_prompt.replace("{context}", context)
                        else:
                            rag_prompt = await get_system_setting(db, "rag_context_prompt")
                            rag_addendum = f"{rag_prompt}\n\n{context}"
                        system_prompt = f"{system_prompt}\n\n{rag_addendum}"
                else:
                    # Regular user documents
                    context = await rag_service.get_context_for_query(
                        db=db,
                        user=user,
                        query=content,
                        document_ids=document_ids,
                        chat_id=chat_id,  # For context-aware query enhancement
                    )
                    
                    if context:
                        rag_had_results = True
                        # Get user docs prompt or fall back to legacy
                        custom_docs_prompt = await get_system_setting(db, "rag_prompt_user_docs")
                        if custom_docs_prompt:
                            rag_addendum = custom_docs_prompt.replace("{context}", context)
                        else:
                            rag_prompt = await get_system_setting(db, "rag_context_prompt")
                            rag_addendum = f"{rag_prompt}\n\n{context}"
                        system_prompt = f"{system_prompt}\n\n{rag_addendum}"
        elif not preemptive_rag_enabled and not is_tool_result:
            logger.debug("[RAG_DEBUG] Skipping assistant/user RAG (preemptive RAG disabled)")
        
        logger.info(f"[RAG_SUMMARY] rag_had_results={rag_had_results}, global_kb_had_results={global_kb_had_results}, is_tool_result={is_tool_result}, preemptive_rag={preemptive_rag_enabled}")
        
        # ==== YOUTUBE URL PROCESSING ====
        # Detect YouTube URLs in user message AND recent conversation history
        youtube_context = ""
        
        # Check current message
        youtube_urls = extract_youtube_urls(content)
        
        # Also check recent messages in conversation history for YouTube URLs
        # This handles "what is this video about?" follow-up questions
        if not youtube_urls:
            try:
                # Fetch recent messages from this chat
                recent_msgs_result = await db.execute(
                    select(Message.content)
                    .where(Message.chat_id == chat_id)
                    .order_by(Message.created_at.desc())
                    .limit(10)
                )
                recent_msgs = recent_msgs_result.scalars().all()
                
                for msg_content in recent_msgs:
                    if msg_content:
                        found_urls = extract_youtube_urls(msg_content)
                        if found_urls:
                            youtube_urls = found_urls
                            logger.info(f"[YOUTUBE] Found {len(found_urls)} YouTube URLs in conversation history")
                            break
            except Exception as e:
                logger.warning(f"[YOUTUBE] Error checking history for URLs: {e}")
        
        if youtube_urls:
            logger.info(f"[YOUTUBE] Processing {len(youtube_urls)} YouTube URLs")
            video_ids = [vid for _, vid in youtube_urls]
            youtube_context = await fetch_youtube_context(video_ids, connection=connection, chat_id=chat_id, db=db)
            if youtube_context:
                logger.info(f"[YOUTUBE] Added {len(youtube_context)} chars of context from video subtitles")
                # Add YouTube context to system prompt
                youtube_addendum = (
                    "\n\n[VIDEO CONTEXT]\n"
                    "The user has shared YouTube video(s). Here are the transcripts/subtitles for reference:\n\n"
                    f"{youtube_context}"
                )
                system_prompt = f"{system_prompt}{youtube_addendum}"
        
        # ==== FILTER CHAIN EXECUTION ====
        # Run enabled filter chains on the user message before LLM processing
        # NC-0.8.0.7: When preemptive RAG is disabled, skip_if_rag_hit logic doesn't apply
        # (rag_had_results will only be True from Global KB, and we want chains to run anyway)
        from app.filters.manager import get_chain_manager
        from app.filters.executor import ChainExecutor
        
        chain_manager = get_chain_manager()
        enabled_chains = chain_manager.get_enabled_chains()
        
        # Track if any chain modified the content
        original_content = content
        filter_context_items = []
        
        logger.info(f"[CONTENT_TRACE] Before filter chains: content='{content[:50]}...' len={len(content)}")
        
        if enabled_chains:
            # NC-0.8.0.7: skip_if_rag_hit only applies when Chat History or Assistant KB found results
            # Global KB results never trigger skip - filter chains should always have a chance to run
            if rag_had_results:
                chains_to_run = [c for c in enabled_chains if not c.get("skip_if_rag_hit", True)]
                skipped_count = len(enabled_chains) - len(chains_to_run)
                if skipped_count > 0:
                    logger.info(f"[FILTER_CHAINS] Skipping {skipped_count} chains due to RAG hit (skip_if_rag_hit=True)")
                enabled_chains = chains_to_run
            
            if enabled_chains:
                logger.info(f"[FILTER_CHAINS] {len(enabled_chains)} chains to execute...")
                # Pre-load MCP/OpenAPI tools for filter chains
                mcp_tools_cache = {}
                
                try:
                    # Load enabled tools from database
                    from app.models.models import Tool
                    query = select(Tool).where(Tool.is_enabled == True)
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
            
            # Check global debug setting for filter chains
            from app.services.settings_service import SettingsService
            debug_filter_chains = await SettingsService.get(db, "debug_filter_chains") == "true"
            
            executor = ChainExecutor(
                llm_func=filter_llm_func,
                tool_func=filter_tool_func,
                global_debug=debug_filter_chains,
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
                        # Log if content is drastically shortened (likely a bug)
                        if len(result.content) < len(content) * 0.3:
                            logger.error(f"[FILTER_BUG] Chain '{chain.get('name', 'unknown')}' drastically shortened content: '{content[:50]}...' ({len(content)} chars) -> '{result.content[:50]}...' ({len(result.content)} chars)")
                        else:
                            logger.warning(f"[FILTER_MODIFIED] Chain '{chain.get('name', 'unknown')}' modified content: '{content[:30]}...' -> '{result.content[:30]}...'")
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
        
        logger.info(f"[CONTENT_TRACE] After filter chains: content='{content[:50]}...' len={len(content)}, modified={content != original_content}")
        
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
        
        # Inject Task Queue context (pending tasks for agentic execution)
        # Only include capability notice if tools are enabled
        try:
            task_queue = TaskQueueService(db, chat_id)
            await task_queue._load_queue()  # Ensure queue is loaded
            task_context = task_queue.get_system_prompt_addition(
                include_capability_notice=enable_tools  # Only show capability if tools enabled
            )
            if task_context:
                system_prompt = f"{system_prompt}\n\n{task_context}"
                logger.debug(f"[TASK_QUEUE] Injected task queue context for chat {chat_id}")
        except Exception as e:
            logger.debug(f"[TASK_QUEUE] Could not get task queue context: {e}")
        
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
        
        # Inject code summary if available (provides overview of project structure)
        if chat.code_summary:
            try:
                summary = chat.code_summary
                summary_lines = ["[PROJECT_CODE_SUMMARY]"]
                
                # Include file signatures
                files = summary.get("files", [])
                if files:
                    summary_lines.append("\nFiles with signatures:")
                    for file_entry in files[:20]:  # Limit to 20 files
                        # Support both "filename" and "path" field names
                        fname = file_entry.get("filename") or file_entry.get("path", "unknown")
                        sigs = file_entry.get("signatures", [])
                        if sigs:
                            summary_lines.append(f"\n  {fname}:")
                            for sig in sigs[:10]:  # Limit signatures per file
                                # Support both "kind" and "type" field names
                                kind = sig.get("kind") or sig.get("type", "")
                                name = sig.get("name", "")
                                line = sig.get("line", 0)
                                summary_lines.append(f"    - {kind} {name} (line {line})")
                            if len(sigs) > 10:
                                summary_lines.append(f"    ... and {len(sigs) - 10} more")
                
                # Include warnings
                if summary.get("warnings"):
                    summary_lines.append("\nCode warnings detected:")
                    for warning in summary.get("warnings", [])[:10]:
                        summary_lines.append(f"  - {warning.get('message', 'Unknown warning')}")
                
                summary_lines.append("\n[END_PROJECT_CODE_SUMMARY]")
                
                summary_context = "\n".join(summary_lines)
                system_prompt = f"{system_prompt}\n\n{summary_context}"
                logger.debug(f"Injected code summary for chat {chat_id} ({len(summary_context)} chars)")
            except Exception as e:
                logger.warning(f"Failed to inject code summary: {e}")
        
        # ==== INJECT ATTACHMENT MANIFEST ====
        # Manifest was generated earlier (before message save) if files were processed
        if attachment_manifest:
            system_prompt = f"{system_prompt}\n\n{attachment_manifest}"
            logger.info(f"[ATTACH_PROC] Added manifest to system prompt ({len(attachment_manifest)} chars)")
        
        # ==== INJECT UPLOAD CONTEXT ====
        # NC-0.8.0.21: Notify LLM about uploaded files
        if _upload_context:
            system_prompt = f"{system_prompt}\n\n{_upload_context}"
            logger.info(f"[UPLOAD_CTX] Added upload context to system prompt ({len(_upload_context)} chars)")
        
        # ==== VISION ROUTING ====
        # If attachments contain images and primary model isn't multimodal,
        # route through vision model to get descriptions
        image_descriptions = []
        if attachments:
            try:
                from app.services.vision_router import get_vision_router, format_image_descriptions_for_llm
                
                vision_router = await get_vision_router(db)
                attachments, image_descriptions = await vision_router.process_attachments_for_routing(
                    attachments=attachments,
                    user_message=content,
                )
                
                if image_descriptions:
                    # Add descriptions to system prompt
                    desc_text = format_image_descriptions_for_llm(image_descriptions)
                    if desc_text:
                        system_prompt = f"{system_prompt}\n\n[IMAGE CONTEXT]\n{desc_text}\n[/IMAGE CONTEXT]"
                        logger.info(f"[VISION_ROUTER] Added {len(image_descriptions)} image descriptions to context")
                        
            except Exception as e:
                logger.error(f"[VISION_ROUTER] Failed to process images: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Update chat system prompt temporarily in memory for LLM call
        # We don't want to persist this change, just use it for the LLM request
        chat.system_prompt = system_prompt
        
        # NC-0.8.0.7: Get chat's active tool categories before expunging
        active_tool_categories = chat.active_tools or []
        
        # Expunge chat from session so SQLAlchemy doesn't track/flush our temporary change
        # We'll use direct SQL for any actual updates later
        db.expunge(chat)
        
        # NC-0.8.0.7: Filter tools based on chat's active_tools setting
        tools = []
        if enable_tools:
            # Get all built-in tools
            all_tools = tool_registry.get_tool_definitions()
            
            # Build list of allowed tool names based on active categories
            allowed_tool_names = set(UTILITY_TOOLS)  # Always include utility tools
            
            for category in active_tool_categories:
                if category in TOOL_CATEGORY_MAP:
                    allowed_tool_names.update(TOOL_CATEGORY_MAP[category])
            
            # Filter tools to only those in allowed list
            # If no categories specified, allow all tools (backward compatibility)
            if active_tool_categories:
                tools = [t for t in all_tools if t["name"] in allowed_tool_names]
                logger.debug(f"Active tool categories: {active_tool_categories}, allowed tools: {[t['name'] for t in tools]}")
            else:
                tools = all_tools
                logger.debug(f"No tool categories set, allowing all {len(tools)} tools")
        
        # Load MCP/OpenAPI tools from database
        mcp_tool_map = {}  # Maps tool name to (tool_db_record, operation_name)
        if enable_tools:
            try:
                tool_service = ToolService()
                mcp_tools, mcp_map = await tool_service.get_tools_for_llm(db, user)
                
                # NC-0.8.0.8: Filter external tools based on categories
                # If specific categories are set, only include external tools if their category is active
                if active_tool_categories:
                    include_mcp = "extern_tools_mcp" in active_tool_categories
                    include_openapi = "extern_tools_openapi" in active_tool_categories
                    
                    if include_mcp or include_openapi:
                        filtered_mcp_tools = []
                        for tool in mcp_tools:
                            tool_type = tool.get("_tool_type", "")
                            if (include_mcp and tool_type == "mcp") or (include_openapi and tool_type == "openapi"):
                                filtered_mcp_tools.append(tool)
                        mcp_tools = filtered_mcp_tools
                        logger.debug(f"Filtered external tools: MCP={include_mcp}, OpenAPI={include_openapi}, count={len(mcp_tools)}")
                    else:
                        # No external tool categories active, exclude all external tools
                        mcp_tools = []
                        logger.debug("No external tool categories active, excluding all MCP/OpenAPI tools")
                
                tools = (tools or []) + mcp_tools
                mcp_tool_map = mcp_map
                if mcp_tools:
                    logger.debug(f"Loaded {len(mcp_tools)} MCP/OpenAPI tools")
            except Exception as e:
                logger.debug(f"No MCP tools available: {e}")
        
        logger.debug(f"Tools enabled: {enable_tools}, tool_count: {len(tools) if tools else 0}")
        
        # NC-0.8.0.8: Debug tool advertisements if enabled
        debug_tool_advertisements = await get_system_setting(db, "debug_tool_advertisements") == "true"
        debug_tool_calls = await get_system_setting(db, "debug_tool_calls") == "true"
        
        if debug_tool_advertisements:
            from app.core.logging import log_tool_debug
            log_tool_debug("=" * 60)
            log_tool_debug("WEBSOCKET TOOL DEBUG - Tools being passed to LLM")
            log_tool_debug("=" * 60)
            log_tool_debug("Context", 
                chat_id=chat_id,
                chat_model=chat.model,
                enable_tools=enable_tools,
                active_tool_categories=active_tool_categories
            )
            if tools:
                log_tool_debug(f"Total tools: {len(tools)}")
                for i, tool in enumerate(tools):
                    log_tool_debug(f"Tool [{i+1}]",
                        name=tool.get('name', 'unknown'),
                        description=tool.get('description', 'N/A')[:200],
                        parameters=list(tool.get('parameters', {}).get('properties', {}).keys())
                    )
            else:
                log_tool_debug("NO TOOLS BEING SENT",
                    reason=f"enable_tools={enable_tools}, filtering may have removed all"
                )
            log_tool_debug("=" * 60)
        
        # Stream response (using database settings)
        llm = await LLMService.from_database(db)
        logger.debug(f"Starting LLM stream for chat {chat_id}")
        
        # Create tool executor function
        # NC-0.8.0.9: Tool delegates - map common hallucinated tool names to actual tools
        # LLMs often hallucinate tools from their training data (Claude, ChatGPT, etc.)
        TOOL_DELEGATES = {
            # Artifact/file creation patterns (Claude, ChatGPT)
            "create_artifact": "create_file",
            "write_file": "create_file",
            "save_file": "create_file",
            "artifact": "create_file",
            "make_file": "create_file",
            "output_file": "create_file",
            
            # Code execution patterns
            "run_code": "execute_python",
            "run_python": "execute_python",
            "python": "execute_python",
            "code_interpreter": "execute_python",
            "execute_code": "execute_python",
            "eval": "execute_python",
            
            # Web/search patterns
            "search_web": "fetch_webpage",
            "browse": "fetch_webpage",
            "browser": "fetch_webpage",
            "get_url": "fetch_webpage",
            "read_url": "fetch_webpage",
            "fetch_url": "fetch_webpage",
            "http_get": "fetch_webpage",
            
            # Image patterns
            "create_image": "generate_image",
            "make_image": "generate_image",
            "dalle": "generate_image",
            "dall-e": "generate_image",
            "image_gen": "generate_image",
            "text_to_image": "generate_image",
            
            # File viewing patterns
            "read_file": "view_file_lines",
            "view_file": "view_file_lines",
            "cat_file": "view_file_lines",
            "get_file": "view_file_lines",
            "open_file": "view_file_lines",
            "file_read": "view_file_lines",
            
            # Calculator patterns
            "math": "calculator",
            "calculate": "calculator",
            "eval_math": "calculator",
            "compute": "calculator",
            
            # Time patterns
            "get_time": "get_current_time",
            "current_time": "get_current_time",
            "time": "get_current_time",
            "date": "get_current_time",
            "now": "get_current_time",
            
            # Memory patterns (legacy names)
            "agent_search": "memory_search",
            "agent_read": "memory_read",
            
            # Search/replace patterns (NC-0.8.0.12)
            "str_replace": "search_replace",
            "find_replace": "search_replace",
            "replace_in_file": "search_replace",
            "edit_file": "search_replace",
            
            # Grep patterns (NC-0.8.0.12)
            "grep": "grep_files",
            "find_in_files": "grep_files",
            "search_files": "grep_files",
            "search_all": "grep_files",
            
            # Sed patterns (NC-0.8.0.12)
            "sed": "sed_files",
            "replace_all": "sed_files",
            "batch_replace": "sed_files",
            
            # Web extract patterns (NC-0.8.0.12)
            "extract_webpage": "web_extract",
            "scrape": "web_extract",
            "scrape_url": "web_extract",
        }
        
        async def execute_tool(tool_name: str, arguments: dict):
            """Execute a tool and return result"""
            # NC-0.8.0.9: Check for delegated tool names first
            original_name = tool_name
            if tool_name in TOOL_DELEGATES:
                delegated_to = TOOL_DELEGATES[tool_name]
                logger.info(f"[LLM_TOOLS] DELEGATE: '{tool_name}' → '{delegated_to}'")
                tool_name = delegated_to
                
                # Map arguments for common patterns
                if delegated_to == "create_file":
                    # Handle various content/path argument names for artifact creation
                    content = (
                        arguments.get("content") or 
                        arguments.get("data") or 
                        arguments.get("text") or
                        arguments.get("code") or
                        arguments.get("body") or
                        arguments.get("source")
                    )
                    path = (
                        arguments.get("path") or 
                        arguments.get("filename") or 
                        arguments.get("file_path") or
                        arguments.get("name") or
                        arguments.get("file")
                    )
                    overwrite = arguments.get("overwrite", True)
                    if isinstance(overwrite, str):
                        overwrite = overwrite.lower() in ("true", "1", "yes")
                    
                    if content and path:
                        arguments = {"path": path, "content": content, "overwrite": overwrite}
                    elif content:
                        # No path provided - generate one from content type
                        arguments = {"path": "output.txt", "content": content, "overwrite": overwrite}
                    
                elif delegated_to == "execute_python":
                    # Handle various code argument names
                    code = (
                        arguments.get("code") or 
                        arguments.get("script") or 
                        arguments.get("python_code") or
                        arguments.get("source") or
                        arguments.get("program")
                    )
                    if code:
                        new_args = {"code": code}
                        # Preserve output options
                        if arguments.get("output_image"):
                            new_args["output_image"] = arguments["output_image"]
                        if arguments.get("output_text"):
                            new_args["output_text"] = arguments["output_text"]
                        if arguments.get("output_filename"):
                            new_args["output_filename"] = arguments["output_filename"]
                        arguments = new_args
                    
                elif delegated_to == "fetch_webpage":
                    # Handle various URL argument names
                    url = (
                        arguments.get("url") or 
                        arguments.get("uri") or 
                        arguments.get("link") or
                        arguments.get("query")  # For search-like calls
                    )
                    if url:
                        arguments = {"url": url}
                        
                elif delegated_to == "generate_image":
                    # Handle various prompt argument names
                    prompt = (
                        arguments.get("prompt") or 
                        arguments.get("description") or 
                        arguments.get("text") or
                        arguments.get("query")
                    )
                    if prompt:
                        arguments = {"prompt": prompt}
                        
                elif delegated_to == "view_file_lines":
                    # Handle various file path argument names
                    path = (
                        arguments.get("path") or 
                        arguments.get("file") or 
                        arguments.get("filename") or
                        arguments.get("file_path")
                    )
                    if path:
                        new_args = {"filename": path}
                        # Preserve line range if provided
                        if "start_line" in arguments:
                            new_args["start_line"] = arguments.get("start_line")
                        if "end_line" in arguments:
                            new_args["end_line"] = arguments.get("end_line")
                        arguments = new_args
                
                elif delegated_to == "search_replace":
                    # Handle various search/replace argument names
                    filename = (
                        arguments.get("filename") or
                        arguments.get("path") or
                        arguments.get("file") or
                        arguments.get("file_path")
                    )
                    search = (
                        arguments.get("search") or
                        arguments.get("old_str") or
                        arguments.get("find") or
                        arguments.get("pattern")
                    )
                    replace = (
                        arguments.get("replace") or
                        arguments.get("new_str") or
                        arguments.get("replacement") or
                        ""
                    )
                    if filename and search is not None:
                        arguments = {"filename": filename, "search": search, "replace": replace}
                
                elif delegated_to == "grep_files":
                    pattern = (
                        arguments.get("pattern") or
                        arguments.get("query") or
                        arguments.get("search") or
                        arguments.get("text")
                    )
                    if pattern:
                        arguments = {"pattern": pattern}
                        if arguments.get("file_pattern"):
                            arguments["file_pattern"] = arguments["file_pattern"]
                
                elif delegated_to == "web_extract":
                    url = (
                        arguments.get("url") or
                        arguments.get("uri") or
                        arguments.get("link")
                    )
                    if url:
                        arguments = {"url": url}
            
            # NC-0.8.0.9: Validate tool exists before execution
            # Prevents LLM from hallucinating tools that don't exist
            is_valid_tool = (
                tool_name in mcp_tool_map or 
                tool_registry.get_handler(tool_name) is not None
            )
            if not is_valid_tool:
                logger.warning(f"[LLM_TOOLS] HALLUCINATED TOOL: '{tool_name}' does not exist. Available: {list(mcp_tool_map.keys()) + [t['function']['name'] for t in tool_registry.get_tool_definitions()]}")
                return {
                    "error": f"TOOL_NOT_FOUND: '{tool_name}' is not an available tool. You may be hallucinating a tool from your training data. Please use only the tools that were provided to you in this conversation.",
                    "available_tools": [t['function']['name'] for t in tool_registry.get_tool_definitions()] + list(mcp_tool_map.keys())
                }
            
            # NC-0.8.0.8: Debug tool calls if enabled
            if debug_tool_calls:
                from app.core.logging import log_tool_debug
                log_tool_debug("=" * 60)
                log_tool_debug("TOOL CALL REQUEST")
                log_tool_debug("=" * 60)
                log_tool_debug("Tool execution request",
                    tool_name=tool_name,
                    arguments=arguments
                )
            
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
                # Built-in tool - NC-0.8.0.7: Include user_id and message_id in context
                result = await tool_registry.execute(
                    tool_name,
                    arguments,
                    {
                        "db": db, 
                        "user": user, 
                        "user_id": user.id,
                        "chat_id": chat_id,
                        "message_id": streaming_handler._current_message_id,
                        "ws_connection": connection,
                    },
                )
            
            # NC-0.8.0.8: Debug tool result if enabled
            if debug_tool_calls:
                from app.core.logging import log_tool_debug
                log_tool_debug("TOOL CALL RESULT",
                    tool_name=tool_name,
                    result_length=len(str(result)),
                    result_preview=str(result)[:2000]
                )
                log_tool_debug("=" * 60)
            
            logger.debug(f"Tool executed, result length: {len(str(result))}")
            return result
        
        try:
            # Log content being sent to LLM
            logger.info(f"[LLM_CALL] Sending to LLM: content_len={len(content)}, chat={chat_id}")
            if len(content) > 200:
                logger.debug(f"[LLM_CALL] Content head: {content[:100]}")
                logger.debug(f"[LLM_CALL] Content tail: {content[-100:]}")
            
            _accumulated_content = ""  # Track streamed content for partial save on cancel
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
                user_timezone=user_timezone,  # NC-0.8.0.13: Browser timezone
            ):
                # Check if stop was requested (backup check)
                if streaming_handler.is_stop_requested():
                    logger.debug(f"Stopping generation for chat {chat_id}")
                    # Save partial content to DB so the message persists in the tree
                    _msg_id = streaming_handler._current_message_id
                    if _msg_id and _accumulated_content:
                        try:
                            await db.execute(
                                update(Message)
                                .where(Message.id == _msg_id)
                                .values(content=_accumulated_content, is_streaming=False)
                            )
                            await db.commit()
                            logger.info(f"[CANCEL] Saved partial content ({len(_accumulated_content)} chars) for message {_msg_id[:8]}")
                        except Exception as save_err:
                            logger.warning(f"[CANCEL] Failed to save partial content: {save_err}")
                    elif _msg_id:
                        try:
                            await db.execute(
                                update(Message)
                                .where(Message.id == _msg_id)
                                .values(is_streaming=False)
                            )
                            await db.commit()
                        except Exception:
                            pass
                    # Send stream_stopped with partial content
                    await ws_manager.send_to_connection(connection, {
                        "type": "stream_stopped",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "parent_id": assistant_parent_id,  # For conversation tree tracking
                            "message": "Generation stopped by user",
                        },
                    })
                    streaming_handler.reset_stop()
                    break
                
                event_type = event.get("type")
                logger.debug(f"LLM event: {event_type}")
                
                if event_type == "message_start":
                    await streaming_handler.start_stream(event["message_id"], chat_id)
                    _accumulated_content = ""  # Track content for partial save on cancel
                
                elif event_type == "text_delta":
                    await streaming_handler.send_chunk(event["text"])
                    _accumulated_content += event["text"]
                
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
                        event.get("parent_id"),
                        ui_events=event.get("tool_groups"),
                    )
                    
                    # Index this chat if chat knowledge is enabled (background, non-blocking)
                    try:
                        from app.api.routes.user_settings import _indexing_executor, index_single_chat_sync
                        # Check if user has chat knowledge enabled before spawning thread
                        # Use getattr for safety in case attribute not loaded
                        if getattr(user, 'all_chats_knowledge_enabled', False):
                            _indexing_executor.submit(index_single_chat_sync, user_id, chat_id)
                            logger.debug(f"[CHAT_KNOWLEDGE] Queued chat {chat_id} for indexing")
                    except Exception as idx_err:
                        logger.debug(f"[CHAT_KNOWLEDGE] Could not queue chat for indexing: {idx_err}")
                
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
                
                # NC-0.8.0.11: Handle direct output from execute_python
                elif event_type == "direct_image":
                    # Send image directly to chat
                    logger.info(f"[DIRECT_OUTPUT] Sending image from {event.get('tool_name')}")
                    await ws_manager.send_to_connection(connection, {
                        "type": "direct_image",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "tool_name": event.get("tool_name"),
                            "image_base64": event.get("image_base64"),
                            "mime_type": event.get("mime_type", "image/png"),
                            "filename": event.get("filename", "output.png"),
                        },
                    })
                
                elif event_type == "direct_text":
                    # Send text directly to chat
                    logger.info(f"[DIRECT_OUTPUT] Sending text from {event.get('tool_name')}")
                    await ws_manager.send_to_connection(connection, {
                        "type": "direct_text",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "tool_name": event.get("tool_name"),
                            "text": event.get("text"),
                            "filename": event.get("filename", "output.txt"),
                        },
                    })
                
                # NC-0.8.0.12: Forward tool activity events for UI timeline
                elif event_type in ("tool_start", "tool_end"):
                    payload_data = {
                        "chat_id": chat_id,
                        "message_id": streaming_handler._current_message_id,
                        "tool": event.get("tool", ""),
                        "round": event.get("round", 0),
                        "ts": event.get("ts", 0),
                        "args_summary": event.get("args_summary", ""),
                        "status": event.get("status", ""),
                        "duration_ms": event.get("duration_ms", 0),
                        "result_summary": event.get("result_summary", ""),
                    }
                    
                    await ws_manager.send_to_connection(connection, {
                        "type": event_type,
                        "payload": payload_data,
                    })
                
                # NC-0.8.0.13: Stream tool content deltas to frontend (live file creation preview)
                elif event_type == "tool_content_delta":
                    await ws_manager.send_to_connection(connection, {
                        "type": "tool_content_delta",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "tool_name": event.get("tool_name", ""),
                            "tool_index": event.get("tool_index", 0),
                            "delta": event.get("delta", ""),
                            "accumulated_length": event.get("accumulated_length", 0),
                        },
                    })
                
                # NC-0.8.0.21: Tool generating — LLM has started producing a tool call
                elif event_type == "tool_generating":
                    await ws_manager.send_to_connection(connection, {
                        "type": "tool_generating",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "tool_name": event.get("tool_name", ""),
                            "tool_index": event.get("tool_index", 0),
                            "round": event.get("round", 0),
                        },
                    })
                
                # NC-0.8.0.12: Forward tool_call events to frontend
                # For file-modifying tools, include updated file content so browser stays in sync
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name", "")
                    tool_args = event.get("arguments", {})
                    tool_result = event.get("result", "")
                    
                    payload = {
                        "chat_id": chat_id,
                        "message_id": streaming_handler._current_message_id,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    }
                    
                    # For file-modifying tools, attach the updated file content
                    FILE_MODIFY_TOOLS = {"create_file", "search_replace", "sed_files"}
                    if tool_name in FILE_MODIFY_TOOLS:
                        from app.tools.registry import get_session_file
                        
                        # Determine which file(s) were modified
                        modified_files = {}
                        if tool_name == "create_file":
                            fname = tool_args.get("path", "")
                            content = tool_args.get("content", "")
                            if fname and content:
                                modified_files[fname] = content
                        elif tool_name == "search_replace":
                            fname = tool_args.get("filename", "")
                            if fname:
                                fc = get_session_file(chat_id, fname)
                                if fc:
                                    modified_files[fname] = fc
                        elif tool_name == "sed_files":
                            # sed_files can modify multiple files — check result for which changed
                            try:
                                result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                                if isinstance(result_data, dict):
                                    for fname in result_data.get("changes", {}).keys():
                                        fc = get_session_file(chat_id, fname)
                                        if fc:
                                            modified_files[fname] = fc
                            except Exception:
                                pass
                        
                        if modified_files:
                            payload["modified_files"] = modified_files
                            logger.info(f"[TOOL_CALL] Sending file updates to frontend: {list(modified_files.keys())}")
                            
                            # NC-0.8.0.12: Rebuild and send associations + call graph
                            try:
                                from app.tools.registry import get_session_files
                                from app.services.zip_processor import (
                                    build_manifest_from_session_files, extract_associations,
                                    extract_call_graph,
                                )
                                session_files = get_session_files(chat_id)
                                if session_files:
                                    mini = build_manifest_from_session_files(session_files)
                                    payload["associations"] = extract_associations(mini)
                                    payload["call_graph"] = extract_call_graph(mini)
                                    payload["signature_index"] = mini.signature_index
                            except Exception as e:
                                logger.debug(f"[TOOL_CALL] Association rebuild skipped: {e}")
                    
                    await ws_manager.send_to_connection(connection, {
                        "type": "tool_call",
                        "payload": payload,
                    })
                
                # NC-0.8.0.13: Forward web retrieval artifact notifications to frontend
                elif event_type == "web_retrieval_artifact":
                    await ws_manager.send_to_connection(connection, {
                        "type": "web_retrieval_artifact",
                        "payload": {
                            "chat_id": chat_id,
                            "message_id": streaming_handler._current_message_id,
                            "filepath": event.get("filepath", ""),
                            "url": event.get("url", ""),
                            "content_length": event.get("content_length", 0),
                        },
                    })
                    
        except asyncio.CancelledError:
            logger.debug(f"Streaming task cancelled for chat {chat_id}")
            # Save partial content to DB so the message persists in the tree
            _msg_id = streaming_handler._current_message_id
            if _msg_id and _accumulated_content:
                try:
                    await db.execute(
                        update(Message)
                        .where(Message.id == _msg_id)
                        .values(content=_accumulated_content, is_streaming=False)
                    )
                    await db.commit()
                    logger.info(f"[CANCEL] Saved partial content ({len(_accumulated_content)} chars) for message {_msg_id[:8]}")
                except Exception as save_err:
                    logger.warning(f"[CANCEL] Failed to save partial content: {save_err}")
            elif _msg_id:
                try:
                    await db.execute(
                        update(Message)
                        .where(Message.id == _msg_id)
                        .values(is_streaming=False)
                    )
                    await db.commit()
                except Exception:
                    pass
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
        
        # ==== AGENTIC TASK QUEUE STATUS ====
        # Send queue status update to frontend after each response
        # LLM manages its own task flow via tools (complete_task, fail_task, etc.)
        try:
            task_queue = TaskQueueService(db, chat_id)
            queue_status = await task_queue.get_queue_status()
            
            if queue_status["has_pending"] or queue_status["completed_count"] > 0:
                # Send status update to frontend
                await ws_manager.send_to_connection(connection, {
                    "type": "task_queue_status",
                    "payload": {
                        "chat_id": chat_id,
                        "queue_length": queue_status["queue_length"],
                        "current_task": queue_status["current_task"],
                        "paused": queue_status["paused"],
                        "completed_count": queue_status["completed_count"],
                    },
                })
                
                # Archive log if getting large
                await task_queue.archive_log_overflow()
            
            # NC-0.8.0.8: Auto-execute tasks if LLM stopped without processing them
            # If there are queued tasks but no current task (LLM didn't pick one up),
            # automatically feed the next task to the LLM
            task_auto_depth = payload.get("_task_auto_depth", 0)
            max_auto_depth = 10  # Prevent infinite recursion
            
            if (queue_status["has_pending"] and 
                not queue_status["current_task"] and 
                not queue_status["paused"] and
                queue_status["queue_length"] > 0 and
                task_auto_depth < max_auto_depth):
                
                logger.info(f"[TASK_QUEUE_AUTO] LLM stopped with {queue_status['queue_length']} pending tasks - auto-executing (depth={task_auto_depth})")
                
                # Get and start the next task
                next_task = await task_queue.get_next_task()
                if next_task:
                    # Mark it as in progress
                    await task_queue._load_queue()
                    started_task = await task_queue._start_next_task()
                    await task_queue._save_queue()
                    
                    if started_task:
                        # Build task instruction message
                        task_prompt = f"""[TASK QUEUE - AUTO EXECUTION]

You have pending tasks in your queue. Please work on the following task:

**Task:** {started_task.description}

**Instructions:** {started_task.instructions}

When you complete this task, call the `complete_task` tool with a brief summary of what you did.
If you cannot complete it, call `fail_task` with the reason.

There are {queue_status['queue_length'] - 1} more tasks waiting after this one."""

                        # Send task notification to frontend
                        await ws_manager.send_to_connection(connection, {
                            "type": "task_auto_execute",
                            "payload": {
                                "chat_id": chat_id,
                                "task_id": started_task.id,
                                "task_description": started_task.description,
                                "remaining_tasks": queue_status['queue_length'] - 1,
                            },
                        })
                        
                        logger.info(f"[TASK_QUEUE_AUTO] Sending task '{started_task.description}' to LLM (depth={task_auto_depth + 1})")
                        
                        # Create task payload with incremented depth
                        task_payload = {
                            "chat_id": chat_id,
                            "content": task_prompt,
                            "enable_tools": True,
                            "_task_auto_depth": task_auto_depth + 1,
                        }
                        
                        # Recursive call to handle_chat_message
                        await handle_chat_message(
                            connection=connection,
                            streaming_handler=streaming_handler,
                            user_id=user_id,
                            payload=task_payload,
                            save_user_message=False,  # Don't save auto-execute as user message
                        )
                        
                        # After returning, check if queue is now empty
                        final_status = await task_queue.get_queue_status()
                        if not final_status["has_pending"] and final_status["queue_length"] == 0 and not final_status["current_task"]:
                            # Queue is empty - notify frontend
                            await ws_manager.send_to_connection(connection, {
                                "type": "task_queue_empty",
                                "payload": {
                                    "chat_id": chat_id,
                                    "completed_count": final_status["completed_count"],
                                },
                            })
                            
                            logger.info(f"[TASK_QUEUE_AUTO] Queue complete - {final_status['completed_count']} tasks processed")
            
            elif task_auto_depth >= max_auto_depth:
                logger.warning(f"[TASK_QUEUE_AUTO] Max recursion depth ({max_auto_depth}) reached, stopping auto-execution")
                
        except Exception as task_err:
            logger.warning(f"[TASK_QUEUE] Task processing error: {task_err}", exc_info=True)
        
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
                logger.info(f"[TITLE_GEN] Generating title for content: '{content[:50]}...' len={len(content)}")
                generated_title = await generate_chat_title(content, db)
                logger.info(f"[TITLE_GEN] Generated: '{generated_title}'")
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


async def generate_chat_title(first_message: str, db: AsyncSession) -> str:
    """Generate a chat title using a fresh LLM instance."""
    from app.api.routes.admin import get_system_setting
    from app.services.llm import LLMService
    
    # Extract meaningful text from first_message if it looks like JSON/search results
    content_for_title = first_message
    if first_message.startswith('{') or first_message.startswith('['):
        # It's JSON - try to extract the original query or meaningful text
        try:
            import json
            data = json.loads(first_message)
            if isinstance(data, dict):
                # Look for query or original message
                content_for_title = data.get('query', data.get('message', data.get('content', '')))
                if not content_for_title and 'results' in data:
                    # Try to get title from first result
                    results = data.get('results', [])
                    if results and isinstance(results[0], dict):
                        content_for_title = results[0].get('title', '')
        except:
            pass
    
    # If we still don't have good content, use a generic fallback
    if not content_for_title or content_for_title.startswith('{'):
        return "New Chat"
    
    try:
        # Get title prompt from admin settings
        title_prompt = await get_system_setting(db, "title_generation_prompt")
        
        # Create LLM instance from database settings
        title_llm = await LLMService.from_database(db)
        
        # Use utility model to avoid vLLM tokenizer conflicts with main model
        title = await title_llm.utility_completion(
            prompt=content_for_title[:500],
            system_prompt=title_prompt,
            max_tokens=30,
            db=db,
        )
        
        # Clean up the title
        title = title.strip().strip('"\'`')
        title = title[:60]
        
        if len(title) < 3:
            return content_for_title[:50] + ("..." if len(content_for_title) > 50 else "")
            
        return title
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        return content_for_title[:50] + ("..." if len(content_for_title) > 50 else "")


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
        await db.flush()
        await db.commit()
        
        # Fallback for created_at if server default wasn't populated
        msg_created_at = msg.created_at or datetime.utcnow()
        
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
                    "created_at": msg_created_at.isoformat(),
                },
            },
            exclude_user=user_id,
        )
        
        # Confirm to sender
        await ws_manager.send_to_connection(connection, {
            "type": "message_sent",
            "payload": {"message_id": msg.id},
        })


async def handle_mermaid_error(
    connection,
    streaming_handler: StreamingHandler,
    user_id: str,
    payload: dict,
):
    """Handle mermaid rendering error - send error back to LLM for auto-fix"""
    
    chat_id = payload.get("chat_id")
    message_id = payload.get("message_id")
    error = payload.get("error", "Unknown mermaid error")
    code = payload.get("code", "")
    
    if not chat_id or not code:
        logger.warning(f"Mermaid error missing required fields: chat_id={chat_id}, code_len={len(code)}")
        return
    
    logger.info(f"[MERMAID_ERROR] chat={chat_id}, error={error[:100]}")
    
    # Create a system message to inform the LLM about the error
    error_content = f"""[MERMAID SYNTAX ERROR]
The mermaid diagram you generated failed to render with the following error:

Error: {error}

Problematic code:
```mermaid
{code}
```

Please fix the mermaid syntax error and regenerate the diagram."""
    
    # Send this as a follow-up message to the LLM
    await handle_chat_message(
        connection,
        streaming_handler,
        user_id,
        {
            "chat_id": chat_id,
            "content": error_content,
            "parent_id": message_id,  # Chain to the problematic message
            "enable_tools": False,  # Don't need tools for fixing mermaid
            "enable_rag": False,
        },
        save_user_message=True,  # Save the error report as a user message
    )
