"""
WebSocket Manager for bidirectional streaming and real-time features
Supports:
- LLM streaming responses
- Client-to-client chat
- Real-time notifications
"""
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timezone
import json
import asyncio
import uuid
import logging
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a WebSocket connection"""
    websocket: WebSocket
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscriptions: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Connection):
            return self.id == other.id
        return False


class WebSocketManager:
    """
    Manages WebSocket connections for real-time features.
    Supports:
    - Per-user connections
    - Chat room subscriptions
    - Broadcast to specific users or rooms
    - LLM streaming
    """
    
    def __init__(self):
        # Map user_id -> set of connections (user can have multiple tabs)
        self._user_connections: Dict[str, Set[Connection]] = {}
        
        # Map chat_id -> set of user_ids subscribed
        self._chat_subscriptions: Dict[str, Set[str]] = {}
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, user_id: str, already_accepted: bool = False) -> Connection:
        """Accept a new WebSocket connection"""
        if not already_accepted:
            await websocket.accept()
        
        connection = Connection(
            websocket=websocket,
            user_id=user_id,
        )
        
        async with self._lock:
            if user_id not in self._user_connections:
                self._user_connections[user_id] = set()
            self._user_connections[user_id].add(connection)
        
        return connection
    
    async def disconnect(self, connection: Connection):
        """Remove a WebSocket connection"""
        async with self._lock:
            user_id = connection.user_id
            
            if user_id in self._user_connections:
                self._user_connections[user_id].discard(connection)
                
                if not self._user_connections[user_id]:
                    del self._user_connections[user_id]
            
            # Remove from chat subscriptions
            for chat_id in list(connection.subscriptions):
                await self._unsubscribe_from_chat(connection, chat_id)
    
    async def subscribe_to_chat(self, connection: Connection, chat_id: str):
        """Subscribe a connection to a chat room"""
        async with self._lock:
            connection.subscriptions.add(chat_id)
            
            if chat_id not in self._chat_subscriptions:
                self._chat_subscriptions[chat_id] = set()
            self._chat_subscriptions[chat_id].add(connection.user_id)
    
    async def _unsubscribe_from_chat(self, connection: Connection, chat_id: str):
        """Unsubscribe from a chat room (internal, assumes lock held)"""
        connection.subscriptions.discard(chat_id)
        
        if chat_id in self._chat_subscriptions:
            self._chat_subscriptions[chat_id].discard(connection.user_id)
            
            if not self._chat_subscriptions[chat_id]:
                del self._chat_subscriptions[chat_id]
    
    async def unsubscribe_from_chat(self, connection: Connection, chat_id: str):
        """Unsubscribe from a chat room"""
        async with self._lock:
            await self._unsubscribe_from_chat(connection, chat_id)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Send a message to all connections of a specific user."""
        connections = self._user_connections.get(user_id, set())
        
        if not connections:
            return 0
        
        message_json = json.dumps(message)
        sent_count = 0
        
        for connection in list(connections):
            try:
                await connection.websocket.send_text(message_json)
                sent_count += 1
            except Exception:
                pass
        
        return sent_count
    
    async def send_to_connection(self, connection: Connection, message: Dict[str, Any]) -> bool:
        """Send a message to a specific connection.
        
        Returns:
            True if sent successfully, False if failed
        """
        try:
            # Check if websocket is still connected
            if connection.websocket.client_state.name != "CONNECTED":
                logger.warning(f"WebSocket not connected for user {connection.user_id}, state: {connection.websocket.client_state.name}")
                return False
            
            await connection.websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Failed to send to connection for user {connection.user_id}: {e}")
            return False
    
    async def send_to_connection_with_retry(
        self, 
        connection: Connection, 
        message: Dict[str, Any],
        max_retries: int = 3,
        retry_delay: float = 0.5
    ) -> bool:
        """Send a message with retry logic for important messages like images.
        
        Returns:
            True if sent successfully, False if all retries failed
        """
        for attempt in range(max_retries):
            try:
                if connection.websocket.client_state.name != "CONNECTED":
                    logger.warning(f"WebSocket not connected (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    return False
                
                await connection.websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.warning(f"Send attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    continue
                return False
        return False
    
    async def broadcast_to_chat(
        self,
        chat_id: str,
        message: Dict[str, Any],
        exclude_user: Optional[str] = None,
    ):
        """Broadcast a message to all users subscribed to a chat"""
        subscribers = self._chat_subscriptions.get(chat_id, set())
        
        for user_id in subscribers:
            if user_id == exclude_user:
                continue
            await self.send_to_user(user_id, message)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected users"""
        for user_id in list(self._user_connections.keys()):
            await self.send_to_user(user_id, message)
    
    def get_online_users(self) -> List[str]:
        """Get list of currently connected user IDs"""
        return list(self._user_connections.keys())
    
    def is_user_online(self, user_id: str) -> bool:
        """Check if a user is online"""
        return user_id in self._user_connections
    
    def get_chat_participants_online(self, chat_id: str) -> List[str]:
        """Get list of online users subscribed to a chat"""
        subscribers = self._chat_subscriptions.get(chat_id, set())
        return [uid for uid in subscribers if self.is_user_online(uid)]


class StreamingHandler:
    """
    Handler for LLM streaming responses over WebSocket.
    Manages the streaming state and message assembly.
    
    Implements aggressive chunk batching to reduce browser load:
    - Large buffer size to minimize WebSocket messages
    - Time-based flushing to ensure responsiveness
    
    Sends messages in the format expected by the frontend:
    { "type": "stream_start", "payload": { "message_id": ..., "chat_id": ... } }
    { "type": "stream_chunk", "payload": { "content": ... } }
    { "type": "stream_end", "payload": { "message_id": ..., "usage": {...} } }
    """
    
    # Aggressive batching - prioritize performance over real-time feel
    MIN_FLUSH_INTERVAL = 0.1  # 100ms - much less frequent than 60fps
    MAX_BUFFER_SIZE = 200  # characters before forced flush
    
    def __init__(self, manager: WebSocketManager, connection: Connection):
        self.manager = manager
        self.connection = connection
        self._is_streaming = False
        self._stop_requested = False
        self._current_message_id: Optional[str] = None
        self._current_chat_id: Optional[str] = None
        self._streaming_task: Optional[asyncio.Task] = None
        self._active_stream: Optional[Any] = None  # Reference to active LLM stream for cancellation
        
        # Batching state
        self._buffer: str = ""
        self._last_flush_time: float = 0
    
    def set_streaming_task(self, task: asyncio.Task):
        """Set the current streaming task for cancellation.
        
        Cancels any existing task before setting the new one to prevent
        multiple concurrent streams from mixing their content.
        """
        # Cancel any existing task first
        if self._streaming_task and not self._streaming_task.done():
            logger.warning("Cancelling existing streaming task before starting new one")
            self._streaming_task.cancel()
        
        # Reset stop flag for the new task
        self._stop_requested = False
        self._streaming_task = task
    
    def set_active_stream(self, stream: Any):
        """Set the active LLM stream for direct cancellation"""
        self._active_stream = stream
        logger.debug(f"Active stream set: {type(stream)}, has_response={hasattr(stream, 'response')}")
    
    async def request_stop(self):
        """Request to stop the current stream - closes stream immediately"""
        logger.info(f"Stop requested - cancelling task and closing stream (has_stream={self._active_stream is not None}, has_task={self._streaming_task is not None})")
        self._stop_requested = True
        
        # Cancel the streaming task FIRST (most reliable method)
        if self._streaming_task and not self._streaming_task.done():
            logger.info("Cancelling streaming task...")
            self._streaming_task.cancel()
            try:
                # Give it a moment to cancel, but don't block
                await asyncio.wait_for(asyncio.shield(self._streaming_task), timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.debug(f"Task cancel completed: {type(e).__name__}")
        
        # Close the active stream to interrupt network I/O immediately
        if self._active_stream:
            try:
                stream = self._active_stream
                
                # Method 1: Close the httpx response directly (most effective)
                if hasattr(stream, 'response'):
                    response = stream.response
                    if hasattr(response, 'aclose'):
                        logger.info("Closing httpx response via aclose()...")
                        try:
                            await asyncio.wait_for(response.aclose(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                    if hasattr(response, 'close'):
                        logger.info("Closing httpx response via close()...")
                        response.close()
                
                # Method 2: Close the stream itself
                if hasattr(stream, 'close'):
                    logger.info("Closing stream via close()...")
                    try:
                        result = stream.close()
                        if hasattr(result, '__await__'):
                            await asyncio.wait_for(result, timeout=0.5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                
                logger.info("Stream close attempted - LLM connection should be terminated")
            except Exception as e:
                logger.info(f"Stream close completed (may have already closed): {type(e).__name__}: {e}")
        
        # Clear references
        self._active_stream = None
        self._streaming_task = None
    
    async def _close_stream(self):
        """Close the active stream"""
        try:
            if self._active_stream:
                if hasattr(self._active_stream, 'response') and hasattr(self._active_stream.response, 'aclose'):
                    await self._active_stream.response.aclose()
                elif hasattr(self._active_stream, 'close'):
                    await self._active_stream.close()
        except Exception:
            pass
    
    def is_stop_requested(self) -> bool:
        """Check if stop has been requested"""
        return self._stop_requested
    
    def is_streaming(self) -> bool:
        """Check if currently streaming"""
        return self._is_streaming or (self._streaming_task and not self._streaming_task.done())
    
    def reset_stop(self):
        """Reset the stop flag"""
        self._stop_requested = False
        self._streaming_task = None
        self._active_stream = None
    
    async def start_stream(self, message_id: str, chat_id: str):
        """Start a new streaming response"""
        self._is_streaming = True
        self._stop_requested = False
        self._current_message_id = message_id
        self._current_chat_id = chat_id
        self._buffer = ""
        self._last_flush_time = asyncio.get_event_loop().time()
        
        await self.manager.send_to_connection(self.connection, {
            "type": "stream_start",
            "payload": {
                "message_id": message_id,
                "chat_id": chat_id,
            },
        })
    
    async def send_chunk(self, text: str):
        """Buffer a text chunk and flush if needed"""
        if not self._is_streaming:
            return
        
        self._buffer += text
        
        current_time = asyncio.get_event_loop().time()
        time_since_flush = current_time - self._last_flush_time
        
        # Flush if buffer is large enough or enough time has passed
        if len(self._buffer) >= self.MAX_BUFFER_SIZE or time_since_flush >= self.MIN_FLUSH_INTERVAL:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Send buffered content to client"""
        if not self._buffer:
            return
        
        await self.manager.send_to_connection(self.connection, {
            "type": "stream_chunk",
            "payload": {
                "message_id": self._current_message_id,
                "chat_id": self._current_chat_id,
                "content": self._buffer,
            },
        })
        
        self._buffer = ""
        self._last_flush_time = asyncio.get_event_loop().time()
    
    async def send_tool_call(self, tool_name: str, tool_id: str, arguments: Dict):
        """Send tool call notification"""
        await self.manager.send_to_connection(self.connection, {
            "type": "tool_call",
            "payload": {
                "message_id": self._current_message_id,
                "tool_call": {
                    "name": tool_name,
                    "id": tool_id,
                    "input": arguments,
                },
            },
        })
    
    async def send_tool_result(self, tool_id: str, result: Any):
        """Send tool result"""
        await self.manager.send_to_connection(self.connection, {
            "type": "tool_result",
            "payload": {
                "message_id": self._current_message_id,
                "tool_id": tool_id,
                "result": result,
            },
        })
    
    async def end_stream(self, input_tokens: int = 0, output_tokens: int = 0, parent_id: str = None, ui_events: list = None):
        """End the streaming response"""
        # Flush any remaining buffered content
        await self._flush_buffer()
        
        self._is_streaming = False
        
        payload = {
            "message_id": self._current_message_id,
            "chat_id": self._current_chat_id,
            "parent_id": parent_id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
        if ui_events:
            payload["ui_events"] = ui_events
        
        await self.manager.send_to_connection(self.connection, {
            "type": "stream_end",
            "payload": payload,
        })
        
        self._current_message_id = None
        self._current_chat_id = None
        self._buffer = ""
    
    async def send_error(self, error: str):
        """Send an error message"""
        await self.manager.send_to_connection(self.connection, {
            "type": "stream_error",
            "payload": {
                "message_id": self._current_message_id,
                "error": error,
            },
        })
        
        self._is_streaming = False
        self._current_message_id = None
        self._current_chat_id = None
        self._buffer = ""


# Global WebSocket manager instance
ws_manager = WebSocketManager()
