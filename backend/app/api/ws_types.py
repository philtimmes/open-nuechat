"""
WebSocket event types for type-safe messaging

This module defines all WebSocket event types used for communication
between the frontend and backend. It provides:

- Type definitions for all events
- Validation via Pydantic models
- Clear documentation of the protocol
"""
from typing import Optional, Dict, Any, List, Literal, Union
from pydantic import BaseModel
from datetime import datetime


# ============ Client -> Server Events ============

class WSSubscribe(BaseModel):
    """Subscribe to a chat room for real-time updates"""
    type: Literal["subscribe"] = "subscribe"
    chat_id: str


class WSUnsubscribe(BaseModel):
    """Unsubscribe from a chat room"""
    type: Literal["unsubscribe"] = "unsubscribe"
    chat_id: str


class WSChatMessage(BaseModel):
    """Send a chat message to the LLM"""
    type: Literal["chat_message"] = "chat_message"
    chat_id: str
    content: str
    attachments: Optional[List[Dict[str, Any]]] = None
    enable_tools: bool = True
    enable_rag: bool = False
    knowledge_store_ids: Optional[List[str]] = None
    parent_id: Optional[str] = None  # For conversation branching


class WSStopGeneration(BaseModel):
    """Request to stop current LLM generation"""
    type: Literal["stop_generation"] = "stop_generation"
    chat_id: str


class WSPing(BaseModel):
    """Heartbeat ping to keep connection alive"""
    type: Literal["ping"] = "ping"


class WSRegenerateMessage(BaseModel):
    """Regenerate an assistant message"""
    type: Literal["regenerate"] = "regenerate"
    chat_id: str
    message_id: str


# Union of all client message types
ClientMessage = Union[
    WSSubscribe, WSUnsubscribe, WSChatMessage, 
    WSStopGeneration, WSPing, WSRegenerateMessage
]


# ============ Server -> Client Events ============

class WSStreamStart(BaseModel):
    """Indicates start of streaming response"""
    type: Literal["stream_start"] = "stream_start"
    payload: Dict[str, Any]  # {message_id, chat_id}


class WSStreamChunk(BaseModel):
    """A chunk of streaming content"""
    type: Literal["stream_chunk"] = "stream_chunk"
    payload: Dict[str, Any]  # {message_id, content}


class WSStreamEnd(BaseModel):
    """Indicates end of streaming response"""
    type: Literal["stream_end"] = "stream_end"
    payload: Dict[str, Any]  # {message_id, chat_id, parent_id, usage}


class WSStreamError(BaseModel):
    """Error during streaming"""
    type: Literal["stream_error"] = "stream_error"
    payload: Dict[str, Any]  # {message_id, error}


class WSToolCall(BaseModel):
    """Notification of a tool being called"""
    type: Literal["tool_call"] = "tool_call"
    payload: Dict[str, Any]  # {message_id, tool_call: {name, id, input}}


class WSToolResult(BaseModel):
    """Result from a tool call"""
    type: Literal["tool_result"] = "tool_result"
    payload: Dict[str, Any]  # {message_id, tool_id, result}


class WSImageGeneration(BaseModel):
    """Image generation status update"""
    type: Literal["image_generation"] = "image_generation"
    payload: Dict[str, Any]  # {message_id, chat_id, status, image_base64?, ...}


class WSMessageSaved(BaseModel):
    """Confirmation that a message was saved with its real ID"""
    type: Literal["message_saved"] = "message_saved"
    payload: Dict[str, Any]  # {temp_id, real_id, parent_id, chat_id}


class WSPong(BaseModel):
    """Response to ping"""
    type: Literal["pong"] = "pong"


class WSError(BaseModel):
    """General error message"""
    type: Literal["error"] = "error"
    payload: Dict[str, Any]  # {message, code}


class WSSubscribed(BaseModel):
    """Confirmation of subscription"""
    type: Literal["subscribed"] = "subscribed"
    payload: Dict[str, Any]  # {chat_id}


class WSUnsubscribed(BaseModel):
    """Confirmation of unsubscription"""
    type: Literal["unsubscribed"] = "unsubscribed"
    payload: Dict[str, Any]  # {chat_id}


# Union of all server message types
ServerMessage = Union[
    WSStreamStart, WSStreamChunk, WSStreamEnd, WSStreamError,
    WSToolCall, WSToolResult, WSImageGeneration, WSMessageSaved,
    WSPong, WSError, WSSubscribed, WSUnsubscribed
]


# ============ Payload Type Definitions ============

class StreamStartPayload(BaseModel):
    """Payload for stream_start event"""
    message_id: str
    chat_id: str


class StreamChunkPayload(BaseModel):
    """Payload for stream_chunk event"""
    message_id: str
    content: str


class StreamEndPayload(BaseModel):
    """Payload for stream_end event"""
    message_id: str
    chat_id: str
    parent_id: Optional[str] = None
    usage: Dict[str, int]  # {input_tokens, output_tokens}


class ToolCallPayload(BaseModel):
    """Payload for tool_call event"""
    message_id: str
    tool_call: Dict[str, Any]  # {name, id, input}


class ImageGenerationPayload(BaseModel):
    """Payload for image_generation event"""
    message_id: str
    chat_id: str
    status: str  # queued, processing, completed, failed
    queue_position: Optional[int] = None
    image_base64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    prompt: Optional[str] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


class MessageSavedPayload(BaseModel):
    """Payload for message_saved event"""
    temp_id: str
    real_id: str
    parent_id: Optional[str] = None
    chat_id: str


# ============ Helper Functions ============

def create_stream_start(message_id: str, chat_id: str) -> dict:
    """Create a stream_start event"""
    return {
        "type": "stream_start",
        "payload": {"message_id": message_id, "chat_id": chat_id}
    }


def create_stream_chunk(message_id: str, content: str, chat_id: Optional[str] = None) -> dict:
    """Create a stream_chunk event"""
    payload = {"message_id": message_id, "content": content}
    if chat_id:
        payload["chat_id"] = chat_id
    return {
        "type": "stream_chunk",
        "payload": payload
    }


def create_stream_end(
    message_id: str, 
    chat_id: str, 
    input_tokens: int, 
    output_tokens: int,
    parent_id: Optional[str] = None
) -> dict:
    """Create a stream_end event"""
    return {
        "type": "stream_end",
        "payload": {
            "message_id": message_id,
            "chat_id": chat_id,
            "parent_id": parent_id,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        }
    }


def create_stream_error(message_id: Optional[str], error: str) -> dict:
    """Create a stream_error event"""
    return {
        "type": "stream_error",
        "payload": {"message_id": message_id, "error": error}
    }


def create_error(message: str, code: Optional[str] = None) -> dict:
    """Create a general error event"""
    payload = {"message": message}
    if code:
        payload["code"] = code
    return {"type": "error", "payload": payload}
