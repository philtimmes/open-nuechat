"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============ Auth Schemas ============

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginResponse(BaseModel):
    """Login response includes user data"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: 'UserResponse'


class TokenRefresh(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    tier: str
    tokens_used_this_month: int
    tokens_limit: int
    theme: str
    is_admin: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


# Forward reference resolution
LoginResponse.model_rebuild()


class UserUpdate(BaseModel):
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    theme: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


# ============ Chat Schemas ============

class ChatCreate(BaseModel):
    title: Optional[str] = "New Chat"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    is_shared: bool = False


class ChatUpdate(BaseModel):
    title: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    title: str
    model: str
    system_prompt: Optional[str]
    is_shared: bool
    total_input_tokens: int
    total_output_tokens: int
    selected_versions: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ChatListResponse(BaseModel):
    chats: List[ChatResponse]
    total: int
    page: int
    page_size: int


# ============ Message Schemas ============

class AttachmentInput(BaseModel):
    type: str  # image, file
    data: Optional[str] = None  # base64 for images
    url: Optional[str] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None


class MessageCreate(BaseModel):
    content: str
    attachments: Optional[List[AttachmentInput]] = None
    enable_tools: bool = True
    enable_rag: bool = False
    document_ids: Optional[List[str]] = None  # For RAG


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str
    content: Optional[str]
    content_type: str
    attachments: Optional[List[Dict]]
    artifacts: Optional[List[Dict]] = None  # Extracted code artifacts with timestamps
    tool_calls: Optional[List[Dict]]
    input_tokens: int
    output_tokens: int
    model: Optional[str]
    is_streaming: bool
    is_error: bool
    parent_id: Optional[str] = None  # For conversation branching
    metadata: Optional[Dict] = None  # Maps to message_metadata column
    created_at: datetime
    
    class Config:
        from_attributes = True
    
    @classmethod
    def model_validate(cls, obj, **kwargs):
        """Custom validator to map message_metadata to metadata"""
        if hasattr(obj, 'message_metadata'):
            # Create dict from ORM object
            data = {
                'id': obj.id,
                'chat_id': obj.chat_id,
                'role': obj.role.value if hasattr(obj.role, 'value') else obj.role,
                'content': obj.content,
                'content_type': obj.content_type.value if hasattr(obj.content_type, 'value') else obj.content_type,
                'attachments': obj.attachments,
                'artifacts': obj.artifacts,  # Include artifacts
                'tool_calls': obj.tool_calls,
                'input_tokens': obj.input_tokens,
                'output_tokens': obj.output_tokens,
                'model': obj.model,
                'is_streaming': obj.is_streaming,
                'is_error': obj.is_error,
                'parent_id': obj.parent_id,
                'metadata': obj.message_metadata,  # Map message_metadata -> metadata
                'created_at': obj.created_at,
            }
            return cls(**data)
        return super().model_validate(obj, **kwargs)


class MessageEdit(BaseModel):
    """For editing a message (creates a branch)"""
    content: str
    regenerate_response: bool = True  # Whether to generate new AI response after edit


# ============ Client Chat Schemas ============

class ClientMessageCreate(BaseModel):
    """For client-to-client chat"""
    content: str
    attachments: Optional[List[AttachmentInput]] = None


class ChatInvite(BaseModel):
    user_id: str
    role: str = "member"


# ============ Document/RAG Schemas ============

class DocumentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    file_type: str
    file_size: int
    is_processed: bool
    chunk_count: int
    knowledge_store_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentSearch(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: int = 5


class SearchResult(BaseModel):
    document_id: str
    document_name: str
    content: str
    similarity: float
    chunk_index: int


# ============ Billing Schemas ============

class UsageSummary(BaseModel):
    user_id: str
    year: int
    month: int
    tier: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_limit: int
    tokens_remaining: int
    usage_percentage: float
    total_cost: float
    request_count: int


class UsageHistory(BaseModel):
    date: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    requests: int


class InvoiceResponse(BaseModel):
    user_id: str
    email: str
    period: str
    tier: str
    base_price: float
    usage_summary: UsageSummary
    overage_tokens: int
    overage_cost: float
    total_amount: float


# ============ Theme Schemas ============

class ThemeColors(BaseModel):
    primary: str
    secondary: str
    background: str
    surface: str
    text: str
    text_secondary: str
    accent: str
    error: str
    success: str
    warning: str
    border: str


class ThemeCreate(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False
    colors: ThemeColors
    fonts: Optional[Dict[str, str]] = None


class ThemeResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    is_public: bool
    is_system: bool
    colors: Dict
    fonts: Dict
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============ Tool Schemas ============

class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolExecution(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


# ============ WebSocket Schemas ============

class WSMessage(BaseModel):
    type: str
    payload: Dict[str, Any]


class WSChatMessage(BaseModel):
    type: str = "chat_message"
    chat_id: str
    content: str
    attachments: Optional[List[AttachmentInput]] = None
    enable_tools: bool = True
    enable_rag: bool = False


class WSSubscribe(BaseModel):
    type: str = "subscribe"
    chat_id: str


class WSUnsubscribe(BaseModel):
    type: str = "unsubscribe"
    chat_id: str


# ============ Code Summary Schemas ============

class CodeSignatureEntry(BaseModel):
    name: str
    type: str  # function, class, method, variable, interface, type, endpoint
    signature: str
    file: str
    line: Optional[int] = None


class FileChange(BaseModel):
    path: str
    action: str  # created, modified, deleted
    language: Optional[str] = None
    signatures: List[CodeSignatureEntry] = []
    timestamp: str


class SignatureWarning(BaseModel):
    type: str  # missing, mismatch, orphan, library_not_found
    message: str
    file: Optional[str] = None
    signature: Optional[str] = None
    suggestion: Optional[str] = None


class CodeSummaryCreate(BaseModel):
    files: List[FileChange] = []
    warnings: List[SignatureWarning] = []
    auto_generated: bool = True


class CodeSummaryResponse(BaseModel):
    id: str
    chat_id: str
    files: List[FileChange]
    warnings: List[SignatureWarning]
    last_updated: str
    auto_generated: bool
