# Open-NueChat - API & Function Signatures

## ⚠️ Development Rule: Software Switches

**ALL software switches/toggles NOT related to deployment should be in the Admin Panel.**

| Goes in .env | Goes in Admin Panel |
|--------------|---------------------|
| URLs, ports, API keys | Feature toggles |
| Database paths | Behavior switches |
| Secrets, credentials | Runtime configuration |
| Debug flags | User-facing options |

---

## Architecture Overview

### Backend Structure (Refactored)

```
backend/app/
├── api/
│   ├── routes/          # FastAPI route handlers
│   ├── helpers.py       # Shared route utilities
│   ├── exception_handlers.py  # Centralized error handling
│   ├── ws_types.py      # WebSocket event types
│   └── schemas.py       # Pydantic request/response schemas
├── core/
│   ├── config.py        # Pydantic settings
│   └── logging.py       # Structured logging utilities
├── db/
│   └── database.py      # SQLAlchemy async engine
├── models/              # SQLAlchemy ORM models (split by domain)
│   ├── __init__.py      # Re-exports all models
│   ├── base.py          # Base, generate_uuid, enums
│   ├── user.py          # User, OAuthAccount, APIKey
│   ├── chat.py          # Chat, Message, ChatParticipant
│   ├── document.py      # Document, DocumentChunk, KnowledgeStore, KnowledgeStoreShare
│   ├── assistant.py     # CustomAssistant, AssistantConversation
│   ├── billing.py       # TokenUsage
│   ├── tool.py          # Tool, ToolUsage
│   ├── filter.py        # ChatFilter
│   ├── filter_chain.py  # FilterChain (JSON-based chain definitions)
│   ├── upload.py        # UploadedFile, UploadedArchive
│   └── settings.py      # SystemSetting, Theme
├── services/
│   ├── auth.py          # JWT + bcrypt
│   ├── billing.py       # Token tracking
│   ├── document_queue.py  # Persistent doc processing
│   ├── llm.py           # OpenAI-compatible client
│   ├── rag.py           # FAISS + embeddings
│   ├── stt.py           # Speech-to-text
│   ├── token_manager.py # JWT blacklisting
│   ├── rate_limiter.py  # Token bucket rate limiting
│   ├── validators.py    # Input validation utilities
│   ├── zip_processor.py # Secure zip extraction
│   └── microservice_utils.py  # Shared microservice utilities
├── filters/             # Bidirectional stream filters
└── tools/               # Built-in LLM tools
```

### Frontend Structure (Refactored)

```
frontend/src/
├── components/          # React components
├── hooks/
│   ├── useMobile.ts
│   ├── useVoice.ts
│   └── useKeyboardShortcuts.ts  # Keyboard shortcuts hook
├── pages/               # Route pages
├── lib/
│   ├── api.ts           # API client
│   ├── artifacts.ts     # Artifact extraction
│   ├── formatters.ts    # Shared formatting utilities
│   └── wsTypes.ts       # WebSocket type guards
├── stores/
│   ├── chatStore.ts     # Legacy (re-exports from chat/)
│   ├── chat/            # Modular chat store
│   │   ├── index.ts     # Composed store
│   │   ├── types.ts     # Type definitions
│   │   ├── chatSlice.ts # Chat CRUD
│   │   ├── messageSlice.ts  # Message handling
│   │   ├── streamSlice.ts   # Streaming state
│   │   ├── artifactSlice.ts # Artifacts
│   │   └── codeSummarySlice.ts  # Code tracking
│   └── modelsStore.ts
└── types/               # TypeScript types
```

---

## Backend Signatures

### app/main.py

```python
SCHEMA_VERSION = "NC-0.6.77"  # Current database schema version

def parse_version(v: str) -> tuple  # Parse "NC-X.Y.Z" to (X, Y, Z)
async def run_migrations(conn)  # Run versioned DB migrations
async def lifespan(app: FastAPI)
  # Startup: imports all models, create_all, run_migrations, load filters, warmup STT
  # Starts: token_reset_checker (hourly), document_queue.worker
  # Shutdown: stops workers
async def health_check() -> { status, service, version, schema_version }
async def list_models(request: Request) -> { api_base, default_model, models, subscribed_assistants }
async def api_info() -> { ..., schema_version, ... }
async def get_shared_chat(share_id: str) -> { id, title, model, created_at, messages, all_messages?, has_branches? }
async def serve_spa(request: Request, full_path: str)  # SPA catch-all
```

### app/api/helpers.py (NEW)

```python
class ResourceNotFoundError(HTTPException)
  # status_code=404, detail="Resource '{id}' not found"

class PermissionDeniedError(HTTPException)
  # status_code=403, detail="You don't have permission to {action} this resource"

class TokenLimitExceededError(HTTPException)
  # status_code=402, detail="Token limit exceeded. Used: X, Limit: Y"

class RateLimitExceededError(HTTPException)
  # status_code=429, headers={"Retry-After": "60"}

async def get_owned_resource(
    db: AsyncSession,
    model_class: Type[T],
    resource_id: str,
    user: User,
    resource_name: str = "Resource",
    owner_field: str = "owner_id"
) -> T
  # Fetch + ownership check in one call

async def get_resource_or_404(
    db: AsyncSession,
    model_class: Type[T],
    resource_id: str,
    resource_name: str = "Resource"
) -> T

def check_token_limit(user, required_tokens: int = 0) -> None
def verify_admin(user) -> None
```

### app/api/exception_handlers.py (NEW)

```python
def setup_exception_handlers(app: FastAPI)
  # Register all exception handlers:
  # - RequestValidationError -> 422 with field details
  # - IntegrityError -> 409 for duplicates, 400 for FK violations
  # - ValueError -> 400
  # - Exception -> 500 (details hidden in production)
```

### app/api/ws_types.py (NEW)

```python
# Client -> Server Events
class WSSubscribe(BaseModel): type, chat_id
class WSUnsubscribe(BaseModel): type, chat_id
class WSChatMessage(BaseModel): type, chat_id, content, attachments?, enable_tools, enable_rag, knowledge_store_ids?, parent_id?
class WSStopGeneration(BaseModel): type, chat_id
class WSPing(BaseModel): type
class WSRegenerateMessage(BaseModel): type, chat_id, message_id

ClientMessage = Union[WSSubscribe, WSUnsubscribe, WSChatMessage, WSStopGeneration, WSPing, WSRegenerateMessage]

# Server -> Client Events
class WSStreamStart(BaseModel): type, payload{message_id, chat_id}
class WSStreamChunk(BaseModel): type, payload{message_id, content}
class WSStreamEnd(BaseModel): type, payload{message_id, chat_id, parent_id?, usage}
class WSStreamError(BaseModel): type, payload{message_id?, error}
class WSToolCall(BaseModel): type, payload{message_id, tool_call{name, id, input}}
class WSToolResult(BaseModel): type, payload{message_id, tool_id, result}
class WSImageGeneration(BaseModel): type, payload{message_id, chat_id, status, ...}
class WSMessageSaved(BaseModel): type, payload{temp_id, real_id, parent_id?, chat_id}
class WSPong(BaseModel): type
class WSError(BaseModel): type, payload{message, code?}
class WSSubscribed(BaseModel): type, payload{chat_id}
class WSUnsubscribed(BaseModel): type, payload{chat_id}

ServerMessage = Union[...]

# Helper functions
def create_stream_start(message_id, chat_id) -> dict
def create_stream_chunk(message_id, content, chat_id?) -> dict
def create_stream_end(message_id, chat_id, input_tokens, output_tokens, parent_id?) -> dict
def create_stream_error(message_id, error) -> dict
def create_error(message, code?) -> dict
```

### app/services/websocket.py (UPDATED NC-0.6.49)

```python
class StreamingHandler:
    """Handler for LLM streaming responses over WebSocket"""
    
    def __init__(self, manager: WebSocketManager, connection: Connection)
    
    def set_streaming_task(self, task: asyncio.Task)
        # Set the current streaming task for cancellation
    
    def set_active_stream(self, stream: Any)
        # Set the active LLM stream for direct cancellation
    
    async def request_stop(self)
        # Request to stop the current stream - AGGRESSIVE STOP (NC-0.6.49)
        # 1. Sets _stop_requested flag
        # 2. Cancels streaming task FIRST (most reliable)
        # 3. Closes httpx response via aclose() with timeout
        # 4. Closes stream via close() with timeout
        # 5. Clears task and stream references
    
    def is_stop_requested(self) -> bool
        # Check if stop has been requested
    
    def reset_stop(self)
        # Reset the stop flag for new stream
    
    async def start_stream(self, message_id: str, chat_id: str)
    async def add_content(self, content: str, force_flush: bool = False)
    async def flush_buffer(self)
    async def end_stream(self, input_tokens: int = 0, output_tokens: int = 0, parent_id: str = None)
    async def send_error(self, error: str, message_id: str = None)
```

### app/services/token_manager.py (NEW)

```python
class TokenBlacklist:
  # In-memory LRU blacklist with auto-cleanup
  async def add(token_jti: str, expires_at: datetime)
  async def is_blacklisted(token_jti: str) -> bool
  async def clear()  # For testing
  property size: int

token_blacklist = TokenBlacklist()  # Global instance

async def blacklist_token(token_jti: str, expires_at: datetime)
async def is_token_blacklisted(token_jti: str) -> bool
def should_rotate_refresh_token(issued_at: datetime) -> bool
def get_token_expiry(token_type: str = "access") -> datetime
```

### app/services/rate_limiter.py (UPDATED)

```python
class RateLimitExceeded(Exception):
  retry_after: int
  message: str

@dataclass
class RateLimitConfig:
  requests: int      # Per window
  window_seconds: int
  burst: int = None  # Defaults to requests

class TokenBucketRateLimiter:
  async def check(key: str, config: RateLimitConfig, cost: int = 1) -> Tuple[bool, Dict[str, int]]
    # Returns: (allowed, {X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, Retry-After?})
  async def check_rate_limit(action: str, identifier: str, cost: int = 1) -> None
    # Raises RateLimitExceeded if limit exceeded
  async def reset(key: str)  # For testing

rate_limiter = TokenBucketRateLimiter()  # Global instance

RATE_LIMITS = {
  "chat_message": RateLimitConfig(60, 60, 10),
  "image_generation": RateLimitConfig(10, 60, 3),
  "file_upload": RateLimitConfig(20, 60, 5),
  "api_key_creation": RateLimitConfig(5, 3600),
  "login_attempt": RateLimitConfig(10, 300),
  "password_reset": RateLimitConfig(3, 3600),
  "document_upload": RateLimitConfig(30, 3600),
  "knowledge_store_creation": RateLimitConfig(10, 3600),
}

async def check_rate_limit(user_id: str, action: str, cost: int = 1) -> Tuple[bool, Dict[str, int]]
async def check_rate_limit_ip(ip_address: str, action: str, cost: int = 1) -> Tuple[bool, Dict[str, int]]
```

### app/core/logging.py (UPDATED)

```python
class StructuredLogger:
  def __init__(name: str)
  def info(message: str, **fields)
  def warning(message: str, **fields)
  def error(message: str, **fields)
  def debug(message: str, **fields)
  def with_fields(**fields) -> StructuredLogger  # Create child logger

def get_logger(name: str) -> StructuredLogger
  # Primary way to get a logger instance

@contextmanager
def log_duration(operation: str, logger?: StructuredLogger, **extra_fields)
  # Context manager for timing operations

def log_llm_request(
  model, input_tokens, output_tokens, duration_ms,
  user_id, chat_id, filters_applied?, tool_calls?, error?, streaming?, logger?
)

def log_websocket_event(event_type, user_id, chat_id?, logger?, **extra_fields)
def log_security_event(event_type, user_id?, ip_address?, success?, logger?, **extra_fields)
```

### app/services/zip_processor.py (SECURITY UPDATED)

```python
# Security limits
MAX_FILES_IN_ARCHIVE = 10000
MAX_TOTAL_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB
MAX_PATH_DEPTH = 50
MAX_PATH_COMPONENT_LENGTH = 255

class ZipSecurityError(Exception):
  pass

def validate_zip_path(filename: str) -> None
  # Raises ZipSecurityError for:
  # - Null bytes
  # - Absolute paths
  # - Directory traversal (.. components)
  # - Excessive path depth
  # - Overly long components

def is_symlink(zip_info: zipfile.ZipInfo) -> bool
  # Check Unix mode for symlink bit

def validate_zip_archive(zf: zipfile.ZipFile) -> None
  # Full archive validation:
  # - File count limit
  # - Total size limit
  # - Path validation for each entry
  # - Symlink detection

class ZipProcessor:
  def process(self, zip_data: bytes) -> ZipManifest
    # Now calls validate_zip_archive() first
```

### app/services/validators.py (INTEGRATED)

```python
# File size constants
MAX_AVATAR_SIZE = 5 * 1024 * 1024   # 5 MB
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_ZIP_SIZE = 100 * 1024 * 1024   # 100 MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS: Set[str]
ALLOWED_DOCUMENT_EXTENSIONS: Set[str]
ALLOWED_ARCHIVE_EXTENSIONS: Set[str]

class FileValidationError(Exception):
  message: str
  details: dict

def validate_file_extension(filename: str, allowed: Set[str], category: str) -> Tuple[str, str]
  # Returns (basename, extension) or raises FileValidationError

def validate_file_size(size: int, max_size: int, category: str) -> None
  # Raises FileValidationError if size > max_size

def is_dangerous_file(filename: str) -> bool
  # Checks against DANGEROUS_PATTERNS (.exe, .dll, .bat, etc.)

# Text validators (available but not yet integrated)
def sanitize_string(s: str, max_length: int, strip_html: bool) -> str
def validate_username(username: str) -> str
def validate_email(email: str) -> str
def validate_slug(slug: str) -> str
def validate_password(password: str, min_length: int) -> str
def validate_hex_color(color: str) -> str
def validate_url(url: str, allowed_schemes: Set[str]) -> str
```

### app/services/image_gen.py (UPDATED NC-0.6.37)

```python
# Configuration
IMAGE_GEN_SERVICE_URL = os.getenv("IMAGE_GEN_SERVICE_URL", "http://localhost:8034")
IMAGE_GEN_TIMEOUT = float(os.getenv("IMAGE_GEN_TIMEOUT", "600"))
IMAGE_CONFIRM_WITH_LLM = os.getenv("IMAGE_CONFIRM_WITH_LLM", "true").lower() == "true"

# Pattern detection (regex pre-filter)
IMAGE_GEN_PATTERNS = [...]  # Wide net - catches potential image requests
NEGATIVE_PATTERNS = [...]   # Quick exclusions for obvious non-requests
# LLM confirmation is the authoritative check for intent

# Detection functions
async def confirm_image_request_with_llm(text: str) -> Tuple[bool, Optional[str]]
  # Uses direct AsyncOpenAI client (NOT LLMService)
  # Does NOT save to chat history
  # Does NOT track tokens against user quota
  # On error: returns (False, None) - SAFE DEFAULT, no image generated
  # Returns: (is_image_request, extracted_prompt)

def detect_image_request_regex(text: str) -> Tuple[bool, Optional[str]]
  # Quick regex-only detection (no LLM)
  # Returns: (is_image_request, extracted_prompt)

def detect_image_request(text: str) -> Tuple[bool, Optional[str]]
  # Synchronous, regex-only (legacy compatibility)
  # Alias for detect_image_request_regex

async def detect_image_request_async(text: str, use_llm: bool = None) -> Tuple[bool, Optional[str]]
  # Main entry point with optional LLM confirmation
  # 1. Regex pre-filter (fast)
  # 2. If regex matches and use_llm=True, confirm with LLM
  # use_llm defaults to IMAGE_CONFIRM_WITH_LLM env var

# Prompt extraction
def extract_image_prompt(text: str) -> str
  # Clean prompt by removing prefixes like "create an image of"

def extract_size_from_text(text: str) -> Tuple[Optional[int], Optional[int]]
  # Extract WIDTHxHEIGHT from text (e.g., "1280x720")

def extract_aspect_ratio_from_text(text: str) -> Tuple[Optional[int], Optional[int]]
  # Extract size from aspect ratio (e.g., "16:9" -> 1280x720)

# Service client
class ImageGenServiceClient:
  def __init__(base_url: str = IMAGE_GEN_SERVICE_URL)
  async def health_check() -> Dict[str, Any]
  async def is_available() -> bool
  async def generate_image(prompt, width=1024, height=1024, seed=None, **kwargs) -> Dict[str, Any]
  async def close()

def get_image_gen_client() -> ImageGenServiceClient  # Singleton
async def reset_image_gen_client()  # Reset on timeout errors

async def handle_image_request(
    prompt: str,
    user_id: str,
    chat_id: str,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]
  # Main entry point for image generation
  # Extracts size from prompt, checks service availability, generates image
```

---

## Model Signatures (Split by Domain)

### app/models/base.py

```python
Base = declarative_base()

def generate_uuid() -> str

class UserTier(str, Enum): FREE, PRO, ENTERPRISE
class MessageRole(str, Enum): USER, ASSISTANT, SYSTEM, TOOL
class ContentType(str, Enum): TEXT, IMAGE, FILE, TOOL_CALL, TOOL_RESULT
class SharePermission(str, Enum): VIEW, EDIT, ADMIN
class APIKeyScope(str, Enum): CHAT, KNOWLEDGE, ASSISTANTS, BILLING, FULL
class ToolType(str, Enum): MCP, OPENAPI
class FilterType(str, Enum): TO_LLM, FROM_LLM, TO_TOOLS, FROM_TOOLS
class FilterPriority(str, Enum): HIGHEST, HIGH, MEDIUM, LOW, LEAST
```

### app/models/user.py

```python
class User(Base):
  id, email, username, hashed_password, full_name, avatar_url
  is_active, is_verified, is_admin
  tier, stripe_customer_id
  tokens_used_this_month, tokens_limit
  theme, preferences
  created_at, updated_at, last_login
  # Relationships: oauth_accounts, chats, documents, token_usage, api_keys, knowledge_stores, custom_assistants
  @property is_unlimited -> bool  # Admin bypass

class OAuthAccount(Base):
  id, user_id, provider, provider_user_id
  access_token, refresh_token, token_expires_at
  created_at, updated_at

class APIKey(Base):
  id, user_id, name, key_prefix, key_hash
  scopes, rate_limit
  allowed_ips, allowed_assistants, allowed_knowledge_stores
  is_active, last_used_at, last_used_ip, usage_count
  expires_at, created_at, updated_at
  def has_scope(scope: APIKeyScope) -> bool
```

### app/models/chat.py

```python
class Chat(Base):
  id, owner_id, title, model, system_prompt
  is_shared, share_id
  total_input_tokens, total_output_tokens
  selected_versions, code_summary
  created_at, updated_at
  @property total_tokens -> int

class Message(Base):
  id, chat_id, sender_id, parent_id
  role, content, content_type
  attachments, tool_calls, tool_call_id
  input_tokens, output_tokens
  model, message_metadata
  is_streaming, is_error
  created_at
  @property total_tokens -> int
  @property has_siblings -> bool

class ChatParticipant(Base):
  id, chat_id, user_id, role
  joined_at, last_read_at
```

### app/models/document.py

```python
class Document(Base):
  id, owner_id, knowledge_store_id
  name, description, file_path, file_type, file_size
  is_processed, chunk_count
  created_at, updated_at

class DocumentChunk(Base):
  id, document_id
  content, chunk_index, embedding, chunk_metadata
  created_at

class KnowledgeStore(Base):
  id, owner_id
  name, description, icon, color
  is_public, is_discoverable
  document_count, total_chunks, total_size_bytes
  embedding_model, chunk_size, chunk_overlap
  created_at, updated_at

class KnowledgeStoreShare(Base):
  id, knowledge_store_id
  shared_with_user_id, share_token
  permission, expires_at, max_uses, use_count
  shared_by_user_id, message
  created_at, accepted_at
```

### app/models/assistant.py

```python
assistant_knowledge_stores = Table(...)  # M2M association

class CustomAssistant(Base):
  id, owner_id
  name, slug, tagline, description
  avatar_url, icon, color
  system_prompt, welcome_message, suggested_prompts
  model, temperature, max_tokens
  enabled_tools, custom_tools
  is_public, is_discoverable, is_featured
  conversation_count, message_count, total_tokens_used
  rating_sum, rating_count
  version, published_at
  created_at, updated_at
  @property average_rating -> float
  @property subscriber_count -> int

class AssistantConversation(Base):
  id, assistant_id, chat_id, user_id
  rating, feedback
  created_at
```

---

## Frontend Signatures

### frontend/src/components/FlowEditor.tsx (NEW NC-0.6.41)

Visual node-based filter chain editor using React Flow (@xyflow/react).

```typescript
interface FlowEditorProps {
  definition: FilterChainDefinition;
  onChange: (definition: FilterChainDefinition) => void;
  availableTools?: Array<{ value: string; label: string; category: string }>;
  filterChains?: Array<{ id: string; name: string }>;
}

// Custom node types:
// - StepNode: Configurable step with type-specific config panel
// - StartNode: Entry point for the flow

// Step type categories:
// - AI: to_llm, query
// - Tools: to_tool
// - Flow: go_to_llm, filter_complete, stop, block
// - Data: set_var, set_array, context_insert, modify
// - Logic: compare, call_chain

// Features:
// - Drag-and-drop node palette
// - Click to expand/configure nodes
// - Visual edge connections for jumps
// - Supports all filter chain step types
// - Real-time definition updates via onChange
```

### frontend/src/contexts/WebSocketContext.tsx (UPDATED NC-0.6.50)

Streaming detection patterns for real-time tool execution and artifact tracking.

```typescript
// Tool detection patterns (interrupt stream when matched)
const STREAM_FIND_LINE_PATTERN = /<find_line\s+path=["']...\/?>/i;
const STREAM_FIND_PATTERN_WITH_PATH = /<find\s+path=["']...["']\s+search=["']...["']\s*\/?>/i;
const STREAM_FIND_PATTERN_NO_PATH = /<find\s+search=["']...["']\s*\/?>/i;
const STREAM_REQUEST_FILE_PATTERN = /<request_file\s+path=["']...["']\s*\/?>/i;
const STREAM_SEARCH_REPLACE_PATTERN = /<search_replace\s+path=...>...SEARCH...REPLACE...</search_replace>/i;

// Artifact detection patterns (track but don't interrupt)
const STREAM_ARTIFACT_XML_PATTERN = /<artifact\s+...title=["']...["']...>...</artifact>/gi;
const STREAM_ARTIFACT_EQUALS_PATTERN = /<artifact=...>...</artifact>/gi;
const STREAM_ARTIFACT_FILENAME_PATTERN = /<filename.ext>...</filename.ext>/gi;
const STREAM_CODE_FENCE_PATTERN = /filename.ext\n```lang\n...\n```/gi;

// Refs for tracking during stream
processedToolTagsRef: Set<string>        // Prevents duplicate processing
savedArtifactsDuringStreamRef: string[]  // Artifact names for notification
toolCallHistoryRef: { key, timestamp }[] // Loop detection (max 3 same calls in 60s)

// Tool result notification includes saved artifacts:
// "[FILES SAVED] The following files were saved during your response: file1.ts, file2.tsx
//  You can reference these files in your continued response."
```

### frontend/src/stores/chat/types.ts (NEW)

```typescript
interface ChatSlice {
  chats: Chat[];
  currentChat: Chat | null;
  isLoadingChats: boolean;
  
  fetchChats: () => Promise<void>;
  createChat: (model?, systemPrompt?) => Promise<Chat>;
  setCurrentChat: (chat: Chat | null) => void;
  deleteChat: (chatId: string) => Promise<void>;
  deleteAllChats: () => Promise<void>;
  updateChatTitle: (chatId: string, title: string) => Promise<void>;
  updateChatLocally: (chatId: string, updates: Partial<Chat>) => void;
}

interface MessageSlice {
  messages: Message[];
  isLoadingMessages: boolean;
  
  fetchMessages: (chatId: string) => Promise<void>;
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  updateMessage: (messageId: string, updates: Partial<Message>) => void;
  replaceMessageId: (tempId: string, realId: string, parentId?) => void;
  retryMessage: (messageId: string) => Promise<void>;
  switchBranch: (messageId: string, branchIndex: number) => void;
}

interface StreamSlice {
  isSending: boolean;
  streamingContent: string;
  streamingToolCall: { name: string; input: string } | null;
  streamingArtifacts: Artifact[];
  error: string | null;
  
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (chunk: string) => void;
  setStreamingToolCall: (toolCall) => void;
  clearStreaming: () => void;
  setIsSending: (isSending: boolean) => void;
  setError: (error: string | null) => void;
  updateStreamingArtifacts: (content: string) => void;
}

interface ArtifactSlice {
  artifacts: Artifact[];
  selectedArtifact: Artifact | null;
  showArtifacts: boolean;
  uploadedArtifacts: Artifact[];
  zipUploadResult: Partial<ZipUploadResult> | null;
  zipContext: string | null;
  generatedImages: Record<string, GeneratedImage>;
  
  setSelectedArtifact: (artifact: Artifact | null) => void;
  setShowArtifacts: (show: boolean) => void;
  collectAllArtifacts: () => Artifact[];
  setZipUploadResult: (result) => void;
  fetchUploadedData: (chatId: string) => Promise<void>;
  setZipContext: (context: string | null) => void;
  setGeneratedImage: (messageId: string, image: GeneratedImage) => void;
}

interface CodeSummarySlice {
  codeSummary: CodeSummary | null;
  showSummary: boolean;
  
  setShowSummary: (show: boolean) => void;
  updateCodeSummary: (files: FileChange[], warnings?) => void;
  addFileToSummary: (file: FileChange) => void;
  addWarning: (warning: SignatureWarning) => void;
  clearSummary: () => void;
  fetchCodeSummary: (chatId: string) => Promise<void>;
  saveCodeSummary: () => Promise<void>;
}

interface ChatStore extends ChatSlice, MessageSlice, StreamSlice, ArtifactSlice, CodeSummarySlice {}
```

### frontend/src/lib/formatters.ts (NEW)

```typescript
function sanitizeCodeFences(text: string): string
function formatFileSize(bytes: number): string  // "1.5 MB"
function formatTokenCount(tokens: number): string  // "1.2K"
function formatDuration(ms: number): string  // "2.5s"
function truncate(text: string, maxLength: number): string
function stringToColor(str: string): string  // Deterministic HSL color
function formatRelativeTime(date: Date | string): string  // "2h ago"
function extensionToLanguage(ext: string): string  // ".py" -> "python"
function formatNumber(num: number): string  // Thousands separators
function formatDate(date: Date | string, options?): string
function formatMessageTime(date: Date | string): string
function escapeHtml(text: string): string
function snakeToTitle(str: string): string
function camelToTitle(str: string): string
```

### frontend/src/lib/wsTypes.ts (NEW)

```typescript
// Server event types
interface WSStreamStart { type: 'stream_start'; payload: { message_id, chat_id } }
interface WSStreamChunk { type: 'stream_chunk'; payload: { message_id, content } }
interface WSStreamEnd { type: 'stream_end'; payload: { message_id, chat_id, parent_id?, usage } }
interface WSStreamError { type: 'stream_error'; payload: { message_id?, error } }
interface WSToolCall { type: 'tool_call'; payload: { message_id, tool_call } }
interface WSToolResult { type: 'tool_result'; payload: { message_id, tool_id, result } }
interface WSImageGeneration { type: 'image_generation'; payload: { ... } }
interface WSMessageSaved { type: 'message_saved'; payload: { temp_id, real_id, parent_id?, chat_id } }
interface WSPong { type: 'pong' }
interface WSError { type: 'error'; payload: { message, code? } }
interface WSSubscribed { type: 'subscribed'; payload: { chat_id } }
interface WSUnsubscribed { type: 'unsubscribed'; payload: { chat_id } }

type ServerMessage = WSStreamStart | WSStreamChunk | ...

// Type guards
function isStreamStart(msg: ServerMessage): msg is WSStreamStart
function isStreamChunk(msg: ServerMessage): msg is WSStreamChunk
// ... (all event types)

function parseServerEvent(data: string): ServerMessage | null
function createClientMessage<T extends ClientMessage>(msg: T): string
function isStreamingEvent(msg: ServerMessage): boolean
function isToolEvent(msg: ServerMessage): boolean
function extractErrorMessage(msg: ServerMessage): string | null
```

### frontend/src/hooks/useKeyboardShortcuts.ts (NEW)

```typescript
interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  handler: (event: KeyboardEvent) => void;
  description?: string;
  preventDefault?: boolean;
  ignoreInInputs?: boolean;
}

function isMac(): boolean
function getModifierKey(): string  // "⌘" or "Ctrl"
function formatShortcut(shortcut: KeyboardShortcut): string

function useKeyboardShortcuts(shortcuts: KeyboardShortcut[], deps?: DependencyList): void

function useChatShortcuts(handlers: {
  onNewChat?: () => void;
  onFocusInput?: () => void;
  onToggleSidebar?: () => void;
  onDeleteChat?: () => void;
  onToggleArtifacts?: () => void;
  onClosePanel?: () => void;
  onSearch?: () => void;
  onSettings?: () => void;
}): void

function getChatShortcuts(): Array<{ shortcut: string; description: string }>
  // Returns: [
  //   { shortcut: "⌘+N", description: "New chat" },
  //   { shortcut: "⌘+/", description: "Focus input" },
  //   { shortcut: "⌘+B", description: "Toggle sidebar" },
  //   ...
  // ]
```

### frontend/src/lib/fileProcessor.ts (NEW NC-0.6.38)

```typescript
// Extension to language/type mapping
const EXT_MAP: Record<string, { language: string; type: Artifact['type'] }>

// Signature extraction patterns by language
const SIGNATURE_PATTERNS: Record<string, Array<{ kind: string; pattern: RegExp }>>

// Extract code signatures from file content
function extractSignatures(content: string, language: string): CodeSignature[]

// File reading utilities
async function readFileAsText(file: File): Promise<string>
async function readFileAsBase64(file: File): Promise<string>

// Process a single file into an artifact
async function processFileToArtifact(file: File): Promise<Artifact | null>

// Process multiple files
async function processFilesToArtifacts(files: File[]): Promise<Artifact[]>

// Generate LLM context manifest
function generateFileManifest(artifacts: Artifact[]): string

// Partial file viewing utilities
function getFileLines(content: string, startLine: number, endLine?: number): string
function searchInFile(content: string, pattern: string, contextLines?: number): string[]
```

---

## Documents API (app/api/routes/documents.py)

### POST /documents/extract-text (NEW NC-0.6.38)

Extract text from binary documents (PDF, DOCX, XLSX, etc) without storing.

```python
@router.post("/extract-text")
async def extract_text_from_file(
    file: UploadFile,
    user: User = Depends(get_current_user),
) -> dict:
    # Returns: { filename, text, chars, lines, warning? }
    # Uses DocumentProcessor.extract_text() from rag.py
```

### DocumentProcessor (app/services/rag.py)

```python
class DocumentProcessor:
    @staticmethod
    async def extract_text(file_path: str, mime_type: str) -> str
        # Extracts text from: PDF (Tika/PyMuPDF), DOCX (python-docx),
        # XLSX (openpyxl), RTF, CSV, JSON, plain text
    
    @staticmethod
    async def extract_text_from_pdf_tika(file_path: str) -> str
    
    @staticmethod
    async def extract_text_from_docx(file_path: str) -> str
    
    @staticmethod
    async def extract_text_from_xlsx(file_path: str) -> str
    
    @staticmethod
    async def extract_text_from_rtf(file_path: str) -> str
```

---

## User Settings API (app/api/routes/user_settings.py) (NEW NC-0.6.45)

### GET /user/chat-knowledge

Get current chat knowledge indexing status.

```python
@router.get("/chat-knowledge")
async def get_chat_knowledge_status(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatKnowledgeStatus:
    # Returns: { enabled, status, indexed_count, total_count, last_indexed }
    # status: "idle" | "processing" | "completed"
```

### POST /user/chat-knowledge

Toggle chat knowledge indexing on/off.

```python
@router.post("/chat-knowledge")
async def toggle_chat_knowledge(
    request: ChatKnowledgeToggle,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatKnowledgeStatus:
    # When enabled: Creates "My Chat History" knowledge store
    # Starts background task to index all chats (newest first)
    # When disabled: Clears indexed flags and deletes knowledge store
```

---

## Chats API (app/api/routes/chats.py)

### POST /chats/{chat_id}/uploaded-files (NEW NC-0.6.39)

Save an individual uploaded file to the database for persistence across page refreshes.

```python
@router.post("/{chat_id}/uploaded-files")
async def save_uploaded_file(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    filename: str = Body(...),
    content: str = Body(...),
    language: str = Body(None),
    signatures: list = Body(default=[]),
) -> dict:
    # Returns: { success: True, filename }
    # Updates existing file if already present
    # Stores in uploaded_files table (archive_name=None for individual uploads)
```

### GET /chats/{chat_id}/uploaded-files

Get all uploaded files for a chat (for restoring artifacts after page refresh).

```python
@router.get("/{chat_id}/uploaded-files")
async def get_chat_uploaded_files(
    chat_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    # Returns: { artifacts: [...], archive: {...} | null }
```

---

## WebSocket Routes (app/api/routes/websocket.py) (UPDATED NC-0.6.40)

### Chat Title Generation (SINGLE PATH)

```python
async def generate_chat_title(first_message: str, db: AsyncSession) -> str:
    # THE ONLY title generation function - no other paths
    # Creates a FRESH LLMService instance to avoid custom assistant pollution
    # Uses admin-configured title_generation_prompt
    # Falls back to first 50 chars of message on error
    # REST API (chats.py) does NOT generate titles - relies on this
```

### Knowledge Store Search Architecture (NC-0.6.40)

```python
# In handle_chat_message() - order matters:

# 1. ALWAYS search global knowledge stores (regardless of RAG settings)
global_results, global_store_names = await rag_service.search_global_stores(db, content)
# Injected as "AUTHORITATIVE KNOWLEDGE BASE" with trusted_knowledge tags

# 2. Check if chat is associated with a Custom GPT (assistant)
assistant_conv = await db.execute(
    select(AssistantConversation).where(AssistantConversation.chat_id == chat_id)
)
if assistant_conv:
    # Search ONLY that assistant's knowledge stores
    context = await rag_service.get_knowledge_store_context(
        knowledge_store_ids=assistant_ks_ids,
        bypass_access_check=True  # Allow access through public GPT
    )
elif enable_rag:
    # 3. Search user's unitemized documents (NOT in any KB)
    context = await rag_service.get_context_for_query(...)
```

**Search Priority:**
1. Global KBs → Always searched, injected as authoritative
2. Custom GPT KBs → Only when using that GPT, bypasses access check
3. User documents → Only when `enable_rag=true` AND no Custom GPT active

---

## Chats API (app/api/routes/chats.py) (UPDATED NC-0.6.40)

### send_message - KB Search (mirrors websocket.py)

```python
@router.post("/{chat_id}/messages")
async def send_message(...):
    # Same 3-tier KB search as websocket.py:
    # 1. search_global_stores() - always
    # 2. get_knowledge_store_context() - if Custom GPT active
    # 3. get_context_for_query() - if enable_rag and no Custom GPT
    
    # NO title generation - handled by websocket.py
```

---

## Admin Routes (app/api/routes/admin.py) (UPDATED NC-0.6.50)

### RAG Model Management

```python
@router.get("/admin/rag/status")
async def get_rag_status(user: User = Depends(get_current_user)) -> dict:
    # Returns: { status, details: {loaded, failed, model_exists}, message? }
    # status: "ok" if loaded, "not_loaded" otherwise
    # message: Hints to use reset endpoint if failed

@router.post("/admin/rag/reset")
async def reset_rag_model(user: User = Depends(get_current_user)) -> dict:
    # Resets model state and retries loading
    # Returns: { status, details, message }
    # Use after model loading failures to retry
```

---

## RAG Service (app/services/rag.py) (UPDATED NC-0.6.51)

### Chat Knowledge Context with Assistant Filtering (NEW NC-0.6.51)

```python
async def get_chat_knowledge_context(
    self,
    db: AsyncSession,
    user: User,
    query: str,
    chat_knowledge_store_id: str,
    current_assistant_id: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Get context from user's chat history knowledge store with assistant filtering.
    
    Filters results to match current assistant context:
    - If current_assistant_id is None: only return results from chats without an assistant
    - If current_assistant_id is set: only return results from chats with that specific assistant
    
    This prevents knowledge leakage between different assistant contexts.
    """
```

### Chunk Metadata for Chat History (UPDATED NC-0.6.51)

```python
# When indexing chats, chunk_metadata now includes:
{
    "chat_id": str,           # ID of the source chat
    "chat_title": str,        # Title of the source chat
    "source": "chat_history", # Constant identifier
    "assistant_id": str|None, # ID of assistant used (None if no assistant)
    "assistant_name": str|None # Name of assistant used
}
```

### Model Loading Fix

```python
# Fixed "meta tensor" errors from accelerate library
# Key change: low_cpu_mem_usage=False in all model loading

class RAGService:
    @classmethod
    def get_model(cls):
        # Multiple loading strategies with fallbacks:
        # 1. model_kwargs={"low_cpu_mem_usage": False}
        # 2. Basic load with device="cpu"
        # 3. trust_remote_code=True
        # 4. Explicit cache_folder
        
    @classmethod
    def reset_model(cls):
        # Reset model state to allow retry of loading
        # Sets _model=None, _model_loaded=False, _model_load_failed=False
        
    @classmethod
    def get_model_status(cls) -> dict:
        # Returns: { loaded, failed, model_exists }
```

---

## Tool Registry (app/tools/registry.py)

### Session File Storage (UPDATED NC-0.6.39)

```python
# Session-based file store with database fallback for uploaded files
# Key: chat_id, Value: Dict[filename, content]
_session_files: Dict[str, Dict[str, str]]

def store_session_file(chat_id: str, filename: str, content: str)
def get_session_file(chat_id: str, filename: str) -> Optional[str]
def get_session_files(chat_id: str) -> Dict[str, str]
def clear_session_files(chat_id: str)

# Database fallback functions (NEW NC-0.6.39)
async def get_session_file_with_db_fallback(chat_id: str, filename: str, db) -> Optional[str]
    # Try in-memory first, then query uploaded_files table
    # Caches DB results to session for future access

async def get_session_files_with_db_fallback(chat_id: str, db) -> Dict[str, str]
    # Merge in-memory files with database files
    # Survives server restarts
```

### File Viewing Tools (UPDATED NC-0.6.39)

```python
class FileViewingTools:
    @staticmethod
    async def view_file_lines(arguments: Dict, context: Dict) -> Dict
        # View specific lines from an uploaded file
        # Now uses DB fallback if file not in memory
        # Args: filename (str), start_line (int), end_line (int)
        # Returns: { filename, total_lines, showing, content }

    @staticmethod
    async def search_in_file(arguments: Dict, context: Dict) -> Dict
        # Search for pattern in file with context
        # Now uses DB fallback if file not in memory
        # Args: filename (str), pattern (str), context_lines (int)
        # Returns: { filename, pattern, total_matches, results[] }

    @staticmethod
    async def list_uploaded_files(arguments: Dict, context: Dict) -> Dict
        # List all uploaded files in session and database
        # Returns: { file_count, files[{filename, lines, size, preview}] }

    @staticmethod
    async def view_signature(arguments: Dict, context: Dict) -> Dict
        # View code around a function/class signature
        # Now uses DB fallback if file not in memory
        # Args: filename (str), signature_name (str), lines_after (int)
        # Returns: { filename, signature, start_line, content }
```

### Built-in Tools

| Tool | Description |
|------|-------------|
| calculator | Mathematical expression evaluation |
| get_current_time | Current time in any timezone |
| search_documents | RAG search in knowledge stores |
| execute_python | Sandboxed Python execution |
| format_json | JSON formatting and validation |
| analyze_text | Word count, character count, readability |
| fetch_webpage | Fetch and parse web page content |
| view_file_lines | View line range from uploaded file |
| search_in_file | Search pattern in uploaded file |
| list_uploaded_files | List all uploaded files |
| view_signature | View code around signature |

---

## Microservice Signatures

### TTS Service (tts-service/main.py)

```python
# Configuration
MAX_QUEUE_SIZE = 100
MAX_CONCURRENT = 2
RESULT_TTL_SECONDS = 300
MAX_TEXT_LENGTH = 10000

# Endpoints
GET  /health -> { status, service, device, gpu?, queue_size, max_queue, max_concurrent }
GET  /voices?gender=&lang= -> { voices: [{id, name, lang, gender}] }
POST /tts/submit -> { job_id, status, queue_position, message }
GET  /tts/status/{job_id} -> { job_id, status, queue_position, created_at, started_at?, completed_at?, error? }
GET  /tts/result/{job_id} -> audio/wav bytes
GET  /tts/stream/{job_id} -> audio/wav streaming
POST /tts/instant -> audio/wav (blocking)
```

### Image Service (image-service/main.py)

```python
# Configuration
MAX_QUEUE_SIZE = 20
MAX_CONCURRENT = 1
RESULT_TTL_SECONDS = 600
MIN_DIMENSION = 256
MAX_DIMENSION = 2048
MAX_PIXELS = 4_000_000
MIN_STEPS = 1
MAX_STEPS = 50

# Endpoints
GET  /health -> { status, service, device, gpu?, queue_size, max_queue }
GET  /sizes -> { sizes: [[width, height], ...] }
POST /generate/submit -> { job_id, status, queue_position, message }
GET  /generate/status/{job_id} -> { job_id, status, queue_position, ... }
GET  /generate/result/{job_id} -> { job_id, status, image_base64, width, height, seed, ... }
```

---

## Current Schema Version

**NC-0.6.68**

Changes:
- NC-0.6.77: **File tree view & breadcrumb fix** - tree/flat toggle in artifacts, fixed breadcrumb nav closing panel, agent file naming aligned to `{AgentNNNN}.md`
- NC-0.6.76: **Billing APIs admin tab** - configure Stripe/PayPal/Google Pay settings from Admin panel with test connection buttons
- NC-0.6.75: **Context overflow protection** - chunk large tool results (>32k chars) into hidden `{AgentNNNN}.md` files, searchable by LLM
- NC-0.6.74: **Auto-close incomplete tool tags** - salvage search_replace/replace_block even without closing tag
- NC-0.6.73: **KaTeX math rendering** - LaTeX notation support ($...$, $$...$$) via remark-math + rehype-katex
- NC-0.6.72: **Tool call closures** - added streaming detection for replace_block, improved inline closing tag handling
- NC-0.6.71: **Sidebar right margin** - added 30px padding so delete icon is reachable past scrollbar
- NC-0.6.70: **Fix filter chain infinite loop** - removed duplicate `_execute_steps()` call in executor.py
- NC-0.6.69: **Enhanced artifact guidance** - improved system prompt with CRITICAL section for artifact editing rules
- NC-0.6.68: **Artifact tools with state tracking** - prevents search_replace confusion, returns actual content on failure
- NC-0.6.67: **Persist LLM on disconnect** - streaming tasks continue when client leaves, content saved to DB, artifacts preserved
- NC-0.6.66: **Complete payment system** - Stripe/PayPal/Google Pay integration, subscriptions, payment methods, transaction history, webhooks
- NC-0.6.65: **Fix artifact closing tag** - detect `</artifact>` even when not alone on line (handles inline content before/after)
- NC-0.6.64: **Gzip & log error support** - client-side gzip decompression (with safety limits), log file error extraction with context, error summary fed to LLM
- NC-0.6.63: **Shared chat images & formatting** - attachments in shared API, matching formatting, user newlines preserved with whitespace-pre-wrap
- NC-0.6.62: **RAG embedding model auto-retry** - time-based retry after 60s, background startup load, retry_in_seconds in status
- NC-0.6.61: **Fix image display & markdown** - setCurrentChat skips clear when same chat (race fix), removed prose classes for explicit styling
- NC-0.6.60: **Immediate image display & markdown formatting** - images show after upload without refresh, added h1-h6/strong/em/hr handlers
- NC-0.6.59: **Fix image base64 truncation** - images were only sending first 50 chars to LLM, now sends full data
- NC-0.6.58: **Fix stream cross-chat contamination** - track streaming chat ID, validate before appending content, clear refs on chat change
- NC-0.6.57: **Thinking tokens support** - Admin LLM tab has think begin/end tokens, content hidden in collapsible panel, Admin scrolling fix
- NC-0.6.56: **Pre-generate assistant message IDs BEFORE streaming** - ID generated upfront, sent to frontend before content, used as parent_id for tool continuations
- NC-0.6.55: **Improved tool continuation parent_id tracking** - validate frontend parent_id, fallback to latest message, comprehensive logging
- NC-0.6.54: **Add kb_search tool** - LLM can search knowledge bases with `<kb_search query="...">` tag
- NC-0.6.53: **Fix tool loops creating branches** - backend now uses latest message as parent for tool continuations, ensuring linear flow
- NC-0.6.52: **Fix artifacts carrying over to New Chat** - added `preserveArtifacts` parameter to `createChat()`
- NC-0.6.51: **Chat Knowledge assistant context filtering** - filter chat history search by current assistant to prevent knowledge leakage between different GPTs
- NC-0.6.50: **RAG model loading fix** (meta tensor errors), RAG admin endpoints, artifact streaming detection, tool result notifications with saved file context
- NC-0.6.49: **Stop button fix** (background tasks), chat deletion removes knowledge index, streaming timeout (httpx.Timeout), All Models Prompt
- NC-0.6.46: Filter chain skip_if_rag_hit option, RAG search order reorganization (Global KB → Chat History KB → Assistant/User KB → Filter Chains)
- NC-0.6.45: Chat Knowledge Base feature - index all chats into personal knowledge store, green dot indicators for indexed chats
- NC-0.6.44: Chat search searches message content, infinite scroll pagination for chat list, import chats sorted newest-to-oldest
- NC-0.6.43: Grok chat import parser fix for nested response structure
- NC-0.6.42: Anonymous sharing option for shared chats
- NC-0.6.41: Visual node-based filter chain editor (FlowEditor.tsx), raw settings API endpoints (/admin/settings/raw, /admin/setting)
- NC-0.6.40: Global KB authoritative injection, unified KB search (Global always, Custom GPT only when active), single-path title generation (websocket.py only)
- NC-0.6.39: Aggressive stop generation (closes HTTP connection), file persistence to database, DB fallback for LLM file tools, remove KB upload rate limit
- NC-0.6.38: File upload to artifacts, partial file viewing tools (view_file_lines, search_in_file, view_signature), removed 100K char filter limit
- NC-0.6.37: LLM confirmation for image generation (safe fallback on error - returns False, not regex)
- NC-0.6.36: API keys table with proper migration, parent_id branching fixes
- NC-0.6.35: Custom assistant chat association (assistant_id, assistant_name on chats)
- NC-0.6.34: Message artifacts JSON column
- NC-0.6.33: Procedural memory skill learning system
- NC-0.6.28: Split models.py into domain modules, security hardening
