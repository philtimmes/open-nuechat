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
SCHEMA_VERSION = "NC-0.6.37"  # Current database schema version

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

**NC-0.6.37**

Changes:
- NC-0.6.37: LLM confirmation for image generation (safe fallback on error - returns False, not regex)
- NC-0.6.36: API keys table with proper migration, parent_id branching fixes
- NC-0.6.35: Custom assistant chat association (assistant_id, assistant_name on chats)
- NC-0.6.34: Message artifacts JSON column
- NC-0.6.33: Procedural memory skill learning system
- NC-0.6.28: Split models.py into domain modules, security hardening
