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
│   ├── assistant.py     # CustomAssistant, AssistantConversation, AssistantCategory
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
│   ├── admin/           # Admin panel components (NC-0.7.15)
│   │   ├── index.ts     # Re-exports components & types
│   │   ├── types.ts     # All admin type definitions
│   │   ├── SystemTab.tsx    # System settings tab
│   │   ├── OAuthTab.tsx     # OAuth settings tab
│   │   ├── FeaturesTab.tsx  # Feature flags tab
│   │   └── CategoriesTab.tsx # GPT categories tab
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
SCHEMA_VERSION = "NC-0.7.15"  # Current database schema version

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

### app/services/websocket.py (UPDATED NC-0.7.08)

```python
class StreamingHandler:
    """Handler for LLM streaming responses over WebSocket"""
    
    def __init__(self, manager: WebSocketManager, connection: Connection)
    
    def set_streaming_task(self, task: asyncio.Task)
        # Set the current streaming task for cancellation
        # NC-0.7.08: Now cancels any existing task before setting new one
        #            to prevent concurrent streams from mixing content
    
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
    
    def is_streaming(self) -> bool  # NC-0.7.08
        # Check if currently streaming (task running or _is_streaming flag)
    
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
  is_public, is_discoverable, is_global
  # Global KB settings
  global_min_score        # Relevance threshold (0-1, default 0.7)
  global_max_results      # Max chunks to return (default 3)
  # Required keywords filter (NC-0.8.0.1)
  require_keywords_enabled, required_keywords  # JSON list
  # Force trigger keywords (NC-0.8.0.8)
  force_trigger_enabled   # When True, bypasses global_min_score when keywords match
  force_trigger_keywords  # JSON list of trigger keywords
  force_trigger_max_chunks  # Max chunks when force triggered (default 5)
  # Stats
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

## Task Queue Service (app/services/task_queue.py) (NEW NC-0.7.17)

```python
# Agentic task queue for multi-step workflows
# LLM manages task flow via tools - NO automatic verification calls
# Queue persisted in Chat.metadata JSON column

class TaskStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

class TaskSource(str, Enum):
    USER = "user"
    LLM = "llm"

@dataclass
class AgentTask:
    id: str                          # UUID
    description: str                 # Short description
    instructions: str                # Detailed instructions (up to 512 tokens)
    status: TaskStatus = QUEUED
    auto_continue: bool = True       # Auto-start next task when complete
    priority: int = 0                # Higher = more urgent
    created_at: str                  # ISO datetime
    completed_at: Optional[str]      # ISO datetime when completed
    result_summary: Optional[str]    # Brief summary when completed
    source: TaskSource = USER

class TaskQueueService:
    def __init__(self, db: AsyncSession, chat_id: str): ...
    
    async def add_task(description, instructions, source, auto_continue=True, priority=0) -> AgentTask
        # Add task, auto-starts if queue not paused
    
    async def add_tasks_batch(tasks: List[Dict], source) -> List[AgentTask]
        # Add multiple tasks with optional priority/auto_continue per task
    
    async def complete_task(result_summary: Optional[str]) -> Tuple[AgentTask, Optional[AgentTask]]
        # Complete current task. Returns (completed, next_task_if_auto_continue)
    
    async def fail_task(reason: str) -> AgentTask
        # Mark current task as failed (no auto-retry)
    
    async def skip_task() -> Tuple[AgentTask, Optional[AgentTask]]
        # Skip current task, returns (skipped, next_task)
    
    async def pause_queue() -> None
        # Pause auto-execution
    
    async def resume_queue() -> Optional[AgentTask]
        # Resume, returns started task if any
    
    async def get_queue_status() -> Dict
        # {queue_length, current_task, queued_tasks, completed_count, 
        #  recent_completed, has_pending, paused}
    
    async def clear_queue() -> int
        # Clear all tasks, return count
    
    def get_system_prompt_addition() -> str
        # System prompt with current/queued/completed tasks

def parse_tasks_from_llm_response(response: str) -> List[Dict]
    # Parse tasks from JSON, XML, or markdown list format
```

### Tools (in tools/registry.py)

```python
add_task(description, instructions, priority=0, auto_continue=True)
    # -> {success, task_id, status, queue_length, current_task}

add_tasks_batch(tasks: [{description, instructions, priority?, auto_continue?}, ...])
    # -> {success, tasks_added, descriptions, queue_length, current_task}

complete_task(result_summary?: str)
    # -> {success, completed, result_summary, next_task?, message}

fail_task(reason?: str)
    # -> {success, failed_task, reason, queue_length}

skip_task()
    # -> {success, skipped, next_task?, queue_length}

get_task_queue()
    # -> {queue_length, current_task, queued_tasks, completed_count, paused, ...}

clear_task_queue()
    # -> {success, tasks_cleared}

pause_task_queue()
    # -> {success, paused, queue_length, current_task}

resume_task_queue()
    # -> {success, paused, queue_length, started_task?}
```

---

## RAG Service (app/services/rag.py) (UPDATED NC-0.7.16)

### Hybrid Search (NEW NC-0.7.16)

```python
# Hybrid search combines semantic FAISS search with exact identifier matching.
# This solves the problem of queries like "Rule 1S-2.053" failing because
# semantic search doesn't match specific codes well.

class RAGService:
    def _normalize_text_for_matching(self, text: str) -> str:
        # Normalize Unicode variants and range words for matching:
        # - All dash variants (em dash, en dash, minus) → hyphen (-)
        # - All quote variants (curly quotes) → straight quotes
        # - Range words: "1 to 9" → "1-9", "5 through 10" → "5-10"
        # Range conversion requires at least one side to have a digit
        # Allows "1—9" in document to match "1 to 9" in query
    
    def _extract_identifiers(self, query: str) -> List[str]:
        # Extract specific identifiers that should be searched exactly
        # Normalizes query first, then extracts patterns:
        # - Legal statutes: "1S-2.053", "768.28", "119.07(1)"
        # - Rule references: "FAC 61G15-30", "Rule 12-345"
        # - Section citations: "§ 119.07", "Section 768.28"
        # - Case numbers: "2024-CF-001234"
        # - Product codes: "ABC-123-XYZ"
        # Returns list of identifier strings found in query
    
    async def _identifier_search(
        self,
        db: AsyncSession,
        identifiers: List[str],
        knowledge_store_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        # Search for chunks containing specific identifiers using substring matching
        # Normalizes chunk content before comparison (handles Unicode dashes)
        # Scores based on: word boundary match (1.0) vs substring match (0.8)
        # Returns results with _match_type="identifier" and _matched_identifiers list
    
    def _merge_search_results(
        self,
        semantic_results: List[Dict[str, Any]],
        identifier_results: List[Dict[str, Any]],
        semantic_threshold: float = 0.7,
        identifier_boost: float = 0.15,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        # Merge semantic and identifier results
        # - Identifier matches get +0.15 score boost
        # - Chunks found by both methods use higher score
        # - Identifier-only matches included even if below semantic threshold
        # Returns merged results sorted by score
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

## NC-0.8.0.0 - Dynamic Tools & Assistant Modes

### Overview

Major architecture change: RAG and tools become **composable via filter chains** instead of hardcoded. Only Global KB remains always-on (invisible guardrails).

### New Database Models

#### AssistantMode (app/models/assistant_mode.py)

```python
class AssistantMode(Base):
    __tablename__ = "assistant_modes"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)       # "Creative Writing"
    description = Column(Text)
    icon = Column(String(500))                       # Path to SVG
    
    active_tools = Column(JSON)                      # ["web_search", "artifacts"]
    advertised_tools = Column(JSON)                  # ["web_search"]
    
    filter_chain_id = Column(String(36), ForeignKey("filter_chains.id"))
    
    sort_order = Column(Integer, default=0)
    enabled = Column(Boolean, default=True)
    is_global = Column(Boolean, default=True)
    created_by = Column(String(36))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
```

#### Schema Changes

```sql
-- New table
CREATE TABLE assistant_modes (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(500),
    active_tools JSON,
    advertised_tools JSON,
    filter_chain_id UUID REFERENCES filter_chains(id),
    sort_order INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    is_global BOOLEAN DEFAULT TRUE,
    created_by UUID,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Modified tables
ALTER TABLE custom_assistants ADD COLUMN mode_id UUID REFERENCES assistant_modes(id);
ALTER TABLE chats ADD COLUMN mode_id UUID REFERENCES assistant_modes(id);
ALTER TABLE chats ADD COLUMN active_tools JSON;
```

### New Filter Chain Primitives (app/filters/executor.py)

#### RAG Evaluators

```python
async def _prim_local_rag(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Search user's uploaded documents for current chat."""
    # config: query_var, output_var, top_k, threshold

async def _prim_kb_rag(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Search current assistant's knowledge base."""
    # config: query_var, output_var, top_k, threshold

async def _prim_global_kb(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Search global knowledge bases (always active, no icon)."""
    # config: query_var, output_var, top_k, threshold, kb_ids (optional)

async def _prim_user_chats_kb(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Search user's chat history knowledge base."""
    # config: query_var, output_var, top_k, threshold
```

#### Dynamic Tool Nodes

```python
async def _prim_export_tool(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Register LLM-triggered tool ($WebSearch="...")."""
    # config:
    #   name: str                    # "WebSearch"
    #   trigger_pattern: str         # r'\$WebSearch="([^"]+)"'
    #   advertise: bool              # Include in system prompt
    #   advertise_text: str          # "Use $WebSearch=..."
    #   erase_from_display: bool     # Hide trigger from UI
    #   keep_in_history: bool        # Preserve in message.content
    #   on_trigger: List[dict]       # Mini filter chain steps

async def _prim_user_hint(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Text selection popup menu item."""
    # config:
    #   label: str                   # "Explain"
    #   icon: str                    # Path to SVG
    #   location: str                # "response" | "query" | "both"
    #   prompt: str                  # "Explain: {$Selected}"
    #   on_trigger: List[dict]       # Optional complex flow

async def _prim_user_action(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
    """Button below chat messages."""
    # config:
    #   label: str                   # "Search"
    #   icon: str                    # Path to SVG
    #   position: str                # "response" | "query" | "input"
    #   prompt: str                  # "Search: {$MessageContent}"
    #   on_trigger: List[dict]       # Optional complex flow
```

### ExecutionContext Enhancement

```python
@dataclass
class ExecutionContext:
    # ... existing fields ...
    
    # Content source flags (NEW)
    from_llm: bool = False    # True when processing LLM output
    from_user: bool = False   # True when processing user input
```

### Special Variables

| Variable | Context | Description |
|----------|---------|-------------|
| `$Selected` | user_hint | User's selected text |
| `$MessageContent` | user_action | Full message content |
| `$MessageId` | user_hint, user_action | Message ID |
| `$MessageRole` | user_hint, user_action | "user" or "assistant" |
| `$1`, `$2`, etc. | export_tool | Regex capture groups |
| `$Query` | all | Original user query |
| `$Var[name]` | all | Named variable |
| `$PreviousResult` | all | Last step output |

### WebSocket Events (NEW)

```python
# Server -> Client: Tools available for this chat
class WSDynamicTools(BaseModel):
    type: Literal["dynamic_tools"]
    payload: {
        chat_id: str,
        tools: List[{
            name: str,
            label: str,
            icon: str,
            location: str,           # "response" | "query" | "both"
            trigger_source: str      # "llm" | "user" | "both"
        }]
    }

# Client -> Server: User triggered a tool
class WSTriggerTool(BaseModel):
    type: Literal["trigger_tool"]
    payload: {
        chat_id: str,
        tool_name: str,
        selected_text: str,
        message_id: str
    }
```

### API Endpoints (NEW)

```python
# Assistant Modes
GET    /api/assistant-modes              # List all modes
POST   /api/assistant-modes              # Create mode (admin)
GET    /api/assistant-modes/{id}         # Get mode
PUT    /api/assistant-modes/{id}         # Update mode (admin)
DELETE /api/assistant-modes/{id}         # Delete mode (admin)

# Chat tools
GET    /api/chats/{id}/tools             # Get active tools for chat
PUT    /api/chats/{id}/tools             # Update active tools
PUT    /api/chats/{id}/mode              # Change chat mode
```

### Tool Categories

```python
class ToolCategory(Enum):
    ALWAYS_ACTIVE = "always"       # Global KB (no icon, invisible)
    MODE_CONTROLLED = "mode"       # Web, artifacts, images
    USER_TOGGLEABLE = "toggle"     # User can enable/disable
```

### Active Tools Bar UI

- Line-based SVG icons (admin uploadable)
- **Active**: Glow effect (brighter, same hue as background)
- **Inactive**: No glow (muted but visible)
- **Hover**: Subtle brightness increase + tooltip
- Global KB has **no icon** (always active, invisible)

### Default Assistant Modes

| Mode | Active Tools | Advertised |
|------|--------------|------------|
| Creative Writing | None | None |
| Coding | artifacts, file_ops, code_exec | artifacts, file_ops |
| Deep Research | web_search, citations | web_search |
| General | web_search, artifacts | None |

### Removed (from websocket.py)

| Removed | Replacement |
|---------|-------------|
| Hardcoded Global KB search | `global_kb` primitive (always active, invisible) |
| Hardcoded User Chats KB search | `user_chats_kb` primitive (mode-controlled) |
| Hardcoded Assistant KB search | `kb_rag` primitive (mode-controlled) |
| Hardcoded Local Docs search | `local_rag` primitive (mode-controlled) |

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

## OpenAI-Compatible API (v1)

NueChat provides an OpenAI-compatible REST API for programmatic access.

### Base URL

```
https://your-nuechat-instance.com/v1
```

### Authentication

All API requests require a Bearer token using a user-generated API key:

```bash
curl https://chat.example.com/v1/chat/completions \
  -H "Authorization: Bearer nxs_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello!"}]}'
```

**Key format:** `nxs_` prefix + 64 hex characters

### API Key Management

Users create/manage API keys via Settings → API Keys or programmatically:

```python
# Endpoints (require JWT auth, not API key)
POST   /api/api-keys              # Create key (returns full key ONCE)
GET    /api/api-keys              # List keys (prefix only)
GET    /api/api-keys/{id}         # Get key details
PATCH  /api/api-keys/{id}         # Update key settings
DELETE /api/api-keys/{id}         # Delete/revoke key
POST   /api/api-keys/{id}/regenerate  # New key value, keep settings
```

**Create Key Request:**
```json
{
  "name": "My App Key",
  "scopes": ["completions", "models", "images"],
  "rate_limit": 100,
  "allowed_ips": ["1.2.3.4"],
  "allowed_assistants": ["assistant-id-1"],
  "expires_in_days": 90
}
```

**Scopes:**
| Scope | Access |
|-------|--------|
| `completions` | `/v1/chat/completions` |
| `models` | `/v1/models` |
| `images` | `/v1/images/generations` |
| `embeddings` | `/v1/embeddings` |
| `full` | All endpoints |

### Chat Completions

```
POST /v1/chat/completions
```

**Request:**
```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false,
  "tools": [...],
  "tool_choice": "auto",
  "knowledge_store_ids": ["kb-id-1"]
}
```

**Model Selection:**
- Base model: `"llama3.2"`, `"gpt-4"`, etc.
- Custom GPT: `"gpt:<assistant-id>"`

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1704825600,
  "model": "llama3.2",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "total_tokens": 15
  }
}
```

**Streaming (`stream: true`):**
```
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"!"}}]}
data: {"id":"chatcmpl-...","choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### Models

```
GET /v1/models
```

Returns base models + user's Custom GPTs (owned + subscribed).

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2",
      "object": "model",
      "owned_by": "system",
      "is_custom_gpt": false
    },
    {
      "id": "gpt:abc123",
      "object": "model",
      "owned_by": "user@example.com",
      "description": "My Custom Assistant",
      "is_custom_gpt": true,
      "root": "llama3.2"
    }
  ]
}
```

**Filtered by API Key:** If `allowed_assistants` is set on the key, only those Custom GPTs appear.

```
GET /v1/models/{model_id}
```

Get details for a specific model.

### Image Generation

```
POST /v1/images/generations
```

**Request:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "model": "z-image-turbo",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard",
  "response_format": "b64_json"
}
```

**Supported Sizes:**
- Square: `512x512`, `768x768`, `1024x1024`, `1080x1080`
- Landscape: `1280x720`, `1920x1080`, `1024x768`
- Portrait: `720x1280`, `1080x1920`, `768x1024`

**Response:**
```json
{
  "created": 1704825600,
  "data": [{
    "b64_json": "iVBORw0KGgo...",
    "revised_prompt": "A beautiful sunset over mountains"
  }]
}
```

```
GET /v1/images/models
```

List available image generation models.

### Embeddings

```
POST /v1/embeddings
```

**Request:**
```json
{
  "model": "text-embedding-ada-002",
  "input": "The food was delicious"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.0023, -0.0091, ...],
    "index": 0
  }],
  "model": "text-embedding-ada-002",
  "usage": {"prompt_tokens": 5, "total_tokens": 5}
}
```

### Error Responses

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "authentication_error",
    "code": 401
  }
}
```

| Code | Meaning |
|------|---------|
| 401 | Invalid/expired API key |
| 403 | Missing scope or IP not allowed |
| 404 | Model/resource not found |
| 429 | Rate limit exceeded |
| 500 | Server error |
| 503 | Service unavailable (image gen down) |

---

## Frontend File Operation Tools (NC-0.8.0.6)

These tools allow the LLM to view and modify files in the Artifacts panel. They are processed client-side in `WebSocketContext.tsx` during streaming.

### Tool Tags (LLM Output)

```xml
<!-- Request file content (chunked, 20KB at a time) -->
<request_file path="src/Main.cpp"/>
<request_file path="src/Main.cpp" offset="20000"/>

<!-- Find line number containing text -->
<find_line path="src/Main.cpp" contains="xbox360.lib"/>

<!-- Search for text in file(s) -->
<find path="src/Main.cpp" search="pragma comment"/>
<find search="xbox360"/>  <!-- Search all artifacts -->

<!-- Replace text in file -->
<search_replace path="src/Main.cpp">
===== SEARCH
#pragma comment(lib, "xbox360.lib")
===== Replace
#pragma comment(lib, "xinput2.lib")
</search_replace>
```

### Processing Flow

1. LLM outputs tool tag during streaming
2. Frontend detects complete tag via regex pattern
3. Frontend sends `stop_generation` to backend
4. Frontend executes tool against artifacts in store
5. Frontend sends tool result via WebSocket `chat_message` with `save_user_message: false`
6. Backend continues LLM generation with tool result injected

### Key Functions (WebSocketContext.tsx)

```typescript
// Pattern matching
STREAM_REQUEST_FILE_PATTERN   // <request_file path="..." offset="..."/>
STREAM_FIND_LINE_PATTERN      // <find_line path="..." contains="..."/>  
STREAM_FIND_PATTERN_WITH_PATH // <find path="..." search="..."/>
STREAM_FIND_PATTERN_NO_PATH   // <find search="..."/>
STREAM_SEARCH_REPLACE_PATTERN // <search_replace>...</search_replace>

// Extraction
extractFileRequests(content): FileRequest[]
extractFindLineOperations(content): FindLineOp[]
extractFindOperations(content): FindOp[]
extractSearchReplaceOperations(content): SearchReplaceOp[]

// Execution
findArtifactByPath(artifacts, path): Artifact | undefined
executeSearchReplaceOperations(artifacts, ops): { updatedArtifacts, results, modifiedFiles }

// Result delivery
sendToolResult(toolName, results): void  // Sends to backend for LLM continuation
```

### Artifact Matching (findArtifactByPath)

Matches request path against artifact `filename` or `title`:
- Exact match (case-insensitive)
- Path suffix match (`src/Main.cpp` matches `project/src/Main.cpp`)
- Basename match (`Main.cpp` matches `src/Main.cpp`)
- Partial match (either direction)

### Tool Result Format

```
[SYSTEM TOOL RESULT - search_replace]
The following results were generated by the system, not typed by the user.

[SEARCH_REPLACE: Successfully replaced in src/Main.cpp (1 lines → 1 lines)]

[FILES SAVED] The following files were saved during your response: Main.cpp
You can reference these files in your continued response.

[END TOOL RESULT]
```

### Display Hiding (MessageBubble.tsx)

Tool tags are stripped from user display via `preprocessContent()`:
- `<request_file .../>` - hidden
- `<find_line .../>` - hidden
- `<find .../>` - hidden
- `<search_replace>...</search_replace>` - hidden

Tags remain in message history for context.

---

## Tool System (NC-0.8.0.7)

### Tool Categories

The frontend ActiveToolsBar uses category IDs that map to backend tool names:

| Category ID | Backend Tools | Description |
|-------------|---------------|-------------|
| `web_search` | `fetch_webpage`, `fetch_urls` | Web page fetching |
| `code_exec` | `execute_python` | Sandboxed Python execution |
| `file_ops` | `view_file_lines`, `search_in_file`, `list_uploaded_files`, `view_signature`, `request_file` | File viewing tools |
| `kb_search` | `search_documents` | Knowledge base RAG search |
| `image_gen` | `generate_image` | LLM-triggered image generation |
| `mcp_install` | `install_mcp_server`, `uninstall_mcp_server`, `list_mcp_servers` | Temporary MCP server installation (4h auto-cleanup) |
| `user_chats_kb` | (always available) | Agent Memory toggle (UI only) |
| `artifacts` | (frontend-only) | Artifact panel display |
| `local_rag` | (auto injection) | Chat document RAG context |
| `citations` | (prompt hint) | Citation formatting hint |

### Utility Tools (Always Available)

These tools are always sent to LLM when `enable_tools=true`:
- `calculator` - Math expressions
- `get_current_time` - Current datetime
- `format_json` - JSON formatting/validation
- `analyze_text` - Text statistics
- `agent_search` - Search archived conversation history
- `agent_read` - Read specific Agent Memory files

### Data Flow

1. **Frontend** stores `active_tools: string[]` in chat via `PUT /chats/{id}/tools`
2. **Frontend** displays ActiveToolsBar buttons based on stored categories
3. **Frontend** sends WebSocket message with `enable_tools: true`
4. **Backend** reads `chat.active_tools` from database
5. **Backend** maps categories to tool names via `TOOL_CATEGORY_MAP`
6. **Backend** filters `tool_registry.get_tool_definitions()` to allowed tools
7. **Backend** sends filtered tools to LLM

### Key Files

- `frontend/src/components/ActiveToolsBar.tsx` - Tool toggle UI
- `frontend/src/lib/api.ts` - `chatToolsApi.getActiveTools()`, `updateActiveTools()`
- `backend/app/api/routes/chats.py` - `GET/PUT /{chat_id}/tools`
- `backend/app/api/routes/websocket.py` - `TOOL_CATEGORY_MAP`, tool filtering logic
- `backend/app/tools/registry.py` - Tool definitions and handlers

---

## Image Generation System (NC-0.8.0.7)

### Admin Settings

New admin panel tab for configuring image generation defaults:

| Setting Key | Type | Default | Description |
|-------------|------|---------|-------------|
| `image_gen_default_width` | int | 1024 | Default width (256-2048) |
| `image_gen_default_height` | int | 1024 | Default height (256-2048) |
| `image_gen_default_aspect_ratio` | string | "1:1" | Default aspect ratio |
| `image_gen_available_resolutions` | JSON | [...] | Array of resolution options |

### Admin Endpoints

```python
GET /api/admin/image-gen-settings
  # Returns: ImageGenerationSettingsSchema
  # - default_width, default_height, default_aspect_ratio
  # - available_resolutions: [{label, width, height}, ...]

PUT /api/admin/image-gen-settings
  # Body: ImageGenerationSettingsSchema
  # Updates all image gen settings

GET /api/admin/public/image-settings
  # Public endpoint for frontend (no auth required)
  # Returns same as admin endpoint
```

### Global KB Admin Endpoints (NC-0.8.0.8)

```python
POST /api/knowledge-stores/admin/global/{store_id}
  # Set or update Global KB settings
  # Query params:
  #   is_global: bool = True
  #   min_score: float = 0.7           # Relevance threshold (0-1)
  #   max_results: int = 3             # Max chunks to return
  #   require_keywords_enabled: bool = False
  #   required_keywords: str           # JSON array string
  #   force_trigger_enabled: bool = False
  #   force_trigger_keywords: str      # JSON array string
  #   force_trigger_max_chunks: int = 5

GET /api/knowledge-stores/admin/all
  # List all knowledge stores (admin only)
  # Returns: List[KnowledgeStoreResponse] with global fields
```

### Global KB Admin UI Settings

In Admin Panel → Global KB tab, each store shows:

| Setting | Description | Range |
|---------|-------------|-------|
| Relevance Threshold | Minimum semantic similarity score | 0-1 (slider + input) |
| Max Results | Maximum chunks to return | 1-20 |
| Require Keywords | Only search when query contains keywords | Toggle + keywords list |
| Force Trigger | Bypass threshold when trigger keywords match | Toggle + keywords list + max chunks |

**Relevance Threshold Labels:**
- 0.8+ = strict (fewer but more relevant results)
- 0.6-0.8 = balanced
- 0.4-0.6 = lenient (more results)
- <0.4 = very lenient (many results)

### generate_image Tool

LLM can call image generation directly:

```python
# Tool definition (registry.py)
{
    "name": "generate_image",
    "description": "Generate an image based on a text prompt",
    "parameters": {
        "prompt": {"type": "string", "required": True},
        "width": {"type": "integer"},  # Optional, uses admin default
        "height": {"type": "integer"}, # Optional, uses admin default
    }
}

# Tool handler flow:
1. Fetch default dimensions from SettingsService
2. Queue image generation task with notify_callback
3. Return status to LLM immediately
4. On completion, notify_callback:
   - Saves image metadata to message_metadata in DB
   - Sends WebSocket "image_generated" event to frontend
```

### Image Persistence

Images persist across reloads via message metadata:

```typescript
// Message.metadata.generated_image
{
  url?: string;      // /api/images/generated/{job_id}.png
  width: number;
  height: number;
  seed: number;
  prompt: string;
  job_id?: string;
}

// Frontend loads on fetchMessages (messageSlice.ts)
for (const msg of messages) {
  if (msg.metadata?.generated_image) {
    generatedImagesFromMessages[msg.id] = {
      url: genImg.url,
      width: genImg.width,
      height: genImg.height,
      seed: genImg.seed ?? 0,
      prompt: genImg.prompt ?? '',
      job_id: genImg.job_id,
    };
  }
}
```

### Artifacts Panel Image Preview

Images now render in Preview tab instead of Code tab:

```typescript
// ArtifactsPanel.tsx
const canPreview = (art: Artifact): boolean => {
  return ['html', 'react', 'svg', 'markdown', 'mermaid', 'image'].includes(art.type);
};
```

### Download All with Images

```typescript
// ChatPage.tsx - handleDownloadAll()
if (artifact.type === 'image') {
  const imgSrc = artifact.imageData?.url || artifact.content;
  if (imgSrc.startsWith('/') || imgSrc.startsWith('http')) {
    const response = await fetch(fullUrl);
    const blob = await response.blob();
    zip.file(filename, blob);
  } else if (imgSrc.startsWith('data:')) {
    const base64Data = imgSrc.split(',')[1];
    zip.file(filename, base64Data, { base64: true });
  }
}
```

---

## Temporary MCP Server Installation (NC-0.8.0.7)

### Overview

LLM can install MCP (Model Context Protocol) servers on-demand when the `mcp_install` tool category is enabled. Servers are automatically removed after 4 hours of non-use.

### Tools

| Tool | Description |
|------|-------------|
| `install_mcp_server` | Install a temporary MCP server by name and URL |
| `uninstall_mcp_server` | Manually remove an installed server |
| `list_mcp_servers` | List all temporary servers with expiry status |

### install_mcp_server

```python
{
    "name": "install_mcp_server",
    "parameters": {
        "name": {"type": "string", "required": True},  # Display name
        "url": {"type": "string", "required": True},   # MCP URL or npx command
        "description": {"type": "string"},             # Optional description
        "api_key": {"type": "string"},                 # Optional API key
    }
}

# Example usage:
install_mcp_server(
    name="GitHub Tools",
    url="npx -y @modelcontextprotocol/server-github",
    description="GitHub repository management"
)
```

### Backend Service

```python
# app/services/temp_mcp_manager.py

class TempMCPManager:
    async def install_temp_server(db, user_id, name, url, description, api_key) -> Dict
    async def uninstall_temp_server(db, user_id, tool_id=None, tool_name=None) -> Dict
    async def list_temp_servers(db, user_id) -> List[Dict]
    async def refresh_server_usage(db, user_id, tool_id) -> bool
    async def cleanup_expired_servers() -> int

# Expiry configuration
TEMP_MCP_EXPIRY_HOURS = 4
CLEANUP_INTERVAL_SECONDS = 15 * 60  # Check every 15 minutes
```

### Storage

Temporary servers stored in `tools` table with special config:

```python
Tool(
    tool_type=ToolType.MCP,
    config={
        "is_temporary": True,
        "installed_at": "2026-01-09T12:00:00Z"
    },
    created_by=user_id,  # Private to installing user
    is_public=False
)
```

### Expiry Tracking

- `updated_at` timestamp refreshed on each tool usage
- Cleanup worker deletes tools where `updated_at < now() - 4 hours`
- Background task runs every 15 minutes

---

## Current Schema Version

**NC-0.8.0.12**

Changes:
- NC-0.8.0.12: **New Tools** - `search_replace` (exact find/replace in session files, unique match required), `web_search` (DuckDuckGo HTML search, no API key), `web_extract` (structured page extraction: headings/links/images/text), `grep_files` (cross-file regex search with context), `sed_files` (batch regex replace with preview mode). Updated `TOOL_CATEGORY_MAP` and `TOOL_DELEGATES` in websocket.py.
- NC-0.8.0.11: **Fresh Install Fix** - (1) `seed_default_settings()` populates `system_settings` from `SETTING_DEFAULTS` on startup if empty. (2) Fresh installs now detected properly and start at `NC-0.0.0` so all migrations run. Detection: no version row + empty system_settings = fresh install.
- NC-0.8.0.21: **Live Tool Call Streaming** - All tool calls now stream arguments into tool bubbles in real-time. New `tool_generating` WebSocket event fires when LLM first emits function name, creating bubble immediately. `tool_content_delta` emitted for ALL tools (removed `STREAMABLE_TOOLS` gate). Frontend `ToolContentPreview` shows raw JSON args for non-file tools, max-height 72px. Spinner icon for generating state. **Interrupted Reply Persistence** - `_accumulated_content` tracked in WebSocket streaming loop. On stop/cancel, partial content saved to assistant message via SQL update with `is_streaming=False`. Next user message chains properly in conversation tree. **Sandbox File Detection** - `execute_python` snapshots sandbox directory before execution, diffs after. New images (png/jpg/gif/webp/svg) base64-encoded and sent as `direct_image` events. New text files stored in session and saved as artifacts via `generated_artifacts`. **Web Fetch Content Pipeline** - Unified fetch handler: detect type from content-type/URL, HTML gets extraction, everything else raw. `full_content` field added to result dict, sent over WebSocket for artifact panel. Stripped from LLM serialization. `_save_web_retrieval_artifact` also calls `store_session_file()` for immediate access. Header removed from saved artifacts. **TOOL_RESULT UI Fix** - `GET /{chat_id}/messages` filters `ContentType.TOOL_RESULT` messages from response. LLM history (`build_messages`) unchanged. **Shift+Enter** - Added `remark-breaks` npm package to `ReactMarkdown` plugins. **Whisper Long Audio** - `_transcribe_sync` uses `model.generate()` directly for audio >30s with 30s chunks and 5s overlap stride. **Rumble** - CSP `frame-src` includes `https://rumble.com`. Proxy retry (up to 5), `--impersonate chrome` via `curl-cffi`. `app.tools.registry` logger set to INFO. New WebSocket events: `tool_generating`. New npm dependency: `remark-breaks`. New pip dependency: `curl-cffi`.
- NC-0.8.0.8: **Force Trigger Keywords for Global KB** - New DB columns: `force_trigger_enabled`, `force_trigger_keywords`, `force_trigger_max_chunks`. When enabled and query contains trigger keywords, KB content is always loaded bypassing the semantic search score threshold. Enhanced Admin UI with relevance threshold slider showing labels (strict/balanced/lenient). Added `KnowledgeStoreResponse` schema fields for force trigger. Backend fallback to direct DB query if FAISS returns empty for force-triggered stores. **Tool Debugging** - New `debug_tool_advertisements` and `debug_tool_calls` settings in Admin Site Dev. When enabled, logs `[DEBUG_TOOL_ADVERTISEMENTS]` (tool definitions) and `[DEBUG_TOOL_CALLS]` (call requests/results) to backend console.
- NC-0.8.0.7: **Admin Image Gen Settings** - New admin tab for default resolution, aspect ratio, available resolutions. Settings stored in `system_settings` table. **Image Persistence** - `generate_image` tool saves metadata to `message_metadata`, frontend loads on `fetchMessages`. **Image Preview** - Added 'image' to `canPreview()` for proper preview rendering. **Download All Images** - Fetches actual image files for ZIP. **Image Context Hidden** - `[IMAGE CONTEXT]` blocks stripped from display. **Temporary MCP Installation** - New `mcp_install` tool category with `install_mcp_server`, `uninstall_mcp_server`, `list_mcp_servers` tools. Servers auto-removed after 4 hours of non-use. New service `temp_mcp_manager.py` with cleanup worker. **Active Tools Filtering** - Fixed tool buttons in ActiveToolsBar not affecting LLM tool availability. WebSocket handler now reads `chat.active_tools` and filters tools based on category. Added `TOOL_CATEGORY_MAP` mapping frontend categories to backend tool names. Added `UTILITY_TOOLS` list (calculator, time, json, text analysis, agent memory) always available. **Streaming Tool Calls** - Fixed tools not being sent to LLM in streaming mode. Added tool call handling in `stream_message()` to execute tools and continue conversation. **API Model Names** - `/v1/models` now exposes assistants by cleaned name instead of UUID (spaces→underscores, non-alphanumeric removed, lowercased). **Performance Indexes** - Added indexes for documents, token_usage, and messages tables.
- NC-0.8.0.6: **Smart RAG Compression** - RAG results now re-ranked by subject relevance (keyword overlap boost), large chunks auto-summarized using extractive summarization (top sentences by query relevance), token budget support for context building. New methods: `_rerank_by_subject()`, `_summarize_chunk()`. `get_knowledge_store_context()` now accepts `max_context_tokens` and `enable_summarization` parameters. **Category filter fix** - Added `category` field to `AssistantPublicInfo`. Updated `/explore`, `/discover`, `/subscribed` to return category. `/assistants/categories` returns slugified mode names.
- NC-0.8.0.5: **Token limits & Agent Memory tools** - Added `max_input_tokens` and `max_output_tokens` columns to `chats` table. Added `agent_search` and `agent_read` tools to tool registry for accessing archived conversation history in Agent Memory files. Error sanitization prevents raw API errors from reaching browser. Max output tokens validated against remaining context.
- NC-0.8.0.4: **Mode/Category unification** - Categories now pull from AssistantModes table. Emojis derived from mode name in code (not stored). Category slugs mapped for backward compatibility.
- NC-0.8.0.3: **Sidebar real-time updates & timestamp preservation** - Sidebar now reloads automatically on chat create/delete/message via `sidebarReloadTrigger`. Removed SQLAlchemy `onupdate` trigger from `Chat.updated_at` - timestamps only update when user actually sends a message. Fixed migration system to only run new migrations (was running all on every startup). Import preserves original timestamps (created_at = first message, updated_at = last message). Fixed chat click removing chat from sidebar (_assignedPeriod preservation).
- NC-0.8.0.1.2: **Chat source field** - Added `source` column to chats table for tracking import origin (native, chatgpt, grok, claude). Migration backfills from title prefixes and cleans titles. Sidebar filtering uses source field instead of title parsing.
- NC-0.8.0.1.1: **Generated images in artifacts** - Images now appear in Artifacts panel as `imageNNN.ext`. Added `image` type to Artifact interface with `imageData` field. **Image context for LLM** - Backend injects generated image metadata into chat history so LLM knows what was previously generated.
- NC-0.8.0.1: **User data export fixes** - Fixed `model_id` → `model` and `display_name` → `username` in export endpoint.
- NC-0.8.0.0: **Dynamic Tools & Assistant Modes** - Major architecture change. See dedicated section below.
- NC-0.7.17: **Agentic Task Queue** - FIFO task queue for multi-step workflows. New service `task_queue.py` with `TaskQueueService`, tools (`add_task`, `add_tasks_batch`, `get_task_queue`, `clear_task_queue`), WebSocket `task_log` events, `chats.chat_metadata` column for queue storage
- NC-0.7.16: **Hybrid RAG search** - combines semantic FAISS search with exact identifier matching for legal statutes, rule references, case numbers, and product codes. New methods: `_extract_identifiers()`, `_identifier_search()`, `_merge_search_results()`
- NC-0.7.15: **Source-specific RAG prompts** (global KB, GPT KB, user docs, chat history), **keyword_check filter node** for quick chain exit, **search_documents tool enhancement** to search all sources, **Admin panel partial refactor** (types.ts, SystemTab, OAuthTab, FeaturesTab, CategoriesTab), **Admin panel scroll fix**, **SQLAlchemy cartesian product warning fix**
- NC-0.7.14: **Migration system fix** - always checks all migrations regardless of stored version, individually validates each change
- NC-0.7.13: **RAG context-aware query enhancement** - follow-up questions use conversation context for better search results
- NC-0.7.12: **Interrupt stream tree fix** - interrupted messages maintain proper parent-child relationships
- NC-0.7.11: **Admin-managed GPT categories** - new AssistantCategory model, admin UI, API endpoints, migration for category column
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
