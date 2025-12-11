# Open-NueChat - API & Function Signatures

## Backend Signatures

app/main.py:
  SCHEMA_VERSION = "NC-0.6.27"  # Current database schema version
  def parse_version(v: str) -> tuple  # Parse "NC-X.Y.Z" to (X, Y, Z)
  async def run_migrations(conn)  # Run versioned DB migrations
  async def lifespan(app: FastAPI)
    # Startup: run_migrations, load filters, warmup STT
    # Starts: token_reset_checker (hourly), document_queue.worker
    # Shutdown: stops workers
  async def health_check() -> { status, service, version, schema_version }
  async def list_models(request: Request) -> { api_base, default_model, models, subscribed_assistants }
  async def api_info() -> { ..., schema_version, ... }
  async def get_shared_chat(share_id: str) -> { id, title, model, created_at, messages }  # Public endpoint
  async def serve_spa(request: Request, full_path: str)  # SPA catch-all route


app/api/dependencies.py:
  async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: AsyncSession = Depends(get_db)) -> User
  async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)), db: AsyncSession = Depends(get_db)) -> Optional[User]
  def require_tier(minimum_tier: str)
  async def check_tier(user: User = Depends(get_current_user))
  async def get_admin_user(user: User = Depends(get_current_user)) -> User


app/api/schemas.py:
  class UserCreate(BaseModel)
  class UserLogin(BaseModel)
  class TokenResponse(BaseModel)
  class TokenRefresh(BaseModel)
  class UserResponse(BaseModel)
  class UserUpdate(BaseModel)
  class LoginResponse(BaseModel)  # { access_token, refresh_token, token_type, expires_in, user }
  class ChatCreate(BaseModel)
  class ChatUpdate(BaseModel)
  class ChatResponse(BaseModel)
  class ChatListResponse(BaseModel)
  class AttachmentInput(BaseModel)
  class MessageCreate(BaseModel)
  class MessageResponse(BaseModel)
  class ClientMessageCreate(BaseModel)
  class ChatInvite(BaseModel)
  class DocumentResponse(BaseModel)
  class DocumentSearch(BaseModel)
  class SearchResult(BaseModel)
  class UsageSummary(BaseModel)
  class UsageHistory(BaseModel)
  class InvoiceResponse(BaseModel)
  class ThemeColors(BaseModel)
  class ThemeCreate(BaseModel)
  class ThemeResponse(BaseModel)
  class ToolDefinition(BaseModel)
  class ToolExecution(BaseModel)
  class WSMessage(BaseModel)
  class WSChatMessage(BaseModel)
  class WSSubscribe(BaseModel)
  class WSUnsubscribe(BaseModel)


app/api/routes/api_keys.py:
  # Admins bypass key limits (free: 3, pro: 10, enterprise: 100)
  class APIKeyCreate(BaseModel)
  class APIKeyResponse(BaseModel)
  class APIKeyCreatedResponse(BaseModel)
  class APIKeyUpdate(BaseModel)
  def generate_api_key() -> tuple[str, str, str]
  def hash_api_key(key: str) -> str
  async def create_api_key(request: APIKeyCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def list_api_keys(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_api_key(key_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def update_api_key(key_id: str, update_data: APIKeyUpdate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def delete_api_key(key_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def regenerate_api_key(key_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_api_key_user(request: Request, db: AsyncSession = Depends(get_db)) -> tuple[User, APIKey]
  def require_scope(required_scope: str)


app/api/routes/assistants.py:
  # Admins bypass assistant limits (free: 3, pro: 20, enterprise: 100)
  class AssistantCreate(BaseModel)
  class AssistantResponse(BaseModel)
  class AssistantUpdate(BaseModel)
  class AssistantPublicInfo(BaseModel)
  class ConversationRating(BaseModel)
  def generate_slug(name: str) -> str
  async def get_unique_slug(db: AsyncSession, base_slug: str, exclude_id: Optional[str] = None) -> str
  async def seed_default_assistant(db: AsyncSession) -> Optional[CustomAssistant]
  async def get_assistant_with_access(assistant_id: str, user: User, db: AsyncSession, require_owner: bool = False) -> CustomAssistant
  async def create_assistant(request: AssistantCreate, current_user: User, db: AsyncSession)
  async def list_my_assistants(current_user: User, db: AsyncSession)
  async def explore_assistants(search: Optional[str], featured_only: bool, limit: int, offset: int, db: AsyncSession)
  async def discover_assistants(search: Optional[str], featured_only: bool, limit: int, offset: int, current_user: User, db: AsyncSession)
  async def list_subscribed_assistants(current_user: User, db: AsyncSession) -> List[AssistantPublicInfo]
  async def get_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def update_assistant(assistant_id: str, update_data: AssistantUpdate, current_user: User, db: AsyncSession)
  async def delete_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def publish_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def unpublish_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def duplicate_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def start_conversation(assistant_id: str, current_user: User, db: AsyncSession)
  async def rate_conversation(assistant_id: str, chat_id: str, rating_data: ConversationRating, current_user: User, db: AsyncSession)
  async def list_my_conversations(assistant_id: str, current_user: User, db: AsyncSession)
  async def subscribe_to_assistant(assistant_id: str, current_user: User, db: AsyncSession)
  async def unsubscribe_from_assistant(assistant_id: str, current_user: User, db: AsyncSession)


app/api/routes/auth.py:
  async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)) -> LoginResponse
  async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)) -> LoginResponse
  async def refresh_token(data: TokenRefresh, db: AsyncSession = Depends(get_db))
  async def get_me(user: User = Depends(get_current_user))
  async def update_me(updates: UserUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def google_login(request: Request)
  async def google_callback(request: Request, code: str, state: str, db: AsyncSession = Depends(get_db))
  async def github_login(request: Request)
  async def github_callback(request: Request, code: str, state: str, db: AsyncSession = Depends(get_db))


app/api/routes/billing.py:
  async def get_usage(year: Optional[int] = None, month: Optional[int] = None, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_usage_history(days: int = Query(30, ge=1, le=90), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_usage_by_chat(limit: int = Query(10, ge=1, le=50), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def check_limit(estimated_tokens: int = Query(0, ge=0), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_invoice(year: int, month: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_tiers()
  async def upgrade_tier(tier: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))


app/services/billing.py:
  class BillingService:
    # Admins bypass all token limits (is_unlimited=True)
    async def check_token_reset(db: AsyncSession) -> Dict[str, Any]
      # Resets all users' tokens_used_this_month if interval has elapsed
      # Uses 'token_refill_interval_hours' system setting (default 720 = 30 days)
      # Stores 'last_token_reset_timestamp' in SystemSetting (ISO datetime)
      # Compares: datetime.now(timezone.utc) vs last_reset timestamp
      # If (now - last_reset).hours >= refill_hours: reset all users
      # Called from: background task (hourly), startup, health_check
      # 
      # When 'debug_token_resets' setting is "true", logs:
      #   - All users' email, tokens_used_this_month, tier
      #   - Current time, last reset, refill interval
      #   - "[Token Reset not necessary]" or "[Token Reset Queued]" + "[Token Reset Completed DATETIME]"
      #
      # When debug disabled: Silent operation (no logging)
      #
      # Returns: { action: 'reset'|'none', refill_interval_hours, hours_since_reset, ... }


app/main.py:
  # Imports: asyncio, contextlib.asynccontextmanager, pathlib.Path, fastapi, logging
  SCHEMA_VERSION = "NC-0.6.27"
  
  # Background Tasks (in lifespan):
  token_reset_checker()  # Runs every hour, checks if token reset needed
    # - Calls BillingService.check_token_reset()
    # - Logs when reset occurs (if debug enabled)
    # - Cancelled on shutdown
  
  document_queue.start_worker()  # Processes pending document uploads
    # - Polls queue for pending tasks
    # - Extracts text, creates embeddings
    # - Survives container restarts
    # - Stopped on shutdown


app/api/routes/admin.py:
  # ... existing endpoints ...
  
  # Debug Settings (NC-0.6.27+)
  GET  /api/admin/debug-settings -> DebugSettingsResponse
    # Returns: { debug_token_resets: bool, debug_document_queue: bool, last_token_reset_timestamp: str|null, token_refill_interval_hours: int }
  
  PUT  /api/admin/debug-settings
    # Body: { debug_token_resets?: bool, debug_document_queue?: bool }
    # Updates debug system settings


app/services/settings_service.py:
  # Default settings include:
  "debug_token_resets": "false"  # Log detailed token info on reset checks
  "debug_document_queue": "false"  # Log document queue processing


app/api/routes/branding.py:
  async def get_public_config()
  async def get_web_manifest()
  async def get_branding_css()
  async def get_available_themes()


app/api/routes/chats.py:
  async def list_chats(...) -> ChatListResponse  # { chats: [...], total, page, page_size }
  async def create_chat(chat_data: ChatCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> ChatResponse
  async def get_chat(chat_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def update_chat(chat_id: str, updates: ChatUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def update_selected_version(chat_id: str, user_message_id: str, version_index: int, ...)  # Persist UI version selection
  async def share_chat(chat_id: str, ...) -> { share_id: str }  # Generate public share link
  async def delete_chat(chat_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def delete_all_chats(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def list_messages(chat_id: str, ...) -> List[MessageResponse]  # Returns array directly
  async def send_message(chat_id: str, message_data: MessageCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def invite_to_chat(chat_id: str, invite: ChatInvite, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def send_client_message(chat_id: str, message_data: ClientMessageCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def _get_user_chat(db: AsyncSession, user: User, chat_id: str) -> Chat


app/api/routes/documents.py:
  async def list_documents(processed_only: bool = False, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def upload_document(file: UploadFile = File(...), description: Optional[str] = Form(None), user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_document(document_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def delete_document(document_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def reprocess_document(document_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def search_documents(search: DocumentSearch, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))


app/api/routes/filters.py:
  class FilterInfo(BaseModel)
  class RegistryStatusResponse(BaseModel)
  class ChatFiltersResponse(BaseModel)
  class FilterConfigUpdate(BaseModel)
  class FilterPresetRequest(BaseModel)
  class TestFilterRequest(BaseModel)
  async def get_registry_status(current_user: User = Depends(get_current_user))
  async def get_chat_filter_status(chat_id: str, current_user: User = Depends(get_current_user))
  async def enable_chat_filter(chat_id: str, filter_name: str, direction: str = "both", current_user: User = Depends(get_admin_user))
  async def disable_chat_filter(chat_id: str, filter_name: str, direction: str = "both", current_user: User = Depends(get_admin_user))
  async def configure_chat_filter(chat_id: str, update: FilterConfigUpdate, direction: str = "both", current_user: User = Depends(get_admin_user))
  def update_filter(chain, name: str, update: FilterConfigUpdate)
  async def apply_preset(request: FilterPresetRequest, current_user: User = Depends(get_admin_user))
  async def get_filter_details(chat_id: str, filter_name: str, direction: str = "to_llm", current_user: User = Depends(get_current_user))
  async def test_filters(request: TestFilterRequest, current_user: User = Depends(get_admin_user))
  async def remove_chat_filters(chat_id: str, current_user: User = Depends(get_admin_user))
  async def reset_registry(current_user: User = Depends(get_admin_user))
  async def list_priorities(current_user: User = Depends(get_current_user))


app/api/routes/tools.py:
  class ToolCreate(BaseModel) - name, description, tool_type, url, api_key, is_public, config, enabled_operations
  class ToolUpdate(BaseModel) - name, description, url, api_key, is_public, is_enabled, config, enabled_operations
  class ToolResponse(BaseModel) - id, name, description, tool_type, url, has_api_key, is_public, is_enabled, config, enabled_operations, schema_cache, last_schema_fetch, created_by, created_at
  class ToolExecuteRequest(BaseModel) - tool_name, params
  class ToolUsageResponse(BaseModel) - id, tool_id, tool_name, operation, success, result_summary, result_url, error_message, duration_ms, created_at
  class ToolUsageStat(BaseModel) - tool_id, tool_name, total_calls, successful_calls, failed_calls, total_duration_ms, avg_duration_ms, last_used, unique_users
  class AllToolsUsageStats(BaseModel) - total_calls, successful_calls, failed_calls, tools: List[ToolUsageStat]
  class ToolProbeRequest(BaseModel) - url, tool_type, api_key, config
  def require_admin(user: User) -> User - dependency for admin-only routes
  def tool_to_response(tool: Tool) -> ToolResponse
  async def list_tools(user: User, db: AsyncSession) -> List[ToolResponse]
  async def create_tool(data: ToolCreate, user: User, db: AsyncSession) -> ToolResponse
  async def get_tool(tool_id: str, user: User, db: AsyncSession) -> ToolResponse
  async def update_tool(tool_id: str, data: ToolUpdate, user: User, db: AsyncSession) -> ToolResponse
  async def delete_tool(tool_id: str, user: User, db: AsyncSession) -> dict
  async def probe_tool_url(data: ToolProbeRequest, user: User) -> dict
  async def refresh_tool_schema(tool_id: str, user: User, db: AsyncSession) -> dict
  async def test_tool(tool_id: str, data: ToolExecuteRequest, user: User, db: AsyncSession) -> dict
  async def get_available_tool_schemas(user: User, db: AsyncSession) -> dict
  async def get_chat_tool_usage(chat_id: str, user: User, db: AsyncSession) -> List[ToolUsageResponse]
  async def get_message_tool_usage(message_id: str, user: User, db: AsyncSession) -> List[ToolUsageResponse]
  async def get_tool_usage_stats(user: User, db: AsyncSession) -> AllToolsUsageStats
  async def reset_all_tool_usage(user: User, db: AsyncSession) -> dict
  async def reset_tool_usage(tool_id: str, user: User, db: AsyncSession) -> dict


app/api/routes/knowledge_stores.py:
  # Admins bypass store limits (free: 3, pro: 20, enterprise: 100)
  class KnowledgeStoreCreate(BaseModel)
  class KnowledgeStoreResponse(BaseModel)
  class KnowledgeStoreUpdate(BaseModel)
  class ShareCreate(BaseModel)
  class ShareResponse(BaseModel)
  class DocumentInfo(BaseModel)
  async def get_store_with_access(store_id: str, user: User, db: AsyncSession, required_permission: SharePermission = SharePermission.VIEW) -> tuple[KnowledgeStore, Optional[SharePermission]]
  async def create_knowledge_store(request: KnowledgeStoreCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def list_knowledge_stores(include_shared: bool = True, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def discover_knowledge_stores(search: Optional[str] = None, limit: int = 20, offset: int = 0, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_knowledge_store(store_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def update_knowledge_store(store_id: str, update_data: KnowledgeStoreUpdate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def delete_knowledge_store(store_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def list_store_documents(store_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def add_document_to_store(store_id: str, file: UploadFile = File(...), description: Optional[str] = Form(None), current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def remove_document_from_store(store_id: str, document_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def share_knowledge_store(store_id: str, request: ShareCreate, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def list_store_shares(store_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def revoke_share(store_id: str, share_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def join_via_share_link(share_token: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))


app/api/routes/themes.py:
  async def list_themes(...) -> List[ThemeResponse]  # Returns array directly
  async def get_system_themes()
  async def create_theme(theme_data: ThemeCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def get_theme(theme_id: str, user: User = Depends(get_optional_user), db: AsyncSession = Depends(get_db))
  async def delete_theme(theme_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def apply_theme(theme_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db))
  async def seed_default_themes(db: AsyncSession)


app/api/routes/websocket.py:
  async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(None))
  async def handle_chat_message(connection, streaming_handler: StreamingHandler, user_id: str, payload: dict)
  async def handle_client_message(connection, user_id: str, payload: dict)
  async def generate_chat_title(llm: LLMService, first_message: str) -> str


app/core/config.py:
  class Settings(BaseSettings)
  # Key settings:
  #   DEFAULT_SYSTEM_PROMPT: str - default system prompt for new chats
  #   TITLE_GENERATION_PROMPT: str - prompt used to generate chat titles
  #   RAG_CONTEXT_PROMPT: str - explanation for RAG context
  def get_branding(self) -> Dict[str, Any]
  def get_settings() -> Settings


app/db/database.py:
  async def get_db()
  async def init_db()
  async def close_db()


app/filters/base.py:
  class Priority(IntEnum)
  class FilterContext
  def clone(self) -> "FilterContext"
  class FilterResult
  def passthrough(cls, content: str) -> "FilterResult"
  def modify(cls, content: str, **metadata) -> "FilterResult"
  def block(cls, reason: str) -> "FilterResult"
  class StreamChunk
  class BaseOverride(ABC)
  def __init__(self, name: str, priority: Priority = Priority.MEDIUM, enabled: bool = True)
  def configure(self, **kwargs) -> "BaseOverride"
  def get_config(self, key: str, default: Any = None) -> Any
  async def process(self, content: str, context: FilterContext) -> FilterResult
  async def process_stream_chunk(self, chunk: StreamChunk, context: FilterContext) -> StreamChunk
  async def on_stream_start(self, context: FilterContext) -> None
  async def on_stream_end(self, context: FilterContext) -> Optional[StreamChunk]
  class OverrideToLLM(BaseOverride)
  class OverrideFromLLM(BaseOverride)
  class FilterChain
  def __init__(self, name: str = "default")
  def add(self, filter_instance: BaseOverride) -> "FilterChain"
  def remove(self, name: str) -> bool
  def get(self, name: str) -> Optional[BaseOverride]
  def enable(self, name: str) -> bool
  def disable(self, name: str) -> bool
  def clear(self) -> None
  async def process(self, content: str, context: FilterContext) -> FilterResult
  async def process_stream(self, stream: AsyncGenerator[StreamChunk, None], context: FilterContext) -> AsyncGenerator[StreamChunk, None]
  def list_filters(self) -> List[Dict[str, Any]]
  class ChatFilterManager
  def __init__(self, chat_id: str)
  def add_to_llm(self, filter_instance: OverrideToLLM) -> "ChatFilterManager"
  def add_from_llm(self, filter_instance: OverrideFromLLM) -> "ChatFilterManager"
  async def process_to_llm(self, content: str, context: FilterContext) -> FilterResult
  async def process_from_llm(self, content: str, context: FilterContext) -> FilterResult
  async def process_from_llm_stream(self, stream: AsyncGenerator[StreamChunk, None], context: FilterContext) -> AsyncGenerator[StreamChunk, None]
  def get_status(self) -> Dict[str, Any]
  class GlobalFilterRegistry
  def register_default_to_llm(self, factory: Callable[[], OverrideToLLM]) -> None
  def register_default_from_llm(self, factory: Callable[[], OverrideFromLLM]) -> None
  def get_chat_manager(self, chat_id: str) -> ChatFilterManager
  def remove_chat_manager(self, chat_id: str) -> bool
  def reset(self) -> None
  def get_status(self) -> Dict[str, Any]
  def get_filter_registry() -> GlobalFilterRegistry
  def get_chat_filters(chat_id: str) -> ChatFilterManager


app/filters/builtin.py:
  class RateLimitToLLM(OverrideToLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class PromptInjectionToLLM(OverrideToLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class ContentModerationToLLM(OverrideToLLM)
  def configure(self, **kwargs) -> "ContentModerationToLLM"
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class InputSanitizerToLLM(OverrideToLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class PIIRedactionToLLM(OverrideToLLM)
  def configure(self, **kwargs) -> "PIIRedactionToLLM"
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class ContextEnhancerToLLM(OverrideToLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class PIIRedactionFromLLM(OverrideFromLLM)
  def configure(self, **kwargs) -> "PIIRedactionFromLLM"
  async def process(self, content: str, context: FilterContext) -> FilterResult
  async def process_stream_chunk(self, chunk: StreamChunk, context: FilterContext) -> StreamChunk
  class SensitiveTopicFromLLM(OverrideFromLLM)
  def configure(self, **kwargs) -> "SensitiveTopicFromLLM"
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class ResponseFormatterFromLLM(OverrideFromLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class TokenCounterFromLLM(OverrideFromLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  async def process_stream_chunk(self, chunk: StreamChunk, context: FilterContext) -> StreamChunk
  class AuditLogFromLLM(OverrideFromLLM)
  async def process(self, content: str, context: FilterContext) -> FilterResult
  class StreamingWordFilterFromLLM(OverrideFromLLM)
  async def on_stream_start(self, context: FilterContext) -> None
  async def process(self, content: str, context: FilterContext) -> FilterResult
  async def process_stream_chunk(self, chunk: StreamChunk, context: FilterContext) -> StreamChunk
  async def on_stream_end(self, context: FilterContext) -> Optional[StreamChunk]


app/models/models.py:
  def generate_uuid()
  class UserTier(str, enum.Enum)
  class MessageRole(str, enum.Enum)
  class ContentType(str, enum.Enum)
  class SharePermission(str, enum.Enum)
  class APIKeyScope(str, enum.Enum)
  class User(Base)
  class OAuthAccount(Base)
  class Chat(Base)
  class ChatParticipant(Base)
  class Message(Base)
  class Document(Base)
  class DocumentChunk(Base)
  class TokenUsage(Base)
  class Theme(Base)
  class APIKey(Base)
  class KnowledgeStore(Base)
  class KnowledgeStoreShare(Base)
  class CustomAssistant(Base)
  def average_rating(self) -> float
  class AssistantConversation(Base)


app/services/auth.py:
  class AuthService
  def verify_password(plain_password: str, hashed_password: str) -> bool
  def hash_password(password: str) -> str
  def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str
  def create_refresh_token(data: dict) -> str
  def decode_token(token: str) -> Optional[dict]
  async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]
  async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]
  async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]
  async def create_user(db: AsyncSession, email: str, username: str, password: Optional[str] = None, full_name: Optional[str] = None, avatar_url: Optional[str] = None) -> User
  def create_tokens(user: User) -> Tuple[str, str]
  class OAuth2Service
  def get_google_client(redirect_uri: str) -> AsyncOAuth2Client
  def get_github_client(redirect_uri: str) -> AsyncOAuth2Client
  async def get_google_auth_url(redirect_uri: str, state: str) -> str
  async def get_github_auth_url(redirect_uri: str, state: str) -> str
  async def handle_google_callback(db: AsyncSession, code: str, redirect_uri: str) -> Tuple[User, str, str]
  async def handle_github_callback(db: AsyncSession, code: str, redirect_uri: str) -> Tuple[User, str, str]


app/services/billing.py:
  class BillingService
  async def get_usage_summary(self, db: AsyncSession, user: User, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]
  async def get_usage_history(self, db: AsyncSession, user: User, days: int = 30) -> List[Dict[str, Any]]
  async def get_usage_by_chat(self, db: AsyncSession, user: User, limit: int = 10) -> List[Dict[str, Any]]
  async def check_usage_limit(self, db: AsyncSession, user: User, estimated_tokens: int = 0) -> Dict[str, Any]
  async def upgrade_tier(self, db: AsyncSession, user: User, new_tier: UserTier) -> User
  async def reset_monthly_usage(self, db: AsyncSession, user: User)
  def estimate_tokens(self, text: str) -> int
  async def get_invoice_data(self, db: AsyncSession, user: User, year: int, month: int) -> Dict[str, Any]


app/services/llm.py:
  class LLMService
  _resolved_default_model: Optional[str] = None  # Class-level cache for resolved model
  def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None)
  async def _get_effective_model(self, requested_model: Optional[str] = None) -> str  # Resolves "default" to actual model name
  def _get_filter_manager(self, chat: Chat) -> ChatFilterManager
  def _create_filter_context(self, user: User, chat: Chat, message_id: Optional[str] = None) -> FilterContext
  async def create_message(self, db: AsyncSession, user: User, chat: Chat, user_message: str, attachments: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_handlers: Optional[Dict[str, Callable]] = None) -> Message
  async def stream_message(self, db: AsyncSession, user: User, chat: Chat, user_message: str, attachments: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_executor: Optional[Callable] = None) -> AsyncGenerator[Dict[str, Any], None]  # Prompt-based tool orchestration
  async def _check_needs_search(self, model: str, messages: List[Dict], user_query: str) -> bool  # Ask LLM if search needed
  async def _get_search_query(self, model: str, messages: List[Dict], user_query: str) -> Optional[str]  # Ask LLM for search query
  async def _build_messages(self, db: AsyncSession, chat: Chat, user_message: str, attachments: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None) -> List[Dict]
  async def _track_usage(self, db: AsyncSession, user: User, chat: Chat, model: str, input_tokens: int, output_tokens: int, message_id: str)
  def _convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]
  def _estimate_tokens(self, content: Any) -> int
  async def list_models(self) -> List[Dict[str, Any]]
  async def health_check(self) -> Dict[str, Any]


app/services/tool_service.py:
  def encrypt_api_key(api_key: str) -> str
  def decrypt_api_key(encrypted_key: str) -> str
  class MCPClient  # Uses fastmcp library
  def __init__(self, base_url: str, api_key: Optional[str] = None, config: Optional[Dict] = None)
  async def discover_tools(self) -> List[Dict[str, Any]]
  async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]
  class OpenAPIClient
  def __init__(self, base_url: str, api_key: Optional[str] = None, config: Optional[Dict] = None)
  async def discover_tools(self) -> List[Dict[str, Any]]
  async def execute_tool(self, operation_id: str, params: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]
  class ToolService
  def __init__(self, db: AsyncSession)
  async def get_available_tools(self, user: User) -> List[Tool]
  async def get_tool_schemas(self, user: User) -> List[Dict]
  async def execute_tool(self, tool_id: str, operation: str, params: Dict, user: User, chat_id: str, message_id: str) -> Dict
  async def record_usage(self, tool: Tool, operation: str, params: Dict, success: bool, result: Any, error: Optional[str], user_id: str, chat_id: str, message_id: str) -> ToolUsage


app/services/document_queue.py:
  # Persistent document processing queue - survives container restarts
  # Queue file: /app/data/document_queue.json
  # Logging controlled by 'debug_document_queue' setting (Admin → Site Dev)
  # When debug enabled: logs all activity including "[DOC_QUEUE] Empty Document Queue"
  # When debug disabled: silent operation
  
  class ProcessingStatus(Enum):
    PENDING, PROCESSING, COMPLETED, FAILED
  
  @dataclass
  class DocumentTask:
    task_id: str
    document_id: str
    user_id: str
    knowledge_store_id: Optional[str]
    file_path: str
    file_type: str
    status: str = "pending"
    created_at: str
    error: Optional[str]
    retry_count: int = 0
  
  class DocumentQueueService:
    # Singleton - use get_document_queue()
    _debug_enabled: bool  # Cached from DB, refreshed periodically
    async _check_debug_enabled() -> bool  # Check 'debug_document_queue' setting
    _load_queue()  # Load from disk on init
    _save_queue()  # Persist to disk
    add_task(task: DocumentTask)  # Add to queue (logs if debug)
    remove_task(task_id: str)  # Remove completed task (logs if debug)
    get_pending_tasks() -> List[DocumentTask]
    update_task_status(task_id, status, error?)
    async process_task(task) -> bool  # Process one document (logs if debug)
    async worker()  # Background loop, logs empty queue if debug
    start_worker()  # Start in lifespan
    stop_worker()  # Stop on shutdown
    get_queue_status() -> { total, by_status, worker_running }
  
  def get_document_queue() -> DocumentQueueService  # Get singleton


app/services/rag.py:
  class FAISSIndexManager
  def __init__(self, index_dir: str = None)
  def build_index(self, index_id: str, embeddings: np.ndarray, chunk_ids: List[str]) -> None
  def load_index(self, index_id: str) -> bool
  def search(self, index_id: str, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]
  def delete_index(self, index_id: str) -> None
  def add_vectors(self, index_id: str, embeddings: np.ndarray, chunk_ids: List[str]) -> None
  class RAGService
  def get_model(cls)
  def get_index_manager(cls) -> FAISSIndexManager
  def embed_text(self, text: str) -> Optional[np.ndarray]
  def embed_texts(self, texts: List[str], batch_size: int = 64) -> Optional[np.ndarray]
  def chunk_text(self, text: str) -> List[Dict[str, Any]]
  async def process_document(self, db: AsyncSession, document: Document, text_content: str) -> int
  async def search(self, db: AsyncSession, user: User, query: str, document_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]
  async def search_knowledge_stores(self, db: AsyncSession, user: User, query: str, knowledge_store_ids: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]
  async def get_knowledge_store_context(self, db: AsyncSession, user: User, query: str, knowledge_store_ids: List[str], top_k: int = 5) -> str
  async def rebuild_all_indexes(self, db: AsyncSession) -> Dict[str, int]
  class DocumentProcessor
  async def extract_text(file_path: str, mime_type: str) -> str


app/services/websocket.py:
  @dataclass
  class Connection:
    websocket: WebSocket
    user_id: str
    subscribed_chats: set[str]
    id: str  # UUID for hashability
    def __hash__(self) -> int
    def __eq__(self, other) -> bool
  class WebSocketManager
  def __init__(self)
  async def connect(self, websocket: WebSocket, user_id: str) -> Connection
  async def disconnect(self, connection: Connection)
  async def subscribe_to_chat(self, connection: Connection, chat_id: str)
  async def unsubscribe_from_chat(self, connection: Connection, chat_id: str)
  async def send_to_user(self, user_id: str, message: Dict[str, Any])
  async def send_to_connection(self, connection: Connection, message: Dict[str, Any])
  async def broadcast_to_chat(self, chat_id: str, message: Dict[str, Any], exclude_user: Optional[str] = None)
  async def broadcast_to_all(self, message: Dict[str, Any])
  def get_online_users(self) -> List[str]
  def is_user_online(self, user_id: str) -> bool
  def get_chat_participants_online(self, chat_id: str) -> List[str]
  class StreamingHandler
  def __init__(self, manager: WebSocketManager, connection: Connection)
  async def start_stream(self, message_id: str, chat_id: str)
  async def send_chunk(self, text: str)
  async def send_tool_call(self, tool_name: str, tool_id: str, arguments: Dict)
  async def send_tool_result(self, tool_id: str, result: Any)
  async def end_stream(self, input_tokens: int = 0, output_tokens: int = 0)
  async def send_error(self, error: str)


app/tools/registry.py:
  class ToolRegistry
  def __init__(self)  # Registers built-in tools: fetch_webpage
  def register(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable)
  def get_tool_definitions(self, tool_names: Optional[List[str]] = None) -> List[Dict]
  def get_handler(self, name: str) -> Optional[Callable]
  async def execute(self, name: str, arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Any
  async def _webpage_fetch_handler(self, url: str, extract_main_content: bool = True) -> Dict  # Built-in web fetcher


## HTTP API Routes Summary

### Authentication (/api/auth)
POST /api/auth/register - Register new user
POST /api/auth/login - Login user
POST /api/auth/refresh - Refresh token
GET /api/auth/me - Get current user
PATCH /api/auth/me - Update current user
GET /api/auth/oauth/google - Start Google OAuth
GET /api/auth/oauth/google/callback - Google OAuth callback
GET /api/auth/oauth/github - Start GitHub OAuth
GET /api/auth/oauth/github/callback - GitHub OAuth callback

### Chats (/api/chats)
GET /api/chats - List user's chats
POST /api/chats - Create new chat
GET /api/chats/{chat_id} - Get chat details
PATCH /api/chats/{chat_id} - Update chat
PATCH /api/chats/{chat_id}/selected-version - Update displayed response version
POST /api/chats/{chat_id}/share - Generate public share link
DELETE /api/chats/{chat_id} - Delete chat
DELETE /api/chats - Delete all chats
GET /api/chats/{chat_id}/messages - List messages
POST /api/chats/{chat_id}/messages - Send message
POST /api/chats/{chat_id}/invite - Invite user to chat
POST /api/chats/{chat_id}/client-message - Send client message

### Shared Chats (Public)
GET /api/shared/{share_id} - Get shared chat (no auth required)

### Documents (/api/documents)
GET /api/documents - List documents
POST /api/documents - Upload document
GET /api/documents/{document_id} - Get document
DELETE /api/documents/{document_id} - Delete document
POST /api/documents/{document_id}/reprocess - Reprocess document
POST /api/documents/search - Search documents

### Billing (/api/billing)
GET /api/billing/usage - Get usage summary
GET /api/billing/usage/history - Get usage history
GET /api/billing/usage/by-chat - Get usage by chat
GET /api/billing/check-limit - Check usage limit
GET /api/billing/invoice/{year}/{month} - Get invoice
GET /api/billing/tiers - Get tier info
POST /api/billing/upgrade - Upgrade tier

### Themes (/api/themes)
GET /api/themes - List themes
GET /api/themes/system - Get system themes
POST /api/themes - Create theme
GET /api/themes/{theme_id} - Get theme
DELETE /api/themes/{theme_id} - Delete theme
POST /api/themes/apply/{theme_id} - Apply theme

### API Keys (/api/keys)
POST /api/keys - Create API key
GET /api/keys - List API keys
GET /api/keys/{key_id} - Get API key
PATCH /api/keys/{key_id} - Update API key
DELETE /api/keys/{key_id} - Delete API key
POST /api/keys/{key_id}/regenerate - Regenerate API key

### Knowledge Stores (/api/knowledge-stores)
POST /api/knowledge-stores - Create store
GET /api/knowledge-stores - List stores
GET /api/knowledge-stores/discover - Discover public stores
GET /api/knowledge-stores/{store_id} - Get store
PATCH /api/knowledge-stores/{store_id} - Update store
DELETE /api/knowledge-stores/{store_id} - Delete store
GET /api/knowledge-stores/{store_id}/documents - List documents
POST /api/knowledge-stores/{store_id}/documents - Add document
DELETE /api/knowledge-stores/{store_id}/documents/{document_id} - Remove document
POST /api/knowledge-stores/{store_id}/share - Share store
GET /api/knowledge-stores/{store_id}/shares - List shares
DELETE /api/knowledge-stores/{store_id}/shares/{share_id} - Revoke share
POST /api/knowledge-stores/join/{share_token} - Join via link

### Assistants (/api/assistants)
POST /api/assistants - Create assistant
GET /api/assistants - List my assistants
GET /api/assistants/discover - Discover public assistants
GET /api/assistants/{assistant_id} - Get assistant
PATCH /api/assistants/{assistant_id} - Update assistant
DELETE /api/assistants/{assistant_id} - Delete assistant
POST /api/assistants/{assistant_id}/publish - Publish assistant
POST /api/assistants/{assistant_id}/unpublish - Unpublish assistant
POST /api/assistants/{assistant_id}/duplicate - Duplicate assistant
POST /api/assistants/{assistant_id}/conversation - Start conversation
POST /api/assistants/{assistant_id}/conversations/{chat_id}/rate - Rate conversation
GET /api/assistants/{assistant_id}/conversations - List conversations

### Filters (/api/filters)
GET /api/filters/status - Get registry status
GET /api/filters/chat/{chat_id} - Get chat filter status
POST /api/filters/chat/{chat_id}/enable - Enable filter
POST /api/filters/chat/{chat_id}/disable - Disable filter
POST /api/filters/chat/{chat_id}/configure - Configure filter
POST /api/filters/preset - Apply preset
GET /api/filters/chat/{chat_id}/details - Get filter details
POST /api/filters/test - Test filters
DELETE /api/filters/chat/{chat_id} - Remove chat filters
POST /api/filters/reset - Reset registry
GET /api/filters/priorities - List priorities

### Tools (/api/tools)
GET /api/tools - List all tools (admin sees all, users see public)
POST /api/tools - Create new tool (admin only)
GET /api/tools/{tool_id} - Get tool by ID
PATCH /api/tools/{tool_id} - Update tool (admin only)
DELETE /api/tools/{tool_id} - Delete tool (admin only)
POST /api/tools/probe - Probe tool URL to test connectivity (admin only)
POST /api/tools/{tool_id}/refresh - Refresh tool schema cache (admin only)
POST /api/tools/{tool_id}/test - Test execute a tool (admin only)
GET /api/tools/available/schemas - Get available tool schemas for current user

### Tool Usage (/api/tools/usage)
GET /api/tools/usage/chat/{chat_id} - Get tool usages for a chat
GET /api/tools/usage/message/{message_id} - Get tool usages for a message
GET /api/tools/usage/stats - Get aggregated usage stats (admin only)
DELETE /api/tools/usage/stats - Reset ALL tool usage stats (admin only)
DELETE /api/tools/usage/stats/{tool_id} - Reset usage stats for specific tool (admin only)

### Branding (/api/branding)
GET /api/branding/config - Get public config
GET /api/branding/manifest.json - Get web manifest
GET /api/branding/theme.css - Get branding CSS
GET /api/branding/themes - Get available themes

### WebSocket (/ws)
WS /ws - WebSocket endpoint

**Message Types (Client → Server):**
- `subscribe` - Subscribe to chat updates
- `unsubscribe` - Unsubscribe from chat
- `chat_message` - Send new message (creates user message + AI response)
- `regenerate_message` - Regenerate AI response (NO new user message)
- `stop_generation` - Stop current generation
- `client_message` - Send client-to-client message

**Message Types (Server → Client):**
- `subscribed` / `unsubscribed` - Subscription confirmations
- `message_saved` - User message saved confirmation
- `stream_start` / `stream_chunk` / `stream_end` - AI response streaming
- `stream_stopped` - Generation stopped
- `chat_updated` - Chat metadata changed (title, etc.)
- `error` - Error message


app/api/routes/admin.py:
  class TierConfig(BaseModel) - id, name, price, tokens, features, popular
  class SystemSettingsSchema(BaseModel) - prompts, pricing, refill interval
  class TiersSchema(BaseModel) - list of TierConfig
  def require_admin(user: User) -> User - dependency for admin-only routes
  async def get_system_setting(db: AsyncSession, key: str) -> str
  async def set_setting(db: AsyncSession, key: str, value: str) -> None
  GET /admin/settings -> SystemSettingsSchema
  PUT /admin/settings -> SystemSettingsSchema
  GET /admin/tiers -> TiersSchema (admin only)
  PUT /admin/tiers -> TiersSchema (admin only)
  GET /admin/public/tiers -> TiersSchema (public, for billing page)

app/models/models.py:
  class SystemSetting(Base)
    key: str (primary key)
    value: Text
    updated_at: DateTime

  class Chat(Base)  # Updated fields
    ...
    selected_versions: JSON  # { user_message_id: version_index }
    share_id: String(36)  # Unique ID for public sharing


## Frontend Signatures

frontend/src/pages/SharedChat.tsx:
  SharedChat()  # Public page for viewing shared chats (no auth)

frontend/src/pages/ChatPage.tsx:
  groupMessages(messages: Message[]) -> MessageGroup[]  # Group assistant responses after user
  MessageList({ messages, chatId, initialSelectedVersions, ... })  # Paginated message display
  handleShareChat()  # Generate and show share link
  handleExportChat()  # Download chat as JSON
  handleDeleteChat()  # Delete with confirmation

frontend/src/contexts/WebSocketContext.tsx:
  sendChatMessage(chatId, content, attachments?)  # Send new message
  regenerateMessage(chatId, content)  # Regenerate without new user message

frontend/src/lib/api.ts:
  chatApi.updateSelectedVersion(chatId, userMessageId, versionIndex)  # Persist version selection
  chatApi.share(chatId) -> { share_id }  # Generate share link

frontend/src/components/MessageBubble.tsx:
  interface VersionInfo { current, total, onPrev, onNext }
  MessageBubble({ ..., versionInfo? })  # Displays < N/T > pagination


app/services/zip_processor.py:
  class CodeSignature - file, language, type, name, line, definition, docstring
  class ZipProcessor:
    @classmethod
    async def process_zip(cls, file_content: bytes, chat_id: str, db: AsyncSession) -> dict
    @classmethod
    def extract_code_signatures(cls, content: str, language: str) -> list[CodeSignature]
    @classmethod
    def extract_python(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_javascript(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_go(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_rust(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_java(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_c(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_cpp(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_csharp(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_ruby(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_php(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_swift(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_kotlin(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_scala(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_shell(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_elixir(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def extract_haskell(cls, content: str, filepath: str) -> list[CodeSignature]
    @classmethod
    def generate_llm_manifest(cls, file_index: list, signature_index: dict, total_size: int, languages: dict) -> str


frontend/src/stores/modelsStore.ts:
  interface Model { id, name?, owned_by?, created? }
  interface SubscribedAssistant { id, name, type, assistant_id, icon, color }
  interface ModelsState:
    models: Model[]
    subscribedAssistants: SubscribedAssistant[]
    defaultModel: string
    selectedModel: string
    isLoading: boolean
    error: string | null
    lastFetched: number | null
    fetchModels(force?: boolean): Promise<void>
    setSelectedModel(modelId: string): void
    getDisplayName(modelId: string): string
    isAssistantModel(modelId: string): boolean
    getAssistantId(modelId: string): string | null
    addSubscribedAssistant(assistant: SubscribedAssistant): void
    removeSubscribedAssistant(assistantId: string): void
  useCurrentModel(): string  # Hook for current model
  useModelDisplayName(modelId: string): string  # Hook for display name


frontend/src/stores/chatStore.ts:
  createChat(model?: string, systemPrompt?: string): Promise<Chat>
    # If model starts with "gpt:", calls /assistants/{id}/start
    # Otherwise creates regular chat with model/systemPrompt


# Code Summary System (NC-0.6.2)

frontend/src/types/index.ts:
  interface CodeSignatureEntry { name, type, signature, file, line? }
  interface FileChange { path, action, language?, signatures, timestamp }
  interface SignatureWarning { type, message, file?, signature?, suggestion? }
  interface CodeSummary { id, chat_id, files, warnings, last_updated, auto_generated }

frontend/src/lib/signatures.ts:
  detectLanguage(filepath: string): string | undefined
  extractSignatures(code: string, filepath: string): CodeSignatureEntry[]
  createFileChangeFromCode(filepath, code, action?): FileChange
  detectLibraryImports(code: string, filepath: string): string[]
  generateSignatureWarnings(currentFiles, newFile, expectedSignatures?): SignatureWarning[]
  # Internal:
  extractTSSignatures(code: string, filepath: string): CodeSignatureEntry[]
  extractPythonSignatures(code: string, filepath: string): CodeSignatureEntry[]

frontend/src/components/SummaryPanel.tsx:
  interface SummaryPanelProps { summary, onClose, onClearWarnings? }
  SummaryPanel({ summary, onClose, onClearWarnings }): JSX.Element
  # Internal helpers:
  getActionColor(action: FileChange['action']): string
  getWarningColor(type: SignatureWarning['type']): string
  getWarningIcon(type: SignatureWarning['type']): string

frontend/src/stores/chatStore.ts (additions):
  codeSummary: CodeSummary | null
  showSummary: boolean
  setShowSummary(show: boolean): void
  updateCodeSummary(files: FileChange[], warnings?: SignatureWarning[]): void
  addFileToSummary(file: FileChange): void
  addWarning(warning: SignatureWarning): void
  clearSummary(): void
  fetchCodeSummary(chatId: string): Promise<void>
  saveCodeSummary(): Promise<void>

backend/app/api/schemas.py:
  class CodeSignatureEntry(BaseModel): name, type, signature, file, line?
  class FileChange(BaseModel): path, action, language?, signatures, timestamp
  class SignatureWarning(BaseModel): type, message, file?, signature?, suggestion?
  class CodeSummaryCreate(BaseModel): files, warnings, auto_generated
  class CodeSummaryResponse(BaseModel): id, chat_id, files, warnings, last_updated, auto_generated

backend/app/api/routes/chats.py (additions):
  @router.get("/{chat_id}/summary") -> CodeSummaryResponse
  @router.put("/{chat_id}/summary", CodeSummaryCreate) -> CodeSummaryResponse
  @router.delete("/{chat_id}/summary") -> {"message": "..."}

backend/app/models/models.py:
  Chat.code_summary: Column(JSON, nullable=True)
    # Format: { "id", "chat_id", "files", "warnings", "last_updated", "auto_generated" }

backend/app/main.py:
  SCHEMA_VERSION = "NC-0.6.2"
  migrations["NC-0.6.2"] = [("ALTER TABLE chats ADD COLUMN code_summary TEXT", "chats.code_summary")]


# Message Edit & Delete (NC-0.6.3)

frontend/src/pages/ChatPage.tsx:
  handleEditMessage(messageId: string, newContent: string): Promise<void>
    # Calls chatApi.editMessage, creates new branch, refreshes messages
  handleDeleteMessage(messageId: string): Promise<void>
    # Calls chatApi.deleteMessage, removes branch, refreshes messages

frontend/src/components/MessageBubble.tsx:
  # Props (already existed):
  onEdit?: (messageId: string, newContent: string) => void
  onDelete?: (messageId: string) => void
  # State:
  isEditing: boolean
  editContent: string
  showDeleteConfirm: boolean
  # Handlers:
  handleStartEdit() -> sets isEditing=true
  handleCancelEdit() -> resets editContent, isEditing=false
  handleSaveEdit() -> calls onEdit, isEditing=false
  handleDelete() -> calls onDelete, showDeleteConfirm=false

frontend/src/lib/api.ts (already existed):
  chatApi.editMessage(chatId, messageId, content, regenerateResponse?) -> PATCH /chats/{chat_id}/messages/{message_id}
  chatApi.deleteMessage(chatId, messageId) -> DELETE /chats/{chat_id}/messages/{message_id}

backend/app/api/routes/chats.py (enhanced NC-0.6.3):
  @router.patch("/{chat_id}/messages/{message_id}") -> edit_message()
    # Creates sibling message (new branch) with same parent_id as original
    # Updates chat.selected_versions to point to new message
    # For user messages with regenerate_response=true: frontend triggers WebSocket regeneration
    # Request: { content: string, regenerate_response: boolean }
    # Response: MessageResponse (the new sibling message)
    
  @router.delete("/{chat_id}/messages/{message_id}") -> delete_message()
    # Deletes message and ALL descendants (children, grandchildren, etc.)
    # Preserves sibling branches (other children of same parent)
    # Updates chat.selected_versions:
    #   - If deleted message was selected: selects next available sibling
    #   - If parent is being deleted: removes that selection entry
    # Response: { status: "deleted", message_id: string, total_deleted: number }

frontend/src/pages/ChatPage.tsx (enhanced NC-0.6.3):
  MessageList - now shows version pagination for BOTH user and assistant messages with siblings
  handleEditMessage(messageId, newContent):
    # 1. Finds message to determine role
    # 2. Calls chatApi.editMessage()
    # 3. For user messages: calls regenerateMessage(chatId, newContent, newMessageId)
    # 4. Refreshes messages to update tree
  handleDeleteMessage(messageId):
    # 1. Calls chatApi.deleteMessage()
    # 2. Refreshes messages to update tree


# Error Handling & File Request Fixes (NC-0.6.22)

backend/app/services/chat_fsm.py (NEW):
  class ChatState(Enum):
    IDLE = auto()           # Ready for new input
    STREAMING = auto()      # LLM is generating response
    AWAITING_FILES = auto() # Waiting for file content from frontend
    COMMITTING = auto()     # Persisting to database
    ERROR = auto()          # Error state, needs recovery

  @dataclass
  class PendingMessage:
    id: str
    chat_id: str
    role: str  # 'user' or 'assistant'
    content: str
    parent_id: Optional[str]
    input_tokens: int
    output_tokens: int
    model: Optional[str]
    tool_calls: Optional[List[Dict]]
    created_at: datetime
    def to_dict() -> Dict[str, Any]

  @dataclass
  class ChatSession:
    chat_id: str
    user_id: str
    state: ChatState
    pending_messages: List[PendingMessage]
    current_message_id: Optional[str]
    streaming_content: str
    pending_file_requests: List[str]
    file_contents: Dict[str, str]
    total_input_tokens: int
    total_output_tokens: int
    error: Optional[str]
    
    def can_accept_input() -> bool
    def can_accept_file_content() -> bool
    async def start_streaming(user_content, parent_id?) -> str  # Returns assistant message ID
    async def append_content(content: str)
    async def complete_streaming(input_tokens, output_tokens, file_requests?) -> bool
    async def add_file_content(path: str, content: str) -> bool  # Returns True when all files received
    async def continue_with_files() -> str  # Returns combined file content
    async def set_error(error: str)
    async def get_pending_for_commit() -> List[PendingMessage]
    async def commit_complete()
    async def reset()

  class ChatSessionManager:
    @classmethod
    async def get_instance() -> ChatSessionManager
    async def get_session(chat_id: str, user_id: str) -> ChatSession
    async def remove_session(chat_id: str)
    async def cleanup_idle_sessions(max_age_seconds: int = 3600)

backend/app/services/llm.py (changes NC-0.6.22):
  # stream_message() now uses direct SQL updates instead of ORM flush:
  # - Prevents "UPDATE expected 1 row, 0 matched" errors when message deleted
  # - Updates Message and Chat tables atomically
  # - Error handling doesn't require rollback

backend/app/api/routes/websocket.py (changes NC-0.6.22):
  # message_end event handler:
  # - Commits transaction BEFORE sending stream_end to frontend
  # - Ensures message is visible to continuation requests
  
  # stream_cancelled handler:
  # - Also commits before notification

frontend/src/contexts/WebSocketContext.tsx (changes NC-0.6.22):
  handleFileRequests(chatId, paths, parentMessageId):
    # Fixed race condition:
    # - Clears streaming at start
    # - Manages isSending state properly
    # - Handles WebSocket not open case
    # - Handles no files fetched case
  
  # stream_end handler:
  # - Breaks early if file requests detected
  # - Doesn't clear streaming state (handleFileRequests manages it)
  # - Extracts artifacts and attaches to message.artifacts

frontend/src/stores/chatStore.ts (changes NC-0.6.22):
  addMessage(message):
    # Now logs artifact extraction for debugging
    # Extracts from both message.artifacts and message.content
  
  updateMessage(messageId, updates):
    # Re-extracts artifacts when content changes
    # Merges uploaded artifacts with message artifacts

frontend/src/components/ZipUploadCard.tsx (changes NC-0.6.22):
  # File size now computed from content.length when size property is 0
  const artifactSize = artifact.size || (artifact.content?.length || 0)

frontend/src/components/ArtifactsPanel.tsx (changes NC-0.6.22):
  # Same file size fix
  const fileSize = group.latestVersion.size || (group.latestVersion.content?.length || 0)


# Filter Chain System (NC-0.6.23)

## Admin API - Filter Chains

app/api/routes/filter_chains.py:
  router = APIRouter(prefix="/filter-chains")
  def require_admin(user: User) -> User  # Dependency to require admin access
  
  # Schemas
  class StepConfig(BaseModel)  # Step-specific configuration
  class Comparison(BaseModel)  # { left, operator, right }
  class LoopConfig(BaseModel)  # { enabled, type, count?, while?, max_iterations, loop_var }
  class ConditionalConfig(BaseModel)  # { enabled, logic, comparisons, on_true?, on_false? }
  class StepDefinition(BaseModel)  # { id, type, name?, enabled?, config, on_error?, conditional?, loop? }
  class ChainDefinition(BaseModel)  # { steps: StepDefinition[] }
  class ChainCreate(BaseModel)  # { name, description?, enabled?, priority?, ... definition }
  class ChainUpdate(BaseModel)  # All fields optional
  class ChainResponse(BaseModel)  # Full chain with id, timestamps
  
  # Endpoints
  GET  /admin/filter-chains/schema -> get_schema()  # UI schema (step types, operators, variables, available_tools)
  GET  /admin/filter-chains -> list_chains()  # List all chains
  POST /admin/filter-chains -> create_chain(data: ChainCreate)  # Create new chain
  GET  /admin/filter-chains/{chain_id} -> get_chain(chain_id)  # Get single chain
  PUT  /admin/filter-chains/{chain_id} -> update_chain(chain_id, data: ChainUpdate)  # Update chain
  DELETE /admin/filter-chains/{chain_id} -> delete_chain(chain_id)  # Delete chain
  POST /admin/filter-chains/validate -> validate_chain(data: ChainDefinition)  # Validate definition
  POST /admin/filter-chains/reload -> reload_chains()  # Hot-reload from DB


## Chain Execution Engine

app/filters/executor.py:
  class FlowSignal(Enum)  # CONTINUE, STOP, FILTER_COMPLETE, ERROR, JUMP
  class CompareOp(Enum)  # EQUALS, NOT_EQUALS, CONTAINS, NOT_CONTAINS, STARTS_WITH, ENDS_WITH, REGEX, GT, LT, GTE, LTE
  
  @dataclass
  class ExecutionContext:
    query: str
    user_id: str
    chat_id: str
    variables: Dict[str, Any]  # Named variables ($Var[name])
    context_items: List[Dict]  # Accumulated context
    signal: FlowSignal
    signal_data: Optional[str]
    current: str  # Current value in pipeline
    execution_log: List[str]
  
  @dataclass
  class ExecutionResult:
    content: str
    proceed_to_llm: bool
    context: ExecutionContext
    error: Optional[str]
  
  class ChainExecutor:
    __init__(llm_func?, tool_func?)
    async execute(chain_def, query, user_id, chat_id) -> ExecutionResult
    async _execute_steps(steps, context) -> ExecutionContext
    async _execute_step(step, context) -> ExecutionContext
    # Primitive handlers:
    async _prim_to_llm(step, context) -> ExecutionContext
    async _prim_query(step, context) -> ExecutionContext
    async _prim_to_tool(step, context) -> ExecutionContext
    async _prim_from_tool(step, context) -> ExecutionContext
    async _prim_context_insert(step, context) -> ExecutionContext
    async _prim_go_to_llm(step, context) -> ExecutionContext
    async _prim_filter_complete(step, context) -> ExecutionContext
    async _prim_set_var(step, context) -> ExecutionContext
    async _prim_compare(step, context) -> ExecutionContext
    async _prim_modify(step, context) -> ExecutionContext
    async _prim_call_chain(step, context) -> ExecutionContext
    async _prim_stop(step, context) -> ExecutionContext
    async _prim_block(step, context) -> ExecutionContext
    async _prim_pass(step, context) -> ExecutionContext
    async _prim_log(step, context) -> ExecutionContext


## Chain Manager (Singleton)

app/filters/manager.py:
  class ChainManager:
    _chains: Dict[str, dict]  # chain_id -> chain data
    _name_map: Dict[str, str]  # name -> chain_id
    _sorted_ids: List[str]  # IDs sorted by priority
    
    async load_from_db(db) -> int  # Load all chains, returns count
    async reload_chain(db, chain_id)  # Hot-reload single chain
    get_chain(chain_id) -> Optional[dict]
    get_chain_by_name(name) -> Optional[dict]
    get_enabled_chains() -> List[dict]  # Sorted by priority
    
    # CRUD
    async create_chain(db, name, definition, ...) -> FilterChainModel
    async update_chain(db, chain_id, **updates) -> Optional[FilterChainModel]
    async delete_chain(db, chain_id) -> bool
    
    # Execution
    async execute_inbound(query, user_id, chat_id, llm_func, tool_func) -> ExecutionResult
    async execute_outbound(response, user_id, chat_id, inbound_chain_id, llm_func, tool_func) -> ExecutionResult
    
    # Validation
    validate_chain(definition) -> List[str]  # Returns list of errors
  
  def get_chain_manager() -> ChainManager  # Get singleton instance


## Database Model

app/models/filter_chain.py:
  class FilterChain(Base):
    __tablename__ = "filter_chains"
    id: str (PK)
    name: str (unique)
    description: Optional[str]
    enabled: bool = True
    priority: int = 100  # Lower = runs first
    retain_history: bool = True  # Hide from chat history
    bidirectional: bool = False  # Process LLM responses too
    outbound_chain_id: Optional[str]  # Chain for response processing
    max_iterations: int = 10  # Loop safety limit
    definition: JSON  # { steps: [...] }
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]


## Frontend Types

frontend/src/contexts/WebSocketContext.tsx:
  sendChatMessage(chatId: string, content: string, attachments?: unknown[], parentId?: string | null) -> void
    # Sends user message via WebSocket
    # Creates temp message with parent_id, sends to backend
    # parentId = ID of last assistant message we're continuing from
  
  regenerateMessage(chatId: string, content: string, parentId: string) -> void
    # Regenerate/retry - parentId is the USER message whose response to regenerate
    # Creates sibling assistant response
  
  # WebSocket message types:
  # 'message_saved' -> { message_id, parent_id }  # Confirms user message saved
  # 'stream_start' -> Start of assistant response
  # 'stream_chunk' -> { content, chat_id }  # Streaming content
  # 'stream_end' -> { message_id, parent_id, chat_id, usage }  # Assistant msg complete


frontend/src/pages/ChatPage.tsx:
  # Message Chain State
  currentLeafAssistantId: string | null  # ID of last assistant msg in current path
  
  # Key Functions
  handleSendMessage(content: string) -> void
    # Entry point for sending messages (including from voice mode)
    # Calculates parentId directly from store to avoid stale closures:
    #   1. Get messages from useChatStore.getState()
    #   2. Filter to current chat, sort by created_at
    #   3. Find last assistant message -> that's the parent
    # Then calls sendChatMessage(chatId, content, undefined, parentId)
  
  handleRetry(userMessageId: string, userContent: string) -> void
    # Regenerate assistant response to a specific user message
    # Calls regenerateMessage(chatId, userContent, userMessageId)
  
  handleVoiceModeRetry() -> void
    # Voice mode swipe-left retry
    # Finds last user message, regenerates its response
  
  # MessageList Component (inside ChatPage.tsx)
  buildConversationPath(messages, selectedVersions) -> ConversationNode[]
    # Builds linear path through message tree following selected branches
  
  getCurrentLeafAssistant(path: ConversationNode[]) -> Message | null
    # Returns last assistant message in the current path
    # Used for UI tracking (not for sending - sending calculates fresh)
  
  onCurrentLeafChange(leafId: string | null) -> void
    # Callback when leaf changes - updates ChatPage state


frontend/src/pages/Admin.tsx:
  interface FilterChainConditional {
    enabled: boolean
    logic?: string  # 'and' | 'or'
    comparisons: Array<{ left: string; operator: string; right: string }>
    on_true?: FilterChainStep[]
    on_false?: FilterChainStep[]
  }
  
  type StepConfig = Record<string, any>  # Flexible config for step types
  
  interface FilterChainStep {
    id: string
    type: string  # 'to_llm' | 'query' | 'to_tool' | 'go_to_llm' | 'filter_complete' | 'set_var' | 'compare' | etc.
    name?: string
    enabled?: boolean
    config: StepConfig
    on_error?: string
    jump_to_step?: string
    conditional?: FilterChainConditional
    loop?: { enabled, type, count?, while?, max_iterations?, loop_var? }
  }
  
  interface FilterChainDef {
    id?: string
    name: string
    description?: string
    enabled: boolean
    priority: number
    retain_history: boolean
    bidirectional: boolean
    outbound_chain_id?: string
    max_iterations: number
    definition: { steps: FilterChainStep[] }
    created_at?: string
    updated_at?: string
  }
  
  interface FilterChainSchema {
    step_types: Record<string, StepTypeSchema>
    comparison_operators: Array<{ value: string; label: string }>
    builtin_variables: Array<{ value: string; label: string }>
  }


## Step Types

Available step types for chain building:
- to_llm: Ask AI a question
- query: Generate search/tool query
- to_tool: Run a tool
- from_tool: Process tool result
- go_to_llm: Send to main AI
- filter_complete: Finish chain
- context_insert: Add to context
- set_var: Set variable
- compare: Compare values
- modify: Transform content
- call_chain: Run another chain
- stop: Stop processing
- block: Block with error
- pass: Pass through
- log: Debug logging


## Comparison Operators

- == (equals), != (not equals)
- contains, not_contains
- starts_with, ends_with
- regex (regex match)
- >, <, >=, <= (numeric)


## Built-in Variables

- $Query: Original user message
- $PreviousResult: Last step output
- $Current: Current pipeline value
- $Var[name]: Named variables from steps


## Debug Mode (NC-0.6.23+)

FilterChain model includes `debug: bool` field. When enabled, executor logs to console:

```
[FilterChain:chain-name] Starting chain execution
[FilterChain:chain-name] Query: <user message>
[FilterChain:chain-name] Steps to execute: 3, max_iterations: 10
[FilterChain:chain-name] ─── Step: Ask AI (type=to_llm, id=abc123...)
[FilterChain:chain-name]   Input value: <current value>
[FilterChain:chain-name]   Output value: <result>
[FilterChain:chain-name]   Variables: {'VarName': 'value'}
[FilterChain:chain-name]   Has CONDITIONAL, evaluating...
[FilterChain:chain-name]     Condition: $PreviousResult (yes) contains yes = True
[FilterChain:chain-name]     Conditional result (and): True
[FilterChain:chain-name] Chain execution complete
[FilterChain:chain-name] Final signal: CONTINUE
[FilterChain:chain-name] Proceed to LLM: True
```

ExecutionContext.debug_log(message, data?) method handles conditional logging.


## LLMService.simple_completion (NC-0.6.24+)

```python
async def simple_completion(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
) -> str:
    """Non-streaming completion for filter chain decisions."""
```

Used by filter chain executor for quick LLM decisions (e.g., "should I search?").


## Filter Chain Integration (websocket.py)

In handle_chat_message(), after user message is saved:
1. Get enabled chains from ChainManager
2. Execute each chain in priority order
3. If chain sets proceed_to_llm=False, send result directly and return
4. If chain modifies content, use modified content for LLM
5. Collect context_items from chains for RAG injection
