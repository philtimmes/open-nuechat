"""
Open-NueChat - Main FastAPI Application
Full-featured LLM chat platform with OAuth2, billing, RAG, real-time streaming, and bidirectional filtering
"""

from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import re
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging

from app.db.database import engine, Base
from app.core.config import settings
from app.api.routes import (
    auth, chats, billing, documents, themes, websocket, filters,
    api_keys, knowledge_stores, assistants, branding, admin, tools, utils, tts, stt, images,
    filter_chains
)
from app.services.rag import RAGService
from app.filters import setup_default_filters, get_filter_registry
from app.api.exception_handlers import setup_exception_handlers

# Configure logging - reduce noise, keep only important messages
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Silence noisy loggers
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.asgi").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Keep websocket logger at INFO for our debug messages
logging.getLogger("app.api.routes.websocket").setLevel(logging.INFO)
logging.getLogger("app.services.llm").setLevel(logging.INFO)
logging.getLogger("app.services.billing").setLevel(logging.INFO)
logging.getLogger("app.services.document_queue").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep main app logger at INFO for startup messages

# Static files directory (frontend build output)
STATIC_DIR = Path(__file__).parent.parent / "static"


# Current schema version
SCHEMA_VERSION = "NC-0.6.32"

def parse_version(v: str) -> tuple:
    """Parse version string like 'NC-0.5.1' into comparable tuple (0, 5, 1)"""
    # Remove 'NC-' prefix if present
    v = v.replace("NC-", "")
    parts = v.split(".")
    return tuple(int(p) for p in parts)


async def run_migrations(conn):
    """
    Run database migrations based on schema version.
    """
    from sqlalchemy import text
    import sys
    
    # Create schema_version table if not exists
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY,
            version VARCHAR(20) NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    
    # Get current version
    result = await conn.execute(text("SELECT version FROM schema_version WHERE id = 1"))
    row = result.fetchone()
    current_version = row[0] if row else "NC-0.5.0"  # Assume 0.5.0 for existing DBs without version
    
    logger.info(f"Database schema version: {current_version}, App schema version: {SCHEMA_VERSION}")
    
    # Compare versions
    current = parse_version(current_version)
    target = parse_version(SCHEMA_VERSION)
    
    if current > target:
        logger.error(f"Database schema ({current_version}) is newer than app ({SCHEMA_VERSION})!")
        logger.error("Please upgrade the application or use a compatible database.")
        sys.exit(1)
    
    if current == target:
        logger.info("Schema is up to date, no migrations needed")
        return
    
    # Define migrations: version -> list of SQL statements
    # Each migration brings the schema TO that version
    migrations = {
        "NC-0.5.0": [
            # Base migrations for older databases
            ("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0", "users.is_admin"),
        ],
        "NC-0.5.1": [
            # Version pagination and sharing
            ("ALTER TABLE chats ADD COLUMN selected_versions TEXT DEFAULT '{}'", "chats.selected_versions"),
            ("ALTER TABLE chats ADD COLUMN share_id VARCHAR(36) DEFAULT NULL", "chats.share_id"),
        ],
        "NC-0.5.2": [
            # Conversation branching - messages form a tree structure
            ("ALTER TABLE messages ADD COLUMN parent_id VARCHAR(36) DEFAULT NULL", "messages.parent_id"),
            ("CREATE INDEX IF NOT EXISTS idx_message_parent ON messages(parent_id)", "messages.idx_parent"),
        ],
        "NC-0.5.3": [
            # Uploaded files from zip archives - server-side persistence
            ("""CREATE TABLE IF NOT EXISTS uploaded_files (
                id VARCHAR(36) PRIMARY KEY,
                chat_id VARCHAR(36) NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                archive_name VARCHAR(255),
                filepath VARCHAR(1000) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                extension VARCHAR(50),
                language VARCHAR(50),
                size INTEGER DEFAULT 0,
                is_binary BOOLEAN DEFAULT 0,
                content TEXT,
                signatures TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""", "uploaded_files table"),
            ("CREATE INDEX IF NOT EXISTS idx_uploaded_file_chat ON uploaded_files(chat_id)", "uploaded_files.idx_chat"),
            ("CREATE INDEX IF NOT EXISTS idx_uploaded_file_path ON uploaded_files(chat_id, filepath)", "uploaded_files.idx_path"),
            # Uploaded archives metadata
            ("""CREATE TABLE IF NOT EXISTS uploaded_archives (
                id VARCHAR(36) PRIMARY KEY,
                chat_id VARCHAR(36) NOT NULL UNIQUE REFERENCES chats(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL,
                total_files INTEGER DEFAULT 0,
                total_size INTEGER DEFAULT 0,
                languages TEXT,
                llm_manifest TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""", "uploaded_archives table"),
            ("CREATE INDEX IF NOT EXISTS idx_uploaded_archive_chat ON uploaded_archives(chat_id)", "uploaded_archives.idx_chat"),
        ],
        "NC-0.6.2": [
            # Code summary tracking for LLM-generated code
            ("ALTER TABLE chats ADD COLUMN code_summary TEXT DEFAULT NULL", "chats.code_summary"),
        ],
        "NC-0.6.15": [
            # Human-readable summary for uploaded archives (helps LLM understand code)
            ("ALTER TABLE uploaded_archives ADD COLUMN summary TEXT DEFAULT NULL", "uploaded_archives.summary"),
        ],
        "NC-0.6.23": [
            # Configurable filter chains (admin-defined agentic flows)
            ("""CREATE TABLE IF NOT EXISTS filter_chains (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                enabled BOOLEAN DEFAULT 1 NOT NULL,
                priority INTEGER DEFAULT 100 NOT NULL,
                retain_history BOOLEAN DEFAULT 1 NOT NULL,
                bidirectional BOOLEAN DEFAULT 0 NOT NULL,
                outbound_chain_id VARCHAR(36),
                max_iterations INTEGER DEFAULT 10 NOT NULL,
                definition TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                created_by VARCHAR(36)
            )""", "filter_chains table"),
            ("CREATE INDEX IF NOT EXISTS idx_filter_chain_name ON filter_chains(name)", "filter_chains.idx_name"),
            ("CREATE INDEX IF NOT EXISTS idx_filter_chain_enabled ON filter_chains(enabled, priority)", "filter_chains.idx_enabled"),
        ],
        "NC-0.6.24": [
            # Add debug column to filter_chains
            ("ALTER TABLE filter_chains ADD COLUMN debug BOOLEAN DEFAULT 0 NOT NULL", "filter_chains.debug"),
        ],
        "NC-0.6.27": [
            # Migrate from monthly calendar-based token reset to interval-based
            # Delete old 'last_token_reset_period' setting (was "2024-12" format)
            # New 'last_token_reset_timestamp' will be created automatically with ISO datetime
            ("DELETE FROM system_settings WHERE key = 'last_token_reset_period'", "remove old token reset setting"),
        ],
        "NC-0.6.29": [
            # Add enable_safety_filters setting (disabled by default)
            # This controls prompt injection and content moderation filters
            ("INSERT INTO system_settings (key, value) VALUES ('enable_safety_filters', 'false')", "enable_safety_filters setting"),
        ],
        "NC-0.6.30": [
            # History compression settings
            ("INSERT INTO system_settings (key, value) VALUES ('history_compression_enabled', 'true')", "history_compression_enabled"),
            ("INSERT INTO system_settings (key, value) VALUES ('history_compression_threshold', '20')", "history_compression_threshold"),
            ("INSERT INTO system_settings (key, value) VALUES ('history_compression_keep_recent', '6')", "history_compression_keep_recent"),
            ("INSERT INTO system_settings (key, value) VALUES ('history_compression_target_tokens', '8000')", "history_compression_target_tokens"),
        ],
    }
    
    # Sort versions and run migrations in order
    sorted_versions = sorted(migrations.keys(), key=parse_version)
    
    for version in sorted_versions:
        version_tuple = parse_version(version)
        
        # Skip if already past this version
        if version_tuple <= current:
            continue
            
        logger.info(f"Running migrations for {version}...")
        
        for sql, name in migrations[version]:
            try:
                # Check if column already exists (for safety)
                if "ADD COLUMN" in sql:
                    parts = sql.split()
                    table_idx = parts.index("TABLE") + 1
                    col_idx = parts.index("COLUMN") + 1
                    table = parts[table_idx]
                    column = parts[col_idx]
                    
                    result = await conn.execute(text(f"PRAGMA table_info({table})"))
                    columns = [row[1] for row in result.fetchall()]
                    
                    if column in columns:
                        logger.info(f"  {name}: already exists, skipping")
                        continue
                
                # Check if system_settings key already exists
                if "INSERT" in sql and "system_settings" in sql:
                    # Extract the key from the INSERT statement
                    key_match = re.search(r"VALUES\s*\(\s*'([^']+)'", sql)
                    if key_match:
                        setting_key = key_match.group(1)
                        result = await conn.execute(
                            text("SELECT 1 FROM system_settings WHERE key = :key"),
                            {"key": setting_key}
                        )
                        if result.fetchone():
                            logger.info(f"  {name}: setting '{setting_key}' already exists, skipping")
                            continue
                
                await conn.execute(text(sql))
                logger.info(f"  {name}: OK")
            except Exception as e:
                logger.warning(f"  {name}: skipped ({e})")
    
    # Update schema version
    if row:
        await conn.execute(text(
            "UPDATE schema_version SET version = :version, updated_at = CURRENT_TIMESTAMP WHERE id = 1"
        ), {"version": SCHEMA_VERSION})
    else:
        await conn.execute(text(
            "INSERT INTO schema_version (id, version) VALUES (1, :version)"
        ), {"version": SCHEMA_VERSION})
    
    logger.info(f"Schema updated to {SCHEMA_VERSION}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events"""
    # Startup
    logger.info(f"Starting Open-NueChat v{settings.APP_VERSION} (schema {SCHEMA_VERSION})...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")
    
    # Run migrations for existing databases
    async with engine.begin() as conn:
        await run_migrations(conn)
    
    # Initialize RAG service (loads embedding model)
    try:
        rag_service = RAGService()
        logger.info(f"RAG service initialized with model: {settings.EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"RAG service initialization deferred: {e}")
    
    # Seed default themes
    from app.db.database import async_session_maker
    from app.api.routes.themes import seed_default_themes
    async with async_session_maker() as db:
        await seed_default_themes(db)
    logger.info("Default themes seeded")
    
    # Seed admin user if configured
    from app.services.auth import seed_admin_user
    async with async_session_maker() as db:
        admin = await seed_admin_user(db)
        if admin:
            logger.info(f"Admin user seeded: {admin.email}")
        elif settings.admin_password:
            logger.warning("Admin seeding failed - check ADMIN_EMAIL/ADMIN_PASS settings")
    
    # Seed default assistant
    from app.api.routes.assistants import seed_default_assistant
    try:
        async with async_session_maker() as db:
            default_gpt = await seed_default_assistant(db)
            if default_gpt:
                logger.info(f"Default assistant ready: {default_gpt.name}")
            else:
                logger.warning("Default assistant seeding returned None")
    except Exception as e:
        logger.error(f"Failed to seed default assistant: {e}")
    
    # Initialize filter system
    logger.info("Initializing filter system...")
    setup_default_filters()
    filter_status = get_filter_registry().get_status()
    logger.info(
        f"Filter system initialized: "
        f"{filter_status['default_to_llm_count']} ToLLM filters, "
        f"{filter_status['default_from_llm_count']} FromLLM filters registered"
    )
    
    # Load configurable filter chains from database
    from app.filters.manager import get_chain_manager
    async with async_session_maker() as db:
        chain_manager = get_chain_manager()
        chain_count = await chain_manager.load_from_db(db)
        logger.info(f"Loaded {chain_count} configurable filter chains")
    
    # Warm up STT model (optional, runs in background)
    try:
        from app.services.stt import get_stt_service
        stt_service = get_stt_service()
        # Run warmup in background to not block startup
        asyncio.create_task(stt_service.warmup())
        logger.info("STT warmup started in background")
    except Exception as e:
        logger.debug(f"STT warmup skipped: {e}")
    
    # Start background task for periodic token reset check
    async def token_reset_checker():
        """Background task that checks token resets every hour."""
        from app.services.billing import BillingService
        billing = BillingService()
        
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                async with async_session_maker() as db:
                    result = await billing.check_token_reset(db)
                    if result.get("action") == "reset":
                        logger.info(f"Token reset completed: {result}")
                    else:
                        logger.debug(f"Token reset check: {result}")
            except asyncio.CancelledError:
                logger.info("Token reset checker stopped")
                break
            except Exception as e:
                logger.error(f"Token reset checker error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying on error
    
    token_reset_task = asyncio.create_task(token_reset_checker())
    logger.info("Token reset background task started (checks every hour)")
    
    # Run initial token reset check on startup
    try:
        from app.services.billing import BillingService
        billing = BillingService()
        async with async_session_maker() as db:
            result = await billing.check_token_reset(db)
            logger.info(f"Initial token reset check: {result}")
    except Exception as e:
        logger.error(f"Initial token reset check failed: {e}")
    
    # Start document processing queue worker
    from app.services.document_queue import get_document_queue
    doc_queue = get_document_queue()
    doc_queue.start_worker()
    queue_status = doc_queue.get_queue_status()
    logger.info(f"Document queue started: {queue_status['total']} tasks in queue")
    
    logger.info("Open-NueChat started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Open-NueChat...")
    
    # Stop document queue worker
    doc_queue.stop_worker()
    
    token_reset_task.cancel()
    try:
        await token_reset_task
    except asyncio.CancelledError:
        pass
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="Open-NueChat",
    description="""
    A full-featured LLM chat platform with:
    - OAuth2 authentication (Google, GitHub)
    - Token tracking and billing
    - RAG with local embeddings
    - Integrated tool calling
    - Bidirectional WebSocket streaming
    - Customizable themes
    - Multi-modal support
    - Client-to-client shared chats
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup centralized exception handlers
setup_exception_handlers(app)


# Request logging middleware - disabled for cleaner logs
# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     logger.info(f"REQUEST: {request.method} {request.url.path}")
#     response = await call_next(request)
#     logger.info(f"RESPONSE: {request.method} {request.url.path} -> {response.status_code}")
#     return response

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chats.router, prefix="/api/chats", tags=["Chats"])
app.include_router(billing.router, prefix="/api/billing", tags=["Billing"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(themes.router, prefix="/api/themes", tags=["Themes"])
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])  # Routes /ws/ws
app.include_router(filters.router, prefix="/api/filters", tags=["Filters"])
app.include_router(api_keys.router, prefix="/api/keys", tags=["API Keys"])
app.include_router(knowledge_stores.router, prefix="/api/knowledge-stores", tags=["Knowledge Stores"])
app.include_router(assistants.router, prefix="/api/assistants", tags=["Custom Assistants"])
logger.info(f"Assistants router registered with {len(assistants.router.routes)} routes")
app.include_router(branding.router, prefix="/api/branding", tags=["Branding"])
app.include_router(admin.router, prefix="/api", tags=["Admin"])  # Routes /api/admin/*
app.include_router(tools.router, prefix="/api", tags=["Tools"])  # Routes /api/tools/*
app.include_router(utils.router, prefix="/api/utils", tags=["Utils"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
app.include_router(stt.router, prefix="/api/stt", tags=["STT"])
app.include_router(images.router, prefix="/api", tags=["Images"])  # Routes /api/images/*
app.include_router(filter_chains.router, prefix="/api/admin", tags=["Filter Chains"])  # Routes /api/admin/filter-chains/*


@app.get("/api/debug/test-admin")
async def debug_test_admin():
    """Debug endpoint to test admin authentication - REMOVE IN PRODUCTION"""
    from app.db.database import async_session_maker
    from app.services.auth import AuthService
    from sqlalchemy import select
    from app.models.models import User
    
    admin_pass = settings.admin_password
    
    async with async_session_maker() as db:
        result = await db.execute(select(User).where(User.email == settings.ADMIN_EMAIL))
        user = result.scalar_one_or_none()
        
        if not user:
            return {
                "error": "Admin user not found",
                "admin_email": settings.ADMIN_EMAIL,
                "admin_pass_configured": bool(admin_pass),
            }
        
        # Test password verification
        if admin_pass and user.hashed_password:
            password_valid = AuthService.verify_password(admin_pass, user.hashed_password)
        else:
            password_valid = None
        
        return {
            "admin_email": settings.ADMIN_EMAIL,
            "admin_pass_configured": bool(admin_pass),
            "admin_pass_length": len(admin_pass) if admin_pass else 0,
            "admin_pass_first_char": admin_pass[0] if admin_pass else None,
            "admin_pass_last_char": admin_pass[-1] if admin_pass else None,
            "user_found": True,
            "user_id": user.id,
            "user_email": user.email,
            "user_is_active": user.is_active,
            "user_is_admin": user.is_admin,
            "user_has_password": bool(user.hashed_password),
            "password_hash_prefix": user.hashed_password[:20] if user.hashed_password else None,
            "password_valid": password_valid,
        }


@app.post("/api/debug/test-post")
async def debug_test_post(data: dict = None):
    """Debug endpoint to test POST requests work"""
    logger.info(f"TEST POST received: {data}")
    return {"status": "ok", "received": data}


@app.get("/api/health")
async def health_check():
    """Health check endpoint - also handles periodic maintenance tasks"""
    from app.services.billing import BillingService
    from app.db.database import async_session_maker
    
    result = {
        "status": "healthy",
        "service": "open-nuechat",
        "version": settings.APP_VERSION,
        "schema_version": SCHEMA_VERSION,
    }
    
    # Check and reset token counts if interval has elapsed
    try:
        async with async_session_maker() as db:
            billing = BillingService()
            token_reset_result = await billing.check_token_reset(db)
            result["token_reset"] = token_reset_result
    except Exception as e:
        logger.error(f"Token reset check failed: {e}")
        result["token_reset"] = {"action": "error", "error": str(e)}
    
    return result


@app.get("/api/models")
async def list_models(request: Request):
    """List available LLM models and subscribed assistants"""
    from app.services.llm import LLMService
    from app.services.settings_service import SettingsService
    from app.api.dependencies import get_current_user_optional
    from app.db.database import async_session_maker
    from app.models.models import CustomAssistant, User as UserModel
    from sqlalchemy import select
    
    # Use database session to get settings
    async with async_session_maker() as db:
        # Create LLM service from database settings
        llm_service = await LLMService.from_database(db)
        models = await llm_service.list_models()
        
        # Get default model from database settings
        llm_settings = await SettingsService.get_llm_settings(db)
        default_model = llm_settings["model"]
        
        if default_model == "default" and models and len(models) > 0:
            # Use first model that isn't an error
            for m in models:
                if "error" not in m and "id" in m:
                    default_model = m["id"]
                    break
    
    # Try to get subscribed assistants if user is authenticated
    subscribed_assistants = []
    try:
        # Get auth header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            from app.services.auth import AuthService
            payload = AuthService.decode_token(token)
            if payload and "sub" in payload:
                user_id = payload["sub"]
                logger.info(f"Fetching subscribed assistants for user {user_id}")
                
                async with async_session_maker() as db:
                    from sqlalchemy import text
                    import json
                    
                    # Get user preferences with raw SQL
                    result = await db.execute(
                        text("SELECT preferences FROM users WHERE id = :user_id"),
                        {"user_id": user_id}
                    )
                    row = result.fetchone()
                    
                    logger.info(f"Raw preferences from DB: {row[0] if row else 'NOT FOUND'}")
                    
                    if row and row[0]:
                        prefs_raw = row[0]
                        if isinstance(prefs_raw, str):
                            prefs = json.loads(prefs_raw)
                        else:
                            prefs = dict(prefs_raw)
                        
                        subscribed_ids = prefs.get("subscribed_assistants", [])
                        logger.info(f"Subscribed IDs: {subscribed_ids}")
                        
                        if subscribed_ids:
                            # Get assistant details
                            result = await db.execute(
                                select(CustomAssistant).where(
                                    CustomAssistant.id.in_(subscribed_ids)
                                )
                            )
                            assistants = result.scalars().all()
                            logger.info(f"Found {len(assistants)} assistants")
                            
                            for asst in assistants:
                                subscribed_assistants.append({
                                    "id": f"gpt:{asst.id}",
                                    "name": f"🤖 {asst.name}",
                                    "type": "assistant",
                                    "assistant_id": asst.id,
                                    "icon": asst.icon,
                                    "color": asst.color,
                                })
    except Exception as e:
        # Silently fail - just don't include subscribed assistants
        logger.error(f"Could not load subscribed assistants: {e}", exc_info=True)
    
    return {
        "api_base": settings.LLM_API_BASE_URL,
        "default_model": default_model,
        "models": models,
        "subscribed_assistants": subscribed_assistants,
    }


@app.get("/api/info")
async def api_info():
    """API information"""
    return {
        "name": f"{settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "features": [
            "OAuth2 (Google, GitHub)",
            "JWT Authentication",
            "Token Tracking & Billing",
            "RAG with Local Embeddings",
            "Tool Calling",
            "WebSocket Streaming",
            "Custom Themes",
            "Multi-modal Support",
            "Shared Chats",
            "Bidirectional Streaming Filters",
            "User API Keys",
            "Knowledge Stores (Personal & Shared)",
            "Custom Assistants (GPTs)"
        ],
        "llm": {
            "api_base": settings.LLM_API_BASE_URL,
            "default_model": settings.LLM_MODEL,
            "description": "OpenAI-compatible API (Ollama, LM Studio, vLLM, etc.)"
        },
        "tiers": {
            "free": {"tokens": 100000, "api_keys": 3, "knowledge_stores": 3, "assistants": 3, "price": 0},
            "pro": {"tokens": 1000000, "api_keys": 10, "knowledge_stores": 20, "assistants": 20, "price": 20},
            "enterprise": {"tokens": 10000000, "api_keys": 100, "knowledge_stores": 100, "assistants": 100, "price": 100}
        },
        "filters": {
            "description": "Bidirectional streaming filter system",
            "to_llm": ["rate_limit", "prompt_injection", "content_moderation", "input_sanitizer", "pii_redaction"],
            "from_llm": ["pii_redaction", "sensitive_topic", "response_formatter", "token_counter"],
            "priorities": ["HIGHEST", "HIGH", "MEDIUM", "LOW", "LEAST"],
            "presets": ["default", "minimal", "strict"]
        },
        "api_keys": {
            "description": "User-generated API keys for programmatic access",
            "scopes": ["chat", "knowledge", "assistants", "billing", "full"],
            "format": "nxs_..."
        },
        "knowledge_stores": {
            "description": "Personal and shareable document collections for RAG",
            "sharing": ["view", "edit", "admin"],
            "supported_formats": ["pdf", "txt", "md", "docx", "html"]
        },
        "custom_assistants": {
            "description": "Custom GPT-like AI assistants with specific objectives",
            "features": ["Custom system prompts", "Knowledge store integration", "Tool selection", "Public sharing"]
        }
    }


@app.get("/api/shared/{share_id}")
async def get_shared_chat(share_id: str):
    """Get a publicly shared chat (no authentication required)"""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from app.db.database import async_session_maker
    from app.models.models import Chat, Message
    
    async with async_session_maker() as db:
        # Find chat by share_id
        result = await db.execute(
            select(Chat)
            .options(selectinload(Chat.messages))
            .where(Chat.share_id == share_id)
        )
        chat = result.scalar_one_or_none()
        
        if not chat:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Shared chat not found")
        
        # Build conversation path using tree structure
        selected_versions = chat.selected_versions or {}
        
        # Build children map: parent_id -> children
        children_by_parent = {}
        for msg in chat.messages:
            parent_key = msg.parent_id or 'root'
            if parent_key not in children_by_parent:
                children_by_parent[parent_key] = []
            children_by_parent[parent_key].append(msg)
        
        # Sort children by created_at
        for children in children_by_parent.values():
            children.sort(key=lambda m: m.created_at)
        
        # Walk the tree from root, following selected branches
        messages = []
        current_parent = 'root'
        
        while True:
            children = children_by_parent.get(current_parent, [])
            if not children:
                break
            
            # Find selected child or default to newest
            selected = children[-1]  # Default to newest
            selected_id = selected_versions.get(current_parent)
            if selected_id:
                found = next((c for c in children if c.id == selected_id), None)
                if found:
                    selected = found
            
            messages.append({
                "id": selected.id,
                "role": selected.role.value if hasattr(selected.role, 'value') else selected.role,
                "content": selected.content,
                "parent_id": selected.parent_id,
                "created_at": selected.created_at.isoformat(),
                "input_tokens": selected.input_tokens,
                "output_tokens": selected.output_tokens,
            })
            
            current_parent = selected.id
        
        return {
            "id": chat.id,
            "title": chat.title,
            "model": chat.model,
            "created_at": chat.created_at.isoformat(),
            "messages": messages,
        }


# ============ Static File Serving ============
# Serve frontend if static directory exists

logger.info(f"Checking for static directory: {STATIC_DIR}")
logger.info(f"Static directory exists: {STATIC_DIR.exists()}")

if STATIC_DIR.exists():
    logger.info(f"Static directory contents: {list(STATIC_DIR.iterdir())}")
    # Mount assets with caching
    if (STATIC_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
        logger.info("Mounted /assets")
    
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve SPA - return index.html for all non-API routes"""
        # Don't serve index.html for API or WebSocket routes
        if full_path.startswith(("api/", "ws/", "docs", "redoc", "openapi")):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not found"}, status_code=404)
        
        # Check if it's a static file that exists
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Otherwise serve index.html for SPA routing
        return FileResponse(STATIC_DIR / "index.html")
else:
    logger.warning(f"Static directory not found: {STATIC_DIR}")
    logger.warning("Frontend will not be served. Build frontend and copy to backend/static/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )
