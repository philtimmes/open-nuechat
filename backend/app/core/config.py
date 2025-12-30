"""
Core configuration for Open-NueChat
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List, Dict, Any
from functools import lru_cache
import secrets


class Settings(BaseSettings):
    # ===========================================
    # BRANDING & CUSTOMIZATION
    # ===========================================
    
    # Application Identity
    APP_NAME: str = "Open-NueChat"
    APP_TAGLINE: str = "AI-Powered Chat Platform"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "A full-featured LLM chat platform"
    
    # Visual Branding
    FAVICON_URL: str = "/favicon.ico"  # URL or path to favicon
    LOGO_URL: Optional[str] = None  # URL or path to logo image
    LOGO_TEXT: Optional[str] = None  # Text to show if no logo image (defaults to APP_NAME)
    
    # Default Theme (must match a theme ID in the database)
    DEFAULT_THEME: str = "dark"  # Options: dark, light, midnight, forest, sunset, ocean
    
    # Custom Colors (override theme defaults for branding)
    BRAND_PRIMARY_COLOR: Optional[str] = None  # e.g., "#6366f1"
    BRAND_SECONDARY_COLOR: Optional[str] = None
    BRAND_ACCENT_COLOR: Optional[str] = None
    
    # Footer/Legal
    FOOTER_TEXT: Optional[str] = None  # e.g., "© 2025 Your Company"
    PRIVACY_URL: Optional[str] = None
    TERMS_URL: Optional[str] = None
    SUPPORT_EMAIL: Optional[str] = None
    
    # Feature Flags
    ENABLE_REGISTRATION: bool = True
    ENABLE_OAUTH_GOOGLE: bool = True
    ENABLE_OAUTH_GITHUB: bool = True
    ENABLE_BILLING: bool = True
    ENABLE_PUBLIC_ASSISTANTS: bool = True
    ENABLE_PUBLIC_KNOWLEDGE_STORES: bool = True
    FREEFORALL: bool = False  # If True, don't enforce token limits for any user
    
    # Welcome Message (shown to new users)
    WELCOME_TITLE: str = ""
    WELCOME_MESSAGE: str = ""
    
    @property
    def welcome_title(self) -> str:
        """Return welcome title with app name substitution"""
        title = self.WELCOME_TITLE.strip() if self.WELCOME_TITLE else ""
        if not title:
            title = "Welcome to {app_name}!"
        return title.format(app_name=self.APP_NAME)
    
    @property
    def welcome_message(self) -> str:
        """Return welcome message or default"""
        msg = self.WELCOME_MESSAGE.strip() if self.WELCOME_MESSAGE else ""
        if not msg:
            msg = "Start a conversation with AI. Ask questions, get help with tasks, or just chat."
        return msg
    
    # ===========================================
    # SERVER & INFRASTRUCTURE
    # ===========================================
    
    DEBUG: bool = False
    BACKEND_HOST: str = "127.0.0.1"  # Bind address (use 0.0.0.0 to expose externally)
    BACKEND_PORT: int = 8000
    
    # Public URL (for OAuth callbacks, emails, etc.)
    # Set this to your external URL when behind a reverse proxy
    # e.g., https://chat.example.com
    PUBLIC_URL: Optional[str] = None
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/nuechat.db"
    
    # Procedural Memory (separate database for skill learning)
    PROCEDURAL_DATABASE_URL: Optional[str] = None  # If not set, derives from DATABASE_URL
    PROCEDURAL_MEMORY_ENABLED: bool = True  # Enable/disable procedural memory feature
    
    # ===========================================
    # SECURITY
    # ===========================================
    
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Flag to track if SECRET_KEY was auto-generated (insecure for production)
    _secret_key_generated: bool = False
    
    def validate_secret_key(self) -> None:
        """
        Warn if SECRET_KEY appears to be auto-generated in production.
        Auto-generated keys change on restart, invalidating all sessions.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if key looks auto-generated (same length as secrets.token_urlsafe(32))
        if len(self.SECRET_KEY) == 43 and not self.DEBUG:  # 32 bytes base64 = 43 chars
            logger.warning(
                "⚠️  SECRET_KEY appears to be auto-generated. "
                "Set a stable SECRET_KEY in production to prevent session invalidation on restart. "
                "Generate one with: openssl rand -hex 32"
            )
    
    # Administrator Account (created on startup if set)
    ADMIN_EMAIL: str = "admin@localhost"
    ADMIN_PASS: Optional[str] = None  # If set, creates/updates admin account on startup
    ADMIN_USERNAME: str = "Administrator"
    
    @property
    def admin_password(self) -> Optional[str]:
        """Return None if ADMIN_PASS is empty or not set"""
        if self.ADMIN_PASS and self.ADMIN_PASS.strip():
            return self.ADMIN_PASS
        return None
    
    # OAuth2 Providers
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    
    # ===========================================
    # LLM CONFIGURATION (OpenAI-Compatible API)
    # ===========================================
    
    LLM_API_BASE_URL: str = "http://localhost:8080/v1"
    LLM_API_KEY: str = "not-needed"
    LLM_MODEL: str = "default"
    LLM_TIMEOUT: int = 300  # 5 minutes for large context requests
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7
    LLM_STREAM_DEFAULT: bool = True
    
    # Default system prompt for new chats
    DEFAULT_SYSTEM_PROMPT: str = "You are a helpful AI assistant. Be concise, accurate, and helpful."
    
    # All models prompt (appended to ALL system prompts including Custom GPTs)
    ALL_MODELS_PROMPT: str = ""
    
    # Title generation prompt (used to auto-generate chat titles)
    TITLE_GENERATION_PROMPT: str = "Generate a short, descriptive title (max 6 words) for a conversation that starts with this message. Return ONLY the title, no quotes or explanation:"
    
    # RAG context prompt (prepended when documents are included)
    RAG_CONTEXT_PROMPT: str = "The following information has been retrieved from the user's documents to help answer their question:"
    
    # ===========================================
    # BILLING & PAYMENTS
    # ===========================================
    
    # Stripe configuration
    STRIPE_API_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    
    # PayPal configuration
    PAYPAL_CLIENT_ID: Optional[str] = None
    PAYPAL_CLIENT_SECRET: Optional[str] = None
    PAYPAL_WEBHOOK_ID: Optional[str] = None
    PAYPAL_MODE: str = "sandbox"  # "sandbox" or "live"
    
    # Google Pay configuration (uses Stripe as processor)
    GOOGLE_PAY_MERCHANT_ID: Optional[str] = None
    GOOGLE_PAY_MERCHANT_NAME: str = "NueChat"
    
    # Payment settings
    PAYMENT_CURRENCY: str = "USD"
    PAYMENT_SUCCESS_URL: str = "/billing?status=success"
    PAYMENT_CANCEL_URL: str = "/billing?status=cancelled"
    
    # Pricing tiers (tokens per tier)
    FREE_TIER_TOKENS: int = 100_000
    PRO_TIER_TOKENS: int = 1_000_000
    ENTERPRISE_TIER_TOKENS: int = 10_000_000
    
    # Tier pricing (monthly, in USD)
    PRO_TIER_PRICE: float = 20.00
    ENTERPRISE_TIER_PRICE: float = 100.00
    
    # Token pricing (per 1M tokens for overage)
    INPUT_TOKEN_PRICE: float = 3.00
    OUTPUT_TOKEN_PRICE: float = 15.00
    
    # ===========================================
    # RAG CONFIGURATION
    # ===========================================
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    # Apache Tika for document extraction
    TIKA_URL: str = "http://beta.offenedaten.de:9998/tika"
    
    # FAISS Index Settings
    FAISS_INDEX_DIR: str = "./faiss_indexes"
    FAISS_NPROBE: int = 32  # Clusters to search (speed vs accuracy tradeoff)
    FAISS_USE_GPU: bool = True
    
    # ===========================================
    # TOOLS (MCP / OpenAPI)
    # ===========================================
    
    # Encryption key for tool API keys (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    TOOL_ENCRYPTION_KEY: Optional[str] = None
    
    # ===========================================
    # FILE UPLOADS
    # ===========================================
    
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = [
        # Images
        "image/png", "image/jpeg", "image/gif", "image/webp",
        # Documents
        "application/pdf", "text/plain", "text/markdown",
        "application/json", "text/csv",
        # Office documents
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel",  # .xls (legacy)
        "application/msword",  # .doc (legacy)
        "application/rtf", "text/rtf",  # .rtf
        # Programming languages - these map to text/plain or application/octet-stream
        "text/x-python", "text/x-java", "text/x-c", "text/x-c++",
        "text/javascript", "application/javascript",
        "text/x-rust", "text/x-go", "text/x-ruby",
        "text/x-typescript", "text/html", "text/css",
        "text/x-yaml", "application/x-yaml",
        "text/xml", "application/xml",
        "application/x-sh", "text/x-shellscript",
    ]
    # File extensions to allow (checked in addition to MIME types)
    ALLOWED_FILE_EXTENSIONS: List[str] = [
        # Documents
        ".pdf", ".txt", ".md", ".json", ".csv",
        # Office documents
        ".docx", ".doc", ".xlsx", ".xls", ".rtf",
        # Python
        ".py", ".pyi", ".pyw",
        # JavaScript/TypeScript
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        # C/C++
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx",
        # Java
        ".java",
        # Rust
        ".rs",
        # Go
        ".go",
        # Ruby
        ".rb",
        # Web
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        # Config/Data
        ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg",
        # Shell
        ".sh", ".bash", ".zsh",
        # Other
        ".sql", ".r", ".swift", ".kt", ".scala", ".php",
        ".lua", ".pl", ".pm", ".ex", ".exs", ".erl",
        ".hs", ".ml", ".fs", ".clj", ".lisp", ".el",
        ".vim", ".dockerfile", ".makefile", ".cmake",
    ]
    UPLOAD_DIR: str = "./uploads"
    
    # ===========================================
    # WEBSOCKET
    # ===========================================
    
    WS_HEARTBEAT_INTERVAL: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_branding(self) -> Dict[str, Any]:
        """Get all branding settings as a dictionary for the frontend"""
        return {
            "app_name": self.APP_NAME,
            "app_tagline": self.APP_TAGLINE,
            "app_version": self.APP_VERSION,
            "app_description": self.APP_DESCRIPTION,
            "favicon_url": self.FAVICON_URL,
            "logo_url": self.LOGO_URL,
            "logo_text": self.LOGO_TEXT or self.APP_NAME,
            "default_theme": self.DEFAULT_THEME,
            "brand_colors": {
                "primary": self.BRAND_PRIMARY_COLOR,
                "secondary": self.BRAND_SECONDARY_COLOR,
                "accent": self.BRAND_ACCENT_COLOR,
            },
            "footer_text": self.FOOTER_TEXT,
            "privacy_url": self.PRIVACY_URL,
            "terms_url": self.TERMS_URL,
            "support_email": self.SUPPORT_EMAIL,
            "features": {
                "registration": self.ENABLE_REGISTRATION,
                "oauth_google": self.ENABLE_OAUTH_GOOGLE and bool(self.GOOGLE_CLIENT_ID),
                "oauth_github": self.ENABLE_OAUTH_GITHUB and bool(self.GITHUB_CLIENT_ID),
                "billing": self.ENABLE_BILLING,
                "public_assistants": self.ENABLE_PUBLIC_ASSISTANTS,
                "public_knowledge_stores": self.ENABLE_PUBLIC_KNOWLEDGE_STORES,
            },
            "welcome": {
                "title": self.welcome_title,
                "message": self.welcome_message,
            },
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
