"""
Centralized settings key definitions - NC-0.8.0.9

All system settings keys should be defined here and imported elsewhere.
This prevents typos and ensures consistency across the codebase.

Usage:
    from app.core.settings_keys import SK, SETTING_DEFAULTS
    
    value = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_WIDTH)
"""

from app.core.config import settings as config_settings


class SK:
    """
    System Settings Keys - All setting key constants.
    
    Naming convention: CATEGORY_SETTING_NAME
    """
    
    # =========================================================================
    # IMAGE GENERATION
    # =========================================================================
    IMAGE_GEN_DEFAULT_WIDTH = "image_gen_default_width"
    IMAGE_GEN_DEFAULT_HEIGHT = "image_gen_default_height"
    IMAGE_GEN_DEFAULT_ASPECT_RATIO = "image_gen_default_aspect_ratio"
    IMAGE_GEN_AVAILABLE_RESOLUTIONS = "image_gen_available_resolutions"
    
    # =========================================================================
    # LLM SETTINGS
    # =========================================================================
    LLM_API_BASE_URL = "llm_api_base_url"
    LLM_API_KEY = "llm_api_key"
    LLM_MODEL = "llm_model"
    LLM_TIMEOUT = "llm_timeout"
    LLM_MAX_TOKENS = "llm_max_tokens"
    LLM_CONTEXT_SIZE = "llm_context_size"
    LLM_TEMPERATURE = "llm_temperature"
    LLM_STREAM_DEFAULT = "llm_stream_default"
    LLM_MULTIMODAL = "llm_multimodal"
    
    # =========================================================================
    # PROMPTS
    # =========================================================================
    DEFAULT_SYSTEM_PROMPT = "default_system_prompt"
    ALL_MODELS_PROMPT = "all_models_prompt"
    TITLE_GENERATION_PROMPT = "title_generation_prompt"
    RAG_CONTEXT_PROMPT = "rag_context_prompt"
    
    # =========================================================================
    # TOKEN LIMITS BY TIER
    # =========================================================================
    FREE_TIER_TOKENS = "free_tier_tokens"
    PRO_TIER_TOKENS = "pro_tier_tokens"
    ENTERPRISE_TIER_TOKENS = "enterprise_tier_tokens"
    TOKEN_REFILL_INTERVAL_HOURS = "token_refill_interval_hours"
    
    # =========================================================================
    # PRICING
    # =========================================================================
    INPUT_TOKEN_PRICE = "input_token_price"
    OUTPUT_TOKEN_PRICE = "output_token_price"
    TIERS = "tiers"
    
    # =========================================================================
    # OAUTH - GOOGLE
    # =========================================================================
    GOOGLE_CLIENT_ID = "google_client_id"
    GOOGLE_CLIENT_SECRET = "google_client_secret"
    GOOGLE_OAUTH_ENABLED = "google_oauth_enabled"
    GOOGLE_OAUTH_TIMEOUT = "google_oauth_timeout"
    
    # =========================================================================
    # OAUTH - GITHUB
    # =========================================================================
    GITHUB_CLIENT_ID = "github_client_id"
    GITHUB_CLIENT_SECRET = "github_client_secret"
    GITHUB_OAUTH_ENABLED = "github_oauth_enabled"
    GITHUB_OAUTH_TIMEOUT = "github_oauth_timeout"
    
    # =========================================================================
    # FEATURE FLAGS
    # =========================================================================
    ENABLE_REGISTRATION = "enable_registration"
    ENABLE_BILLING = "enable_billing"
    FREEFORALL = "freeforall"
    ENABLE_SAFETY_FILTERS = "enable_safety_filters"
    
    # =========================================================================
    # HISTORY COMPRESSION
    # =========================================================================
    HISTORY_COMPRESSION_ENABLED = "history_compression_enabled"
    HISTORY_COMPRESSION_THRESHOLD = "history_compression_threshold"
    HISTORY_COMPRESSION_KEEP_RECENT = "history_compression_keep_recent"
    HISTORY_COMPRESSION_TARGET_TOKENS = "history_compression_target_tokens"
    MODEL_CONTEXT_SIZE = "model_context_size"
    
    # =========================================================================
    # API RATE LIMITS
    # =========================================================================
    API_RATE_LIMIT_COMPLETIONS = "api_rate_limit_completions"
    API_RATE_LIMIT_EMBEDDINGS = "api_rate_limit_embeddings"
    API_RATE_LIMIT_IMAGES = "api_rate_limit_images"
    API_RATE_LIMIT_MODELS = "api_rate_limit_models"
    
    # =========================================================================
    # STORAGE LIMITS
    # =========================================================================
    MAX_UPLOAD_SIZE_MB = "max_upload_size_mb"
    MAX_KNOWLEDGE_STORE_SIZE_MB = "max_knowledge_store_size_mb"
    MAX_KNOWLEDGE_STORES_FREE = "max_knowledge_stores_free"
    MAX_KNOWLEDGE_STORES_PRO = "max_knowledge_stores_pro"
    MAX_KNOWLEDGE_STORES_ENTERPRISE = "max_knowledge_stores_enterprise"
    
    # =========================================================================
    # DEBUG SETTINGS
    # =========================================================================
    DEBUG_TOKEN_RESETS = "debug_token_resets"
    DEBUG_DOCUMENT_QUEUE = "debug_document_queue"
    DEBUG_RAG = "debug_rag"
    DEBUG_FILTER_CHAINS = "debug_filter_chains"
    DEBUG_TOOL_ADVERTISEMENTS = "debug_tool_advertisements"
    DEBUG_TOOL_CALLS = "debug_tool_calls"
    
    # =========================================================================
    # WEB SEARCH (Google Custom Search Engine)
    # =========================================================================
    WEB_SEARCH_GOOGLE_API_KEY = "web_search_google_api_key"
    WEB_SEARCH_GOOGLE_CX_ID = "web_search_google_cx_id"
    WEB_SEARCH_ENABLED = "web_search_enabled"


# Alias for convenience
SettingsKeys = SK


def _get_setting_defaults():
    """
    Build SETTING_DEFAULTS dict.
    
    This is a function to avoid circular import issues with config_settings.
    Called once at module load time.
    """
    import json
    
    # Default tiers for billing
    DEFAULT_TIERS = [
        {
            "name": "Free",
            "price": 0,
            "tokens": 100000,
            "features": ["100K tokens/month", "Basic models", "Community support"],
            "popular": False,
        },
        {
            "name": "Pro",
            "price": 20,
            "tokens": 1000000,
            "features": ["1M tokens/month", "All models", "Priority support", "RAG storage: 100MB"],
            "popular": True,
        },
        {
            "name": "Enterprise",
            "price": 100,
            "tokens": 10000000,
            "features": ["10M tokens/period", "All models", "Dedicated support", "RAG storage: 1GB", "Custom integrations"],
            "popular": False,
        },
    ]
    
    return {
        # Image Generation
        SK.IMAGE_GEN_DEFAULT_WIDTH: "1024",
        SK.IMAGE_GEN_DEFAULT_HEIGHT: "1024",
        SK.IMAGE_GEN_DEFAULT_ASPECT_RATIO: "1:1",
        SK.IMAGE_GEN_AVAILABLE_RESOLUTIONS: json.dumps([
            {"width": 512, "height": 512, "label": "Small Square (512x512)"},
            {"width": 768, "height": 768, "label": "Medium Square (768x768)"},
            {"width": 1024, "height": 1024, "label": "Large Square (1024x1024)"},
            {"width": 768, "height": 1024, "label": "Portrait (768x1024)"},
            {"width": 1024, "height": 768, "label": "Landscape (1024x768)"},
        ]),
        
        # LLM Settings
        SK.LLM_API_BASE_URL: config_settings.LLM_API_BASE_URL,
        SK.LLM_API_KEY: config_settings.LLM_API_KEY,
        SK.LLM_MODEL: config_settings.LLM_MODEL,
        SK.LLM_TIMEOUT: str(config_settings.LLM_TIMEOUT),
        SK.LLM_MAX_TOKENS: str(config_settings.LLM_MAX_TOKENS),
        SK.LLM_CONTEXT_SIZE: "200000",
        SK.LLM_TEMPERATURE: str(config_settings.LLM_TEMPERATURE),
        SK.LLM_STREAM_DEFAULT: str(config_settings.LLM_STREAM_DEFAULT).lower(),
        SK.LLM_MULTIMODAL: "false",
        
        # Prompts
        SK.DEFAULT_SYSTEM_PROMPT: config_settings.DEFAULT_SYSTEM_PROMPT,
        SK.ALL_MODELS_PROMPT: config_settings.ALL_MODELS_PROMPT,
        SK.TITLE_GENERATION_PROMPT: config_settings.TITLE_GENERATION_PROMPT,
        SK.RAG_CONTEXT_PROMPT: config_settings.RAG_CONTEXT_PROMPT,
        
        # Token limits
        SK.FREE_TIER_TOKENS: str(config_settings.FREE_TIER_TOKENS),
        SK.PRO_TIER_TOKENS: str(config_settings.PRO_TIER_TOKENS),
        SK.ENTERPRISE_TIER_TOKENS: str(config_settings.ENTERPRISE_TIER_TOKENS),
        SK.TOKEN_REFILL_INTERVAL_HOURS: "720",  # 30 days
        
        # Pricing
        SK.INPUT_TOKEN_PRICE: str(config_settings.INPUT_TOKEN_PRICE),
        SK.OUTPUT_TOKEN_PRICE: str(config_settings.OUTPUT_TOKEN_PRICE),
        SK.TIERS: json.dumps(DEFAULT_TIERS),
        
        # OAuth - Google
        SK.GOOGLE_CLIENT_ID: config_settings.GOOGLE_CLIENT_ID or "",
        SK.GOOGLE_CLIENT_SECRET: config_settings.GOOGLE_CLIENT_SECRET or "",
        SK.GOOGLE_OAUTH_ENABLED: str(config_settings.ENABLE_OAUTH_GOOGLE).lower(),
        SK.GOOGLE_OAUTH_TIMEOUT: "30",
        
        # OAuth - GitHub
        SK.GITHUB_CLIENT_ID: config_settings.GITHUB_CLIENT_ID or "",
        SK.GITHUB_CLIENT_SECRET: config_settings.GITHUB_CLIENT_SECRET or "",
        SK.GITHUB_OAUTH_ENABLED: str(config_settings.ENABLE_OAUTH_GITHUB).lower(),
        SK.GITHUB_OAUTH_TIMEOUT: "30",
        
        # Feature flags
        SK.ENABLE_REGISTRATION: str(config_settings.ENABLE_REGISTRATION).lower(),
        SK.ENABLE_BILLING: str(config_settings.ENABLE_BILLING).lower(),
        SK.FREEFORALL: str(config_settings.FREEFORALL).lower(),
        SK.ENABLE_SAFETY_FILTERS: "false",
        
        # History compression
        SK.HISTORY_COMPRESSION_ENABLED: "true",
        SK.HISTORY_COMPRESSION_THRESHOLD: "20",
        SK.HISTORY_COMPRESSION_KEEP_RECENT: "10",
        SK.HISTORY_COMPRESSION_TARGET_TOKENS: "8000",
        SK.MODEL_CONTEXT_SIZE: "128000",
        
        # API Rate Limits
        SK.API_RATE_LIMIT_COMPLETIONS: "60",
        SK.API_RATE_LIMIT_EMBEDDINGS: "200",
        SK.API_RATE_LIMIT_IMAGES: "10",
        SK.API_RATE_LIMIT_MODELS: "100",
        
        # Storage limits
        SK.MAX_UPLOAD_SIZE_MB: str(config_settings.MAX_UPLOAD_SIZE_MB),
        SK.MAX_KNOWLEDGE_STORE_SIZE_MB: "500",
        SK.MAX_KNOWLEDGE_STORES_FREE: "3",
        SK.MAX_KNOWLEDGE_STORES_PRO: "20",
        SK.MAX_KNOWLEDGE_STORES_ENTERPRISE: "100",
        
        # Debug settings
        SK.DEBUG_TOKEN_RESETS: "false",
        SK.DEBUG_DOCUMENT_QUEUE: "false",
        SK.DEBUG_RAG: "false",
        SK.DEBUG_FILTER_CHAINS: "false",
        SK.DEBUG_TOOL_ADVERTISEMENTS: "false",
        SK.DEBUG_TOOL_CALLS: "false",
        
        # Web Search (Google Custom Search Engine)
        SK.WEB_SEARCH_GOOGLE_API_KEY: "",
        SK.WEB_SEARCH_GOOGLE_CX_ID: "",
        SK.WEB_SEARCH_ENABLED: "true",
    }


# Single source of truth for all setting defaults
SETTING_DEFAULTS = _get_setting_defaults()
