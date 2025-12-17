"""
Runtime settings service - reads settings from database with fallback to config
"""
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.models.models import SystemSetting
from app.core.config import settings as config_settings


# Default values from config (fallback when not in DB)
SETTING_DEFAULTS = {
    # Prompts
    "default_system_prompt": config_settings.DEFAULT_SYSTEM_PROMPT,
    "title_generation_prompt": config_settings.TITLE_GENERATION_PROMPT,
    "rag_context_prompt": config_settings.RAG_CONTEXT_PROMPT,
    
    # Token limits by tier
    "free_tier_tokens": str(config_settings.FREE_TIER_TOKENS),
    "pro_tier_tokens": str(config_settings.PRO_TIER_TOKENS),
    "enterprise_tier_tokens": str(config_settings.ENTERPRISE_TIER_TOKENS),
    
    # Pricing
    "input_token_price": str(config_settings.INPUT_TOKEN_PRICE),
    "output_token_price": str(config_settings.OUTPUT_TOKEN_PRICE),
    
    # Token refill (in hours)
    "token_refill_interval_hours": "720",  # 30 days default
    
    # Debug settings
    "debug_token_resets": "false",  # Log detailed token info on reset checks
    "debug_document_queue": "false",  # Log document queue processing
    "debug_rag": "false",  # Log all RAG queries, results, and context retrieval
    "debug_filter_chains": "false",  # Log filter chain execution with full input/output
    
    # OAuth - Google
    "google_client_id": config_settings.GOOGLE_CLIENT_ID or "",
    "google_client_secret": config_settings.GOOGLE_CLIENT_SECRET or "",
    "google_oauth_enabled": str(config_settings.ENABLE_OAUTH_GOOGLE).lower(),
    "google_oauth_timeout": "30",
    
    # OAuth - GitHub
    "github_client_id": config_settings.GITHUB_CLIENT_ID or "",
    "github_client_secret": config_settings.GITHUB_CLIENT_SECRET or "",
    "github_oauth_enabled": str(config_settings.ENABLE_OAUTH_GITHUB).lower(),
    "github_oauth_timeout": "30",
    
    # LLM Settings
    "llm_api_base_url": config_settings.LLM_API_BASE_URL,
    "llm_api_key": config_settings.LLM_API_KEY,
    "llm_model": config_settings.LLM_MODEL,
    "llm_timeout": str(config_settings.LLM_TIMEOUT),
    "llm_max_tokens": str(config_settings.LLM_MAX_TOKENS),
    "llm_temperature": str(config_settings.LLM_TEMPERATURE),
    "llm_stream_default": str(config_settings.LLM_STREAM_DEFAULT).lower(),
    
    # Feature flags
    "enable_registration": str(config_settings.ENABLE_REGISTRATION).lower(),
    "enable_billing": str(config_settings.ENABLE_BILLING).lower(),
    "freeforall": str(config_settings.FREEFORALL).lower(),
    
    # Safety filters (prompt injection, content moderation)
    "enable_safety_filters": "false",  # Disabled by default - most self-hosted users don't need this
    
    # History compression (reduce context window usage for long conversations)
    "history_compression_enabled": "true",  # Compress old messages into summaries
    "history_compression_threshold": "20",  # Compress after this many messages
    "history_compression_keep_recent": "6",  # Keep this many recent message pairs intact
    "history_compression_target_tokens": "8000",  # Target total tokens after compression
    
    # API Rate Limits (per minute, per API key)
    "api_rate_limit_completions": "60",  # Chat completions per minute
    "api_rate_limit_embeddings": "200",  # Embeddings requests per minute
    "api_rate_limit_images": "10",  # Image generations per minute
    "api_rate_limit_models": "100",  # Model list requests per minute
}


class SettingsService:
    """Service to read settings from database with config fallback"""
    
    @staticmethod
    async def get(db: AsyncSession, key: str) -> str:
        """Get a system setting value with fallback to defaults."""
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting:
            return setting.value
        return SETTING_DEFAULTS.get(key, "")
    
    @staticmethod
    async def get_bool(db: AsyncSession, key: str) -> bool:
        """Get a boolean system setting."""
        value = await SettingsService.get(db, key)
        return value.lower() in ("true", "1", "yes", "on")
    
    @staticmethod
    async def get_int(db: AsyncSession, key: str) -> int:
        """Get an integer system setting."""
        value = await SettingsService.get(db, key)
        try:
            return int(value)
        except (ValueError, TypeError):
            default = SETTING_DEFAULTS.get(key, "0")
            try:
                return int(default)
            except (ValueError, TypeError):
                return 0
    
    @staticmethod
    async def get_float(db: AsyncSession, key: str) -> float:
        """Get a float system setting."""
        value = await SettingsService.get(db, key)
        try:
            return float(value)
        except (ValueError, TypeError):
            default = SETTING_DEFAULTS.get(key, "0.0")
            try:
                return float(default)
            except (ValueError, TypeError):
                return 0.0
    
    @staticmethod
    async def set(db: AsyncSession, key: str, value: str) -> None:
        """Set a system setting value."""
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        
        if setting:
            setting.value = value
        else:
            setting = SystemSetting(key=key, value=value)
            db.add(setting)
    
    @staticmethod
    async def get_many(db: AsyncSession, keys: list) -> Dict[str, str]:
        """Get multiple settings at once."""
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key.in_(keys))
        )
        settings = {s.key: s.value for s in result.scalars().all()}
        
        # Fill in defaults for missing keys
        for key in keys:
            if key not in settings:
                settings[key] = SETTING_DEFAULTS.get(key, "")
        
        return settings
    
    # =========================================================================
    # Convenience methods for specific setting groups
    # =========================================================================
    
    @staticmethod
    async def get_llm_settings(db: AsyncSession) -> Dict[str, Any]:
        """Get all LLM settings."""
        settings = await SettingsService.get_many(db, [
            "llm_api_base_url",
            "llm_api_key",
            "llm_model",
            "llm_timeout",
            "llm_max_tokens",
            "llm_temperature",
            "llm_stream_default",
        ])
        
        return {
            "base_url": settings["llm_api_base_url"],
            "api_key": settings["llm_api_key"],
            "model": settings["llm_model"],
            "timeout": int(settings["llm_timeout"]) if settings["llm_timeout"] else 120,
            "max_tokens": int(settings["llm_max_tokens"]) if settings["llm_max_tokens"] else 4096,
            "temperature": float(settings["llm_temperature"]) if settings["llm_temperature"] else 0.7,
            "stream": settings["llm_stream_default"].lower() in ("true", "1", "yes", "on"),
        }
    
    @staticmethod
    async def get_google_oauth_settings(db: AsyncSession) -> Dict[str, Any]:
        """Get Google OAuth settings."""
        settings = await SettingsService.get_many(db, [
            "google_client_id",
            "google_client_secret",
            "google_oauth_enabled",
            "google_oauth_timeout",
        ])
        
        return {
            "client_id": settings["google_client_id"],
            "client_secret": settings["google_client_secret"],
            "enabled": settings["google_oauth_enabled"].lower() in ("true", "1", "yes", "on"),
            "timeout": int(settings["google_oauth_timeout"]) if settings["google_oauth_timeout"] else 30,
        }
    
    @staticmethod
    async def get_github_oauth_settings(db: AsyncSession) -> Dict[str, Any]:
        """Get GitHub OAuth settings."""
        settings = await SettingsService.get_many(db, [
            "github_client_id",
            "github_client_secret",
            "github_oauth_enabled",
            "github_oauth_timeout",
        ])
        
        return {
            "client_id": settings["github_client_id"],
            "client_secret": settings["github_client_secret"],
            "enabled": settings["github_oauth_enabled"].lower() in ("true", "1", "yes", "on"),
            "timeout": int(settings["github_oauth_timeout"]) if settings["github_oauth_timeout"] else 30,
        }
    
    @staticmethod
    async def get_tier_tokens(db: AsyncSession, tier: str) -> int:
        """Get token limit for a tier."""
        key = f"{tier}_tier_tokens"
        return await SettingsService.get_int(db, key)
    
    @staticmethod
    async def get_token_refill_hours(db: AsyncSession) -> int:
        """Get token refill interval in hours."""
        return await SettingsService.get_int(db, "token_refill_interval_hours")
    
    @staticmethod
    async def is_freeforall(db: AsyncSession) -> bool:
        """Check if freeforall mode is enabled."""
        return await SettingsService.get_bool(db, "freeforall")
    
    @staticmethod
    async def is_registration_enabled(db: AsyncSession) -> bool:
        """Check if registration is enabled."""
        return await SettingsService.get_bool(db, "enable_registration")
    
    @staticmethod
    async def is_billing_enabled(db: AsyncSession) -> bool:
        """Check if billing is enabled."""
        return await SettingsService.get_bool(db, "enable_billing")
