"""
Runtime settings service - reads settings from database with fallback to config

NC-0.8.0.9: Now uses centralized SETTING_DEFAULTS from settings_keys.py
"""
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from app.models.models import SystemSetting
from app.core.settings_keys import SK, SETTING_DEFAULTS


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
    async def get_bool(db: AsyncSession, key: str, default: bool = False) -> bool:
        """Get a boolean system setting."""
        value = await SettingsService.get(db, key)
        if not value:
            return default
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
            SK.LLM_API_BASE_URL,
            SK.LLM_API_KEY,
            SK.LLM_MODEL,
            SK.LLM_TIMEOUT,
            SK.LLM_MAX_TOKENS,
            SK.LLM_TEMPERATURE,
            SK.LLM_STREAM_DEFAULT,
        ])
        
        return {
            "base_url": settings[SK.LLM_API_BASE_URL],
            "api_key": settings[SK.LLM_API_KEY],
            "model": settings[SK.LLM_MODEL],
            "timeout": int(settings[SK.LLM_TIMEOUT]) if settings[SK.LLM_TIMEOUT] else 300,
            "max_tokens": int(settings[SK.LLM_MAX_TOKENS]) if settings[SK.LLM_MAX_TOKENS] else 4096,
            "temperature": float(settings[SK.LLM_TEMPERATURE]) if settings[SK.LLM_TEMPERATURE] else 0.7,
            "stream": settings[SK.LLM_STREAM_DEFAULT].lower() in ("true", "1", "yes", "on"),
        }
    
    @staticmethod
    async def get_google_oauth_settings(db: AsyncSession) -> Dict[str, Any]:
        """Get Google OAuth settings."""
        settings = await SettingsService.get_many(db, [
            SK.GOOGLE_CLIENT_ID,
            SK.GOOGLE_CLIENT_SECRET,
            SK.GOOGLE_OAUTH_ENABLED,
            SK.GOOGLE_OAUTH_TIMEOUT,
        ])
        
        return {
            "client_id": settings[SK.GOOGLE_CLIENT_ID],
            "client_secret": settings[SK.GOOGLE_CLIENT_SECRET],
            "enabled": settings[SK.GOOGLE_OAUTH_ENABLED].lower() in ("true", "1", "yes", "on"),
            "timeout": int(settings[SK.GOOGLE_OAUTH_TIMEOUT]) if settings[SK.GOOGLE_OAUTH_TIMEOUT] else 30,
        }
    
    @staticmethod
    async def get_github_oauth_settings(db: AsyncSession) -> Dict[str, Any]:
        """Get GitHub OAuth settings."""
        settings = await SettingsService.get_many(db, [
            SK.GITHUB_CLIENT_ID,
            SK.GITHUB_CLIENT_SECRET,
            SK.GITHUB_OAUTH_ENABLED,
            SK.GITHUB_OAUTH_TIMEOUT,
        ])
        
        return {
            "client_id": settings[SK.GITHUB_CLIENT_ID],
            "client_secret": settings[SK.GITHUB_CLIENT_SECRET],
            "enabled": settings[SK.GITHUB_OAUTH_ENABLED].lower() in ("true", "1", "yes", "on"),
            "timeout": int(settings[SK.GITHUB_OAUTH_TIMEOUT]) if settings[SK.GITHUB_OAUTH_TIMEOUT] else 30,
        }
    
    @staticmethod
    async def get_tier_tokens(db: AsyncSession, tier: str) -> int:
        """Get token limit for a tier."""
        key = f"{tier}_tier_tokens"
        return await SettingsService.get_int(db, key)
    
    @staticmethod
    async def get_token_refill_hours(db: AsyncSession) -> int:
        """Get token refill interval in hours."""
        return await SettingsService.get_int(db, SK.TOKEN_REFILL_INTERVAL_HOURS)
    
    @staticmethod
    async def is_freeforall(db: AsyncSession) -> bool:
        """Check if freeforall mode is enabled."""
        return await SettingsService.get_bool(db, SK.FREEFORALL)
    
    @staticmethod
    async def is_registration_enabled(db: AsyncSession) -> bool:
        """Check if registration is enabled."""
        return await SettingsService.get_bool(db, SK.ENABLE_REGISTRATION)
    
    @staticmethod
    async def is_billing_enabled(db: AsyncSession) -> bool:
        """Check if billing is enabled."""
        return await SettingsService.get_bool(db, SK.ENABLE_BILLING)
