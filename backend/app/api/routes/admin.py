"""
Admin routes for system settings management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import json
import uuid
import shutil

from app.api.dependencies import get_current_user, get_db
from app.models.models import User, SystemSetting, UserTier, Chat, Message
from app.core.config import settings
from app.services.billing import BillingService
from sqlalchemy import select, func, or_

router = APIRouter(prefix="/admin", tags=["admin"])


def get_branding_dir() -> Path:
    """
    Get the branding assets directory, located in the same directory as the database.
    This ensures branding images persist with the database.
    """
    # Parse database path from DATABASE_URL
    # Format: sqlite+aiosqlite:///./data/nuechat.db or sqlite+aiosqlite:////absolute/path/nuechat.db
    db_url = settings.DATABASE_URL
    
    if "sqlite" in db_url:
        # Extract path after sqlite+aiosqlite:///
        if ":///" in db_url:
            db_path = db_url.split(":///", 1)[1]
            # Handle relative vs absolute paths
            if db_path.startswith("/"):
                # Absolute path (4 slashes total: sqlite+aiosqlite:////path)
                db_dir = Path(db_path).parent
            else:
                # Relative path (3 slashes: sqlite+aiosqlite:///./data/db)
                db_dir = Path(db_path).parent
        else:
            # Fallback to default
            db_dir = Path("./data")
    else:
        # Non-SQLite database - use a default data directory
        db_dir = Path("./data")
    
    branding_dir = db_dir / "branding_assets"
    branding_dir.mkdir(parents=True, exist_ok=True)
    return branding_dir


# Directory for branding assets - lazily initialized
_branding_dir: Optional[Path] = None

def get_branding_assets_dir() -> Path:
    """Get or create the branding assets directory."""
    global _branding_dir
    if _branding_dir is None:
        _branding_dir = get_branding_dir()
    return _branding_dir


# Default tier configuration
DEFAULT_TIERS = [
    {
        "id": "free",
        "name": "Free",
        "price": 0,
        "tokens": 100000,
        "features": ["100K tokens/period", "Basic models", "Email support"],
        "popular": False,
    },
    {
        "id": "pro",
        "name": "Pro",
        "price": 20,
        "tokens": 1000000,
        "features": ["1M tokens/period", "All models", "Priority support", "RAG storage: 100MB"],
        "popular": True,
    },
    {
        "id": "enterprise",
        "name": "Enterprise",
        "price": 100,
        "tokens": 10000000,
        "features": ["10M tokens/period", "All models", "Dedicated support", "RAG storage: 1GB", "Custom integrations"],
        "popular": False,
    },
]


# Default values from config (fallback when not in DB)
SETTING_DEFAULTS = {
    # Prompts
    "default_system_prompt": settings.DEFAULT_SYSTEM_PROMPT,
    "all_models_prompt": settings.ALL_MODELS_PROMPT,
    "title_generation_prompt": settings.TITLE_GENERATION_PROMPT,
    "rag_context_prompt": settings.RAG_CONTEXT_PROMPT,
    
    # Token limits by tier
    "free_tier_tokens": str(settings.FREE_TIER_TOKENS),
    "pro_tier_tokens": str(settings.PRO_TIER_TOKENS),
    "enterprise_tier_tokens": str(settings.ENTERPRISE_TIER_TOKENS),
    
    # Pricing
    "input_token_price": str(settings.INPUT_TOKEN_PRICE),
    "output_token_price": str(settings.OUTPUT_TOKEN_PRICE),
    
    # Token refill (in hours)
    "token_refill_interval_hours": "720",  # 30 days default
    
    # Tiers JSON
    "tiers": json.dumps(DEFAULT_TIERS),
    
    # OAuth - Google
    "google_client_id": settings.GOOGLE_CLIENT_ID or "",
    "google_client_secret": settings.GOOGLE_CLIENT_SECRET or "",
    "google_oauth_enabled": str(settings.ENABLE_OAUTH_GOOGLE).lower(),
    "google_oauth_timeout": "30",
    
    # OAuth - GitHub
    "github_client_id": settings.GITHUB_CLIENT_ID or "",
    "github_client_secret": settings.GITHUB_CLIENT_SECRET or "",
    "github_oauth_enabled": str(settings.ENABLE_OAUTH_GITHUB).lower(),
    "github_oauth_timeout": "30",
    
    # LLM Settings
    "llm_api_base_url": settings.LLM_API_BASE_URL,
    "llm_api_key": settings.LLM_API_KEY,
    "llm_model": settings.LLM_MODEL,
    "llm_timeout": str(settings.LLM_TIMEOUT),
    "llm_max_tokens": str(settings.LLM_MAX_TOKENS),
    "llm_temperature": str(settings.LLM_TEMPERATURE),
    "llm_stream_default": str(settings.LLM_STREAM_DEFAULT).lower(),
    "llm_multimodal": "false",  # Whether legacy model supports vision/images
    
    # Feature flags
    "enable_registration": str(settings.ENABLE_REGISTRATION).lower(),
    "enable_billing": str(settings.ENABLE_BILLING).lower(),
    "freeforall": str(settings.FREEFORALL).lower(),
    
    # API Rate Limits (per minute, per API key)
    "api_rate_limit_completions": "60",
    "api_rate_limit_embeddings": "200",
    "api_rate_limit_images": "10",
    "api_rate_limit_models": "100",
    
    # Storage limits
    "max_upload_size_mb": str(settings.MAX_UPLOAD_SIZE_MB),
    "max_knowledge_store_size_mb": "500",  # 500MB default per knowledge store
    "max_knowledge_stores_free": "3",
    "max_knowledge_stores_pro": "20",
    "max_knowledge_stores_enterprise": "100",
    
    # Billing API Settings - Stripe
    "stripe_enabled": "false",
    "stripe_api_key": settings.STRIPE_API_KEY or "",
    "stripe_webhook_secret": settings.STRIPE_WEBHOOK_SECRET or "",
    "stripe_publishable_key": settings.STRIPE_PUBLISHABLE_KEY or "",
    "stripe_pro_price_id": "",
    "stripe_enterprise_price_id": "",
    
    # Billing API Settings - PayPal
    "paypal_enabled": "false",
    "paypal_client_id": settings.PAYPAL_CLIENT_ID or "",
    "paypal_client_secret": settings.PAYPAL_CLIENT_SECRET or "",
    "paypal_webhook_id": settings.PAYPAL_WEBHOOK_ID or "",
    "paypal_mode": settings.PAYPAL_MODE or "sandbox",
    "paypal_pro_plan_id": "",
    "paypal_enterprise_plan_id": "",
    
    # Billing API Settings - Google Pay
    "google_pay_enabled": "false",
    "google_pay_merchant_id": settings.GOOGLE_PAY_MERCHANT_ID or "",
    "google_pay_merchant_name": settings.GOOGLE_PAY_MERCHANT_NAME or "NueChat",
    
    # Branding
    "app_name": "Open-NueChat",
    "app_tagline": "AI-Powered Chat",
    "favicon_url": "",
    "logo_url": "",
    "custom_css": "",
    "custom_themes": "[]",
}


async def get_system_setting(db: AsyncSession, key: str) -> str:
    """Get a system setting value with fallback to defaults."""
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting:
        return setting.value
    return SETTING_DEFAULTS.get(key, "")


async def get_system_setting_bool(db: AsyncSession, key: str) -> bool:
    """Get a boolean system setting."""
    value = await get_system_setting(db, key)
    return value.lower() in ("true", "1", "yes", "on")


async def get_system_setting_int(db: AsyncSession, key: str, default: int = 0) -> int:
    """Get an integer system setting."""
    value = await get_system_setting(db, key)
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


async def get_system_setting_float(db: AsyncSession, key: str, default: float = 0.0) -> float:
    """Get a float system setting."""
    value = await get_system_setting(db, key)
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


class TierConfig(BaseModel):
    """Configuration for a pricing tier"""
    id: str
    name: str
    price: float
    tokens: int
    features: List[str]
    popular: bool = False


class SystemSettingsSchema(BaseModel):
    """System settings that can be modified by admins"""
    # Prompts
    default_system_prompt: str
    all_models_prompt: str = ""  # Appended to all system prompts including Custom GPTs
    title_generation_prompt: str
    rag_context_prompt: str
    
    # Image classification
    image_confirm_with_llm: bool = True  # Whether to use LLM to confirm image requests
    image_classification_prompt: str = ""
    image_classification_true_response: str = "YES"
    
    # Pricing
    input_token_price: float
    output_token_price: float
    
    # Token refill (in hours)
    token_refill_interval_hours: int = Field(ge=1, description="Token refill interval in hours")
    
    # Token limits by tier
    free_tier_tokens: int = Field(ge=0)
    pro_tier_tokens: int = Field(ge=0)
    enterprise_tier_tokens: int = Field(ge=0)
    
    # Storage limits
    max_upload_size_mb: int = Field(ge=1, le=10000, description="Maximum file upload size in MB")
    max_knowledge_store_size_mb: int = Field(ge=1, le=100000, description="Maximum knowledge store size in MB")
    max_knowledge_stores_free: int = Field(ge=0, description="Max knowledge stores for free tier")
    max_knowledge_stores_pro: int = Field(ge=0, description="Max knowledge stores for pro tier")
    max_knowledge_stores_enterprise: int = Field(ge=0, description="Max knowledge stores for enterprise tier")


class OAuthSettingsSchema(BaseModel):
    """OAuth provider settings"""
    # Google
    google_client_id: str = ""
    google_client_secret: str = ""
    google_oauth_enabled: bool = True
    google_oauth_timeout: int = Field(default=30, ge=5, le=300)
    
    # GitHub
    github_client_id: str = ""
    github_client_secret: str = ""
    github_oauth_enabled: bool = True
    github_oauth_timeout: int = Field(default=30, ge=5, le=300)


class LLMSettingsSchema(BaseModel):
    """LLM connectivity settings"""
    llm_api_base_url: str
    llm_api_key: str = ""
    llm_model: str = "default"
    llm_timeout: int = Field(default=120, ge=10, le=600)
    llm_max_tokens: int = Field(default=4096, ge=1)
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_stream_default: bool = True
    llm_multimodal: bool = False  # Whether legacy model supports vision/images
    # Thinking tokens (for models that output reasoning)
    think_begin_token: str = ""
    think_end_token: str = ""
    # History compression settings
    history_compression_enabled: bool = True
    history_compression_threshold: int = Field(default=20, ge=5, le=100)
    history_compression_keep_recent: int = Field(default=6, ge=2, le=20)
    history_compression_target_tokens: int = Field(default=8000, ge=2000, le=128000)


class FeatureFlagsSchema(BaseModel):
    """Feature flags"""
    enable_registration: bool = True
    enable_billing: bool = True
    freeforall: bool = False
    enable_safety_filters: bool = False  # Prompt injection & content moderation filters


class BillingApiSettingsSchema(BaseModel):
    """Payment provider API settings"""
    # Stripe
    stripe_enabled: bool = False
    stripe_api_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_publishable_key: str = ""
    stripe_pro_price_id: str = ""
    stripe_enterprise_price_id: str = ""
    
    # PayPal
    paypal_enabled: bool = False
    paypal_client_id: str = ""
    paypal_client_secret: str = ""
    paypal_webhook_id: str = ""
    paypal_mode: str = "sandbox"  # "sandbox" or "live"
    paypal_pro_plan_id: str = ""
    paypal_enterprise_plan_id: str = ""
    
    # Google Pay
    google_pay_enabled: bool = False
    google_pay_merchant_id: str = ""
    google_pay_merchant_name: str = "NueChat"


class APIRateLimitsSchema(BaseModel):
    """API rate limit settings (requests per minute)"""
    api_rate_limit_completions: int = Field(default=60, ge=1, le=1000, description="Chat completions per minute")
    api_rate_limit_embeddings: int = Field(default=200, ge=1, le=1000, description="Embeddings per minute")
    api_rate_limit_images: int = Field(default=10, ge=1, le=100, description="Image generations per minute")
    api_rate_limit_models: int = Field(default=100, ge=1, le=1000, description="Model list requests per minute")


class BrandingSettingsSchema(BaseModel):
    """Branding and customization settings"""
    app_name: str = Field(default="Open-NueChat", description="Application name")
    app_tagline: str = Field(default="AI-Powered Chat", description="Application tagline")
    favicon_url: str = Field(default="", description="URL to favicon")
    logo_url: str = Field(default="", description="URL to logo image")
    custom_css: str = Field(default="", description="Custom CSS rules")
    custom_themes: str = Field(default="[]", description="JSON array of custom themes")


class TiersSchema(BaseModel):
    """Pricing tiers configuration"""
    tiers: List[TierConfig]


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency that requires the user to be an admin"""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


async def set_setting(db: AsyncSession, key: str, value: str) -> None:
    """Set a system setting value"""
    result = await db.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    
    if setting:
        setting.value = value
    else:
        setting = SystemSetting(key=key, value=value)
        db.add(setting)


# =============================================================================
# RAW SYSTEM SETTINGS (individual key-value access)
# =============================================================================

class RawSettingSchema(BaseModel):
    """Single setting key-value pair"""
    key: str
    value: str


class RawSettingsListSchema(BaseModel):
    """List of raw settings"""
    settings: List[RawSettingSchema]


@router.get("/settings/raw", response_model=List[RawSettingSchema])
async def get_raw_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get all raw system settings as key-value pairs"""
    result = await db.execute(select(SystemSetting))
    settings = result.scalars().all()
    return [RawSettingSchema(key=s.key, value=s.value) for s in settings]


@router.get("/setting/{key}")
async def get_single_setting(
    key: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get a single setting by key"""
    value = await get_system_setting(db, key)
    return {"key": key, "value": value}


@router.put("/setting")
async def set_single_setting(
    data: RawSettingSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a single setting by key"""
    await set_setting(db, data.key, data.value)
    await db.commit()
    return {"key": data.key, "value": data.value}


# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

@router.get("/settings", response_model=SystemSettingsSchema)
async def get_admin_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get all system settings"""
    image_confirm_setting = await get_system_setting(db, "image_confirm_with_llm")
    return SystemSettingsSchema(
        default_system_prompt=await get_system_setting(db, "default_system_prompt"),
        all_models_prompt=await get_system_setting(db, "all_models_prompt"),
        title_generation_prompt=await get_system_setting(db, "title_generation_prompt"),
        rag_context_prompt=await get_system_setting(db, "rag_context_prompt"),
        image_confirm_with_llm=image_confirm_setting.lower() == "true" if image_confirm_setting else True,
        image_classification_prompt=await get_system_setting(db, "image_classification_prompt") or "",
        image_classification_true_response=await get_system_setting(db, "image_classification_true_response") or "YES",
        input_token_price=await get_system_setting_float(db, "input_token_price"),
        output_token_price=await get_system_setting_float(db, "output_token_price"),
        token_refill_interval_hours=await get_system_setting_int(db, "token_refill_interval_hours"),
        free_tier_tokens=await get_system_setting_int(db, "free_tier_tokens"),
        pro_tier_tokens=await get_system_setting_int(db, "pro_tier_tokens"),
        enterprise_tier_tokens=await get_system_setting_int(db, "enterprise_tier_tokens"),
        max_upload_size_mb=await get_system_setting_int(db, "max_upload_size_mb", settings.MAX_UPLOAD_SIZE_MB),
        max_knowledge_store_size_mb=await get_system_setting_int(db, "max_knowledge_store_size_mb", 500),
        max_knowledge_stores_free=await get_system_setting_int(db, "max_knowledge_stores_free", 3),
        max_knowledge_stores_pro=await get_system_setting_int(db, "max_knowledge_stores_pro", 20),
        max_knowledge_stores_enterprise=await get_system_setting_int(db, "max_knowledge_stores_enterprise", 100),
    )


@router.put("/settings", response_model=SystemSettingsSchema)
async def update_admin_settings(
    data: SystemSettingsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update system settings"""
    await set_setting(db, "default_system_prompt", data.default_system_prompt)
    await set_setting(db, "all_models_prompt", data.all_models_prompt)
    await set_setting(db, "title_generation_prompt", data.title_generation_prompt)
    await set_setting(db, "rag_context_prompt", data.rag_context_prompt)
    await set_setting(db, "image_confirm_with_llm", "true" if data.image_confirm_with_llm else "false")
    await set_setting(db, "image_classification_prompt", data.image_classification_prompt)
    await set_setting(db, "image_classification_true_response", data.image_classification_true_response)
    await set_setting(db, "input_token_price", str(data.input_token_price))
    await set_setting(db, "output_token_price", str(data.output_token_price))
    await set_setting(db, "token_refill_interval_hours", str(data.token_refill_interval_hours))
    await set_setting(db, "free_tier_tokens", str(data.free_tier_tokens))
    await set_setting(db, "pro_tier_tokens", str(data.pro_tier_tokens))
    await set_setting(db, "enterprise_tier_tokens", str(data.enterprise_tier_tokens))
    await set_setting(db, "max_upload_size_mb", str(data.max_upload_size_mb))
    await set_setting(db, "max_knowledge_store_size_mb", str(data.max_knowledge_store_size_mb))
    await set_setting(db, "max_knowledge_stores_free", str(data.max_knowledge_stores_free))
    await set_setting(db, "max_knowledge_stores_pro", str(data.max_knowledge_stores_pro))
    await set_setting(db, "max_knowledge_stores_enterprise", str(data.max_knowledge_stores_enterprise))
    
    await db.commit()
    
    return data


# =============================================================================
# BRANDING SETTINGS
# =============================================================================

@router.get("/settings/branding", response_model=BrandingSettingsSchema)
async def get_branding_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get branding settings"""
    return BrandingSettingsSchema(
        app_name=await get_system_setting(db, "app_name"),
        app_tagline=await get_system_setting(db, "app_tagline"),
        favicon_url=await get_system_setting(db, "favicon_url"),
        logo_url=await get_system_setting(db, "logo_url"),
        custom_css=await get_system_setting(db, "custom_css"),
        custom_themes=await get_system_setting(db, "custom_themes"),
    )


@router.post("/settings/branding", response_model=BrandingSettingsSchema)
async def update_branding_settings(
    data: BrandingSettingsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update branding settings"""
    await set_setting(db, "app_name", data.app_name)
    await set_setting(db, "app_tagline", data.app_tagline)
    await set_setting(db, "favicon_url", data.favicon_url)
    await set_setting(db, "logo_url", data.logo_url)
    await set_setting(db, "custom_css", data.custom_css)
    await set_setting(db, "custom_themes", data.custom_themes)
    
    await db.commit()
    
    return data


@router.post("/settings/branding/favicon")
async def upload_favicon(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Upload a favicon image (PNG, JPG, ICO, or SVG)"""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/x-icon", "image/svg+xml", "image/vnd.microsoft.icon"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: PNG, JPG, ICO, SVG"
        )
    
    # Validate file size (max 1MB)
    contents = await file.read()
    if len(contents) > 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 1MB"
        )
    
    # Determine extension
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/x-icon": ".ico",
        "image/vnd.microsoft.icon": ".ico",
        "image/svg+xml": ".svg",
    }
    ext = ext_map.get(file.content_type, ".png")
    
    # Save file with unique name
    filename = f"favicon{ext}"
    filepath = get_branding_assets_dir() / filename
    
    with open(filepath, "wb") as f:
        f.write(contents)
    
    # Update favicon_url setting to point to the uploaded file
    favicon_url = f"/api/admin/branding/favicon{ext}"
    await set_setting(db, "favicon_url", favicon_url)
    await db.commit()
    
    return {"url": favicon_url, "filename": filename}


@router.post("/settings/branding/logo")
async def upload_logo(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Upload a logo image (PNG, JPG, or SVG)"""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/svg+xml"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: PNG, JPG, SVG"
        )
    
    # Validate file size (max 2MB)
    contents = await file.read()
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File too large. Maximum size is 2MB"
        )
    
    # Determine extension
    ext_map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/svg+xml": ".svg",
    }
    ext = ext_map.get(file.content_type, ".png")
    
    # Save file
    filename = f"logo{ext}"
    filepath = get_branding_assets_dir() / filename
    
    with open(filepath, "wb") as f:
        f.write(contents)
    
    # Update logo_url setting
    logo_url = f"/api/admin/branding/logo{ext}"
    await set_setting(db, "logo_url", logo_url)
    await db.commit()
    
    return {"url": logo_url, "filename": filename}


@router.get("/branding/{filename}")
async def serve_branding_asset(filename: str):
    """Serve uploaded branding assets (favicon, logo)"""
    # Sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    filepath = get_branding_assets_dir() / safe_filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    
    return FileResponse(filepath)


# =============================================================================
# OAUTH SETTINGS
# =============================================================================

@router.get("/oauth-settings", response_model=OAuthSettingsSchema)
async def get_oauth_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get OAuth provider settings"""
    return OAuthSettingsSchema(
        google_client_id=await get_system_setting(db, "google_client_id"),
        google_client_secret=await get_system_setting(db, "google_client_secret"),
        google_oauth_enabled=await get_system_setting_bool(db, "google_oauth_enabled"),
        google_oauth_timeout=await get_system_setting_int(db, "google_oauth_timeout"),
        github_client_id=await get_system_setting(db, "github_client_id"),
        github_client_secret=await get_system_setting(db, "github_client_secret"),
        github_oauth_enabled=await get_system_setting_bool(db, "github_oauth_enabled"),
        github_oauth_timeout=await get_system_setting_int(db, "github_oauth_timeout"),
    )


@router.put("/oauth-settings", response_model=OAuthSettingsSchema)
async def update_oauth_settings(
    data: OAuthSettingsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update OAuth provider settings"""
    await set_setting(db, "google_client_id", data.google_client_id)
    await set_setting(db, "google_client_secret", data.google_client_secret)
    await set_setting(db, "google_oauth_enabled", str(data.google_oauth_enabled).lower())
    await set_setting(db, "google_oauth_timeout", str(data.google_oauth_timeout))
    await set_setting(db, "github_client_id", data.github_client_id)
    await set_setting(db, "github_client_secret", data.github_client_secret)
    await set_setting(db, "github_oauth_enabled", str(data.github_oauth_enabled).lower())
    await set_setting(db, "github_oauth_timeout", str(data.github_oauth_timeout))
    
    await db.commit()
    
    return data


# =============================================================================
# LLM SETTINGS
# =============================================================================

@router.get("/llm-settings", response_model=LLMSettingsSchema)
async def get_llm_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get LLM connectivity settings"""
    return LLMSettingsSchema(
        llm_api_base_url=await get_system_setting(db, "llm_api_base_url"),
        llm_api_key=await get_system_setting(db, "llm_api_key"),
        llm_model=await get_system_setting(db, "llm_model"),
        llm_timeout=await get_system_setting_int(db, "llm_timeout"),
        llm_max_tokens=await get_system_setting_int(db, "llm_max_tokens"),
        llm_temperature=await get_system_setting_float(db, "llm_temperature"),
        llm_stream_default=await get_system_setting_bool(db, "llm_stream_default"),
        llm_multimodal=await get_system_setting_bool(db, "llm_multimodal"),
        # Thinking tokens
        think_begin_token=await get_system_setting(db, "think_begin_token") or "",
        think_end_token=await get_system_setting(db, "think_end_token") or "",
        # History compression
        history_compression_enabled=await get_system_setting_bool(db, "history_compression_enabled"),
        history_compression_threshold=await get_system_setting_int(db, "history_compression_threshold") or 20,
        history_compression_keep_recent=await get_system_setting_int(db, "history_compression_keep_recent") or 6,
        history_compression_target_tokens=await get_system_setting_int(db, "history_compression_target_tokens") or 8000,
    )


@router.put("/llm-settings", response_model=LLMSettingsSchema)
async def update_llm_settings(
    data: LLMSettingsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update LLM connectivity settings"""
    await set_setting(db, "llm_api_base_url", data.llm_api_base_url)
    await set_setting(db, "llm_api_key", data.llm_api_key)
    await set_setting(db, "llm_model", data.llm_model)
    await set_setting(db, "llm_timeout", str(data.llm_timeout))
    await set_setting(db, "llm_max_tokens", str(data.llm_max_tokens))
    await set_setting(db, "llm_temperature", str(data.llm_temperature))
    await set_setting(db, "llm_stream_default", str(data.llm_stream_default).lower())
    await set_setting(db, "llm_multimodal", str(data.llm_multimodal).lower())
    # Thinking tokens
    await set_setting(db, "think_begin_token", data.think_begin_token)
    await set_setting(db, "think_end_token", data.think_end_token)
    # History compression
    await set_setting(db, "history_compression_enabled", str(data.history_compression_enabled).lower())
    await set_setting(db, "history_compression_threshold", str(data.history_compression_threshold))
    await set_setting(db, "history_compression_keep_recent", str(data.history_compression_keep_recent))
    await set_setting(db, "history_compression_target_tokens", str(data.history_compression_target_tokens))
    
    await db.commit()
    
    return data


@router.post("/llm-settings/test")
async def test_llm_connection(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test LLM connection with current settings"""
    import httpx
    
    base_url = await get_system_setting(db, "llm_api_base_url")
    api_key = await get_system_setting(db, "llm_api_key")
    timeout = await get_system_setting_int(db, "llm_timeout")
    
    try:
        async with httpx.AsyncClient(timeout=min(timeout, 10)) as client:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Try to fetch models list
            response = await client.get(f"{base_url}/models", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", []) if isinstance(data, dict) else data
                model_names = [m.get("id", m.get("name", "unknown")) for m in models[:10]]
                return {
                    "success": True,
                    "message": f"Connected successfully. Found {len(models)} model(s).",
                    "models": model_names,
                }
            else:
                return {
                    "success": False,
                    "message": f"Server returned status {response.status_code}",
                    "models": [],
                }
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": "Connection timed out",
            "models": [],
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "models": [],
        }


# =============================================================================
# FEATURE FLAGS
# =============================================================================

@router.get("/feature-flags", response_model=FeatureFlagsSchema)
async def get_feature_flags(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get feature flags"""
    return FeatureFlagsSchema(
        enable_registration=await get_system_setting_bool(db, "enable_registration"),
        enable_billing=await get_system_setting_bool(db, "enable_billing"),
        freeforall=await get_system_setting_bool(db, "freeforall"),
        enable_safety_filters=await get_system_setting_bool(db, "enable_safety_filters"),
    )


@router.put("/feature-flags", response_model=FeatureFlagsSchema)
async def update_feature_flags(
    data: FeatureFlagsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update feature flags"""
    await set_setting(db, "enable_registration", str(data.enable_registration).lower())
    await set_setting(db, "enable_billing", str(data.enable_billing).lower())
    await set_setting(db, "freeforall", str(data.freeforall).lower())
    await set_setting(db, "enable_safety_filters", str(data.enable_safety_filters).lower())
    
    await db.commit()
    
    return data


# =============================================================================
# BILLING API SETTINGS
# =============================================================================

@router.get("/billing-api-settings", response_model=BillingApiSettingsSchema)
async def get_billing_api_settings(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get payment provider API settings"""
    return BillingApiSettingsSchema(
        # Stripe
        stripe_enabled=await get_system_setting_bool(db, "stripe_enabled"),
        stripe_api_key=await get_system_setting(db, "stripe_api_key"),
        stripe_webhook_secret=await get_system_setting(db, "stripe_webhook_secret"),
        stripe_publishable_key=await get_system_setting(db, "stripe_publishable_key"),
        stripe_pro_price_id=await get_system_setting(db, "stripe_pro_price_id"),
        stripe_enterprise_price_id=await get_system_setting(db, "stripe_enterprise_price_id"),
        # PayPal
        paypal_enabled=await get_system_setting_bool(db, "paypal_enabled"),
        paypal_client_id=await get_system_setting(db, "paypal_client_id"),
        paypal_client_secret=await get_system_setting(db, "paypal_client_secret"),
        paypal_webhook_id=await get_system_setting(db, "paypal_webhook_id"),
        paypal_mode=await get_system_setting(db, "paypal_mode") or "sandbox",
        paypal_pro_plan_id=await get_system_setting(db, "paypal_pro_plan_id"),
        paypal_enterprise_plan_id=await get_system_setting(db, "paypal_enterprise_plan_id"),
        # Google Pay
        google_pay_enabled=await get_system_setting_bool(db, "google_pay_enabled"),
        google_pay_merchant_id=await get_system_setting(db, "google_pay_merchant_id"),
        google_pay_merchant_name=await get_system_setting(db, "google_pay_merchant_name") or "NueChat",
    )


@router.put("/billing-api-settings", response_model=BillingApiSettingsSchema)
async def update_billing_api_settings(
    data: BillingApiSettingsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update payment provider API settings"""
    # Stripe
    await set_setting(db, "stripe_enabled", str(data.stripe_enabled).lower())
    await set_setting(db, "stripe_api_key", data.stripe_api_key)
    await set_setting(db, "stripe_webhook_secret", data.stripe_webhook_secret)
    await set_setting(db, "stripe_publishable_key", data.stripe_publishable_key)
    await set_setting(db, "stripe_pro_price_id", data.stripe_pro_price_id)
    await set_setting(db, "stripe_enterprise_price_id", data.stripe_enterprise_price_id)
    # PayPal
    await set_setting(db, "paypal_enabled", str(data.paypal_enabled).lower())
    await set_setting(db, "paypal_client_id", data.paypal_client_id)
    await set_setting(db, "paypal_client_secret", data.paypal_client_secret)
    await set_setting(db, "paypal_webhook_id", data.paypal_webhook_id)
    await set_setting(db, "paypal_mode", data.paypal_mode)
    await set_setting(db, "paypal_pro_plan_id", data.paypal_pro_plan_id)
    await set_setting(db, "paypal_enterprise_plan_id", data.paypal_enterprise_plan_id)
    # Google Pay
    await set_setting(db, "google_pay_enabled", str(data.google_pay_enabled).lower())
    await set_setting(db, "google_pay_merchant_id", data.google_pay_merchant_id)
    await set_setting(db, "google_pay_merchant_name", data.google_pay_merchant_name)
    
    await db.commit()
    
    return data


@router.post("/billing-api-settings/test-stripe")
async def test_stripe_connection(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test Stripe API connection"""
    api_key = await get_system_setting(db, "stripe_api_key")
    if not api_key:
        return {"success": False, "message": "Stripe API key not configured"}
    
    try:
        import stripe
        stripe.api_key = api_key
        # Try to fetch account info
        account = stripe.Account.retrieve()
        return {
            "success": True,
            "message": f"Connected to Stripe account: {account.get('email', account.get('id', 'Unknown'))}",
            "account_id": account.get("id"),
        }
    except ImportError:
        return {"success": False, "message": "Stripe library not installed"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/billing-api-settings/test-paypal")
async def test_paypal_connection(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test PayPal API connection"""
    import httpx
    
    client_id = await get_system_setting(db, "paypal_client_id")
    client_secret = await get_system_setting(db, "paypal_client_secret")
    mode = await get_system_setting(db, "paypal_mode") or "sandbox"
    
    if not client_id or not client_secret:
        return {"success": False, "message": "PayPal credentials not configured"}
    
    base_url = "https://api-m.sandbox.paypal.com" if mode == "sandbox" else "https://api-m.paypal.com"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Get access token
            response = await client.post(
                f"{base_url}/v1/oauth2/token",
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": f"Connected to PayPal ({mode} mode)",
                    "app_id": data.get("app_id"),
                }
            else:
                return {"success": False, "message": f"PayPal returned status {response.status_code}"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# =============================================================================
# API RATE LIMITS
# =============================================================================

@router.get("/api-rate-limits", response_model=APIRateLimitsSchema)
async def get_api_rate_limits(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get API rate limit settings"""
    return APIRateLimitsSchema(
        api_rate_limit_completions=await get_system_setting_int(db, "api_rate_limit_completions", 60),
        api_rate_limit_embeddings=await get_system_setting_int(db, "api_rate_limit_embeddings", 200),
        api_rate_limit_images=await get_system_setting_int(db, "api_rate_limit_images", 10),
        api_rate_limit_models=await get_system_setting_int(db, "api_rate_limit_models", 100),
    )


@router.put("/api-rate-limits", response_model=APIRateLimitsSchema)
async def update_api_rate_limits(
    data: APIRateLimitsSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update API rate limit settings"""
    await set_setting(db, "api_rate_limit_completions", str(data.api_rate_limit_completions))
    await set_setting(db, "api_rate_limit_embeddings", str(data.api_rate_limit_embeddings))
    await set_setting(db, "api_rate_limit_images", str(data.api_rate_limit_images))
    await set_setting(db, "api_rate_limit_models", str(data.api_rate_limit_models))
    
    await db.commit()
    
    return data


# =============================================================================
# TIERS
# =============================================================================

@router.get("/tiers", response_model=TiersSchema)
async def get_tiers(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get pricing tier configuration"""
    tiers_json = await get_system_setting(db, "tiers")
    try:
        tiers = json.loads(tiers_json)
    except json.JSONDecodeError:
        tiers = DEFAULT_TIERS
    return TiersSchema(tiers=[TierConfig(**t) for t in tiers])


@router.put("/tiers", response_model=TiersSchema)
async def update_tiers(
    data: TiersSchema,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update pricing tier configuration"""
    # Validate tier IDs are unique
    tier_ids = [t.id for t in data.tiers]
    if len(tier_ids) != len(set(tier_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tier IDs must be unique"
        )
    
    # Ensure required tiers exist
    required_tiers = {"free", "pro", "enterprise"}
    if not required_tiers.issubset(set(tier_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must include free, pro, and enterprise tiers"
        )
    
    tiers_json = json.dumps([t.model_dump() for t in data.tiers])
    await set_setting(db, "tiers", tiers_json)
    await db.commit()
    
    return data


# Public endpoint for billing page (no admin required)
@router.get("/public/tiers", response_model=TiersSchema)
async def get_public_tiers(
    db: AsyncSession = Depends(get_db),
):
    """Get pricing tier configuration (public endpoint for billing page)"""
    tiers_json = await get_system_setting(db, "tiers")
    try:
        tiers = json.loads(tiers_json)
    except json.JSONDecodeError:
        tiers = DEFAULT_TIERS
    return TiersSchema(tiers=[TierConfig(**t) for t in tiers])


@router.get("/public/branding", response_model=BrandingSettingsSchema)
async def get_public_branding(
    db: AsyncSession = Depends(get_db),
):
    """Get branding settings (public endpoint for frontend)"""
    return BrandingSettingsSchema(
        app_name=await get_system_setting(db, "app_name"),
        app_tagline=await get_system_setting(db, "app_tagline"),
        favicon_url=await get_system_setting(db, "favicon_url"),
        logo_url=await get_system_setting(db, "logo_url"),
        custom_css=await get_system_setting(db, "custom_css"),
        custom_themes=await get_system_setting(db, "custom_themes"),
    )


# =============================================================================
# USER MANAGEMENT
# =============================================================================

class UserListItem(BaseModel):
    """User summary for admin list view"""
    id: str
    email: str
    username: str
    tier: str
    is_admin: bool
    is_active: bool
    tokens_limit: int
    tokens_used: int
    chat_count: int
    created_at: str


class UserListResponse(BaseModel):
    """Paginated user list"""
    users: List[UserListItem]
    total: int
    page: int
    page_size: int


class UserUpdateRequest(BaseModel):
    """Request to update user settings"""
    tier: Optional[str] = None
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None
    tokens_limit: Optional[int] = None


class SetUserPasswordRequest(BaseModel):
    """Request to set user password (admin only)"""
    password: str


class TokenResetResponse(BaseModel):
    """Response after resetting user tokens"""
    user_id: str
    tokens_remaining: int
    tokens_limit: int
    message: str


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all users with pagination and search"""
    from app.models.models import TokenUsage
    from datetime import datetime, timezone, timezone
    
    billing = BillingService()
    now = datetime.now(timezone.utc)
    
    # Base query
    query = select(User)
    
    # Search filter
    if search:
        search_filter = f"%{search}%"
        query = query.where(
            (User.email.ilike(search_filter)) |
            (User.username.ilike(search_filter))
        )
    
    # Count total
    count_result = await db.execute(
        select(func.count(User.id)).select_from(query.subquery())
    )
    total = count_result.scalar() or 0
    
    # Paginate
    query = query.order_by(User.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Get usage and chat count for each user
    user_items = []
    for u in users:
        # Get current month usage
        usage_result = await db.execute(
            select(func.sum(TokenUsage.input_tokens + TokenUsage.output_tokens))
            .where(
                TokenUsage.user_id == u.id,
                TokenUsage.year == now.year,
                TokenUsage.month == now.month,
            )
        )
        tokens_used = usage_result.scalar() or 0
        
        # Get chat count
        chat_count_result = await db.execute(
            select(func.count(Chat.id)).where(Chat.owner_id == u.id)
        )
        chat_count = chat_count_result.scalar() or 0
        
        user_items.append(UserListItem(
            id=u.id,
            email=u.email,
            username=u.username,
            tier=u.tier.value,
            is_admin=u.is_admin,
            is_active=u.is_active,
            tokens_limit=u.tokens_limit,
            tokens_used=tokens_used,
            chat_count=chat_count,
            created_at=u.created_at.isoformat() if u.created_at else "",
        ))
    
    return UserListResponse(
        users=user_items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str,
    data: UserUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update user settings (tier, admin status, active status, token limit)"""
    result = await db.execute(select(User).where(User.id == user_id))
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Prevent self-demotion from admin
    if target_user.id == admin.id and data.is_admin is False:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove your own admin status"
        )
    
    if data.tier is not None:
        try:
            target_user.tier = UserTier(data.tier)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tier: {data.tier}"
            )
    
    if data.is_admin is not None:
        target_user.is_admin = data.is_admin
    
    if data.is_active is not None:
        target_user.is_active = data.is_active
    
    if data.tokens_limit is not None:
        target_user.tokens_limit = data.tokens_limit
    
    await db.commit()
    
    return {
        "id": target_user.id,
        "email": target_user.email,
        "tier": target_user.tier.value,
        "is_admin": target_user.is_admin,
        "is_active": target_user.is_active,
        "tokens_limit": target_user.tokens_limit,
    }


@router.post("/users/{user_id}/set-password")
async def set_user_password(
    user_id: str,
    data: SetUserPasswordRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Set a user's password (admin only). For non-OAuth users."""
    from app.services.auth import AuthService
    
    result = await db.execute(select(User).where(User.id == user_id))
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate password length
    if len(data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    
    # Hash and set the new password
    target_user.hashed_password = AuthService.hash_password(data.password)
    
    await db.commit()
    
    return {
        "message": f"Password updated for user {target_user.email}",
        "user_id": target_user.id,
    }


@router.post("/users/{user_id}/reset-tokens", response_model=TokenResetResponse)
async def reset_user_tokens(
    user_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Reset a user's token usage for the current period"""
    result = await db.execute(select(User).where(User.id == user_id))
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    billing = BillingService()
    summary = await billing.reset_user_tokens(db, target_user)
    
    return TokenResetResponse(
        user_id=user_id,
        tokens_remaining=summary["tokens_remaining"],
        tokens_limit=summary["tokens_limit"],
        message=f"Token usage reset for {target_user.email}",
    )


# =============================================================================
# USER CHATS (Admin view of other users' chats)
# =============================================================================

class ChatListItem(BaseModel):
    """Chat summary for admin list view"""
    id: str
    title: str
    model: Optional[str]
    owner_email: Optional[str] = None
    message_count: int
    created_at: str
    updated_at: Optional[str]


class ChatListResponse(BaseModel):
    """Paginated chat list"""
    chats: List[ChatListItem]
    total: int
    page: int
    page_size: int


class MessageItem(BaseModel):
    """Message for admin view"""
    id: str
    role: str
    content: str
    created_at: str


class ChatDetailResponse(BaseModel):
    """Detailed chat view for admin"""
    id: str
    title: str
    model: Optional[str]
    owner_id: str
    owner_email: str
    owner_username: str
    created_at: str
    updated_at: Optional[str]
    messages: List[MessageItem]


@router.get("/users/{user_id}/chats", response_model=ChatListResponse)
async def list_user_chats(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all chats for a specific user (admin only)"""
    # Verify user exists
    user_result = await db.execute(select(User).where(User.id == user_id))
    target_user = user_result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Base query
    query = select(Chat).where(Chat.owner_id == user_id)
    
    # Search filter
    if search:
        query = query.where(Chat.title.ilike(f"%{search}%"))
    
    # Count total
    count_result = await db.execute(
        select(func.count(Chat.id)).select_from(query.subquery())
    )
    total = count_result.scalar() or 0
    
    # Paginate
    query = query.order_by(Chat.updated_at.desc().nullsfirst(), Chat.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    chats = result.scalars().all()
    
    # Get message counts
    chat_items = []
    for chat in chats:
        msg_count_result = await db.execute(
            select(func.count(Message.id)).where(Message.chat_id == chat.id)
        )
        msg_count = msg_count_result.scalar() or 0
        
        chat_items.append(ChatListItem(
            id=chat.id,
            title=chat.title or "Untitled",
            model=chat.model,
            message_count=msg_count,
            created_at=chat.created_at.isoformat() if chat.created_at else "",
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
        ))
    
    return ChatListResponse(
        chats=chat_items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat_detail(
    chat_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed chat view with messages (admin only)"""
    # Get chat with owner
    result = await db.execute(
        select(Chat).where(Chat.id == chat_id)
    )
    chat = result.scalar_one_or_none()
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Get owner
    owner_result = await db.execute(select(User).where(User.id == chat.owner_id))
    owner = owner_result.scalar_one_or_none()
    
    # Get messages
    msg_result = await db.execute(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
    )
    messages = msg_result.scalars().all()
    
    return ChatDetailResponse(
        id=chat.id,
        title=chat.title or "Untitled",
        model=chat.model,
        owner_id=chat.owner_id,
        owner_email=owner.email if owner else "Unknown",
        owner_username=owner.username if owner else "Unknown",
        created_at=chat.created_at.isoformat() if chat.created_at else "",
        updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
        messages=[
            MessageItem(
                id=m.id,
                role=m.role,
                content=m.content or "",
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ],
    )


@router.get("/all-chats", response_model=ChatListResponse)
async def list_all_chats(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    user_id: Optional[str] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all chats across all users (admin only)"""
    # Base query with user join for email
    query = select(Chat, User.email).outerjoin(User, Chat.owner_id == User.id)
    
    # Filter by user if specified
    if user_id:
        query = query.where(Chat.owner_id == user_id)
    
    # Search filter
    if search:
        query = query.where(Chat.title.ilike(f"%{search}%"))
    
    # Count total
    count_query = select(func.count(Chat.id))
    if user_id:
        count_query = count_query.where(Chat.owner_id == user_id)
    if search:
        count_query = count_query.where(Chat.title.ilike(f"%{search}%"))
    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0
    
    # Paginate
    query = query.order_by(Chat.updated_at.desc().nullsfirst(), Chat.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    rows = result.all()
    
    # Get message counts
    chat_items = []
    for chat, owner_email in rows:
        msg_count_result = await db.execute(
            select(func.count(Message.id)).where(Message.chat_id == chat.id)
        )
        msg_count = msg_count_result.scalar() or 0
        
        chat_items.append(ChatListItem(
            id=chat.id,
            title=chat.title or "Untitled",
            model=chat.model,
            owner_email=owner_email,
            message_count=msg_count,
            created_at=chat.created_at.isoformat() if chat.created_at else "",
            updated_at=chat.updated_at.isoformat() if chat.updated_at else None,
        ))
    
    return ChatListResponse(
        chats=chat_items,
        total=total,
        page=page,
        page_size=page_size,
    )


# =============================================================================
# FILTER MANAGEMENT
# =============================================================================

from app.models.models import ChatFilter, FilterType, FilterPriority


class FilterSchema(BaseModel):
    """Schema for a chat filter"""
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    filter_type: str  # to_llm, from_llm, to_tools, from_tools
    priority: str = "medium"  # highest, high, medium, low, least
    enabled: bool = True
    filter_mode: str = "pattern"  # pattern, code, llm
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    word_list: Optional[List[str]] = None
    case_sensitive: bool = False
    action: str = "modify"  # modify, block, log, passthrough
    block_message: Optional[str] = None
    code: Optional[str] = None
    llm_prompt: Optional[str] = None
    config: Optional[dict] = None
    is_global: bool = True


class FilterCreateRequest(BaseModel):
    """Request to create a filter"""
    name: str
    description: Optional[str] = None
    filter_type: str
    priority: str = "medium"
    enabled: bool = True
    filter_mode: str = "pattern"
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    word_list: Optional[List[str]] = None
    case_sensitive: bool = False
    action: str = "modify"
    block_message: Optional[str] = None
    code: Optional[str] = None
    llm_prompt: Optional[str] = None
    config: Optional[dict] = None
    is_global: bool = True


class FilterUpdateRequest(BaseModel):
    """Request to update a filter"""
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    enabled: Optional[bool] = None
    filter_mode: Optional[str] = None
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    word_list: Optional[List[str]] = None
    case_sensitive: Optional[bool] = None
    action: Optional[str] = None
    block_message: Optional[str] = None
    code: Optional[str] = None
    llm_prompt: Optional[str] = None
    config: Optional[dict] = None
    is_global: Optional[bool] = None


class FilterListResponse(BaseModel):
    """Response containing list of filters"""
    filters: List[FilterSchema]
    total: int


def filter_to_schema(f: ChatFilter) -> FilterSchema:
    """Convert ChatFilter model to schema"""
    return FilterSchema(
        id=f.id,
        name=f.name,
        description=f.description,
        filter_type=f.filter_type.value,
        priority=f.priority.value if f.priority else "medium",
        enabled=f.enabled,
        filter_mode=f.filter_mode or "pattern",
        pattern=f.pattern,
        replacement=f.replacement,
        word_list=f.word_list,
        case_sensitive=f.case_sensitive,
        action=f.action or "modify",
        block_message=f.block_message,
        code=f.code,
        llm_prompt=f.llm_prompt,
        config=f.config,
        is_global=f.is_global,
    )


@router.get("/filters", response_model=FilterListResponse)
async def list_filters(
    filter_type: Optional[str] = None,
    enabled: Optional[bool] = None,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List all filters, optionally filtered by type or enabled status"""
    query = select(ChatFilter)
    
    if filter_type:
        try:
            ft = FilterType(filter_type)
            query = query.where(ChatFilter.filter_type == ft)
        except ValueError:
            pass
    
    if enabled is not None:
        query = query.where(ChatFilter.enabled == enabled)
    
    query = query.order_by(ChatFilter.filter_type, ChatFilter.priority)
    
    result = await db.execute(query)
    filters = result.scalars().all()
    
    return FilterListResponse(
        filters=[filter_to_schema(f) for f in filters],
        total=len(filters),
    )


@router.get("/filters/{filter_id}", response_model=FilterSchema)
async def get_filter(
    filter_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific filter by ID"""
    result = await db.execute(select(ChatFilter).where(ChatFilter.id == filter_id))
    filter_obj = result.scalar_one_or_none()
    
    if not filter_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Filter not found"
        )
    
    return filter_to_schema(filter_obj)


@router.post("/filters", response_model=FilterSchema)
async def create_filter(
    data: FilterCreateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new filter"""
    # Validate filter type
    try:
        filter_type = FilterType(data.filter_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid filter_type: {data.filter_type}. Must be one of: to_llm, from_llm, to_tools, from_tools"
        )
    
    # Validate priority
    try:
        priority = FilterPriority(data.priority)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid priority: {data.priority}. Must be one of: highest, high, medium, low, least"
        )
    
    # Check for duplicate name
    existing = await db.execute(select(ChatFilter).where(ChatFilter.name == data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filter with name '{data.name}' already exists"
        )
    
    filter_obj = ChatFilter(
        name=data.name,
        description=data.description,
        filter_type=filter_type,
        priority=priority,
        enabled=data.enabled,
        filter_mode=data.filter_mode,
        pattern=data.pattern,
        replacement=data.replacement,
        word_list=data.word_list,
        case_sensitive=data.case_sensitive,
        action=data.action,
        block_message=data.block_message,
        code=data.code,
        llm_prompt=data.llm_prompt,
        config=data.config,
        is_global=data.is_global,
        owner_id=admin.id,
    )
    
    db.add(filter_obj)
    await db.commit()
    await db.refresh(filter_obj)
    
    return filter_to_schema(filter_obj)


@router.patch("/filters/{filter_id}", response_model=FilterSchema)
async def update_filter(
    filter_id: str,
    data: FilterUpdateRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing filter"""
    result = await db.execute(select(ChatFilter).where(ChatFilter.id == filter_id))
    filter_obj = result.scalar_one_or_none()
    
    if not filter_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Filter not found"
        )
    
    # Update fields
    if data.name is not None:
        # Check for duplicate name
        existing = await db.execute(
            select(ChatFilter).where(ChatFilter.name == data.name, ChatFilter.id != filter_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filter with name '{data.name}' already exists"
            )
        filter_obj.name = data.name
    
    if data.description is not None:
        filter_obj.description = data.description
    
    if data.priority is not None:
        try:
            filter_obj.priority = FilterPriority(data.priority)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid priority: {data.priority}"
            )
    
    if data.enabled is not None:
        filter_obj.enabled = data.enabled
    
    if data.filter_mode is not None:
        filter_obj.filter_mode = data.filter_mode
    
    if data.pattern is not None:
        filter_obj.pattern = data.pattern
    
    if data.replacement is not None:
        filter_obj.replacement = data.replacement
    
    if data.word_list is not None:
        filter_obj.word_list = data.word_list
    
    if data.case_sensitive is not None:
        filter_obj.case_sensitive = data.case_sensitive
    
    if data.action is not None:
        filter_obj.action = data.action
    
    if data.block_message is not None:
        filter_obj.block_message = data.block_message
    
    if data.code is not None:
        filter_obj.code = data.code
    
    if data.llm_prompt is not None:
        filter_obj.llm_prompt = data.llm_prompt
    
    if data.config is not None:
        filter_obj.config = data.config
    
    if data.is_global is not None:
        filter_obj.is_global = data.is_global
    
    await db.commit()
    await db.refresh(filter_obj)
    
    return filter_to_schema(filter_obj)


@router.delete("/filters/{filter_id}")
async def delete_filter(
    filter_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a filter"""
    result = await db.execute(select(ChatFilter).where(ChatFilter.id == filter_id))
    filter_obj = result.scalar_one_or_none()
    
    if not filter_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Filter not found"
        )
    
    await db.delete(filter_obj)
    await db.commit()
    
    return {"message": f"Filter '{filter_obj.name}' deleted"}


@router.post("/filters/{filter_id}/toggle")
async def toggle_filter(
    filter_id: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Toggle a filter's enabled status"""
    result = await db.execute(select(ChatFilter).where(ChatFilter.id == filter_id))
    filter_obj = result.scalar_one_or_none()
    
    if not filter_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Filter not found"
        )
    
    filter_obj.enabled = not filter_obj.enabled
    await db.commit()
    
    return {
        "id": filter_obj.id,
        "name": filter_obj.name,
        "enabled": filter_obj.enabled,
    }


@router.post("/filters/test")
async def test_filter(
    filter_id: str,
    content: str,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Test a filter with sample content"""
    import re
    
    result = await db.execute(select(ChatFilter).where(ChatFilter.id == filter_id))
    filter_obj = result.scalar_one_or_none()
    
    if not filter_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Filter not found"
        )
    
    original = content
    modified = False
    blocked = False
    block_reason = None
    
    if filter_obj.filter_mode == "pattern" and filter_obj.pattern:
        try:
            flags = 0 if filter_obj.case_sensitive else re.IGNORECASE
            
            if filter_obj.action == "block":
                if re.search(filter_obj.pattern, content, flags):
                    blocked = True
                    block_reason = filter_obj.block_message or "Content blocked by filter"
            elif filter_obj.action == "modify" and filter_obj.replacement is not None:
                new_content = re.sub(filter_obj.pattern, filter_obj.replacement, content, flags=flags)
                if new_content != content:
                    content = new_content
                    modified = True
        except re.error as e:
            return {
                "error": f"Invalid regex pattern: {e}",
                "original": original,
            }
    
    elif filter_obj.filter_mode == "pattern" and filter_obj.word_list:
        for word in filter_obj.word_list:
            if filter_obj.case_sensitive:
                if word in content:
                    if filter_obj.action == "block":
                        blocked = True
                        block_reason = filter_obj.block_message or f"Blocked word: {word}"
                        break
                    elif filter_obj.action == "modify" and filter_obj.replacement is not None:
                        content = content.replace(word, filter_obj.replacement)
                        modified = True
            else:
                if word.lower() in content.lower():
                    if filter_obj.action == "block":
                        blocked = True
                        block_reason = filter_obj.block_message or f"Blocked word: {word}"
                        break
                    elif filter_obj.action == "modify" and filter_obj.replacement is not None:
                        # Case-insensitive replacement
                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                        content = pattern.sub(filter_obj.replacement, content)
                        modified = True
    
    return {
        "original": original,
        "result": content,
        "modified": modified,
        "blocked": blocked,
        "block_reason": block_reason,
        "filter_name": filter_obj.name,
        "filter_type": filter_obj.filter_type.value,
    }


# =============================================================================
# Debug Settings
# =============================================================================

class DebugSettingsResponse(BaseModel):
    debug_token_resets: bool
    debug_document_queue: bool
    debug_rag: bool
    debug_filter_chains: bool
    last_token_reset_timestamp: Optional[str] = None
    token_refill_interval_hours: int


class DebugSettingsUpdate(BaseModel):
    debug_token_resets: Optional[bool] = None
    debug_document_queue: Optional[bool] = None
    debug_rag: Optional[bool] = None
    debug_filter_chains: Optional[bool] = None


@router.get("/debug-settings", response_model=DebugSettingsResponse)
async def get_debug_settings(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get debug settings for the Site Dev tab."""
    debug_token_resets = await get_system_setting(db, "debug_token_resets") == "true"
    debug_document_queue = await get_system_setting(db, "debug_document_queue") == "true"
    debug_rag = await get_system_setting(db, "debug_rag") == "true"
    debug_filter_chains = await get_system_setting(db, "debug_filter_chains") == "true"
    last_token_reset = await get_system_setting(db, "last_token_reset_timestamp")
    refill_hours = await get_system_setting_int(db, "token_refill_interval_hours")
    
    return DebugSettingsResponse(
        debug_token_resets=debug_token_resets,
        debug_document_queue=debug_document_queue,
        debug_rag=debug_rag,
        debug_filter_chains=debug_filter_chains,
        last_token_reset_timestamp=last_token_reset if last_token_reset else None,
        token_refill_interval_hours=refill_hours,
    )


@router.put("/debug-settings")
async def update_debug_settings(
    data: DebugSettingsUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update debug settings."""
    if data.debug_token_resets is not None:
        await set_setting(db, "debug_token_resets", "true" if data.debug_token_resets else "false")
    if data.debug_document_queue is not None:
        await set_setting(db, "debug_document_queue", "true" if data.debug_document_queue else "false")
    if data.debug_rag is not None:
        await set_setting(db, "debug_rag", "true" if data.debug_rag else "false")
    if data.debug_filter_chains is not None:
        await set_setting(db, "debug_filter_chains", "true" if data.debug_filter_chains else "false")
    
    await db.commit()
    
    return {"status": "ok"}


# ============================================================
# SECURITY SETTINGS
# ============================================================

class SecuritySettingsResponse(BaseModel):
    """Response for security settings"""
    secret_key: str  # Masked or actual
    debug_level: str


class SecuritySettingsUpdate(BaseModel):
    """Update security settings"""
    secret_key: Optional[str] = None
    debug_level: Optional[str] = None


@router.get("/security-settings", response_model=SecuritySettingsResponse)
async def get_security_settings(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Get security settings for the Site Dev tab."""
    import logging
    
    # Get stored SECRET_KEY (or show current from settings)
    stored_key = await get_system_setting(db, "secret_key")
    if stored_key:
        # Show first 8 and last 4 chars, mask the rest
        if len(stored_key) > 16:
            masked = stored_key[:8] + "..." + stored_key[-4:]
        else:
            masked = stored_key
    else:
        # Using env/default - show indicator
        masked = "(using environment variable or default)"
    
    # Get debug level
    debug_level = await get_system_setting(db, "debug_level") or "INFO"
    
    return SecuritySettingsResponse(
        secret_key=masked,
        debug_level=debug_level,
    )


@router.put("/security-settings")
async def update_security_settings(
    data: SecuritySettingsUpdate,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update security settings. SECRET_KEY changes require restart."""
    import logging
    
    if data.secret_key is not None:
        if len(data.secret_key) < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SECRET_KEY must be at least 32 characters"
            )
        await set_setting(db, "secret_key", data.secret_key)
        logger.info("SECRET_KEY updated via admin panel - restart required for changes to take effect")
    
    if data.debug_level is not None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if data.debug_level not in valid_levels:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid debug level. Must be one of: {', '.join(valid_levels)}"
            )
        await set_setting(db, "debug_level", data.debug_level)
        
        # Apply debug level immediately
        log_level = getattr(logging, data.debug_level)
        logging.getLogger().setLevel(log_level)
        # Also update specific loggers
        for name in ['app', 'uvicorn', 'uvicorn.access', 'uvicorn.error']:
            logging.getLogger(name).setLevel(log_level)
        
        logger.info(f"Debug level changed to {data.debug_level}")
    
    await db.commit()
    
    return {"status": "ok"}


# ============================================================
# LLM PROVIDERS
# ============================================================

class LLMProviderCreate(BaseModel):
    """Create/update an LLM provider"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    model_id: str
    is_multimodal: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    is_default: bool = False
    is_vision_default: bool = False
    is_enabled: bool = True
    timeout: int = 300
    max_tokens: int = 8192
    context_size: int = 128000
    temperature: str = "0.7"
    vision_prompt: Optional[str] = None


@router.get("/llm-providers")
async def list_llm_providers(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all LLM providers."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.models.llm_provider import LLMProvider
    
    result = await db.execute(
        select(LLMProvider).order_by(LLMProvider.is_default.desc(), LLMProvider.name)
    )
    providers = result.scalars().all()
    
    return {"providers": [p.to_dict() for p in providers]}


@router.post("/llm-providers")
async def create_llm_provider(
    data: LLMProviderCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new LLM provider."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    import uuid
    from app.models.llm_provider import LLMProvider
    
    # If setting as default, clear other defaults
    if data.is_default:
        await db.execute(
            select(LLMProvider).where(LLMProvider.is_default == True)
        )
        result = await db.execute(select(LLMProvider).where(LLMProvider.is_default == True))
        for p in result.scalars().all():
            p.is_default = False
    
    # If setting as vision default, clear other vision defaults
    if data.is_vision_default:
        result = await db.execute(select(LLMProvider).where(LLMProvider.is_vision_default == True))
        for p in result.scalars().all():
            p.is_vision_default = False
    
    provider = LLMProvider(
        id=str(uuid.uuid4()),
        name=data.name,
        base_url=data.base_url.rstrip('/'),
        api_key=data.api_key,
        model_id=data.model_id,
        is_multimodal=data.is_multimodal,
        supports_tools=data.supports_tools,
        supports_streaming=data.supports_streaming,
        is_default=data.is_default,
        is_vision_default=data.is_vision_default,
        is_enabled=data.is_enabled,
        timeout=data.timeout,
        max_tokens=data.max_tokens,
        context_size=data.context_size,
        temperature=data.temperature,
        vision_prompt=data.vision_prompt,
    )
    
    db.add(provider)
    await db.commit()
    await db.refresh(provider)
    
    return {"provider": provider.to_dict()}


@router.put("/llm-providers/{provider_id}")
async def update_llm_provider(
    provider_id: str,
    data: LLMProviderCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing LLM provider."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.models.llm_provider import LLMProvider
    
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # If setting as default, clear other defaults
    if data.is_default and not provider.is_default:
        result = await db.execute(select(LLMProvider).where(LLMProvider.is_default == True))
        for p in result.scalars().all():
            p.is_default = False
    
    # If setting as vision default, clear other vision defaults
    if data.is_vision_default and not provider.is_vision_default:
        result = await db.execute(select(LLMProvider).where(LLMProvider.is_vision_default == True))
        for p in result.scalars().all():
            p.is_vision_default = False
    
    provider.name = data.name
    provider.base_url = data.base_url.rstrip('/')
    provider.api_key = data.api_key
    provider.model_id = data.model_id
    provider.is_multimodal = data.is_multimodal
    provider.supports_tools = data.supports_tools
    provider.supports_streaming = data.supports_streaming
    provider.is_default = data.is_default
    provider.is_vision_default = data.is_vision_default
    provider.is_enabled = data.is_enabled
    provider.timeout = data.timeout
    provider.max_tokens = data.max_tokens
    provider.context_size = data.context_size
    provider.temperature = data.temperature
    provider.vision_prompt = data.vision_prompt
    
    await db.commit()
    await db.refresh(provider)
    
    return {"provider": provider.to_dict()}


@router.delete("/llm-providers/{provider_id}")
async def delete_llm_provider(
    provider_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an LLM provider."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.models.llm_provider import LLMProvider
    
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    await db.delete(provider)
    await db.commit()
    
    return {"status": "deleted"}


@router.post("/llm-providers/{provider_id}/test")
async def test_llm_provider(
    provider_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Test connection to an LLM provider."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.models.llm_provider import LLMProvider
    import httpx
    
    result = await db.execute(select(LLMProvider).where(LLMProvider.id == provider_id))
    provider = result.scalar_one_or_none()
    
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    try:
        # Try to list models
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Content-Type": "application/json"}
            if provider.api_key:
                headers["Authorization"] = f"Bearer {provider.api_key}"
            
            response = await client.get(
                f"{provider.base_url}/models",
                headers=headers,
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                model_names = [m.get("id", "unknown") for m in models[:10]]
                return {
                    "status": "ok",
                    "message": f"Connected successfully. Found {len(models)} models.",
                    "models": model_names,
                }
            else:
                return {
                    "status": "error",
                    "message": f"API returned status {response.status_code}: {response.text[:200]}",
                }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "message": "Connection timed out",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


# ============ RAG Model Management ============

@router.get("/rag/status")
async def get_rag_status(
    user: User = Depends(get_current_user),
):
    """Get RAG embedding model status."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.services.rag import RAGService
    
    status = RAGService.get_model_status()
    return {
        "status": "ok" if status["loaded"] else "not_loaded",
        "details": status,
        "message": "Model failed to load - use POST /admin/rag/reset to retry" if status["failed"] else None,
    }


@router.post("/rag/reset")
async def reset_rag_model(
    user: User = Depends(get_current_user),
):
    """Reset RAG model state and retry loading."""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    from app.services.rag import RAGService
    
    # Reset the failure flag
    RAGService.reset_model()
    
    # Try to load the model
    model = RAGService.get_model()
    
    status = RAGService.get_model_status()
    return {
        "status": "ok" if status["loaded"] else "failed",
        "details": status,
        "message": "Model loaded successfully" if status["loaded"] else "Model failed to load again",
    }
