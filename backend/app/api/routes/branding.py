"""
Branding & Configuration API Routes

Exposes public configuration and branding settings to the frontend.
These endpoints do not require authentication.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
import os as import_os

from app.core.config import settings
from app.db.database import get_db
from app.services.settings_service import SettingsService

router = APIRouter()


@router.get("/config")
async def get_public_config(db: AsyncSession = Depends(get_db)):
    """
    Get public configuration for the frontend.
    
    This endpoint exposes branding and feature flags that the frontend
    needs to render the UI correctly. No authentication required.
    
    Returns all customizable branding settings, with DB overrides for
    site name, tagline, URLs, favicon, copyright, and branding image.
    """
    # Get base branding from config (.env defaults)
    branding = settings.get_branding()
    
    # NC-0.8.0.27: Override with DB-stored branding settings
    from app.api.routes.admin import get_system_setting
    
    db_site_name = await get_system_setting(db, "branding_site_name")
    db_tagline = await get_system_setting(db, "branding_tagline")
    db_main_url = await get_system_setting(db, "branding_main_url")
    db_tos_url = await get_system_setting(db, "branding_tos_url")
    db_user_agreement_url = await get_system_setting(db, "branding_user_agreement_url")
    db_branding_image = await get_system_setting(db, "branding_image_url")
    db_copyright = await get_system_setting(db, "branding_copyright")
    db_favicon = await get_system_setting(db, "branding_favicon_url")
    
    if db_site_name:
        branding["app_name"] = db_site_name
        branding["logo_text"] = db_site_name
    if db_tagline:
        branding["app_tagline"] = db_tagline
    if db_main_url:
        branding["main_url"] = db_main_url
    if db_tos_url:
        branding["terms_url"] = db_tos_url
    if db_user_agreement_url:
        branding["user_agreement_url"] = db_user_agreement_url
    if db_branding_image:
        branding["branding_image_url"] = db_branding_image
    if db_copyright:
        branding["footer_text"] = db_copyright
    if db_favicon:
        branding["favicon_url"] = db_favicon
    elif import_os.path.exists("/app/data/favicon.png"):
        branding["favicon_url"] = "/api/branding/favicon.png"
    
    # Override feature flags with database values
    google_settings = await SettingsService.get_google_oauth_settings(db)
    github_settings = await SettingsService.get_github_oauth_settings(db)
    
    # OAuth is enabled if: DB enabled flag is True AND client_id is set (either in DB or config)
    google_enabled = google_settings["enabled"] and (google_settings["client_id"] or settings.GOOGLE_CLIENT_ID)
    github_enabled = github_settings["enabled"] and (github_settings["client_id"] or settings.GITHUB_CLIENT_ID)
    
    # Get other feature flags from database
    registration_enabled = await SettingsService.is_registration_enabled(db)
    billing_enabled = await SettingsService.is_billing_enabled(db)
    
    # Get mermaid rendering setting (default to True)
    from app.api.routes.admin import get_system_setting_bool
    mermaid_enabled = await get_system_setting_bool(db, "enable_mermaid_rendering", default=True)
    
    branding["features"] = {
        "registration": registration_enabled,
        "oauth_google": google_enabled,
        "oauth_github": github_enabled,
        "billing": billing_enabled,
        "public_assistants": settings.ENABLE_PUBLIC_ASSISTANTS,
        "public_knowledge_stores": settings.ENABLE_PUBLIC_KNOWLEDGE_STORES,
        "mermaid_rendering": mermaid_enabled,
    }
    
    return branding


@router.get("/favicon.png")
async def serve_favicon():
    """Serve the custom favicon from /app/data/favicon.png"""
    from fastapi.responses import FileResponse
    favicon_path = "/app/data/favicon.png"
    if import_os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/png")
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="No custom favicon")


@router.get("/image/{filename}")
async def serve_branding_image(filename: str):
    """Serve a branding image from /app/data/"""
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    # Sanitize
    safe = filename.replace("..", "").replace("/", "").replace("\\", "")
    path = f"/app/data/{safe}"
    if import_os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/manifest.json")
async def get_web_manifest():
    """
    Generate a dynamic web app manifest based on branding settings.
    
    This allows PWA support with customized app name and colors.
    """
    return {
        "name": settings.APP_NAME,
        "short_name": settings.APP_NAME[:12],
        "description": settings.APP_DESCRIPTION,
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": settings.BRAND_PRIMARY_COLOR or "#6366f1",
        "icons": [
            {
                "src": settings.FAVICON_URL,
                "sizes": "any",
                "type": "image/x-icon"
            }
        ]
    }


@router.get("/branding/css")
async def get_branding_css():
    """
    Generate CSS custom properties for brand colors.
    
    Can be loaded as a stylesheet to apply brand colors globally.
    """
    css_vars = []
    
    if settings.BRAND_PRIMARY_COLOR:
        css_vars.append(f"--brand-primary: {settings.BRAND_PRIMARY_COLOR};")
    if settings.BRAND_SECONDARY_COLOR:
        css_vars.append(f"--brand-secondary: {settings.BRAND_SECONDARY_COLOR};")
    if settings.BRAND_ACCENT_COLOR:
        css_vars.append(f"--brand-accent: {settings.BRAND_ACCENT_COLOR};")
    
    if css_vars:
        css = f":root {{\n  {chr(10).join(css_vars)}\n}}"
    else:
        css = "/* No custom brand colors configured */"
    
    from fastapi.responses import Response
    return Response(content=css, media_type="text/css")


@router.get("/themes")
async def get_available_themes():
    """
    Get list of available themes with their preview colors.
    
    Returns the default theme ID as specified in .env
    """
    # These match the seeded themes in the database
    themes = [
        {
            "id": "dark",
            "name": "Dark",
            "preview": {
                "background": "#0f172a",
                "primary": "#6366f1",
                "text": "#f8fafc"
            }
        },
        {
            "id": "light",
            "name": "Light",
            "preview": {
                "background": "#ffffff",
                "primary": "#6366f1",
                "text": "#0f172a"
            }
        },
        {
            "id": "midnight",
            "name": "Midnight",
            "preview": {
                "background": "#020617",
                "primary": "#8b5cf6",
                "text": "#e2e8f0"
            }
        },
        {
            "id": "forest",
            "name": "Forest",
            "preview": {
                "background": "#14532d",
                "primary": "#22c55e",
                "text": "#f0fdf4"
            }
        },
        {
            "id": "sunset",
            "name": "Sunset",
            "preview": {
                "background": "#1c1917",
                "primary": "#f97316",
                "text": "#fef3c7"
            }
        },
        {
            "id": "ocean",
            "name": "Ocean",
            "preview": {
                "background": "#0c4a6e",
                "primary": "#0ea5e9",
                "text": "#e0f2fe"
            }
        }
    ]
    
    return {
        "default_theme": settings.DEFAULT_THEME,
        "themes": themes
    }
