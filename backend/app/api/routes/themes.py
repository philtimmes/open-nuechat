"""
Themes API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List
from datetime import datetime

from app.db.database import get_db
from app.api.dependencies import get_current_user, get_optional_user
from app.api.schemas import ThemeCreate, ThemeResponse
from app.models.models import User, Theme


router = APIRouter(tags=["Themes"])


# Built-in system themes
SYSTEM_THEMES = [
    {
        "name": "Midnight",
        "description": "Deep dark theme with purple accents",
        "colors": {
            "primary": "#8B5CF6",
            "secondary": "#A78BFA",
            "background": "#0F0F1A",
            "surface": "#1A1A2E",
            "text": "#FAFAFA",
            "text_secondary": "#A1A1AA",
            "accent": "#C084FC",
            "error": "#EF4444",
            "success": "#22C55E",
            "warning": "#F59E0B",
            "border": "#27273A",
        },
        "fonts": {
            "heading": "Cal Sans",
            "body": "Inter",
            "code": "JetBrains Mono",
        }
    },
    {
        "name": "Ocean",
        "description": "Calm blue theme inspired by the sea",
        "colors": {
            "primary": "#0EA5E9",
            "secondary": "#38BDF8",
            "background": "#0C1222",
            "surface": "#162032",
            "text": "#F8FAFC",
            "text_secondary": "#94A3B8",
            "accent": "#22D3EE",
            "error": "#F87171",
            "success": "#34D399",
            "warning": "#FBBF24",
            "border": "#1E3A5F",
        },
        "fonts": {
            "heading": "Outfit",
            "body": "Inter",
            "code": "Fira Code",
        }
    },
    {
        "name": "Forest",
        "description": "Natural green theme",
        "colors": {
            "primary": "#22C55E",
            "secondary": "#4ADE80",
            "background": "#0A1410",
            "surface": "#14261C",
            "text": "#F0FDF4",
            "text_secondary": "#86EFAC",
            "accent": "#10B981",
            "error": "#FB7185",
            "success": "#22C55E",
            "warning": "#FACC15",
            "border": "#1F3D2B",
        },
        "fonts": {
            "heading": "Bricolage Grotesque",
            "body": "DM Sans",
            "code": "Source Code Pro",
        }
    },
    {
        "name": "Sunset",
        "description": "Warm orange and pink theme",
        "colors": {
            "primary": "#F97316",
            "secondary": "#FB923C",
            "background": "#1C0F0F",
            "surface": "#2D1A1A",
            "text": "#FFF7ED",
            "text_secondary": "#FED7AA",
            "accent": "#EC4899",
            "error": "#EF4444",
            "success": "#84CC16",
            "warning": "#F59E0B",
            "border": "#422020",
        },
        "fonts": {
            "heading": "Sora",
            "body": "Work Sans",
            "code": "IBM Plex Mono",
        }
    },
    {
        "name": "Light",
        "description": "Clean light theme",
        "colors": {
            "primary": "#6366F1",
            "secondary": "#818CF8",
            "background": "#FFFFFF",
            "surface": "#F8FAFC",
            "text": "#0F172A",
            "text_secondary": "#475569",
            "accent": "#8B5CF6",
            "error": "#DC2626",
            "success": "#16A34A",
            "warning": "#D97706",
            "border": "#E2E8F0",
        },
        "fonts": {
            "heading": "Plus Jakarta Sans",
            "body": "Inter",
            "code": "JetBrains Mono",
        }
    },
    {
        "name": "Noir",
        "description": "Pure black and white minimal theme",
        "colors": {
            "primary": "#FFFFFF",
            "secondary": "#E5E5E5",
            "background": "#000000",
            "surface": "#1A1A1A",
            "text": "#FFFFFF",
            "text_secondary": "#A3A3A3",
            "accent": "#FFFFFF",
            "error": "#FF4444",
            "success": "#00FF88",
            "warning": "#FFAA00",
            "border": "#333333",
            "button": "#3A3A3A",
            "button_text": "#FFFFFF",
        },
        "fonts": {
            "heading": "Space Grotesk",
            "body": "Space Grotesk",
            "code": "Space Mono",
        }
    },
]


@router.get("", response_model=List[ThemeResponse])
async def list_themes(
    user: User = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """List all available themes (system + public + user's own + admin custom themes)"""
    from app.api.routes.admin import get_system_setting
    import json
    
    # Get system and public themes
    query = select(Theme).where(
        or_(Theme.is_system == True, Theme.is_public == True)
    )
    
    # Add user's private themes if authenticated
    if user:
        query = select(Theme).where(
            or_(
                Theme.is_system == True,
                Theme.is_public == True,
                Theme.creator_id == user.id,
            )
        )
    
    result = await db.execute(query)
    themes = result.scalars().all()
    
    theme_responses = [ThemeResponse.model_validate(t) for t in themes]
    
    # Also include custom themes from admin branding settings
    try:
        custom_themes_json = await get_system_setting(db, "custom_themes")
        if custom_themes_json:
            custom_themes = json.loads(custom_themes_json)
            for ct in custom_themes:
                if not ct.get('id') or not ct.get('name'):
                    continue
                
                # Convert CSS variable format to colors dict
                colors = {
                    'primary': ct.get('--color-primary', '#8B5CF6'),
                    'secondary': ct.get('--color-secondary', '#A78BFA'),
                    'background': ct.get('--color-background', '#0F0F1A'),
                    'surface': ct.get('--color-surface', '#1A1A2E'),
                    'text': ct.get('--color-text', '#FAFAFA'),
                    'text_secondary': ct.get('--color-text-secondary', '#A1A1AA'),
                    'accent': ct.get('--color-accent', '#C084FC'),
                    'error': ct.get('--color-error', '#EF4444'),
                    'success': ct.get('--color-success', '#22C55E'),
                    'warning': ct.get('--color-warning', '#F59E0B'),
                    'border': ct.get('--color-border', '#27273A'),
                    'button': ct.get('--color-button', ct.get('--color-primary', '#8B5CF6')),
                    'button_text': ct.get('--color-button-text', '#FFFFFF'),
                }
                
                # Create a ThemeResponse-like dict
                theme_responses.append(ThemeResponse(
                    id=f"custom-{ct['id']}",
                    name=ct['name'],
                    description=ct.get('description', 'Custom admin theme'),
                    is_public=True,
                    is_system=False,  # Mark as non-system so they appear in custom section
                    colors=colors,
                    fonts={'heading': 'Inter', 'body': 'Inter', 'code': 'JetBrains Mono'},
                    created_at=datetime.now(),
                ))
    except Exception as e:
        # Silently ignore errors parsing custom themes
        pass
    
    return theme_responses


@router.get("/system", response_model=List[dict])
async def get_system_themes():
    """Get built-in system themes"""
    return SYSTEM_THEMES


@router.post("", response_model=ThemeResponse)
async def create_theme(
    theme_data: ThemeCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a custom theme"""
    
    theme = Theme(
        creator_id=user.id,
        name=theme_data.name,
        description=theme_data.description,
        is_public=theme_data.is_public,
        is_system=False,
        colors=theme_data.colors.model_dump(),
        fonts=theme_data.fonts or {},
    )
    db.add(theme)
    await db.commit()
    await db.refresh(theme)
    
    return ThemeResponse.model_validate(theme)


@router.get("/{theme_id}", response_model=ThemeResponse)
async def get_theme(
    theme_id: str,
    user: User = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific theme"""
    
    result = await db.execute(select(Theme).where(Theme.id == theme_id))
    theme = result.scalar_one_or_none()
    
    if not theme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Theme not found",
        )
    
    # Check access
    if not theme.is_system and not theme.is_public:
        if not user or theme.creator_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Theme not found",
            )
    
    return ThemeResponse.model_validate(theme)


@router.delete("/{theme_id}")
async def delete_theme(
    theme_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a custom theme"""
    
    result = await db.execute(select(Theme).where(Theme.id == theme_id))
    theme = result.scalar_one_or_none()
    
    if not theme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Theme not found",
        )
    
    if theme.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete system themes",
        )
    
    if theme.creator_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete themes created by others",
        )
    
    await db.delete(theme)
    await db.commit()
    
    return {"status": "deleted", "theme_id": theme_id}


@router.post("/apply/{theme_id}")
async def apply_theme(
    theme_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Apply a theme to user's account"""
    
    # Get theme to verify it exists and user has access
    result = await db.execute(select(Theme).where(Theme.id == theme_id))
    theme = result.scalar_one_or_none()
    
    if not theme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Theme not found",
        )
    
    if not theme.is_system and not theme.is_public and theme.creator_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access this theme",
        )
    
    # Update user's theme
    user.theme = theme_id
    await db.commit()
    
    return {
        "status": "applied",
        "theme_id": theme_id,
        "theme_name": theme.name,
    }


async def seed_default_themes(db: AsyncSession):
    """Seed system themes if they don't exist, or update if they do"""
    for theme_data in SYSTEM_THEMES:
        # Check if theme exists
        result = await db.execute(
            select(Theme).where(
                Theme.name == theme_data["name"],
                Theme.is_system == True
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing system theme
            existing.description = theme_data["description"]
            existing.colors = theme_data["colors"]
            existing.fonts = theme_data["fonts"]
        else:
            theme = Theme(
                name=theme_data["name"],
                description=theme_data["description"],
                is_system=True,
                is_public=True,
                colors=theme_data["colors"],
                fonts=theme_data["fonts"],
            )
            db.add(theme)
    
    await db.commit()
