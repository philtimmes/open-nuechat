"""
Authentication API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone, timezone
import secrets

from app.db.database import get_db
from app.services.auth import AuthService, OAuth2Service
from app.api.schemas import (
    UserCreate, UserLogin, TokenResponse, TokenRefresh, 
    UserResponse, UserUpdate, LoginResponse
)
from app.api.dependencies import get_current_user
from app.models.models import User
from app.core.config import settings


router = APIRouter(tags=["Authentication"])


def get_oauth_callback_url(request: Request, route_name: str) -> str:
    """
    Generate OAuth callback URL, respecting PUBLIC_URL setting.
    
    When behind a reverse proxy (nginx), the backend sees http://localhost:8000
    but the actual URL is https://chat.example.com. PUBLIC_URL fixes this.
    """
    if settings.PUBLIC_URL:
        # Use configured public URL
        base = settings.PUBLIC_URL.rstrip("/")
        path = request.app.url_path_for(route_name)
        return f"{base}{path}"
    else:
        # Fall back to request-based URL (works when not behind proxy)
        return str(request.url_for(route_name))


@router.post("/register", response_model=LoginResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user with email/password"""
    
    # Check if email exists
    existing = await AuthService.get_user_by_email(db, user_data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create user
    user = await AuthService.create_user(
        db=db,
        email=user_data.email,
        username=user_data.username,
        password=user_data.password,
        full_name=user_data.full_name,
    )
    
    # Generate tokens
    access_token, refresh_token = AuthService.create_tokens(user)
    
    await db.commit()
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """Login with email/password"""
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"LOGIN ATTEMPT: {credentials.email}")
    
    user = await AuthService.authenticate_user(
        db=db,
        email=credentials.email,
        password=credentials.password,
    )
    
    if not user:
        logger.warning(f"LOGIN FAILED: {credentials.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    # Update last login
    user.last_login = datetime.now(timezone.utc)
    
    # Generate tokens
    access_token, refresh_token = AuthService.create_tokens(user)
    
    await db.commit()
    
    logger.debug(f"LOGIN SUCCESS: {credentials.email}")
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    data: TokenRefresh,
    db: AsyncSession = Depends(get_db),
):
    """Refresh access token using refresh token"""
    
    payload = AuthService.decode_token(data.refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    user = await AuthService.get_user_by_id(db, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Generate new tokens
    access_token, refresh_token = AuthService.create_tokens(user)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    user: User = Depends(get_current_user),
):
    """Get current user profile"""
    return user


@router.patch("/me", response_model=UserResponse)
async def update_me(
    updates: UserUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user profile"""
    
    if updates.username:
        user.username = updates.username
    if updates.full_name is not None:
        user.full_name = updates.full_name
    if updates.avatar_url is not None:
        user.avatar_url = updates.avatar_url
    if updates.theme:
        user.theme = updates.theme
    if updates.preferences is not None:
        user.preferences = updates.preferences
    
    await db.commit()
    return user


# ============ OAuth Routes ============

@router.get("/oauth/google")
async def google_login(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Initiate Google OAuth flow"""
    from app.services.settings_service import SettingsService
    
    # Check database settings first, fall back to config
    oauth_settings = await SettingsService.get_google_oauth_settings(db)
    
    client_id = oauth_settings["client_id"] or settings.GOOGLE_CLIENT_ID
    enabled = oauth_settings["enabled"] if oauth_settings["client_id"] else settings.ENABLE_OAUTH_GOOGLE
    
    if not client_id or not enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth not configured or disabled",
        )
    
    redirect_uri = get_oauth_callback_url(request, "google_callback")
    state = secrets.token_urlsafe(32)
    
    # Store state in session/cookie in production
    url = await OAuth2Service.get_google_auth_url(redirect_uri, state, db)
    
    return RedirectResponse(url=url)


@router.get("/oauth/google/callback")
async def google_callback(
    request: Request,
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db),
):
    """Handle Google OAuth callback"""
    redirect_uri = get_oauth_callback_url(request, "google_callback")
    
    try:
        user, access_token, refresh_token = await OAuth2Service.handle_google_callback(
            db=db,
            code=code,
            redirect_uri=redirect_uri,
        )
        
        await db.commit()
        
        # Redirect to frontend with tokens in URL fragment
        # The frontend will extract these and store them
        base_url = settings.PUBLIC_URL or str(request.base_url).rstrip("/")
        redirect_url = f"{base_url}/oauth/callback#access_token={access_token}&refresh_token={refresh_token}"
        
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        # Redirect to login with error
        base_url = settings.PUBLIC_URL or str(request.base_url).rstrip("/")
        error_url = f"{base_url}/login?error={str(e)}"
        return RedirectResponse(url=error_url)


@router.get("/oauth/github")
async def github_login(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Initiate GitHub OAuth flow"""
    from app.services.settings_service import SettingsService
    
    # Check database settings first, fall back to config
    oauth_settings = await SettingsService.get_github_oauth_settings(db)
    
    client_id = oauth_settings["client_id"] or settings.GITHUB_CLIENT_ID
    enabled = oauth_settings["enabled"] if oauth_settings["client_id"] else settings.ENABLE_OAUTH_GITHUB
    
    if not client_id or not enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="GitHub OAuth not configured or disabled",
        )
    
    redirect_uri = get_oauth_callback_url(request, "github_callback")
    state = secrets.token_urlsafe(32)
    
    url = await OAuth2Service.get_github_auth_url(redirect_uri, state, db)
    
    return RedirectResponse(url=url)


@router.get("/oauth/github/callback")
async def github_callback(
    request: Request,
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db),
):
    """Handle GitHub OAuth callback"""
    redirect_uri = get_oauth_callback_url(request, "github_callback")
    
    try:
        user, access_token, refresh_token = await OAuth2Service.handle_github_callback(
            db=db,
            code=code,
            redirect_uri=redirect_uri,
        )
        
        await db.commit()
        
        # Redirect to frontend with tokens in URL fragment
        base_url = settings.PUBLIC_URL or str(request.base_url).rstrip("/")
        redirect_url = f"{base_url}/oauth/callback#access_token={access_token}&refresh_token={refresh_token}"
        
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        # Redirect to login with error
        base_url = settings.PUBLIC_URL or str(request.base_url).rstrip("/")
        error_url = f"{base_url}/login?error={str(e)}"
        return RedirectResponse(url=error_url)
