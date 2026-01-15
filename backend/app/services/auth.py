"""
Authentication service with OAuth2 support
"""
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from jose import jwt, JWTError
import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from authlib.integrations.httpx_client import AsyncOAuth2Client
import httpx

from app.core.config import settings
from app.models.models import User, OAuthAccount, UserTier


class AuthService:
    """Handle authentication, tokens, and OAuth"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(
            password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except JWTError:
            return None
    
    @staticmethod
    async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Authenticating user: {email}")
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user:
            logger.warning(f"User not found: {email}")
            return None
            
        if not user.hashed_password:
            logger.warning(f"User has no password (OAuth only?): {email}")
            return None
            
        logger.info(f"Found user: id={user.id}, is_active={user.is_active}, has_password=True")
        logger.info(f"Stored hash (first 20 chars): {user.hashed_password[:20]}...")
        
        if not AuthService.verify_password(password, user.hashed_password):
            logger.warning(f"Password verification failed for: {email}")
            return None
            
        logger.info(f"Authentication successful for: {email}")
        return user
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        username: str,
        password: Optional[str] = None,
        full_name: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        user = User(
            email=email,
            username=username,
            hashed_password=AuthService.hash_password(password) if password else None,
            full_name=full_name,
            avatar_url=avatar_url,
            tier=UserTier.FREE,
            tokens_limit=settings.FREE_TIER_TOKENS,
        )
        db.add(user)
        await db.flush()
        return user
    
    @staticmethod
    def create_tokens(user: User) -> Tuple[str, str]:
        """Create both access and refresh tokens"""
        token_data = {"sub": user.id, "email": user.email}
        access_token = AuthService.create_access_token(token_data)
        refresh_token = AuthService.create_refresh_token(token_data)
        return access_token, refresh_token


class OAuth2Service:
    """Handle OAuth2 provider integrations"""
    
    GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_USERINFO_URL = "https://api.github.com/user"
    GITHUB_EMAIL_URL = "https://api.github.com/user/emails"
    
    @staticmethod
    def get_google_client(redirect_uri: str, client_id: str = None, client_secret: str = None) -> AsyncOAuth2Client:
        """Get Google OAuth client. Uses provided credentials or falls back to config."""
        return AsyncOAuth2Client(
            client_id=client_id or settings.GOOGLE_CLIENT_ID,
            client_secret=client_secret or settings.GOOGLE_CLIENT_SECRET,
            redirect_uri=redirect_uri,
            scope="openid email profile",
        )
    
    @staticmethod
    def get_github_client(redirect_uri: str, client_id: str = None, client_secret: str = None) -> AsyncOAuth2Client:
        """Get GitHub OAuth client. Uses provided credentials or falls back to config."""
        return AsyncOAuth2Client(
            client_id=client_id or settings.GITHUB_CLIENT_ID,
            client_secret=client_secret or settings.GITHUB_CLIENT_SECRET,
            redirect_uri=redirect_uri,
            scope="user:email",
        )
    
    @staticmethod
    async def get_google_auth_url(redirect_uri: str, state: str, db: AsyncSession = None) -> str:
        """Get Google authorization URL. Reads credentials from DB if available."""
        from app.services.settings_service import SettingsService
        
        client_id = None
        client_secret = None
        
        if db:
            oauth_settings = await SettingsService.get_google_oauth_settings(db)
            if oauth_settings["client_id"]:
                client_id = oauth_settings["client_id"]
                client_secret = oauth_settings["client_secret"]
        
        client = OAuth2Service.get_google_client(redirect_uri, client_id, client_secret)
        url, _ = client.create_authorization_url(
            OAuth2Service.GOOGLE_AUTHORIZE_URL,
            state=state,
        )
        return url
    
    @staticmethod
    async def get_github_auth_url(redirect_uri: str, state: str, db: AsyncSession = None) -> str:
        """Get GitHub authorization URL. Reads credentials from DB if available."""
        from app.services.settings_service import SettingsService
        
        client_id = None
        client_secret = None
        
        if db:
            oauth_settings = await SettingsService.get_github_oauth_settings(db)
            if oauth_settings["client_id"]:
                client_id = oauth_settings["client_id"]
                client_secret = oauth_settings["client_secret"]
        
        client = OAuth2Service.get_github_client(redirect_uri, client_id, client_secret)
        url, _ = client.create_authorization_url(
            OAuth2Service.GITHUB_AUTHORIZE_URL,
            state=state,
        )
        return url
    
    @staticmethod
    async def handle_google_callback(
        db: AsyncSession, 
        code: str, 
        redirect_uri: str
    ) -> Tuple[User, str, str]:
        """Handle Google OAuth callback, return user and tokens"""
        from app.services.settings_service import SettingsService
        
        # Get OAuth settings from database
        oauth_settings = await SettingsService.get_google_oauth_settings(db)
        client_id = oauth_settings["client_id"] if oauth_settings["client_id"] else None
        client_secret = oauth_settings["client_secret"] if oauth_settings["client_secret"] else None
        timeout = oauth_settings["timeout"]
        
        client = OAuth2Service.get_google_client(redirect_uri, client_id, client_secret)
        
        # Exchange code for token
        token = await client.fetch_token(
            OAuth2Service.GOOGLE_TOKEN_URL,
            code=code,
        )
        
        # Get user info
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            response = await http_client.get(
                OAuth2Service.GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {token['access_token']}"},
            )
            user_info = response.json()
        
        # Find or create user
        user = await OAuth2Service._get_or_create_oauth_user(
            db=db,
            provider="google",
            provider_user_id=user_info["id"],
            email=user_info["email"],
            full_name=user_info.get("name"),
            avatar_url=user_info.get("picture"),
            access_token=token["access_token"],
            refresh_token=token.get("refresh_token"),
        )
        
        access_token, refresh_token = AuthService.create_tokens(user)
        return user, access_token, refresh_token
    
    @staticmethod
    async def handle_github_callback(
        db: AsyncSession, 
        code: str, 
        redirect_uri: str
    ) -> Tuple[User, str, str]:
        """Handle GitHub OAuth callback, return user and tokens"""
        from app.services.settings_service import SettingsService
        
        # Get OAuth settings from database
        oauth_settings = await SettingsService.get_github_oauth_settings(db)
        client_id = oauth_settings["client_id"] if oauth_settings["client_id"] else None
        client_secret = oauth_settings["client_secret"] if oauth_settings["client_secret"] else None
        timeout = oauth_settings["timeout"]
        
        client = OAuth2Service.get_github_client(redirect_uri, client_id, client_secret)
        
        # Exchange code for token
        token = await client.fetch_token(
            OAuth2Service.GITHUB_TOKEN_URL,
            code=code,
        )
        
        # Get user info
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            headers = {
                "Authorization": f"Bearer {token['access_token']}",
                "Accept": "application/json",
            }
            
            response = await http_client.get(OAuth2Service.GITHUB_USERINFO_URL, headers=headers)
            user_info = response.json()
            
            # Get primary email
            email_response = await http_client.get(OAuth2Service.GITHUB_EMAIL_URL, headers=headers)
            emails = email_response.json()
            primary_email = next((e["email"] for e in emails if e["primary"]), None)
        
        # Find or create user
        user = await OAuth2Service._get_or_create_oauth_user(
            db=db,
            provider="github",
            provider_user_id=str(user_info["id"]),
            email=primary_email or user_info.get("email"),
            full_name=user_info.get("name"),
            avatar_url=user_info.get("avatar_url"),
            access_token=token["access_token"],
            refresh_token=None,
        )
        
        access_token, refresh_token = AuthService.create_tokens(user)
        return user, access_token, refresh_token
    
    @staticmethod
    async def _get_or_create_oauth_user(
        db: AsyncSession,
        provider: str,
        provider_user_id: str,
        email: str,
        full_name: Optional[str],
        avatar_url: Optional[str],
        access_token: str,
        refresh_token: Optional[str],
    ) -> User:
        """Find existing user by OAuth or email, or create new one"""
        
        # Check for existing OAuth account
        result = await db.execute(
            select(OAuthAccount).where(
                OAuthAccount.provider == provider,
                OAuthAccount.provider_user_id == provider_user_id,
            )
        )
        oauth_account = result.scalar_one_or_none()
        
        if oauth_account:
            # Update tokens
            oauth_account.access_token = access_token
            oauth_account.refresh_token = refresh_token
            
            # Get and return user
            result = await db.execute(select(User).where(User.id == oauth_account.user_id))
            user = result.scalar_one()
            user.last_login = datetime.now(timezone.utc)
            await db.flush()
            return user
        
        # Check for existing user by email
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            username = email.split("@")[0]
            # Ensure unique username
            base_username = username
            counter = 1
            while True:
                result = await db.execute(select(User).where(User.username == username))
                if not result.scalar_one_or_none():
                    break
                username = f"{base_username}{counter}"
                counter += 1
            
            user = await AuthService.create_user(
                db=db,
                email=email,
                username=username,
                full_name=full_name,
                avatar_url=avatar_url,
            )
        
        # Create OAuth account link
        oauth_account = OAuthAccount(
            user_id=user.id,
            provider=provider,
            provider_user_id=provider_user_id,
            access_token=access_token,
            refresh_token=refresh_token,
        )
        db.add(oauth_account)
        
        user.last_login = datetime.now(timezone.utc)
        await db.flush()
        
        return user


async def seed_admin_user(db: AsyncSession) -> Optional[User]:
    """
    Create or update admin user from environment settings.
    Called on application startup if ADMIN_PASS is set.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    admin_pass = settings.admin_password
    if not admin_pass:
        logger.info("No ADMIN_PASS configured, skipping admin seeding")
        return None
    
    logger.info(f"Seeding admin user: email={settings.ADMIN_EMAIL}, username={settings.ADMIN_USERNAME}")
    
    # Check if admin user exists
    result = await db.execute(select(User).where(User.email == settings.ADMIN_EMAIL))
    admin = result.scalar_one_or_none()
    
    hashed_password = AuthService.hash_password(admin_pass)
    logger.info(f"Generated password hash (first 20 chars): {hashed_password[:20]}...")
    
    if admin:
        logger.info(f"Updating existing admin user: id={admin.id}, is_active={admin.is_active}")
        # Update existing admin password and username
        admin.hashed_password = hashed_password
        admin.username = settings.ADMIN_USERNAME  # Also update username from env
        admin.is_admin = True
        admin.is_active = True  # Ensure admin is active
        admin.tier = UserTier.ENTERPRISE  # Admin gets enterprise tier
    else:
        logger.info("Creating new admin user")
        # Create new admin user
        admin = User(
            email=settings.ADMIN_EMAIL,
            username=settings.ADMIN_USERNAME,
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True,
            tier=UserTier.ENTERPRISE,
        )
        db.add(admin)
    
    await db.commit()
    await db.refresh(admin)
    logger.info(f"Admin user saved: id={admin.id}, email={admin.email}, is_active={admin.is_active}")
    return admin
