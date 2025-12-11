"""
API endpoint tests for Open-NueChat
"""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for the health check endpoint"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test that health endpoint returns OK"""
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestInfoEndpoint:
    """Tests for the info endpoint"""
    
    @pytest.mark.asyncio
    async def test_info_endpoint(self, client: AsyncClient):
        """Test that info endpoint returns app information"""
        response = await client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data


class TestBrandingEndpoint:
    """Tests for branding configuration"""
    
    @pytest.mark.asyncio
    async def test_branding_config(self, client: AsyncClient):
        """Test that branding config is returned"""
        response = await client.get("/api/branding/config")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "default_theme" in data
        assert "features" in data
    
    @pytest.mark.asyncio
    async def test_themes_endpoint(self, client: AsyncClient):
        """Test that themes are returned"""
        response = await client.get("/api/branding/themes")
        assert response.status_code == 200
        data = response.json()
        assert "default_theme" in data
        assert "themes" in data
        assert len(data["themes"]) > 0


class TestAuthEndpoints:
    """Tests for authentication endpoints"""
    
    @pytest.mark.asyncio
    async def test_register_user(self, client: AsyncClient, test_user_data: dict):
        """Test user registration"""
        response = await client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 200
        data = response.json()
        assert "user" in data
        assert "access_token" in data
        assert data["user"]["email"] == test_user_data["email"]
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user_data: dict):
        """Test that duplicate email registration fails"""
        # First registration
        await client.post("/api/auth/register", json=test_user_data)
        
        # Second registration with same email
        response = await client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_login_user(self, client: AsyncClient, test_user_data: dict):
        """Test user login"""
        # First register
        await client.post("/api/auth/register", json=test_user_data)
        
        # Then login
        login_data = {
            "email": test_user_data["email"],
            "password": test_user_data["password"]
        }
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, client: AsyncClient):
        """Test login with invalid credentials"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "wrongpassword"
        }
        response = await client.post("/api/auth/login", json=login_data)
        assert response.status_code == 401


class TestProtectedEndpoints:
    """Tests for protected endpoints"""
    
    @pytest.mark.asyncio
    async def test_access_without_auth(self, client: AsyncClient):
        """Test that protected endpoints require authentication"""
        response = await client.get("/api/chats")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_access_with_auth(self, client: AsyncClient, test_user_data: dict):
        """Test that protected endpoints work with authentication"""
        # Register and get token
        register_response = await client.post("/api/auth/register", json=test_user_data)
        token = register_response.json()["access_token"]
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get("/api/chats", headers=headers)
        assert response.status_code == 200
