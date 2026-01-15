"""
Temporary MCP Server Manager

Manages temporary MCP server installations that auto-expire after 4 hours of non-use.

Features:
- LLM can install MCP servers on-demand when mcp_install tool is enabled
- Tracks last usage time per server per user
- Background task cleans up expired servers
- Stores temporary servers in database with expiry tracking
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, text

from app.models.tool import Tool
from app.models.base import ToolType, generate_uuid

logger = logging.getLogger(__name__)

# Expiry time for temporary MCP servers (4 hours of non-use)
TEMP_MCP_EXPIRY_HOURS = 4

# Background cleanup interval (check every 15 minutes)
CLEANUP_INTERVAL_SECONDS = 15 * 60


class TempMCPManager:
    """
    Manages temporary MCP server installations.
    
    Temporary servers are marked with a special config flag and track
    their last usage time. A background task periodically removes
    servers that haven't been used in TEMP_MCP_EXPIRY_HOURS.
    """
    
    def __init__(self, session_maker):
        self.session_maker = session_maker
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start_cleanup_worker(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Temporary MCP cleanup worker started")
    
    async def stop_cleanup_worker(self):
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Temporary MCP cleanup worker stopped")
    
    async def _cleanup_loop(self):
        """Background loop that cleans up expired temporary MCP servers"""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired_servers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in temporary MCP cleanup: {e}")
    
    async def cleanup_expired_servers(self) -> int:
        """Remove temporary MCP servers that haven't been used in 4 hours"""
        async with self.session_maker() as db:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=TEMP_MCP_EXPIRY_HOURS)
                
                # Find expired temporary servers
                # Temporary servers have config->>'is_temporary' = 'true'
                result = await db.execute(
                    select(Tool).where(
                        Tool.tool_type == ToolType.MCP,
                        Tool.config['is_temporary'].as_boolean() == True,
                        Tool.updated_at < cutoff_time
                    )
                )
                expired_tools = result.scalars().all()
                
                if expired_tools:
                    tool_ids = [t.id for t in expired_tools]
                    tool_names = [t.name for t in expired_tools]
                    
                    # Delete expired tools
                    await db.execute(
                        delete(Tool).where(Tool.id.in_(tool_ids))
                    )
                    await db.commit()
                    
                    logger.info(f"Cleaned up {len(expired_tools)} expired temporary MCP servers: {tool_names}")
                    return len(expired_tools)
                
                return 0
            except Exception as e:
                logger.error(f"Failed to cleanup expired MCP servers: {e}")
                await db.rollback()
                return 0
    
    async def install_temp_server(
        self,
        db: AsyncSession,
        user_id: str,
        name: str,
        url: str,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Install a temporary MCP server for a user.
        
        Args:
            db: Database session
            user_id: ID of the user installing the server
            name: Display name for the server
            url: MCP server URL (e.g., npx command or HTTP endpoint)
            description: Optional description
            api_key: Optional API key for the server
            config: Additional configuration
        
        Returns:
            Dict with success status and tool info
        """
        try:
            # Check if server with same URL already exists for this user
            result = await db.execute(
                select(Tool).where(
                    Tool.url == url,
                    Tool.created_by == user_id
                )
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update last usage time (updated_at)
                existing.updated_at = datetime.now(timezone.utc)
                await db.commit()
                return {
                    "success": True,
                    "message": f"MCP server '{existing.name}' already installed, refreshed expiry",
                    "tool_id": existing.id,
                    "tool_name": existing.name,
                    "expires_in_hours": TEMP_MCP_EXPIRY_HOURS
                }
            
            # Create new temporary MCP server
            tool_config = config or {}
            tool_config["is_temporary"] = True
            tool_config["installed_at"] = datetime.now(timezone.utc).isoformat()
            
            new_tool = Tool(
                id=generate_uuid(),
                name=name,
                description=description or f"Temporary MCP server: {name}",
                tool_type=ToolType.MCP,
                url=url,
                api_key_encrypted=api_key,  # Should be encrypted by caller if needed
                is_public=False,  # Temporary servers are private
                is_enabled=True,
                config=tool_config,
                created_by=user_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            db.add(new_tool)
            await db.commit()
            await db.refresh(new_tool)
            
            logger.info(f"Installed temporary MCP server '{name}' for user {user_id}")
            
            return {
                "success": True,
                "message": f"Successfully installed temporary MCP server '{name}'",
                "tool_id": new_tool.id,
                "tool_name": new_tool.name,
                "expires_in_hours": TEMP_MCP_EXPIRY_HOURS,
                "note": f"This server will be automatically removed after {TEMP_MCP_EXPIRY_HOURS} hours of non-use"
            }
            
        except Exception as e:
            logger.error(f"Failed to install temporary MCP server: {e}")
            await db.rollback()
            return {
                "success": False,
                "error": str(e)
            }
    
    async def uninstall_temp_server(
        self,
        db: AsyncSession,
        user_id: str,
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Uninstall a temporary MCP server.
        
        Args:
            db: Database session
            user_id: ID of the user (must own the server)
            tool_id: ID of the tool to uninstall
            tool_name: Name of the tool to uninstall (if tool_id not provided)
        
        Returns:
            Dict with success status
        """
        try:
            if tool_id:
                result = await db.execute(
                    select(Tool).where(
                        Tool.id == tool_id,
                        Tool.created_by == user_id,
                        Tool.config['is_temporary'].as_boolean() == True
                    )
                )
            elif tool_name:
                result = await db.execute(
                    select(Tool).where(
                        Tool.name == tool_name,
                        Tool.created_by == user_id,
                        Tool.config['is_temporary'].as_boolean() == True
                    )
                )
            else:
                return {"success": False, "error": "Must provide tool_id or tool_name"}
            
            tool = result.scalar_one_or_none()
            
            if not tool:
                return {"success": False, "error": "Temporary MCP server not found"}
            
            tool_name_deleted = tool.name
            await db.delete(tool)
            await db.commit()
            
            logger.info(f"Uninstalled temporary MCP server '{tool_name_deleted}' for user {user_id}")
            
            return {
                "success": True,
                "message": f"Successfully uninstalled '{tool_name_deleted}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to uninstall temporary MCP server: {e}")
            await db.rollback()
            return {"success": False, "error": str(e)}
    
    async def list_temp_servers(
        self,
        db: AsyncSession,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        List all temporary MCP servers for a user.
        
        Returns list of server info dicts with expiry status.
        """
        try:
            result = await db.execute(
                select(Tool).where(
                    Tool.created_by == user_id,
                    Tool.config['is_temporary'].as_boolean() == True
                ).order_by(Tool.updated_at.desc())
            )
            tools = result.scalars().all()
            
            now = datetime.now(timezone.utc)
            servers = []
            
            for tool in tools:
                # Calculate time until expiry
                expires_at = tool.updated_at + timedelta(hours=TEMP_MCP_EXPIRY_HOURS)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                
                time_remaining = expires_at - now
                hours_remaining = max(0, time_remaining.total_seconds() / 3600)
                
                servers.append({
                    "id": tool.id,
                    "name": tool.name,
                    "url": tool.url,
                    "description": tool.description,
                    "installed_at": tool.config.get("installed_at") if tool.config else None,
                    "last_used": tool.updated_at.isoformat() if tool.updated_at else None,
                    "hours_until_expiry": round(hours_remaining, 1),
                    "is_expired": hours_remaining <= 0
                })
            
            return servers
            
        except Exception as e:
            logger.error(f"Failed to list temporary MCP servers: {e}")
            return []
    
    async def refresh_server_usage(
        self,
        db: AsyncSession,
        user_id: str,
        tool_id: str
    ) -> bool:
        """
        Refresh the last usage time for a temporary MCP server.
        Called whenever the server is used.
        """
        try:
            result = await db.execute(
                update(Tool)
                .where(
                    Tool.id == tool_id,
                    Tool.created_by == user_id,
                    Tool.config['is_temporary'].as_boolean() == True
                )
                .values(updated_at=datetime.now(timezone.utc))
            )
            await db.commit()
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to refresh MCP server usage: {e}")
            return False


# Global instance (initialized in main.py)
temp_mcp_manager: Optional[TempMCPManager] = None


def get_temp_mcp_manager() -> Optional[TempMCPManager]:
    """Get the global temporary MCP manager instance"""
    return temp_mcp_manager


def init_temp_mcp_manager(session_maker) -> TempMCPManager:
    """Initialize the global temporary MCP manager"""
    global temp_mcp_manager
    temp_mcp_manager = TempMCPManager(session_maker)
    return temp_mcp_manager
