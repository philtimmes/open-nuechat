"""
Tool Service - MCP and OpenAPI tool execution

Handles:
- MCP (Model Context Protocol) server connections
- OpenAPI specification parsing and execution
- Tool schema caching and discovery
- Secure API key management
"""
import httpx
import json
import asyncio
import hashlib
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.models import Tool, ToolUsage, ToolType

logger = logging.getLogger(__name__)

# Encryption key for API keys (generate once and store in env)
# In production, use: Fernet.generate_key()
ENCRYPTION_KEY = getattr(settings, 'TOOL_ENCRYPTION_KEY', None)
if ENCRYPTION_KEY:
    _fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)
else:
    _fernet = None
    logger.warning("TOOL_ENCRYPTION_KEY not set - API keys will not be encrypted")


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage"""
    if not api_key:
        return ""
    if _fernet:
        return _fernet.encrypt(api_key.encode()).decode()
    return api_key  # Fallback: store plaintext (not recommended)


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key for use"""
    if not encrypted_key:
        return ""
    if _fernet:
        try:
            return _fernet.decrypt(encrypted_key.encode()).decode()
        except Exception:
            return encrypted_key  # Might be plaintext from before encryption
    return encrypted_key


class MCPClient:
    """
    Model Context Protocol client using the official fastmcp library.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30)
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from MCP server"""
        from fastmcp import Client
        
        try:
            async with Client(self.base_url) as client:
                tools = await client.list_tools()
                
                # Convert to our format
                result = []
                for tool in tools:
                    tool_schema = {}
                    if hasattr(tool, 'inputSchema') and tool.inputSchema:
                        tool_schema = tool.inputSchema
                    elif hasattr(tool, 'parameters') and tool.parameters:
                        tool_schema = tool.parameters
                    
                    result.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool_schema,
                        "inputSchema": tool_schema,
                    })
                logger.info(f"Discovered {len(result)} tools from MCP server")
                return result
                
        except Exception as e:
            logger.error(f"MCP discovery error: {e}")
            raise
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Any, Optional[str]]:
        """Execute a tool on the MCP server"""
        from fastmcp import Client
        
        try:
            async with Client(self.base_url) as client:
                result = await client.call_tool(tool_name, params)
                
                # Extract content
                result_data = None
                result_url = None
                
                if hasattr(result, 'content') and result.content:
                    # MCP returns content as a list of content blocks
                    contents = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            contents.append(item.text)
                        elif hasattr(item, 'url'):
                            result_url = item.url
                    result_data = "\n".join(contents) if contents else str(result)
                else:
                    result_data = str(result)
                
                return True, result_data, result_url
                
        except Exception as e:
            logger.error(f"MCP tool execution error: {e}")
            return False, {"error": str(e)}, None


class OpenAPIClient:
    """
    OpenAPI/Swagger client
    
    Parses OpenAPI specs and executes operations
    """
    
    def __init__(self, spec_url: str, api_key: Optional[str] = None, config: Optional[Dict] = None):
        self.spec_url = spec_url
        self.api_key = api_key
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30)
        self._spec: Optional[Dict] = None
        self._base_url: Optional[str] = None
    
    async def load_spec(self) -> Dict[str, Any]:
        """Load and parse OpenAPI specification"""
        if self._spec:
            return self._spec
        
        headers = {
            "Accept": "application/json, application/yaml, text/yaml, */*",
            "User-Agent": "Open-NueChat/1.0",
        }
        if self.api_key:
            auth_type = self.config.get('auth_type', 'bearer')
            if auth_type == 'bearer':
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif auth_type == 'api_key':
                headers[self.config.get('api_key_header', 'X-API-Key')] = self.api_key
            
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            response = await client.get(self.spec_url, headers=headers)
            response.raise_for_status()
            
            # Handle JSON or YAML
            content_type = response.headers.get('content-type', '')
            if 'yaml' in content_type or self.spec_url.endswith(('.yaml', '.yml')):
                import yaml
                self._spec = yaml.safe_load(response.text)
            else:
                self._spec = response.json()
            
            # Determine base URL
            if 'servers' in self._spec and self._spec['servers']:
                self._base_url = self._spec['servers'][0].get('url', '')
            elif 'host' in self._spec:  # OpenAPI 2.0
                scheme = self._spec.get('schemes', ['https'])[0]
                self._base_url = f"{scheme}://{self._spec['host']}{self._spec.get('basePath', '')}"
            
            # NC-0.8.0.12: If no base URL in spec, derive from spec URL
            if not self._base_url:
                # Extract base URL from spec_url (e.g., http://localhost:8000/openapi.json -> http://localhost:8000)
                from urllib.parse import urlparse
                parsed = urlparse(self.spec_url)
                self._base_url = f"{parsed.scheme}://{parsed.netloc}"
                logger.info(f"Derived base URL from spec URL: {self._base_url}")
            
            return self._spec
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Convert OpenAPI operations to tool definitions"""
        spec = await self.load_spec()
        tools = []
        
        paths = spec.get('paths', {})
        for path, methods in paths.items():
            for method, operation in methods.items():
                if method.lower() not in ['get', 'post', 'put', 'patch', 'delete']:
                    continue
                
                op_id = operation.get('operationId', f"{method}_{path}".replace('/', '_'))
                
                # Build parameter schema
                parameters = []
                for param in operation.get('parameters', []):
                    parameters.append({
                        "name": param.get('name'),
                        "type": param.get('schema', {}).get('type', 'string'),
                        "required": param.get('required', False),
                        "description": param.get('description', ''),
                        "in": param.get('in', 'query'),
                    })
                
                # Handle request body (OpenAPI 3.0)
                if 'requestBody' in operation:
                    content = operation['requestBody'].get('content', {})
                    json_schema = content.get('application/json', {}).get('schema', {})
                    if json_schema:
                        # NC-0.8.0.12: Resolve $ref if present
                        if '$ref' in json_schema:
                            ref_path = json_schema['$ref']
                            # Parse reference like "#/components/schemas/RetrievalQueryInput"
                            if ref_path.startswith('#/'):
                                parts = ref_path[2:].split('/')
                                resolved = spec
                                for part in parts:
                                    resolved = resolved.get(part, {})
                                json_schema = resolved
                        
                        # NC-0.8.0.12: Expose body properties directly instead of wrapping in 'body'
                        # This makes it clearer to the LLM what parameters to send
                        body_props = json_schema.get('properties', {})
                        body_required = json_schema.get('required', [])
                        
                        if body_props:
                            for prop_name, prop_schema in body_props.items():
                                # Resolve nested $ref in property schema
                                if '$ref' in prop_schema:
                                    ref_path = prop_schema['$ref']
                                    if ref_path.startswith('#/'):
                                        parts = ref_path[2:].split('/')
                                        resolved = spec
                                        for part in parts:
                                            resolved = resolved.get(part, {})
                                        prop_schema = resolved
                                
                                param_type = prop_schema.get('type', 'string')
                                # Handle array types
                                if param_type == 'array':
                                    items = prop_schema.get('items', {})
                                    item_type = items.get('type', 'string')
                                    param_desc = prop_schema.get('description', f'Array of {item_type}')
                                else:
                                    param_desc = prop_schema.get('description', '')
                                
                                parameters.append({
                                    "name": prop_name,
                                    "type": param_type,
                                    "required": prop_name in body_required,
                                    "description": param_desc,
                                    "in": "body",
                                })
                        else:
                            # Fallback: wrap entire schema as 'body' param
                            parameters.append({
                                "name": "body",
                                "type": "object",
                                "required": operation['requestBody'].get('required', False),
                                "description": "Request body",
                                "in": "body",
                                "schema": json_schema,
                            })
                
                tools.append({
                    "name": op_id,
                    "description": operation.get('summary', '') or operation.get('description', ''),
                    "parameters": parameters,
                    "method": method.upper(),
                    "path": path,
                    "tags": operation.get('tags', []),
                })
        
        return tools
    
    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, */*",
            "User-Agent": "Open-NueChat/1.0",
        }
        if self.api_key:
            auth_type = self.config.get('auth_type', 'bearer')
            auth_header = self.config.get('auth_header', 'Authorization')
            
            if auth_type == 'bearer':
                headers[auth_header] = f"Bearer {self.api_key}"
            elif auth_type == 'api_key':
                headers[self.config.get('api_key_header', 'X-API-Key')] = self.api_key
            elif auth_type == 'basic':
                import base64
                headers["Authorization"] = f"Basic {base64.b64encode(self.api_key.encode()).decode()}"
        
        custom_headers = self.config.get('headers', {})
        headers.update(custom_headers)
        
        return headers
    
    async def execute_operation(
        self,
        operation_id: str,
        params: Dict[str, Any]
    ) -> Tuple[bool, Any, Optional[str]]:
        """Execute an OpenAPI operation"""
        spec = await self.load_spec()
        
        # Find the operation
        operation = None
        op_method = None
        op_path = None
        
        for path, methods in spec.get('paths', {}).items():
            for method, op in methods.items():
                if op.get('operationId') == operation_id:
                    operation = op
                    op_method = method.upper()
                    op_path = path
                    break
            if operation:
                break
        
        if not operation:
            return False, {"error": f"Operation {operation_id} not found"}, None
        
        # Build request
        url = f"{self._base_url}{op_path}"
        query_params = {}
        path_params = {}
        body = None
        
        for param in operation.get('parameters', []):
            param_name = param.get('name')
            if param_name in params:
                if param.get('in') == 'query':
                    query_params[param_name] = params[param_name]
                elif param.get('in') == 'path':
                    path_params[param_name] = params[param_name]
                elif param.get('in') == 'header':
                    pass  # Handle separately
        
        # Handle body parameter - NC-0.8.0.12: Better request body handling
        if 'body' in params:
            body = params['body']
        elif 'requestBody' in operation:
            # If operation has a requestBody but no 'body' param was passed,
            # assume all non-path/query params are meant for the body
            body_schema = operation['requestBody'].get('content', {}).get('application/json', {}).get('schema', {})
            if body_schema:
                # Resolve $ref if present
                if '$ref' in body_schema:
                    ref_path = body_schema['$ref']
                    if ref_path.startswith('#/'):
                        parts = ref_path[2:].split('/')
                        resolved = spec
                        for part in parts:
                            resolved = resolved.get(part, {})
                        body_schema = resolved
                
                # Get properties from the resolved schema
                schema_props = body_schema.get('properties', {})
                if schema_props:
                    # Build body from schema properties
                    body = {}
                    for prop_name in schema_props:
                        if prop_name in params:
                            body[prop_name] = params[prop_name]
                    # If no matching props found, pass all remaining params
                    if not body:
                        body = {k: v for k, v in params.items() if k not in path_params and k not in query_params}
                else:
                    # No properties defined, pass all remaining params
                    body = {k: v for k, v in params.items() if k not in path_params and k not in query_params}
        
        # Substitute path parameters
        for name, value in path_params.items():
            url = url.replace(f"{{{name}}}", str(value))
        
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            try:
                response = await client.request(
                    method=op_method,
                    url=url,
                    params=query_params if query_params else None,
                    json=body,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                result = response.json() if response.content else {}
                
                # Try to extract URL from result
                result_url = None
                if isinstance(result, dict):
                    result_url = result.get('url') or result.get('link') or result.get('href')
                
                return True, result, result_url
                
            except httpx.HTTPStatusError as e:
                error_msg = f"API call failed: {e.response.status_code}"
                try:
                    error_detail = e.response.json()
                    error_msg = str(error_detail)
                except:
                    pass
                return False, {"error": error_msg}, None
            except Exception as e:
                return False, {"error": str(e)}, None


class ToolService:
    """
    Main tool service coordinating MCP and OpenAPI tools
    """
    
    SCHEMA_CACHE_HOURS = 24  # How long to cache tool schemas
    
    async def get_available_tools(
        self,
        db: AsyncSession,
        user_id: str,
        is_admin: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all tools available to a user"""
        
        query = select(Tool).where(Tool.is_enabled == True)
        
        if not is_admin:
            # Non-admins only see public tools
            query = query.where(Tool.is_public == True)
        
        result = await db.execute(query)
        tools = result.scalars().all()
        
        all_tool_defs = []
        
        for tool in tools:
            # Use cached schema if fresh
            if tool.schema_cache and tool.last_schema_fetch:
                # Ensure timezone-aware comparison (SQLite stores naive datetimes)
                last_fetch = tool.last_schema_fetch
                if last_fetch.tzinfo is None:
                    last_fetch = last_fetch.replace(tzinfo=timezone.utc)
                cache_age = datetime.now(timezone.utc) - last_fetch
                if cache_age < timedelta(hours=self.SCHEMA_CACHE_HOURS):
                    for t in tool.schema_cache:
                        t['_tool_id'] = tool.id
                        t['_tool_name'] = tool.name
                        t['_tool_type'] = tool.tool_type.value
                    all_tool_defs.extend(tool.schema_cache)
                    continue
            
            # Fetch fresh schema
            try:
                schemas = await self.refresh_tool_schema(db, tool)
                for t in schemas:
                    t['_tool_id'] = tool.id
                    t['_tool_name'] = tool.name
                    t['_tool_type'] = tool.tool_type.value
                all_tool_defs.extend(schemas)
            except Exception as e:
                logger.error(f"Failed to fetch schema for tool {tool.id}: {e}")
        
        return all_tool_defs
    
    async def refresh_tool_schema(
        self,
        db: AsyncSession,
        tool: Tool
    ) -> List[Dict[str, Any]]:
        """Refresh cached tool schema"""
        api_key = decrypt_api_key(tool.api_key_encrypted) if tool.api_key_encrypted else None
        
        try:
            if tool.tool_type == ToolType.MCP:
                client = MCPClient(tool.url, api_key, tool.config)
                schemas = await client.discover_tools()
            else:
                client = OpenAPIClient(tool.url, api_key, tool.config)
                schemas = await client.discover_tools()
                
                # Filter to enabled operations if specified
                if tool.enabled_operations:
                    schemas = [s for s in schemas if s['name'] in tool.enabled_operations]
            
            # Update cache
            tool.schema_cache = schemas
            tool.last_schema_fetch = datetime.now(timezone.utc)
            await db.commit()
            
            return schemas
            
        except Exception as e:
            logger.error(f"Schema refresh failed for tool {tool.id}: {e}")
            raise
    
    async def get_tools_for_llm(
        self,
        db: AsyncSession,
        user: Any,  # User model
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[Any, str]]]:
        """
        Get tools formatted for LLM API and a map to their database records.
        
        Returns: (tool_definitions, tool_map)
        - tool_definitions: List of tool defs in LLM format
        - tool_map: Dict mapping tool_name -> (Tool db record, operation_name)
        """
        query = select(Tool).where(Tool.is_enabled == True)
        
        if not user.is_admin:
            query = query.where(Tool.is_public == True)
        
        result = await db.execute(query)
        tools = result.scalars().all()
        
        tool_definitions = []
        tool_map = {}
        
        for tool in tools:
            # Get schema (cached or fresh)
            schemas = tool.schema_cache or []
            
            if not schemas or not tool.last_schema_fetch:
                try:
                    schemas = await self.refresh_tool_schema(db, tool)
                except Exception as e:
                    logger.warning(f"Failed to get schema for tool {tool.name}: {e}")
                    continue
            elif tool.last_schema_fetch:
                # Ensure timezone-aware comparison (SQLite stores naive datetimes)
                last_fetch = tool.last_schema_fetch
                if last_fetch.tzinfo is None:
                    last_fetch = last_fetch.replace(tzinfo=timezone.utc)
                cache_age = datetime.now(timezone.utc) - last_fetch
                if cache_age > timedelta(hours=self.SCHEMA_CACHE_HOURS):
                    try:
                        schemas = await self.refresh_tool_schema(db, tool)
                    except Exception as e:
                        logger.warning(f"Failed to refresh schema for tool {tool.name}: {e}")
            
            # Convert each operation/tool to LLM format
            for schema in schemas:
                # Create unique tool name (prefix with tool name to avoid collisions)
                operation_name = schema.get('name', 'unknown')
                llm_tool_name = f"{tool.name}_{operation_name}" if len(schemas) > 1 else operation_name
                
                # Build input schema from parameters
                input_schema = schema.get('parameters', schema.get('inputSchema', {}))
                if isinstance(input_schema, list):
                    # Convert parameter list to JSON schema
                    properties = {}
                    required = []
                    for param in input_schema:
                        param_name = param.get('name', 'param')
                        properties[param_name] = {
                            "type": param.get('type', 'string'),
                            "description": param.get('description', ''),
                        }
                        if param.get('required'):
                            required.append(param_name)
                    input_schema = {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                elif not isinstance(input_schema, dict):
                    input_schema = {"type": "object", "properties": {}}
                
                # Ensure input_schema has required structure
                if "type" not in input_schema:
                    input_schema["type"] = "object"
                if "properties" not in input_schema:
                    input_schema["properties"] = {}
                
                tool_def = {
                    "name": llm_tool_name,
                    "description": f"[{tool.name}] {schema.get('description', '')}",
                    "input_schema": input_schema,
                    "_tool_type": tool.tool_type.value,  # NC-0.8.0.12: Add tool type for filtering
                    "_tool_id": tool.id,
                    "_tool_name": tool.name,
                }
                
                tool_definitions.append(tool_def)
                tool_map[llm_tool_name] = (tool, operation_name)
        
        return tool_definitions, tool_map
    
    async def execute_tool(
        self,
        db: AsyncSession,
        tool: Tool,
        tool_name: str,
        params: Dict[str, Any],
        user_id: str,
        chat_id: str,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool and record usage
        
        Returns: result dict
        """
        import time
        start_time = time.time()
        
        if not tool or not tool.is_enabled:
            return {"error": "Tool not found or disabled"}
        
        api_key = decrypt_api_key(tool.api_key_encrypted) if tool.api_key_encrypted else None
        
        # Execute based on type
        try:
            if tool.tool_type == ToolType.MCP:
                client = MCPClient(tool.url, api_key, tool.config)
                success, result_data, result_url = await client.execute_tool(tool_name, params)
            else:
                client = OpenAPIClient(tool.url, api_key, tool.config)
                success, result_data, result_url = await client.execute_operation(tool_name, params)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Create usage record
            usage = ToolUsage(
                tool_id=tool.id,
                message_id=message_id,
                chat_id=chat_id,
                user_id=user_id,
                tool_name=tool_name,
                operation=tool_name if tool.tool_type == ToolType.OPENAPI else None,
                input_params=params,
                success=success,
                result_summary=self._summarize_result(result_data) if success else None,
                result_url=result_url,
                error_message=str(result_data.get('error')) if not success and isinstance(result_data, dict) else None,
                duration_ms=duration_ms
            )
            db.add(usage)
            await db.commit()
            
            if success:
                return result_data
            else:
                return {"error": result_data.get('error', 'Tool execution failed') if isinstance(result_data, dict) else str(result_data)}
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            
            # Record failed usage
            usage = ToolUsage(
                tool_id=tool.id,
                message_id=message_id,
                chat_id=chat_id,
                user_id=user_id,
                tool_name=tool_name,
                input_params=params,
                success=False,
                error_message=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            db.add(usage)
            await db.commit()
            
            return {"error": str(e)}
    
    def _summarize_result(self, result: Any, max_length: int = 200) -> str:
        """Create a brief summary of tool result for citations"""
        if isinstance(result, str):
            return result[:max_length] + "..." if len(result) > max_length else result
        elif isinstance(result, dict):
            # Try to extract meaningful summary
            for key in ['summary', 'message', 'result', 'data', 'content', 'text']:
                if key in result:
                    val = str(result[key])
                    return val[:max_length] + "..." if len(val) > max_length else val
            return json.dumps(result)[:max_length]
        elif isinstance(result, list):
            return f"[{len(result)} items]"
        else:
            return str(result)[:max_length]
    
    async def get_tool_usage_for_message(
        self,
        db: AsyncSession,
        message_id: str
    ) -> List[ToolUsage]:
        """Get all tool usages for a message (for citations)"""
        result = await db.execute(
            select(ToolUsage)
            .where(ToolUsage.message_id == message_id)
            .order_by(ToolUsage.created_at)
        )
        return list(result.scalars().all())
    
    async def get_chat_tool_usage(
        self,
        db: AsyncSession,
        chat_id: str
    ) -> List[ToolUsage]:
        """Get all tool usages for a chat"""
        result = await db.execute(
            select(ToolUsage)
            .where(ToolUsage.chat_id == chat_id)
            .order_by(ToolUsage.created_at)
        )
        return list(result.scalars().all())


# Global instance
tool_service = ToolService()
