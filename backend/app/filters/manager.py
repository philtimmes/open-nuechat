"""
Filter Chain Manager

Manages filter chains:
- Loads from database to memory on startup
- Hot-reload on changes
- Provides chain lookup for executor
- Handles chain CRUD operations
"""

from typing import Any, Callable, Coroutine, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, text
from datetime import datetime, timezone
import logging
import asyncio
import json

from app.models.filter_chain import FilterChain as FilterChainModel
from app.filters.executor import ChainExecutor, ExecutionResult

logger = logging.getLogger(__name__)


# Flag to track if new columns exist in database
_new_columns_exist: Optional[bool] = None


def reset_column_cache():
    """Reset the column existence cache (call after migrations)."""
    global _new_columns_exist
    _new_columns_exist = None


async def _check_new_columns_exist(db: AsyncSession) -> bool:
    """Check if the new export_as_tool columns exist in the database."""
    global _new_columns_exist
    if _new_columns_exist is not None:
        return _new_columns_exist
    
    try:
        result = await db.execute(text("PRAGMA table_info(filter_chains)"))
        columns = [row[1] for row in result.fetchall()]
        _new_columns_exist = "export_as_tool" in columns
        logger.info(f"Column check: export_as_tool exists = {_new_columns_exist}, columns = {columns}")
        return _new_columns_exist
    except Exception as e:
        logger.error(f"Column check failed: {e}")
        _new_columns_exist = False
        return False


def _make_chain_dict_from_row(row, has_new_columns: bool) -> dict:
    """Create a chain dict from a raw SQL row."""
    if has_new_columns:
        # Full schema with new columns
        raw_definition = row[25]  # definition is at index 25
        logger.debug(f"Raw definition (new schema): type={type(raw_definition)}, value={repr(raw_definition)[:200]}")
        
        if isinstance(raw_definition, str):
            try:
                definition = json.loads(raw_definition)
            except Exception as e:
                logger.error(f"Failed to parse definition JSON: {e}")
                definition = {"steps": [], "_parse_error": str(e)}
        elif isinstance(raw_definition, dict):
            definition = raw_definition
        else:
            logger.warning(f"Definition is not str or dict: {type(raw_definition)}")
            definition = {"steps": []}
            
        tool_variables = row[24]  # tool_variables is at index 24
        if isinstance(tool_variables, str):
            try:
                tool_variables = json.loads(tool_variables)
            except:
                tool_variables = []
        elif not isinstance(tool_variables, list):
            tool_variables = []
        
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "enabled": bool(row[3]),
            "priority": row[4],
            "retain_history": bool(row[5]),
            "bidirectional": bool(row[6]),
            "outbound_chain_id": row[7],
            "max_iterations": row[8],
            "debug": bool(row[9]),
            "skip_if_rag_hit": bool(row[10]),
            "export_as_tool": bool(row[11]),
            "tool_name": row[12],
            "tool_label": row[13],
            "advertise_to_llm": bool(row[14]),
            "advertise_text": row[15],
            "trigger_pattern": row[16],
            "trigger_source": row[17] or "both",
            "erase_from_display": bool(row[18]) if row[18] is not None else True,
            "keep_in_history": bool(row[19]) if row[19] is not None else True,
            "button_enabled": bool(row[20]),
            "button_icon": row[21],
            "button_location": row[22] or "response",
            "button_trigger_mode": row[23] or "immediate",
            "tool_variables": tool_variables,
            "definition": definition,
            "created_at": row[26],
            "updated_at": row[27],
            "created_by": row[28],
        }
    else:
        # Old schema without new columns
        # Column order: id(0), name(1), description(2), enabled(3), priority(4),
        # retain_history(5), bidirectional(6), outbound_chain_id(7), max_iterations(8),
        # definition(9), created_at(10), updated_at(11), created_by(12), debug(13), skip_if_rag_hit(14)
        raw_definition = row[9]  # definition is at index 9
        logger.debug(f"Raw definition (old schema): type={type(raw_definition)}, value={repr(raw_definition)[:200]}")
        
        if isinstance(raw_definition, str):
            try:
                definition = json.loads(raw_definition)
            except Exception as e:
                logger.error(f"Failed to parse definition JSON: {e}")
                definition = {"steps": [], "_parse_error": str(e)}
        elif isinstance(raw_definition, dict):
            definition = raw_definition
        else:
            logger.warning(f"Definition is not str or dict: {type(raw_definition)}")
            definition = {"steps": []}
        
        return {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "enabled": bool(row[3]),
            "priority": row[4],
            "retain_history": bool(row[5]),
            "bidirectional": bool(row[6]),
            "outbound_chain_id": row[7],
            "max_iterations": row[8],
            "debug": bool(row[13]) if row[13] is not None else False,
            "skip_if_rag_hit": bool(row[14]) if row[14] is not None else True,
            "definition": definition,
            "created_at": row[10],
            "updated_at": row[11],
            "created_by": row[12],
            # Default values for new fields
            "export_as_tool": False,
            "tool_name": None,
            "tool_label": None,
            "advertise_to_llm": False,
            "advertise_text": None,
            "trigger_pattern": None,
            "trigger_source": "both",
            "erase_from_display": True,
            "keep_in_history": True,
            "button_enabled": False,
            "button_icon": None,
            "button_location": "response",
            "button_trigger_mode": "immediate",
            "tool_variables": [],
        }


async def _safe_query_all_chains(db: AsyncSession) -> List[dict]:
    """Query all filter chains safely, handling missing columns."""
    has_new = await _check_new_columns_exist(db)
    logger.info(f"_safe_query_all_chains: has_new_columns = {has_new}")
    
    if has_new:
        sql = """
            SELECT id, name, description, enabled, priority, retain_history,
                   bidirectional, outbound_chain_id, max_iterations, debug,
                   skip_if_rag_hit, export_as_tool, tool_name, tool_label,
                   advertise_to_llm, advertise_text, trigger_pattern, trigger_source,
                   erase_from_display, keep_in_history, button_enabled, button_icon,
                   button_location, button_trigger_mode, tool_variables, definition,
                   created_at, updated_at, created_by
            FROM filter_chains
        """
    else:
        # Query columns in the order they actually exist in the old schema
        # Based on PRAGMA table_info: id, name, description, enabled, priority,
        # retain_history, bidirectional, outbound_chain_id, max_iterations,
        # definition, created_at, updated_at, created_by, debug, skip_if_rag_hit
        sql = """
            SELECT id, name, description, enabled, priority, retain_history,
                   bidirectional, outbound_chain_id, max_iterations, definition,
                   created_at, updated_at, created_by, debug, skip_if_rag_hit
            FROM filter_chains
        """
    
    logger.debug(f"Executing SQL: {sql.strip()}")
    result = await db.execute(text(sql))
    rows = result.fetchall()
    logger.info(f"Query returned {len(rows)} rows")
    
    chains = []
    for i, row in enumerate(rows):
        try:
            chain_dict = _make_chain_dict_from_row(row, has_new)
            chains.append(chain_dict)
        except Exception as e:
            logger.error(f"Failed to parse row {i}: {e}, row data: {row}")
    
    return chains


async def _safe_query_chain_by_id(db: AsyncSession, chain_id: str) -> Optional[dict]:
    """Query a single filter chain by ID safely, handling missing columns."""
    has_new = await _check_new_columns_exist(db)
    
    if has_new:
        sql = """
            SELECT id, name, description, enabled, priority, retain_history,
                   bidirectional, outbound_chain_id, max_iterations, debug,
                   skip_if_rag_hit, export_as_tool, tool_name, tool_label,
                   advertise_to_llm, advertise_text, trigger_pattern, trigger_source,
                   erase_from_display, keep_in_history, button_enabled, button_icon,
                   button_location, button_trigger_mode, tool_variables, definition,
                   created_at, updated_at, created_by
            FROM filter_chains WHERE id = :chain_id
        """
    else:
        # Query columns in the order they actually exist in the old schema
        sql = """
            SELECT id, name, description, enabled, priority, retain_history,
                   bidirectional, outbound_chain_id, max_iterations, definition,
                   created_at, updated_at, created_by, debug, skip_if_rag_hit
            FROM filter_chains WHERE id = :chain_id
        """
    
    result = await db.execute(text(sql), {"chain_id": chain_id})
    row = result.fetchone()
    if row:
        return _make_chain_dict_from_row(row, has_new)
    return None


# Type aliases
LLMFunc = Callable[[str, Optional[str]], Coroutine[Any, Any, str]]
ToolFunc = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]


class ChainManager:
    """
    Manages filter chains with in-memory caching.
    
    Usage:
        manager = ChainManager()
        await manager.load_from_db(db_session)
        
        result = await manager.execute_inbound(
            query="User question",
            user_id="...",
            chat_id="...",
            llm_func=my_llm,
            tool_func=my_tool,
        )
    """
    
    _instance: Optional["ChainManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # In-memory cache: {chain_id: chain_dict}
        self._chains: Dict[str, dict] = {}
        
        # Name -> ID mapping for quick lookup
        self._name_to_id: Dict[str, str] = {}
        
        # Sorted list of enabled chain IDs by priority
        self._sorted_chain_ids: List[str] = []
        
        # Executor instance
        self._executor: Optional[ChainExecutor] = None
        
        # Lock for thread-safe updates
        self._lock = asyncio.Lock()
        
        self._initialized = True
    
    def _get_executor(
        self,
        llm_func: Optional[LLMFunc] = None,
        tool_func: Optional[ToolFunc] = None,
    ) -> ChainExecutor:
        """Get or create executor with provided functions."""
        if not self._executor:
            self._executor = ChainExecutor(
                chain_loader=self.get_chain_by_name
            )
        
        if llm_func:
            self._executor.set_llm_func(llm_func)
        if tool_func:
            self._executor.set_tool_func(tool_func)
        
        return self._executor
    
    # =========================================================================
    # CACHE OPERATIONS
    # =========================================================================
    
    async def load_from_db(self, db: AsyncSession) -> int:
        """
        Load all chains from database into memory.
        
        Returns number of chains loaded.
        """
        async with self._lock:
            try:
                # First check if table exists and has data
                try:
                    count_result = await db.execute(text("SELECT COUNT(*) FROM filter_chains"))
                    row_count = count_result.scalar()
                    logger.info(f"filter_chains table has {row_count} rows")
                except Exception as ce:
                    logger.error(f"Could not count filter_chains: {ce}")
                    row_count = 0
                
                chains = await _safe_query_all_chains(db)
                logger.info(f"_safe_query_all_chains returned {len(chains)} chains")
                
                self._chains.clear()
                self._name_to_id.clear()
                
                for chain_dict in chains:
                    self._chains[chain_dict["id"]] = chain_dict
                    self._name_to_id[chain_dict["name"]] = chain_dict["id"]
                    logger.debug(f"Loaded chain: {chain_dict['name']} (id={chain_dict['id']})")
                
                self._rebuild_sorted_list()
                
                logger.info(f"Loaded {len(self._chains)} filter chains into memory")
                return len(self._chains)
            except Exception as e:
                logger.error(f"Failed to load filter chains: {e}", exc_info=True)
                return 0
    
    def _rebuild_sorted_list(self):
        """Rebuild the sorted list of enabled chain IDs."""
        enabled = [
            (cid, c) for cid, c in self._chains.items()
            if c.get("enabled", True)
        ]
        enabled.sort(key=lambda x: x[1].get("priority", 100))
        self._sorted_chain_ids = [cid for cid, _ in enabled]
    
    async def reload_chain(self, db: AsyncSession, chain_id: str) -> bool:
        """Reload a specific chain from database."""
        async with self._lock:
            chain_dict = await _safe_query_chain_by_id(db, chain_id)
            
            if chain_dict:
                # Remove old name mapping if name changed
                old_chain = self._chains.get(chain_id)
                if old_chain and old_chain.get("name") != chain_dict["name"]:
                    old_name = old_chain.get("name")
                    if old_name in self._name_to_id:
                        del self._name_to_id[old_name]
                
                self._chains[chain_id] = chain_dict
                self._name_to_id[chain_dict["name"]] = chain_id
                self._rebuild_sorted_list()
                
                logger.info(f"Reloaded chain: {chain_dict['name']}")
                return True
            else:
                # Chain was deleted
                if chain_id in self._chains:
                    old_name = self._chains[chain_id].get("name")
                    del self._chains[chain_id]
                    if old_name in self._name_to_id:
                        del self._name_to_id[old_name]
                    self._rebuild_sorted_list()
                return False
    
    # =========================================================================
    # LOOKUP
    # =========================================================================
    
    def get_chain(self, chain_id: str) -> Optional[dict]:
        """Get a chain by ID."""
        return self._chains.get(chain_id)
    
    def get_chain_by_name(self, name: str) -> Optional[dict]:
        """Get a chain by name."""
        chain_id = self._name_to_id.get(name)
        if chain_id:
            return self._chains.get(chain_id)
        return None
    
    def get_all_chains(self) -> List[dict]:
        """Get all chains (for admin UI)."""
        return list(self._chains.values())
    
    def get_enabled_chains(self) -> List[dict]:
        """Get enabled chains in priority order."""
        return [self._chains[cid] for cid in self._sorted_chain_ids]
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    async def create_chain(
        self,
        db: AsyncSession,
        name: str,
        description: str = "",
        definition: dict = None,
        enabled: bool = True,
        priority: int = 100,
        retain_history: bool = True,
        bidirectional: bool = False,
        outbound_chain_id: str = None,
        max_iterations: int = 10,
        debug: bool = False,
        created_by: str = None,
    ) -> dict:
        """Create a new chain."""
        
        chain = FilterChainModel(
            name=name,
            description=description,
            definition=definition or {"steps": []},
            enabled=enabled,
            priority=priority,
            retain_history=retain_history,
            bidirectional=bidirectional,
            outbound_chain_id=outbound_chain_id,
            max_iterations=max_iterations,
            debug=debug,
            created_by=created_by,
        )
        
        db.add(chain)
        await db.commit()
        await db.refresh(chain)
        
        # Update cache
        chain_dict = chain.to_dict()
        async with self._lock:
            self._chains[chain.id] = chain_dict
            self._name_to_id[chain.name] = chain.id
            self._rebuild_sorted_list()
        
        logger.info(f"Created filter chain: {name}")
        return chain_dict
    
    async def update_chain(
        self,
        db: AsyncSession,
        chain_id: str,
        **updates,
    ) -> Optional[dict]:
        """Update a chain."""
        
        # SAFETY: Never allow updating definition to empty
        if "definition" in updates:
            def_value = updates["definition"]
            if def_value is None or def_value == {} or def_value == {"steps": []}:
                # Check if this would wipe existing data
                existing = await _safe_query_chain_by_id(db, chain_id)
                if existing and existing.get("definition", {}).get("steps"):
                    logger.warning(f"Blocked attempt to wipe definition for chain {chain_id}")
                    del updates["definition"]  # Remove the dangerous update
        
        # First check if new columns exist - if so, use ORM, otherwise use raw SQL
        has_new = await _check_new_columns_exist(db)
        
        if has_new:
            # Use ORM when schema is up to date
            result = await db.execute(
                select(FilterChainModel).where(FilterChainModel.id == chain_id)
            )
            chain = result.scalar_one_or_none()
            
            if not chain:
                return None
            
            # Track name change
            old_name = chain.name
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(chain, key):
                    setattr(chain, key, value)
            
            chain.updated_at = datetime.now(timezone.utc)
            
            await db.commit()
            await db.refresh(chain)
            
            # Update cache
            chain_dict = chain.to_dict()
        else:
            # Use raw SQL when schema doesn't have new columns yet
            # First verify chain exists
            existing = await _safe_query_chain_by_id(db, chain_id)
            if not existing:
                return None
            
            old_name = existing["name"]
            
            # Build UPDATE statement for old schema columns only
            allowed_cols = {"name", "description", "enabled", "priority", "retain_history",
                          "bidirectional", "outbound_chain_id", "max_iterations", "debug",
                          "skip_if_rag_hit", "definition"}
            set_parts = []
            params = {"chain_id": chain_id}
            
            for key, value in updates.items():
                if key in allowed_cols:
                    if key == "definition" and isinstance(value, dict):
                        value = json.dumps(value)
                    set_parts.append(f"{key} = :{key}")
                    params[key] = value
            
            if set_parts:
                set_parts.append("updated_at = :updated_at")
                params["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                sql = f"UPDATE filter_chains SET {', '.join(set_parts)} WHERE id = :chain_id"
                await db.execute(text(sql), params)
                await db.commit()
            
            # Refresh from DB
            chain_dict = await _safe_query_chain_by_id(db, chain_id)
            if not chain_dict:
                return None
        
        async with self._lock:
            # Handle name change
            if old_name != chain_dict["name"] and old_name in self._name_to_id:
                del self._name_to_id[old_name]
            
            self._chains[chain_dict["id"]] = chain_dict
            self._name_to_id[chain_dict["name"]] = chain_dict["id"]
            self._rebuild_sorted_list()
        
        logger.info(f"Updated filter chain: {chain_dict['name']}")
        return chain_dict
    
    async def delete_chain(self, db: AsyncSession, chain_id: str) -> bool:
        """Delete a chain."""
        
        # Get chain info for cache cleanup
        chain_dict = await _safe_query_chain_by_id(db, chain_id)
        
        if not chain_dict:
            return False
        
        name = chain_dict["name"]
        
        # Delete using raw SQL (works regardless of schema)
        await db.execute(
            text("DELETE FROM filter_chains WHERE id = :chain_id"),
            {"chain_id": chain_id}
        )
        await db.commit()
        
        # Update cache
        async with self._lock:
            if chain_id in self._chains:
                del self._chains[chain_id]
            if name in self._name_to_id:
                del self._name_to_id[name]
            self._rebuild_sorted_list()
        
        logger.info(f"Deleted filter chain: {name}")
        return True
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    async def execute_inbound(
        self,
        query: str,
        user_id: str,
        chat_id: str,
        llm_func: LLMFunc,
        tool_func: ToolFunc,
    ) -> ExecutionResult:
        """
        Execute all enabled inbound chains on a query.
        
        Chains are executed in priority order.
        Each chain's output becomes the next chain's input.
        """
        
        executor = self._get_executor(llm_func, tool_func)
        
        current_query = query
        final_result: Optional[ExecutionResult] = None
        
        for chain_id in self._sorted_chain_ids:
            chain = self._chains.get(chain_id)
            if not chain:
                continue
            
            try:
                result = await executor.execute(
                    chain,
                    current_query,
                    user_id,
                    chat_id,
                )
                
                final_result = result
                current_query = result.content
                
                # Check if chain signaled to stop
                if not result.proceed_to_llm:
                    break
                    
            except Exception as e:
                logger.error(f"Error executing chain '{chain.get('name')}': {e}")
                continue
        
        # If no chains ran, return passthrough result
        if final_result is None:
            from app.filters.executor import ExecutionContext
            final_result = ExecutionResult(
                content=query,
                context=ExecutionContext(user_id=user_id, chat_id=chat_id, original_query=query),
                proceed_to_llm=True,
            )
        
        return final_result
    
    async def execute_outbound(
        self,
        response: str,
        user_id: str,
        chat_id: str,
        inbound_chain_id: str,
        llm_func: LLMFunc,
        tool_func: ToolFunc,
    ) -> ExecutionResult:
        """
        Execute outbound chain on LLM response (for bidirectional chains).
        """
        
        inbound_chain = self._chains.get(inbound_chain_id)
        if not inbound_chain:
            from app.filters.executor import ExecutionContext
            return ExecutionResult(
                content=response,
                context=ExecutionContext(user_id=user_id, chat_id=chat_id, original_query=response),
                proceed_to_llm=True,
            )
        
        # Get outbound chain
        outbound_id = inbound_chain.get("outbound_chain_id")
        if not outbound_id:
            from app.filters.executor import ExecutionContext
            return ExecutionResult(
                content=response,
                context=ExecutionContext(user_id=user_id, chat_id=chat_id, original_query=response),
                proceed_to_llm=True,
            )
        
        outbound_chain = self._chains.get(outbound_id)
        if not outbound_chain:
            from app.filters.executor import ExecutionContext
            return ExecutionResult(
                content=response,
                context=ExecutionContext(user_id=user_id, chat_id=chat_id, original_query=response),
                proceed_to_llm=True,
            )
        
        executor = self._get_executor(llm_func, tool_func)
        
        return await executor.execute(
            outbound_chain,
            response,
            user_id,
            chat_id,
        )
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_chain(self, definition: dict) -> List[str]:
        """
        Validate a chain definition.
        
        Returns list of errors (empty if valid).
        """
        errors = []
        
        steps = definition.get("steps", [])
        
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
            return errors
        
        valid_types = {
            "pass", "stop", "block", "to_llm", "from_llm", "query",
            "to_tool", "from_tool", "context_insert", "go_to_llm",
            "filter_complete", "set_var", "set_array", "compare", "call_chain",
            "modify", "log", "branch",
        }
        
        def validate_steps(steps: list, path: str = ""):
            for i, step in enumerate(steps):
                step_path = f"{path}[{i}]"
                
                if not isinstance(step, dict):
                    errors.append(f"{step_path}: step must be an object")
                    continue
                
                step_type = step.get("type")
                if step_type and step_type not in valid_types:
                    errors.append(f"{step_path}: unknown type '{step_type}'")
                
                # Validate conditional branches
                conditional = step.get("conditional")
                if conditional and conditional.get("enabled"):
                    on_true = conditional.get("on_true", [])
                    on_false = conditional.get("on_false", [])
                    
                    if on_true:
                        validate_steps(on_true, f"{step_path}.conditional.on_true")
                    if on_false:
                        validate_steps(on_false, f"{step_path}.conditional.on_false")
                
                # Validate call_chain references
                if step_type == "call_chain":
                    chain_name = step.get("config", {}).get("chain_name")
                    if chain_name and chain_name not in self._name_to_id:
                        errors.append(f"{step_path}: referenced chain '{chain_name}' not found")
        
        validate_steps(steps)
        return errors


# Global instance
_manager: Optional[ChainManager] = None


def get_chain_manager() -> ChainManager:
    """Get the global chain manager instance."""
    global _manager
    if _manager is None:
        _manager = ChainManager()
    return _manager
