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
from sqlalchemy import select, delete
from datetime import datetime, timezone
import logging
import asyncio

from app.models.filter_chain import FilterChain as FilterChainModel
from app.filters.executor import ChainExecutor, ExecutionResult

logger = logging.getLogger(__name__)


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
            result = await db.execute(select(FilterChainModel))
            chains = result.scalars().all()
            
            self._chains.clear()
            self._name_to_id.clear()
            
            for chain in chains:
                chain_dict = chain.to_dict()
                self._chains[chain.id] = chain_dict
                self._name_to_id[chain.name] = chain.id
            
            self._rebuild_sorted_list()
            
            logger.info(f"Loaded {len(self._chains)} filter chains into memory")
            return len(self._chains)
    
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
            result = await db.execute(
                select(FilterChainModel).where(FilterChainModel.id == chain_id)
            )
            chain = result.scalar_one_or_none()
            
            if chain:
                chain_dict = chain.to_dict()
                
                # Remove old name mapping if name changed
                old_chain = self._chains.get(chain_id)
                if old_chain and old_chain.get("name") != chain.name:
                    old_name = old_chain.get("name")
                    if old_name in self._name_to_id:
                        del self._name_to_id[old_name]
                
                self._chains[chain_id] = chain_dict
                self._name_to_id[chain.name] = chain_id
                self._rebuild_sorted_list()
                
                logger.info(f"Reloaded chain: {chain.name}")
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
        async with self._lock:
            # Handle name change
            if old_name != chain.name and old_name in self._name_to_id:
                del self._name_to_id[old_name]
            
            self._chains[chain.id] = chain_dict
            self._name_to_id[chain.name] = chain.id
            self._rebuild_sorted_list()
        
        logger.info(f"Updated filter chain: {chain.name}")
        return chain_dict
    
    async def delete_chain(self, db: AsyncSession, chain_id: str) -> bool:
        """Delete a chain."""
        
        result = await db.execute(
            select(FilterChainModel).where(FilterChainModel.id == chain_id)
        )
        chain = result.scalar_one_or_none()
        
        if not chain:
            return False
        
        name = chain.name
        
        await db.execute(
            delete(FilterChainModel).where(FilterChainModel.id == chain_id)
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
