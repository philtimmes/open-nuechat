"""
Filter Chain System

A simple, ordered filter chain. Parts execute in the order they're added.

Usage:
    chain = FilterChain()
    
    # Add parts in order of execution
    chain.add(ToLLMPart("sanitizer", sanitize_fn))
    chain.add(ToToolPart("tool_prep", prep_fn))
    chain.add(FromToolPart("tool_cleanup", cleanup_fn))
    chain.add(FromLLMPart("formatter", format_fn))
    
    # Process by type
    result = await chain.process_to_llm(content, ctx)
    result = await chain.process_to_tool(tool_name, params, ctx)
    result = await chain.process_from_tool(tool_name, result, ctx)
    result = await chain.process_from_llm(content, ctx)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PartType(Enum):
    """Types of chain parts."""
    TO_LLM = "to_llm"
    FROM_LLM = "from_llm"
    TO_TOOL = "to_tool"
    FROM_TOOL = "from_tool"
    LOGIC = "logic"


@dataclass
class ChainContext:
    """Context passed through the chain."""
    user_id: str
    chat_id: str
    message_id: Optional[str] = None
    model: Optional[str] = None
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    data: Dict[str, Any] = field(default_factory=dict)
    stop_chain: bool = False
    skip_llm: bool = False


@dataclass
class ChainResult:
    """Result from chain processing."""
    content: Any
    modified: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, content: Any) -> "ChainResult":
        return cls(content=content)
    
    @classmethod
    def modify(cls, content: Any, **meta) -> "ChainResult":
        return cls(content=content, modified=True, metadata=meta)
    
    @classmethod
    def block(cls, reason: str) -> "ChainResult":
        return cls(content=None, blocked=True, block_reason=reason)


# Handler function type
Handler = Callable[[Any, ChainContext], Coroutine[Any, Any, ChainResult]]


class ChainPart(ABC):
    """Base class for chain parts."""
    
    def __init__(self, name: str, part_type: PartType, enabled: bool = True):
        self.name = name
        self.part_type = part_type
        self.enabled = enabled
        self._config: Dict[str, Any] = {}
    
    def configure(self, **kwargs) -> "ChainPart":
        self._config.update(kwargs)
        return self
    
    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    @abstractmethod
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        pass


class ToLLMPart(ChainPart):
    """Processes content before sending to LLM."""
    def __init__(self, name: str, handler: Optional[Handler] = None, enabled: bool = True):
        super().__init__(name, PartType.TO_LLM, enabled)
        self._handler = handler
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if self._handler:
            return await self._handler(content, ctx)
        return ChainResult.ok(content)


class FromLLMPart(ChainPart):
    """Processes content after receiving from LLM."""
    def __init__(self, name: str, handler: Optional[Handler] = None, enabled: bool = True):
        super().__init__(name, PartType.FROM_LLM, enabled)
        self._handler = handler
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if self._handler:
            return await self._handler(content, ctx)
        return ChainResult.ok(content)


class ToToolPart(ChainPart):
    """Processes data before tool execution."""
    def __init__(self, name: str, handler: Optional[Handler] = None, enabled: bool = True):
        super().__init__(name, PartType.TO_TOOL, enabled)
        self._handler = handler
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if self._handler:
            return await self._handler(content, ctx)
        return ChainResult.ok(content)


class FromToolPart(ChainPart):
    """Processes data after tool execution."""
    def __init__(self, name: str, handler: Optional[Handler] = None, enabled: bool = True):
        super().__init__(name, PartType.FROM_TOOL, enabled)
        self._handler = handler
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if self._handler:
            return await self._handler(content, ctx)
        return ChainResult.ok(content)


class LogicPart(ChainPart):
    """Conditional logic or transformations."""
    def __init__(self, name: str, handler: Optional[Handler] = None, enabled: bool = True):
        super().__init__(name, PartType.LOGIC, enabled)
        self._handler = handler
    
    async def process(self, content: Any, ctx: ChainContext) -> ChainResult:
        if self._handler:
            return await self._handler(content, ctx)
        return ChainResult.ok(content)


class FilterChain:
    """
    Ordered list of chain parts.
    Parts execute in the order they are added.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._parts: List[ChainPart] = []
    
    def add(self, part: ChainPart) -> "FilterChain":
        """Add a part to the end of the chain."""
        self._parts.append(part)
        return self
    
    def insert(self, index: int, part: ChainPart) -> "FilterChain":
        """Insert a part at a specific index."""
        self._parts.insert(index, part)
        return self
    
    def remove(self, name: str) -> bool:
        """Remove a part by name."""
        for i, p in enumerate(self._parts):
            if p.name == name:
                self._parts.pop(i)
                return True
        return False
    
    def get(self, name: str) -> Optional[ChainPart]:
        """Get a part by name."""
        for p in self._parts:
            if p.name == name:
                return p
        return None
    
    def clear(self) -> "FilterChain":
        """Remove all parts."""
        self._parts = []
        return self
    
    def enable(self, name: str) -> bool:
        """Enable a part."""
        p = self.get(name)
        if p:
            p.enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a part."""
        p = self.get(name)
        if p:
            p.enabled = False
            return True
        return False
    
    async def _run(
        self,
        content: Any,
        ctx: ChainContext,
        part_type: Optional[PartType] = None,
    ) -> ChainResult:
        """Run parts through the chain."""
        current = content
        modified = False
        meta: Dict[str, Any] = {}
        
        for part in self._parts:
            if not part.enabled:
                continue
            if part_type and part.part_type != part_type:
                continue
            if ctx.stop_chain:
                break
            
            try:
                result = await part.process(current, ctx)
                
                if result.blocked:
                    logger.info(f"Blocked by '{part.name}': {result.block_reason}")
                    return result
                
                if result.modified:
                    modified = True
                    current = result.content
                    meta.update(result.metadata)
                    
            except Exception as e:
                logger.error(f"Error in '{part.name}': {e}")
                continue
        
        return ChainResult(content=current, modified=modified, metadata=meta)
    
    async def process_to_llm(self, content: str, ctx: ChainContext) -> ChainResult:
        """Process before sending to LLM."""
        return await self._run(content, ctx, PartType.TO_LLM)
    
    async def process_from_llm(self, content: str, ctx: ChainContext) -> ChainResult:
        """Process after receiving from LLM."""
        return await self._run(content, ctx, PartType.FROM_LLM)
    
    async def process_to_tool(
        self, tool_name: str, params: Dict[str, Any], ctx: ChainContext
    ) -> ChainResult:
        """Process before tool execution."""
        ctx.tool_name = tool_name
        ctx.tool_params = params
        return await self._run(params, ctx, PartType.TO_TOOL)
    
    async def process_from_tool(
        self, tool_name: str, result: Any, ctx: ChainContext
    ) -> ChainResult:
        """Process after tool execution."""
        ctx.tool_name = tool_name
        return await self._run(result, ctx, PartType.FROM_TOOL)
    
    async def process_logic(self, content: Any, ctx: ChainContext) -> ChainResult:
        """Process logic parts only."""
        return await self._run(content, ctx, PartType.LOGIC)
    
    async def process_all(self, content: Any, ctx: ChainContext) -> ChainResult:
        """Process all parts regardless of type."""
        return await self._run(content, ctx, None)
    
    def list_parts(self) -> List[Dict[str, Any]]:
        """List all parts."""
        return [
            {"index": i, "name": p.name, "type": p.part_type.value, "enabled": p.enabled}
            for i, p in enumerate(self._parts)
        ]
    
    def __len__(self) -> int:
        return len(self._parts)
    
    def __repr__(self) -> str:
        return f"FilterChain({self.name}, {len(self._parts)} parts)"


# =============================================================================
# DECORATOR HELPERS
# =============================================================================

def to_llm(name: str, enabled: bool = True):
    """Decorator to create a ToLLMPart."""
    def decorator(fn: Handler) -> ToLLMPart:
        return ToLLMPart(name, fn, enabled)
    return decorator


def from_llm(name: str, enabled: bool = True):
    """Decorator to create a FromLLMPart."""
    def decorator(fn: Handler) -> FromLLMPart:
        return FromLLMPart(name, fn, enabled)
    return decorator


def to_tool(name: str, enabled: bool = True):
    """Decorator to create a ToToolPart."""
    def decorator(fn: Handler) -> ToToolPart:
        return ToToolPart(name, fn, enabled)
    return decorator


def from_tool(name: str, enabled: bool = True):
    """Decorator to create a FromToolPart."""
    def decorator(fn: Handler) -> FromToolPart:
        return FromToolPart(name, fn, enabled)
    return decorator


def logic_part(name: str, enabled: bool = True):
    """Decorator to create a LogicPart."""
    def decorator(fn: Handler) -> LogicPart:
        return LogicPart(name, fn, enabled)
    return decorator


# =============================================================================
# GLOBAL CHAIN REGISTRY
# =============================================================================

class ChainRegistry:
    """Registry for managing filter chains per chat."""
    
    _instance: Optional["ChainRegistry"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._default = FilterChain("default")
            cls._instance._chains: Dict[str, FilterChain] = {}
        return cls._instance
    
    @property
    def default(self) -> FilterChain:
        """The default chain template."""
        return self._default
    
    def get(self, chat_id: str) -> FilterChain:
        """Get or create a chain for a chat."""
        if chat_id not in self._chains:
            # Create new chain with same parts as default
            chain = FilterChain(f"chat_{chat_id}")
            for part in self._default._parts:
                chain.add(part)
            self._chains[chat_id] = chain
        return self._chains[chat_id]
    
    def remove(self, chat_id: str) -> bool:
        """Remove a chat's chain."""
        if chat_id in self._chains:
            del self._chains[chat_id]
            return True
        return False
    
    def reset(self):
        """Reset registry."""
        self._default = FilterChain("default")
        self._chains = {}


# Global instance
_registry = ChainRegistry()


def get_registry() -> ChainRegistry:
    """Get the chain registry."""
    return _registry


def get_default_chain() -> FilterChain:
    """Get the default chain template."""
    return _registry.default


def get_chat_chain(chat_id: str) -> FilterChain:
    """Get or create a chain for a chat."""
    return _registry.get(chat_id)
