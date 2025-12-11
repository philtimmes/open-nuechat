"""
Bidirectional Streaming Filter System

Provides overridable filters for processing data flowing to and from the LLM.
Supports streaming in both directions with async generators.

Filter naming convention:
- OverrideToLLM: Filters that process user input BEFORE it goes to the LLM
- OverrideFromLLM: Filters that process LLM output BEFORE it goes to the user
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Any,
)
import asyncio
import logging
from datetime import datetime, timezone, timezone

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """
    Priority levels for filter execution order.
    
    Filters execute from HIGHEST (0) to LEAST (100).
    Lower numbers run first.
    """
    HIGHEST = 0
    HIGH = 25
    MEDIUM = 50
    LOW = 75
    LEAST = 100


@dataclass
class FilterContext:
    """Context passed through the filter chain."""
    user_id: str
    chat_id: str
    message_id: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Accumulated state for streaming
    accumulated_content: str = ""
    chunk_index: int = 0
    
    # Control flags
    should_stop: bool = False
    skip_remaining_filters: bool = False
    
    def clone(self) -> "FilterContext":
        """Create a copy of the context."""
        return FilterContext(
            user_id=self.user_id,
            chat_id=self.chat_id,
            message_id=self.message_id,
            model=self.model,
            metadata=self.metadata.copy(),
            timestamp=self.timestamp,
            accumulated_content=self.accumulated_content,
            chunk_index=self.chunk_index,
            should_stop=self.should_stop,
            skip_remaining_filters=self.skip_remaining_filters,
        )


@dataclass
class FilterResult:
    """Result from a filter operation."""
    content: str
    modified: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def passthrough(cls, content: str) -> "FilterResult":
        """Create a passthrough result (no modification)."""
        return cls(content=content, modified=False)
    
    @classmethod
    def modify(cls, content: str, **metadata) -> "FilterResult":
        """Create a modified result."""
        return cls(content=content, modified=True, metadata=metadata)
    
    @classmethod
    def block(cls, reason: str) -> "FilterResult":
        """Create a blocked result."""
        return cls(content="", blocked=True, block_reason=reason)


@dataclass
class StreamChunk:
    """A chunk of streaming data."""
    content: str
    chunk_type: str = "text"  # text, tool_call, tool_result, metadata
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOverride(ABC):
    """
    Abstract base class for all override filters.
    
    Filters can process both complete messages and streaming chunks.
    """
    
    def __init__(
        self,
        name: str,
        priority: Priority = Priority.MEDIUM,
        enabled: bool = True,
    ):
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self._config: Dict[str, Any] = {}
    
    def configure(self, **kwargs) -> "BaseOverride":
        """Configure the filter with custom settings."""
        self._config.update(kwargs)
        return self
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    @abstractmethod
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        """
        Process a complete message.
        
        Args:
            content: The message content to process
            context: Filter context with metadata
            
        Returns:
            FilterResult with processed content
        """
        pass
    
    async def process_stream_chunk(
        self, chunk: StreamChunk, context: FilterContext
    ) -> StreamChunk:
        """
        Process a single streaming chunk.
        
        Default implementation passes through unchanged.
        Override for chunk-level processing.
        """
        return chunk
    
    async def on_stream_start(self, context: FilterContext) -> None:
        """Called when a stream begins. Override for initialization."""
        pass
    
    async def on_stream_end(self, context: FilterContext) -> Optional[StreamChunk]:
        """
        Called when a stream ends.
        Can return a final chunk to append (e.g., for buffered content).
        """
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, priority={self.priority.name})"


class OverrideToLLM(BaseOverride):
    """
    Base class for filters that process user input BEFORE it goes to the LLM.
    
    Use cases:
    - Input validation and sanitization
    - Prompt injection detection
    - PII redaction from user messages
    - Rate limiting
    - Content moderation
    - Context enhancement
    
    Example:
        class MyInputFilter(OverrideToLLM):
            def __init__(self):
                super().__init__(name="my_input_filter", priority=Priority.HIGH)
            
            async def process(self, content: str, context: FilterContext) -> FilterResult:
                # Modify or block the content
                return FilterResult.passthrough(content)
    """
    pass


class OverrideFromLLM(BaseOverride):
    """
    Base class for filters that process LLM output BEFORE it goes to the user.
    
    Use cases:
    - Response formatting
    - PII redaction from responses
    - Adding disclaimers
    - Token counting
    - Content filtering
    - Branding/signatures
    
    Example:
        class MyOutputFilter(OverrideFromLLM):
            def __init__(self):
                super().__init__(name="my_output_filter", priority=Priority.LOW)
            
            async def process(self, content: str, context: FilterContext) -> FilterResult:
                return FilterResult.modify(content + "\\n\\n— AI Assistant")
    """
    pass


class FilterChain:
    """
    Manages a chain of filters for processing messages and streams.
    
    Filters are executed in priority order (HIGHEST first, LEAST last).
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._filters: List[BaseOverride] = []
        self._sorted = False
    
    def add(self, filter_instance: BaseOverride) -> "FilterChain":
        """Add a filter to the chain."""
        self._filters.append(filter_instance)
        self._sorted = False
        return self
    
    def remove(self, name: str) -> bool:
        """Remove a filter by name."""
        initial_len = len(self._filters)
        self._filters = [f for f in self._filters if f.name != name]
        self._sorted = False
        return len(self._filters) < initial_len
    
    def get(self, name: str) -> Optional[BaseOverride]:
        """Get a filter by name."""
        for f in self._filters:
            if f.name == name:
                return f
        return None
    
    def enable(self, name: str) -> bool:
        """Enable a filter by name."""
        f = self.get(name)
        if f:
            f.enabled = True
            return True
        return False
    
    def disable(self, name: str) -> bool:
        """Disable a filter by name."""
        f = self.get(name)
        if f:
            f.enabled = False
            return True
        return False
    
    def clear(self) -> None:
        """Remove all filters from the chain."""
        self._filters = []
        self._sorted = False
    
    def _get_sorted_filters(self) -> List[BaseOverride]:
        """Get filters sorted by priority (HIGHEST first)."""
        if not self._sorted:
            self._filters.sort(key=lambda f: f.priority.value)
            self._sorted = True
        
        return [f for f in self._filters if f.enabled]
    
    async def process(self, content: str, context: FilterContext) -> FilterResult:
        """
        Process content through all filters in priority order.
        
        Args:
            content: The content to process
            context: Filter context
            
        Returns:
            Final FilterResult after all filters
        """
        filters = self._get_sorted_filters()
        current_content = content
        combined_metadata: Dict[str, Any] = {}
        any_modified = False
        
        for f in filters:
            if context.skip_remaining_filters:
                break
                
            try:
                result = await f.process(current_content, context)
                
                if result.blocked:
                    logger.info(f"Content blocked by filter '{f.name}': {result.block_reason}")
                    return result
                
                if result.modified:
                    any_modified = True
                    current_content = result.content
                    combined_metadata.update(result.metadata)
                    combined_metadata[f"filter_{f.name}_applied"] = True
                    logger.debug(f"Content modified by filter '{f.name}'")
                    
            except Exception as e:
                logger.error(f"Error in filter '{f.name}': {e}")
                continue
        
        return FilterResult(
            content=current_content,
            modified=any_modified,
            metadata=combined_metadata,
        )
    
    async def process_stream(
        self,
        stream: AsyncGenerator[StreamChunk, None],
        context: FilterContext,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a stream through all filters in priority order.
        
        Args:
            stream: Async generator of StreamChunks
            context: Filter context (will be updated with accumulated state)
            
        Yields:
            Processed StreamChunks
        """
        filters = self._get_sorted_filters()
        
        # Notify filters of stream start (in priority order)
        for f in filters:
            try:
                await f.on_stream_start(context)
            except Exception as e:
                logger.error(f"Error in on_stream_start for '{f.name}': {e}")
        
        # Process each chunk through the filter chain
        async for chunk in stream:
            if context.should_stop:
                break
            
            current_chunk = chunk
            context.chunk_index += 1
            
            # Apply each filter in priority order
            for f in filters:
                if context.skip_remaining_filters:
                    break
                    
                try:
                    current_chunk = await f.process_stream_chunk(current_chunk, context)
                except Exception as e:
                    logger.error(f"Error processing chunk in '{f.name}': {e}")
                    continue
            
            # Update accumulated content
            if current_chunk.chunk_type == "text":
                context.accumulated_content += current_chunk.content
            
            yield current_chunk
        
        # Notify filters of stream end and collect any final chunks
        for f in filters:
            try:
                final_chunk = await f.on_stream_end(context)
                if final_chunk:
                    yield final_chunk
            except Exception as e:
                logger.error(f"Error in on_stream_end for '{f.name}': {e}")
    
    def list_filters(self) -> List[Dict[str, Any]]:
        """List all registered filters with their status."""
        self._get_sorted_filters()  # Ensure sorted
        
        return [
            {
                "name": f.name,
                "priority": f.priority.name,
                "priority_value": f.priority.value,
                "enabled": f.enabled,
                "type": f.__class__.__name__,
            }
            for f in self._filters
        ]


class ChatFilterManager:
    """
    Manages filter chains for a chat session.
    
    Provides separate chains for:
    - ToLLM: user input -> LLM (processed in priority order)
    - FromLLM: LLM output -> user (processed in priority order)
    """
    
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.to_llm_chain = FilterChain(name=f"to_llm_{chat_id}")
        self.from_llm_chain = FilterChain(name=f"from_llm_{chat_id}")
    
    def add_to_llm(self, filter_instance: OverrideToLLM) -> "ChatFilterManager":
        """Add a filter to the ToLLM chain (user input -> LLM)."""
        self.to_llm_chain.add(filter_instance)
        return self
    
    def add_from_llm(self, filter_instance: OverrideFromLLM) -> "ChatFilterManager":
        """Add a filter to the FromLLM chain (LLM output -> user)."""
        self.from_llm_chain.add(filter_instance)
        return self
    
    async def process_to_llm(
        self, content: str, context: FilterContext
    ) -> FilterResult:
        """Process user input through ToLLM filters before sending to LLM."""
        return await self.to_llm_chain.process(content, context)
    
    async def process_from_llm(
        self, content: str, context: FilterContext
    ) -> FilterResult:
        """Process LLM output through FromLLM filters before sending to user."""
        return await self.from_llm_chain.process(content, context)
    
    async def process_from_llm_stream(
        self,
        stream: AsyncGenerator[StreamChunk, None],
        context: FilterContext,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Process streaming LLM output through FromLLM filters."""
        async for chunk in self.from_llm_chain.process_stream(stream, context):
            yield chunk
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all filter chains."""
        return {
            "chat_id": self.chat_id,
            "to_llm_filters": self.to_llm_chain.list_filters(),
            "from_llm_filters": self.from_llm_chain.list_filters(),
        }


class GlobalFilterRegistry:
    """
    Global registry for default filter configurations.
    
    Provides default filters that are applied to all chats,
    plus methods to create per-chat filter managers.
    """
    
    _instance: Optional["GlobalFilterRegistry"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._default_to_llm: List[Callable[[], OverrideToLLM]] = []
        self._default_from_llm: List[Callable[[], OverrideFromLLM]] = []
        self._chat_managers: Dict[str, ChatFilterManager] = {}
        self._initialized = True
    
    def register_default_to_llm(self, factory: Callable[[], OverrideToLLM]) -> None:
        """Register a factory for default ToLLM filters."""
        self._default_to_llm.append(factory)
    
    def register_default_from_llm(self, factory: Callable[[], OverrideFromLLM]) -> None:
        """Register a factory for default FromLLM filters."""
        self._default_from_llm.append(factory)
    
    def get_chat_manager(self, chat_id: str) -> ChatFilterManager:
        """
        Get or create a filter manager for a chat.
        
        New managers are initialized with default filters.
        """
        if chat_id not in self._chat_managers:
            manager = ChatFilterManager(chat_id)
            
            # Add default filters
            for factory in self._default_to_llm:
                try:
                    manager.add_to_llm(factory())
                except Exception as e:
                    logger.error(f"Error creating default ToLLM filter: {e}")
            
            for factory in self._default_from_llm:
                try:
                    manager.add_from_llm(factory())
                except Exception as e:
                    logger.error(f"Error creating default FromLLM filter: {e}")
            
            self._chat_managers[chat_id] = manager
        
        return self._chat_managers[chat_id]
    
    def remove_chat_manager(self, chat_id: str) -> bool:
        """Remove a chat's filter manager (e.g., on chat deletion)."""
        if chat_id in self._chat_managers:
            del self._chat_managers[chat_id]
            return True
        return False
    
    def reset(self) -> None:
        """Reset the registry (clears all defaults and chat managers)."""
        self._default_to_llm = []
        self._default_from_llm = []
        self._chat_managers = {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            "default_to_llm_count": len(self._default_to_llm),
            "default_from_llm_count": len(self._default_from_llm),
            "active_chat_managers": len(self._chat_managers),
            "chat_ids": list(self._chat_managers.keys()),
        }


# Global registry instance
_global_registry = GlobalFilterRegistry()


def get_filter_registry() -> GlobalFilterRegistry:
    """Get the global filter registry."""
    return _global_registry


def get_chat_filters(chat_id: str) -> ChatFilterManager:
    """Convenience function to get filter manager for a chat."""
    return _global_registry.get_chat_manager(chat_id)
