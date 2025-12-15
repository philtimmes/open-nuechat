"""
Open-NueChat Filter Chain System

Two systems:

1. Configurable Chains (Admin UI) - JSON-defined flows
   
   from app.filters.manager import get_chain_manager
   manager = get_chain_manager()
   result = await manager.execute_inbound(query, user_id, chat_id, llm_func, tool_func)

2. Simple Filter Chain (Code-defined)
   
   chain = FilterChain()
   chain.add(SanitizePart())
   result = await chain.process_to_llm(content, ctx)
"""

# Configurable chain system (Admin UI)
from .executor import (
    ChainExecutor,
    ExecutionContext,
    ExecutionResult,
    FlowSignal,
    CompareOp,
)

from .manager import (
    ChainManager,
    get_chain_manager,
)

# Simple chain (code-defined)
from .chain import (
    FilterChain,
    ChainContext,
    ChainResult,
    ChainPart,
    PartType,
    ToLLMPart,
    FromLLMPart,
    ToToolPart,
    FromToolPart,
    LogicPart,
    to_llm,
    from_llm,
    to_tool,
    from_tool,
    logic_part,
    ChainRegistry,
    get_registry,
    get_default_chain,
    get_chat_chain,
)

# Built-in parts for simple chain
from .parts import (
    RateLimitPart,
    SanitizePart,
    PromptGuardPart,
    PIIRedactPart,
    OutputFormatterPart,
    WordFilterPart,
    AuditLogPart,
    ToolLogPart,
    ToolSanitizePart,
    ToolResultLogPart,
    ToolResultTruncatePart,
    ConditionalSkipPart,
)

# Legacy imports for backward compatibility
from .base import (
    FilterContext,
    FilterResult,
    StreamChunk,
    Priority,
    ChatFilterManager,
    get_chat_filters,
    get_filter_registry,
)


def setup_default_chain() -> FilterChain:
    """Initialize the default simple chain with standard parts."""
    chain = get_default_chain()
    chain.clear()
    chain.add(RateLimitPart(max_requests=30, window_seconds=60))
    chain.add(PromptGuardPart(threshold=2))
    chain.add(SanitizePart(max_length=100000))
    chain.add(ToolLogPart())
    chain.add(ToolSanitizePart())
    chain.add(ToolResultTruncatePart(max_length=50000))
    chain.add(OutputFormatterPart())
    return chain


def setup_minimal_chain() -> FilterChain:
    """Minimal chain for low-latency."""
    chain = get_default_chain()
    chain.clear()
    chain.add(PromptGuardPart(threshold=3))
    chain.add(SanitizePart(max_length=50000))
    return chain


def setup_strict_chain() -> FilterChain:
    """Strict chain for high-security."""
    chain = get_default_chain()
    chain.clear()
    chain.add(RateLimitPart(max_requests=10, window_seconds=60))
    chain.add(PromptGuardPart(threshold=1))
    chain.add(PIIRedactPart())
    chain.add(SanitizePart(max_length=10000))
    chain.add(ToolLogPart())
    chain.add(ToolSanitizePart())
    chain.add(ToolResultLogPart())
    chain.add(ToolResultTruncatePart(max_length=20000))
    chain.add(AuditLogPart(log_level="info"))
    chain.add(OutputFormatterPart())
    return chain


# Legacy setup for backward compat
from .builtin import (
    RateLimitToLLM,
    PromptInjectionToLLM,
    InputSanitizerToLLM,
    ResponseFormatterFromLLM,
    AuditLogFromLLM,
)

def setup_default_filters():
    """Legacy: Set up default filter configuration."""
    registry = get_filter_registry()
    registry.reset()
    
    registry.register_default_to_llm(
        lambda: RateLimitToLLM(priority=Priority.HIGHEST).configure(
            window_seconds=60, max_requests=30,
        )
    )
    registry.register_default_to_llm(
        lambda: PromptInjectionToLLM(priority=Priority.HIGHEST).configure(
            threshold=2, block_on_detection=True,
        )
    )
    registry.register_default_to_llm(
        lambda: InputSanitizerToLLM(priority=Priority.MEDIUM).configure(
            normalize_whitespace=True, trim=True, max_length=0,  # 0 = no limit, let LLM context window be the constraint
        )
    )
    registry.register_default_from_llm(
        lambda: ResponseFormatterFromLLM(priority=Priority.LOW).configure(
            normalize_code_blocks=True, clean_newlines=True,
        )
    )
    
    return registry


def setup_minimal_filters():
    """Legacy: Set up minimal filter configuration."""
    registry = get_filter_registry()
    registry.reset()
    
    registry.register_default_to_llm(
        lambda: PromptInjectionToLLM(priority=Priority.HIGHEST).configure(
            threshold=3, block_on_detection=True,
        )
    )
    registry.register_default_to_llm(
        lambda: InputSanitizerToLLM(priority=Priority.MEDIUM).configure(
            normalize_whitespace=False, trim=True, max_length=0,
        )
    )
    
    return registry


def setup_strict_filters():
    """Legacy: Set up strict filter configuration."""
    registry = get_filter_registry()
    registry.reset()
    
    registry.register_default_to_llm(
        lambda: RateLimitToLLM(priority=Priority.HIGHEST).configure(
            window_seconds=60, max_requests=10,
        )
    )
    registry.register_default_to_llm(
        lambda: PromptInjectionToLLM(priority=Priority.HIGHEST).configure(
            threshold=1, block_on_detection=True,
        )
    )
    registry.register_default_to_llm(
        lambda: InputSanitizerToLLM(priority=Priority.MEDIUM).configure(
            normalize_whitespace=True, trim=True, max_length=0,
        )
    )
    registry.register_default_from_llm(
        lambda: ResponseFormatterFromLLM(priority=Priority.LOW).configure(
            normalize_code_blocks=True, clean_newlines=True,
        )
    )
    registry.register_default_from_llm(
        lambda: AuditLogFromLLM(priority=Priority.LEAST).configure(
            log_level="info", truncate_content=500,
        )
    )
    
    return registry


__all__ = [
    # Configurable chains (Admin UI)
    "ChainExecutor",
    "ExecutionContext", 
    "ExecutionResult",
    "FlowSignal",
    "CompareOp",
    "ChainManager",
    "get_chain_manager",
    
    # Simple chains (code-defined)
    "FilterChain",
    "ChainContext",
    "ChainResult",
    "ChainPart",
    "PartType",
    "ToLLMPart",
    "FromLLMPart",
    "ToToolPart",
    "FromToolPart",
    "LogicPart",
    "to_llm",
    "from_llm",
    "to_tool",
    "from_tool",
    "logic_part",
    "ChainRegistry",
    "get_registry",
    "get_default_chain",
    "get_chat_chain",
    
    # Parts
    "RateLimitPart",
    "SanitizePart",
    "PromptGuardPart",
    "PIIRedactPart",
    "OutputFormatterPart",
    "WordFilterPart",
    "AuditLogPart",
    "ToolLogPart",
    "ToolSanitizePart",
    "ToolResultLogPart",
    "ToolResultTruncatePart",
    "ConditionalSkipPart",
    
    # Setup
    "setup_default_chain",
    "setup_minimal_chain",
    "setup_strict_chain",
    "setup_default_filters",
    "setup_minimal_filters",
    "setup_strict_filters",
    
    # Legacy
    "FilterContext",
    "FilterResult",
    "StreamChunk",
    "Priority",
    "ChatFilterManager",
    "get_chat_filters",
    "get_filter_registry",
]
