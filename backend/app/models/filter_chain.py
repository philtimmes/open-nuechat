"""
Filter Chain Database Model

Stores configurable filter chains as JSON for maximum flexibility.
Chains are loaded into memory on startup for fast execution.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON
from datetime import datetime, timezone
import uuid

from .base import Base


class FilterChain(Base):
    """
    A configurable filter chain.
    
    The entire chain definition is stored as JSON to support:
    - Arbitrary nesting depth
    - Complex conditionals
    - Loops
    - Chain composition
    - Export as dynamic tool
    """
    __tablename__ = "filter_chains"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic info
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, default=True, nullable=False)
    
    # Execution order (lower = runs first)
    priority = Column(Integer, default=100, nullable=False)
    
    # Global settings
    retain_history = Column(Boolean, default=True, nullable=False)  # vs modify history
    bidirectional = Column(Boolean, default=False, nullable=False)
    outbound_chain_id = Column(String(36), nullable=True)  # Chain to use for LLM response
    max_iterations = Column(Integer, default=10, nullable=False)  # Safety limit for loops
    debug = Column(Boolean, default=False, nullable=False)  # Log execution to console
    skip_if_rag_hit = Column(Boolean, default=True, nullable=False)  # Skip chain if RAG found results
    
    # Export as Dynamic Tool (NC-0.8.0.0)
    # When enabled, this filter chain becomes a tool that can be advertised to the LLM
    export_as_tool = Column(Boolean, default=False, nullable=False)  # Enable tool export
    tool_name = Column(String(100), nullable=True)  # Tool identifier (e.g., "WebSearch")
    tool_label = Column(String(100), nullable=True)  # Display label (e.g., "🔍 Search Web")
    
    # Advertising to LLM
    advertise_to_llm = Column(Boolean, default=False, nullable=False)  # Include in system prompt
    advertise_text = Column(Text, nullable=True)  # e.g., "Use $WebSearch=\"query\" to search"
    
    # Triggering
    trigger_pattern = Column(String(500), nullable=True)  # Regex pattern e.g., r'\$WebSearch="([^"]+)"'
    trigger_source = Column(String(20), default='both', nullable=False)  # 'llm', 'user', 'both'
    
    # Display behavior
    erase_from_display = Column(Boolean, default=True, nullable=False)  # Hide trigger from chat UI
    keep_in_history = Column(Boolean, default=True, nullable=False)  # Preserve in message history
    
    # UI Button (for user hints)
    button_enabled = Column(Boolean, default=False, nullable=False)
    button_icon = Column(String(50), nullable=True)  # Emoji or icon name
    button_location = Column(String(20), default='response', nullable=False)  # 'response', 'query', 'both'
    button_trigger_mode = Column(String(20), default='immediate', nullable=False)  # 'immediate', 'modal', 'selection'
    
    # Variables that this tool accepts (JSON array of {name, label, default, type})
    tool_variables = Column(JSON, nullable=True, default=list)
    
    # The chain definition (JSON)
    # Structure: {"steps": [...], "variables": [...]}
    definition = Column(JSON, nullable=False, default=dict)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    created_by = Column(String(36), nullable=True)  # Admin user ID
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API/cache."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "priority": self.priority,
            "retain_history": self.retain_history,
            "bidirectional": self.bidirectional,
            "outbound_chain_id": self.outbound_chain_id,
            "max_iterations": self.max_iterations,
            "debug": self.debug,
            "skip_if_rag_hit": self.skip_if_rag_hit,
            # Export as tool fields
            "export_as_tool": self.export_as_tool,
            "tool_name": self.tool_name,
            "tool_label": self.tool_label,
            "advertise_to_llm": self.advertise_to_llm,
            "advertise_text": self.advertise_text,
            "trigger_pattern": self.trigger_pattern,
            "trigger_source": self.trigger_source,
            "erase_from_display": self.erase_from_display,
            "keep_in_history": self.keep_in_history,
            "button_enabled": self.button_enabled,
            "button_icon": self.button_icon,
            "button_location": self.button_location,
            "button_trigger_mode": self.button_trigger_mode,
            "tool_variables": self.tool_variables,
            # Definition and metadata
            "definition": self.definition,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
        }


"""
Step Definition Schema:

{
    "id": "uuid",
    "type": "to_llm|from_llm|to_tool|from_tool|query|context_insert|go_to_llm|filter_complete|call_chain|set_var|compare",
    "name": "Human readable name",
    "enabled": true,
    
    # Type-specific config
    "config": {
        # For to_llm, query:
        "prompt": "The prompt text with {$Query} variables",
        "system_prompt": "Optional system prompt",
        "input_var": "$Query",
        "output_var": "ResultName",  # Creates $Var[ResultName]
        
        # For to_tool:
        "tool_name": "web_search",
        "params": {"query": "{$Var[SearchQuery]}"},
        "output_var": "SearchResults",
        
        # For context_insert:
        "source_var": "$Var[SearchResults]",
        "target": "to_llm",
        "label": "Search Results",
        
        # For go_to_llm:
        "input_var": "$Query",
        
        # For filter_complete:
        "action": "go_to_llm",
        "input_var": "$Query",
        
        # For call_chain:
        "chain_name": "OtherChainName",
        
        # For set_var:
        "var_name": "MyVar",
        "value": "static value or {$Var[Other]}",
        
        # For compare (standalone comparison, not conditional):
        "left": "$Var[Something]",
        "operator": "==|!=|contains|not_contains|starts_with|ends_with|regex|>|<|>=|<=",
        "right": "value or {$Var[Other]}",
        "output_var": "CompareResult",  # Stores true/false
    },
    
    # Error handling
    "on_error": "stop|skip|jump",
    "jump_to_step": "step_id",  # If on_error is "jump"
    
    # Conditional branching (optional)
    "conditional": {
        "enabled": true,
        "logic": "and|or",  # How to combine multiple comparisons
        "comparisons": [
            {
                "left": "$Var[NeedsContext]",
                "operator": "contains",
                "right": "yes"
            }
        ],
        "on_true": [
            # Nested steps array
        ],
        "on_false": [
            # Nested steps array
        ]
    },
    
    # Loop (optional)
    "loop": {
        "enabled": true,
        "type": "counter|condition",
        
        # For counter:
        "count": 3,
        
        # For condition:
        "while": {
            "left": "$Var[Done]",
            "operator": "!=",
            "right": "true"
        },
        
        "max_iterations": 5,  # Override chain default
        "loop_var": "i",  # Creates $Var[i] with current iteration
    }
}
"""
