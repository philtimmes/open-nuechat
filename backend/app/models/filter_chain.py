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
