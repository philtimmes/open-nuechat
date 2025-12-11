"""
Filter Chain Admin API

CRUD operations for filter chains.
Also provides schema information for the admin UI.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.models.models import User
from app.filters.manager import get_chain_manager


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require admin user for endpoint."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


router = APIRouter(prefix="/filter-chains", tags=["filter-chains"])


# =============================================================================
# SCHEMAS
# =============================================================================

class StepConfig(BaseModel):
    """Configuration for a step."""
    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    input_var: Optional[str] = None
    output_var: Optional[str] = None
    tool_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    add_to_context: Optional[bool] = None
    context_label: Optional[str] = None
    source_var: Optional[str] = None
    target: Optional[str] = None
    include_context: Optional[bool] = None
    action: Optional[str] = None
    chain_name: Optional[str] = None
    var_name: Optional[str] = None
    value: Optional[str] = None
    values: Optional[List[str]] = None  # For set_array step
    template: Optional[str] = None
    left: Optional[str] = None
    operator: Optional[str] = None
    right: Optional[str] = None
    response: Optional[str] = None
    proceed_to_llm: Optional[bool] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    level: Optional[str] = None


class Comparison(BaseModel):
    """A comparison for conditionals."""
    left: str
    operator: str
    right: str


class LoopConfig(BaseModel):
    """Loop configuration."""
    enabled: bool = False
    type: str = "counter"  # counter, condition
    count: Optional[int] = None
    while_cond: Optional[Comparison] = Field(None, alias="while")
    max_iterations: Optional[int] = None
    loop_var: Optional[str] = "i"


class ConditionalConfig(BaseModel):
    """Conditional configuration."""
    enabled: bool = False
    logic: str = "and"  # and, or
    comparisons: List[Comparison] = []
    on_true: List["StepDefinition"] = []
    on_false: List["StepDefinition"] = []


class StepDefinition(BaseModel):
    """Definition of a single step."""
    id: Optional[str] = None
    type: str
    name: Optional[str] = None
    enabled: bool = True
    config: StepConfig = Field(default_factory=StepConfig)
    on_error: str = "skip"  # stop, skip, jump
    jump_to_step: Optional[str] = None
    conditional: Optional[ConditionalConfig] = None
    loop: Optional[LoopConfig] = None


# Allow recursive definition
ConditionalConfig.model_rebuild()


class ChainDefinition(BaseModel):
    """The chain definition structure."""
    steps: List[StepDefinition] = []
    variables: List[str] = []  # Custom variable names for reference


class ChainCreate(BaseModel):
    """Create a new filter chain."""
    name: str
    description: str = ""
    enabled: bool = True
    priority: int = 100
    retain_history: bool = True
    bidirectional: bool = False
    outbound_chain_id: Optional[str] = None
    max_iterations: int = 10
    debug: bool = False
    definition: ChainDefinition = Field(default_factory=ChainDefinition)


class ChainUpdate(BaseModel):
    """Update a filter chain."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    retain_history: Optional[bool] = None
    bidirectional: Optional[bool] = None
    outbound_chain_id: Optional[str] = None
    max_iterations: Optional[int] = None
    debug: Optional[bool] = None
    definition: Optional[ChainDefinition] = None


class ChainResponse(BaseModel):
    """Filter chain response."""
    id: str
    name: str
    description: Optional[str]
    enabled: bool
    priority: int
    retain_history: bool
    bidirectional: bool
    outbound_chain_id: Optional[str]
    max_iterations: int
    debug: bool
    definition: Dict[str, Any]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    created_by: Optional[str]


# =============================================================================
# UI SCHEMA - Tells frontend how to render each step type
# =============================================================================

STEP_TYPE_SCHEMA = {
    "pass": {
        "label": "Pass Through",
        "description": "Pass content through unchanged",
        "category": "flow",
        "fields": [],
    },
    "stop": {
        "label": "Stop Chain",
        "description": "Stop chain execution",
        "category": "flow",
        "fields": [
            {"name": "response", "type": "textarea", "label": "Response (optional)", "placeholder": "Response to return"},
            {"name": "proceed_to_llm", "type": "checkbox", "label": "Still proceed to LLM", "default": False},
        ],
    },
    "block": {
        "label": "Block Request",
        "description": "Block the request with an error",
        "category": "flow",
        "fields": [
            {"name": "reason", "type": "text", "label": "Block Reason", "required": True},
        ],
    },
    "to_llm": {
        "label": "Ask LLM",
        "description": "Ask the LLM a question",
        "category": "llm",
        "fields": [
            {"name": "prompt", "type": "textarea", "label": "Prompt", "required": True, "placeholder": "Enter prompt with {$Query} variables"},
            {"name": "system_prompt", "type": "textarea", "label": "System Prompt (optional)"},
            {"name": "input_var", "type": "variable", "label": "Input Variable", "default": "$Query"},
            {"name": "output_var", "type": "text", "label": "Store Result As", "placeholder": "VariableName"},
        ],
    },
    "query": {
        "label": "Generate Query",
        "description": "Ask LLM to generate a search/tool query",
        "category": "llm",
        "fields": [
            {"name": "prompt", "type": "textarea", "label": "Prompt", "required": True, "placeholder": "Generate a search query for: {$Query}"},
            {"name": "output_var", "type": "text", "label": "Store Query As", "default": "GeneratedQuery"},
        ],
    },
    "to_tool": {
        "label": "Call Tool",
        "description": "Execute a tool",
        "category": "tool",
        "fields": [
            {"name": "tool_name", "type": "tool_select", "label": "Tool", "required": True},
            {"name": "params", "type": "key_value", "label": "Parameters", "placeholder": "Use {$Var[Name]} for variables"},
            {"name": "output_var", "type": "text", "label": "Store Result As"},
            {"name": "add_to_context", "type": "checkbox", "label": "Add result to context", "default": True},
            {"name": "context_label", "type": "text", "label": "Context Label"},
        ],
    },
    "from_tool": {
        "label": "Process Tool Result",
        "description": "Transform tool result",
        "category": "tool",
        "fields": [
            {"name": "transform", "type": "textarea", "label": "Transform Template", "placeholder": "Use {$PreviousResult}"},
        ],
    },
    "from_llm": {
        "label": "Process LLM Output",
        "description": "Process LLM response (bidirectional)",
        "category": "llm",
        "fields": [],
    },
    "context_insert": {
        "label": "Insert Context",
        "description": "Add content to accumulated context",
        "category": "context",
        "fields": [
            {"name": "source_var", "type": "variable", "label": "Source", "default": "$PreviousResult"},
            {"name": "label", "type": "text", "label": "Context Label"},
        ],
    },
    "go_to_llm": {
        "label": "Go To LLM",
        "description": "Prepare final message for main LLM call",
        "category": "flow",
        "fields": [
            {"name": "input_var", "type": "variable", "label": "Input", "default": "$Query"},
            {"name": "include_context", "type": "checkbox", "label": "Include accumulated context", "default": True},
        ],
    },
    "filter_complete": {
        "label": "Filter Complete",
        "description": "Mark filter as complete, skip remaining steps",
        "category": "flow",
        "fields": [
            {"name": "action", "type": "select", "label": "Action", "options": ["go_to_llm", "stop"], "default": "go_to_llm"},
            {"name": "input_var", "type": "variable", "label": "Input (for go_to_llm)", "default": "$Query"},
        ],
    },
    "set_var": {
        "label": "Set Variable",
        "description": "Set a variable value",
        "category": "variable",
        "fields": [
            {"name": "var_name", "type": "text", "label": "Variable Name", "required": True},
            {"name": "value", "type": "textarea", "label": "Value", "placeholder": "Static value or {$Var[Other]}"},
        ],
    },
    "set_array": {
        "label": "Set Array",
        "description": "Create an array variable from multiple values",
        "category": "variable",
        "fields": [
            {"name": "var_name", "type": "text", "label": "Array Name", "required": True},
            {"name": "values", "type": "array", "label": "Values", "placeholder": "Value or variable reference"},
        ],
    },
    "compare": {
        "label": "Compare",
        "description": "Compare values and store result",
        "category": "logic",
        "fields": [
            {"name": "left", "type": "variable", "label": "Left Value", "required": True},
            {"name": "operator", "type": "select", "label": "Operator", "required": True, "options": [
                {"value": "==", "label": "equals"},
                {"value": "!=", "label": "not equals"},
                {"value": "contains", "label": "contains"},
                {"value": "not_contains", "label": "not contains"},
                {"value": "starts_with", "label": "starts with"},
                {"value": "ends_with", "label": "ends with"},
                {"value": "regex", "label": "regex match"},
                {"value": ">", "label": "greater than"},
                {"value": "<", "label": "less than"},
                {"value": ">=", "label": "greater or equal"},
                {"value": "<=", "label": "less or equal"},
            ]},
            {"name": "right", "type": "text", "label": "Right Value", "required": True},
            {"name": "output_var", "type": "text", "label": "Store Result As", "default": "CompareResult"},
        ],
    },
    "call_chain": {
        "label": "Call Chain",
        "description": "Execute another chain as subroutine",
        "category": "flow",
        "fields": [
            {"name": "chain_name", "type": "chain_select", "label": "Chain", "required": True},
        ],
    },
    "modify": {
        "label": "Modify Content",
        "description": "Modify current content with template",
        "category": "variable",
        "fields": [
            {"name": "template", "type": "textarea", "label": "Template", "required": True, "placeholder": "{$Current} modified"},
            {"name": "output_var", "type": "text", "label": "Also store as variable"},
        ],
    },
    "log": {
        "label": "Log Message",
        "description": "Log a message for debugging",
        "category": "debug",
        "fields": [
            {"name": "message", "type": "textarea", "label": "Message", "placeholder": "Value: {$Var[Name]}"},
            {"name": "level", "type": "select", "label": "Level", "options": ["debug", "info", "warning", "error"], "default": "debug"},
        ],
    },
}

COMPARISON_OPERATORS = [
    {"value": "==", "label": "equals"},
    {"value": "!=", "label": "not equals"},
    {"value": "contains", "label": "contains"},
    {"value": "not_contains", "label": "not contains"},
    {"value": "starts_with", "label": "starts with"},
    {"value": "ends_with", "label": "ends with"},
    {"value": "regex", "label": "regex match"},
    {"value": ">", "label": "greater than"},
    {"value": "<", "label": "less than"},
    {"value": ">=", "label": "greater or equal"},
    {"value": "<=", "label": "less or equal"},
]

BUILTIN_VARIABLES = [
    {"value": "$Query", "label": "Original Query"},
    {"value": "$PreviousResult", "label": "Previous Step Result"},
    {"value": "$Current", "label": "Current Value"},
]


# =============================================================================
# ROUTES
# =============================================================================

@router.get("/schema")
async def get_schema(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Get the UI schema for building chains, including available tools."""
    from app.models.models import Tool
    
    # Fetch enabled tools
    result = await db.execute(
        select(Tool).where(Tool.is_enabled == True)
    )
    tools = result.scalars().all()
    
    # Build flat tool list
    available_tools = [
        # Built-in tools
        {"value": "web_search", "label": "🌐 Web Search", "category": "Built-in"},
        {"value": "web_fetch", "label": "📄 Fetch Web Page", "category": "Built-in"},
        {"value": "calculator", "label": "🔢 Calculator", "category": "Built-in"},
        {"value": "code_interpreter", "label": "💻 Run Code", "category": "Built-in"},
    ]
    
    # Add MCP/OpenAPI tools and sub-tools
    for tool in tools:
        icon = "🔌" if tool.tool_type.value == "mcp" else "🔗"
        
        if tool.schema_cache and len(tool.schema_cache) > 0:
            # Tool has sub-tools
            for sub_tool in tool.schema_cache:
                available_tools.append({
                    "value": f"{tool.name}:{sub_tool.get('name', '')}",
                    "label": f"{icon} {tool.name} → {sub_tool.get('name', '')}",
                    "category": tool.name,
                    "description": sub_tool.get("description", ""),
                })
        else:
            # Single operation tool
            available_tools.append({
                "value": tool.name,
                "label": f"{icon} {tool.name}",
                "category": "External",
                "description": tool.description or "",
            })
    
    return {
        "step_types": STEP_TYPE_SCHEMA,
        "comparison_operators": COMPARISON_OPERATORS,
        "builtin_variables": BUILTIN_VARIABLES,
        "available_tools": available_tools,
        "error_handlers": [
            {"value": "skip", "label": "Skip to next step"},
            {"value": "stop", "label": "Stop chain"},
            {"value": "jump", "label": "Jump to step"},
        ],
        "loop_types": [
            {"value": "counter", "label": "Fixed count"},
            {"value": "condition", "label": "While condition"},
        ],
    }


@router.get("", response_model=List[ChainResponse])
async def list_chains(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """List all filter chains."""
    manager = get_chain_manager()
    chains = manager.get_all_chains()
    return chains


@router.get("/{chain_id}", response_model=ChainResponse)
async def get_chain(
    chain_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Get a specific filter chain."""
    manager = get_chain_manager()
    chain = manager.get_chain(chain_id)
    
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    return chain


@router.post("", response_model=ChainResponse, status_code=status.HTTP_201_CREATED)
async def create_chain(
    data: ChainCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Create a new filter chain."""
    manager = get_chain_manager()
    
    # Check for duplicate name
    if manager.get_chain_by_name(data.name):
        raise HTTPException(status_code=400, detail="Chain with this name already exists")
    
    # Validate definition
    errors = manager.validate_chain(data.definition.model_dump())
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    chain = await manager.create_chain(
        db=db,
        name=data.name,
        description=data.description,
        definition=data.definition.model_dump(),
        enabled=data.enabled,
        priority=data.priority,
        retain_history=data.retain_history,
        bidirectional=data.bidirectional,
        outbound_chain_id=data.outbound_chain_id,
        max_iterations=data.max_iterations,
        debug=data.debug,
        created_by=str(user.id),
    )
    
    return chain


@router.put("/{chain_id}", response_model=ChainResponse)
async def update_chain(
    chain_id: str,
    data: ChainUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Update a filter chain."""
    manager = get_chain_manager()
    
    existing = manager.get_chain(chain_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    # Check for duplicate name
    if data.name and data.name != existing.get("name"):
        if manager.get_chain_by_name(data.name):
            raise HTTPException(status_code=400, detail="Chain with this name already exists")
    
    # Validate definition if provided
    if data.definition:
        errors = manager.validate_chain(data.definition.model_dump())
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors})
    
    updates = data.model_dump(exclude_unset=True)
    if "definition" in updates and updates["definition"]:
        updates["definition"] = data.definition.model_dump()
    
    chain = await manager.update_chain(db=db, chain_id=chain_id, **updates)
    
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    return chain


@router.delete("/{chain_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chain(
    chain_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Delete a filter chain."""
    manager = get_chain_manager()
    
    success = await manager.delete_chain(db=db, chain_id=chain_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chain not found")


@router.post("/{chain_id}/validate")
async def validate_chain(
    chain_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Validate a chain definition."""
    manager = get_chain_manager()
    
    chain = manager.get_chain(chain_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    errors = manager.validate_chain(chain.get("definition", {}))
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


@router.post("/validate")
async def validate_chain_definition(
    data: ChainDefinition,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Validate a chain definition without saving."""
    manager = get_chain_manager()
    
    errors = manager.validate_chain(data.model_dump())
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


@router.post("/{chain_id}/test")
async def test_chain(
    chain_id: str,
    test_query: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """
    Test a chain with a sample query.
    
    Note: This requires LLM and tool functions to be configured.
    """
    manager = get_chain_manager()
    
    chain = manager.get_chain(chain_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Chain not found")
    
    # For testing, we'd need mock LLM/tool functions
    # This is a placeholder that returns the chain structure
    return {
        "message": "Test execution not implemented - requires runtime LLM/tool configuration",
        "chain": chain,
        "test_query": test_query,
    }


@router.post("/reload")
async def reload_chains(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Reload all chains from database."""
    manager = get_chain_manager()
    count = await manager.load_from_db(db)
    
    return {"message": f"Reloaded {count} chains"}
