"""
Filter Chain Executor

Interprets and executes chain definitions.
Handles all primitives, flow control, variables, and nesting.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from enum import Enum
import re
import logging
import copy
import json

logger = logging.getLogger(__name__)


# Type aliases
LLMFunc = Callable[[str, Optional[str]], Coroutine[Any, Any, str]]
ToolFunc = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, Any]]


class FlowSignal(Enum):
    """Signals for flow control."""
    CONTINUE = "continue"
    STOP = "stop"
    FILTER_COMPLETE = "filter_complete"
    ERROR = "error"
    JUMP = "jump"


class CompareOp(Enum):
    """Comparison operators."""
    EQ = "=="
    NE = "!="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="


@dataclass
class ExecutionContext:
    """
    Runtime context for chain execution.
    
    Holds all variables, accumulated context, and execution state.
    """
    user_id: str
    chat_id: str
    
    # Chain info
    chain_name: str = ""
    debug: bool = False
    
    # The original query
    original_query: str = ""
    
    # Current value being passed through the chain
    current_value: Any = None
    
    # Previous step result
    previous_result: Any = None
    
    # Named variables: {"VarName": value}
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Accumulated context to inject
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution state
    signal: FlowSignal = FlowSignal.CONTINUE
    jump_target: Optional[str] = None
    error_message: Optional[str] = None
    
    # Loop tracking
    iteration_counts: Dict[str, int] = field(default_factory=dict)
    
    # History of what happened (for debugging/logging)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Final message to send to LLM
    final_message: Optional[str] = None
    
    # Should we call the main LLM after chain completes?
    proceed_to_llm: bool = True
    
    def debug_log(self, message: str, data: Any = None):
        """Log debug message to console if debug is enabled."""
        if self.debug:
            prefix = f"[FilterChain:{self.chain_name}]" if self.chain_name else "[FilterChain]"
            if data is not None:
                # Truncate long data
                data_str = str(data)
                if len(data_str) > 200:
                    data_str = data_str[:200] + "..."
                print(f"{prefix} {message}: {data_str}")
            else:
                print(f"{prefix} {message}")
    
    def resolve_variable(self, ref: str) -> Any:
        """
        Resolve a variable reference.
        
        Formats:
            $Query -> original_query
            $PreviousResult -> previous_result
            $Var[Name] -> variables["Name"]
            $Current -> current_value
        """
        if not ref:
            return ref
            
        if not isinstance(ref, str):
            return ref
        
        # Direct variable references
        if ref == "$Query":
            return self.original_query
        elif ref == "$PreviousResult":
            return self.previous_result
        elif ref == "$Current":
            return self.current_value
        elif ref.startswith("$Var[") and ref.endswith("]"):
            var_name = ref[5:-1]
            return self.variables.get(var_name, "")
        
        return ref
    
    def _extract_json_path(self, data: Any, path: str) -> Any:
        """
        Extract value from data using a JSON path.
        
        Examples:
            .results[0].url -> data["results"][0]["url"]
            .title -> data["title"]
            [0].name -> data[0]["name"]
        """
        if not path:
            return data
        
        # Parse JSON string if needed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return data
        
        # Navigate the path
        current = data
        # Split on . but keep array indices
        parts = re.split(r'\.(?![^\[]*\])', path.lstrip('.'))
        
        for part in parts:
            if not part:
                continue
            
            # Check for array index like "results[0]" or just "[0]"
            array_match = re.match(r'([^\[]*)\[(\d+)\](.*)$', part)
            if array_match:
                key = array_match.group(1)
                index = int(array_match.group(2))
                remaining = array_match.group(3)
                
                # Get the key first if present
                if key:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return None
                
                # Then get the index
                if isinstance(current, (list, tuple)) and index < len(current):
                    current = current[index]
                else:
                    return None
                
                # Handle nested paths after index like [0].url
                if remaining:
                    current = self._extract_json_path(current, remaining)
            else:
                # Simple key access
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    
    def interpolate(self, template: str) -> str:
        """
        Interpolate variables in a template string.
        
        Supports both legacy and simplified syntax:
            {Query} or {$Query} -> original_query
            {PreviousResult} or {$PreviousResult} -> previous_result
            {VarName} or {VarName.path} -> variables["VarName"]
            {$Var[Name]} -> variables["Name"] (legacy)
        """
        if not template or not isinstance(template, str):
            return template or ""
        
        result = template
        
        # Replace {Query}, {$Query}, $Query
        result = result.replace("{Query}", str(self.original_query or ""))
        result = result.replace("{$Query}", str(self.original_query or ""))
        result = result.replace("$Query", str(self.original_query or ""))
        
        # Replace {PreviousResult} and {PreviousResult.path}
        def replace_prev(match):
            path = match.group(1) or ""
            if path:
                return str(self._extract_json_path(self.previous_result, path) or "")
            return str(self.previous_result or "")
        
        result = re.sub(r'\{PreviousResult(\.[^\}]+)?\}', replace_prev, result)
        result = re.sub(r'\{\$PreviousResult(\.[^\}]+)?\}', replace_prev, result)
        result = re.sub(r'\$PreviousResult(\.[a-zA-Z0-9_\[\]]+)?', replace_prev, result)
        
        # Replace {Current} and {Current.path}
        def replace_current(match):
            path = match.group(1) or ""
            if path:
                return str(self._extract_json_path(self.current_value, path) or "")
            return str(self.current_value or "")
        
        result = re.sub(r'\{Current(\.[^\}]+)?\}', replace_current, result)
        result = re.sub(r'\{\$Current(\.[^\}]+)?\}', replace_current, result)
        result = re.sub(r'\$Current(\.[a-zA-Z0-9_\[\]]+)?', replace_current, result)
        
        # Replace {$Var[Name].path} (legacy with braces)
        def replace_var_legacy(match):
            var_name = match.group(1)
            path = match.group(2) or ""
            value = self.variables.get(var_name, "")
            if path:
                extracted = self._extract_json_path(value, path)
                return str(extracted) if extracted is not None else ""
            return str(value)
        result = re.sub(r'\{\$Var\[([^\]]+)\](\.[^\}]+)?\}', replace_var_legacy, result)
        result = re.sub(r'\$Var\[([^\]]+)\](\.[a-zA-Z0-9_\[\].]+)?', replace_var_legacy, result)
        
        # Replace {VarName} or {VarName.path} - NEW simplified syntax
        # Match {word} or {word.path.to[0].field} where word is a known variable
        def replace_var_simple(match):
            full_match = match.group(1)
            # Split into var name and path
            if '.' in full_match:
                parts = full_match.split('.', 1)
                var_name = parts[0]
                path = '.' + parts[1]
            else:
                var_name = full_match
                path = ""
            
            # Only replace if it's a known variable
            if var_name in self.variables:
                value = self.variables.get(var_name, "")
                if path:
                    extracted = self._extract_json_path(value, path)
                    return str(extracted) if extracted is not None else ""
                return str(value)
            # Not a known variable, leave unchanged
            return match.group(0)
        
        result = re.sub(r'\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[^\}]+)?)\}', replace_var_simple, result)
        
        return result
    
    def interpolate_value(self, template: str) -> Any:
        """
        Interpolate a value, preserving type when possible.
        
        Supports:
            - Direct variable names: SearchResults.results[0].url
            - Legacy syntax: $Var[Name].path
            - Array syntax: [SearchResults.results[0].url] -> ["https://..."]
            - Special vars: Query, PreviousResult, Current
        
        Returns the actual typed value when the template is a simple variable reference,
        or a string when it contains mixed content.
        """
        if not template or not isinstance(template, str):
            return template
        
        template = template.strip()
        
        # Check for array syntax: [value] or [value1, value2] (simplified)
        array_match = re.match(r'^\[(.+)\]$', template)
        if array_match:
            inner = array_match.group(1).strip()
            # Interpolate the inner value and wrap in array
            inner_value = self.interpolate_value(inner)
            if inner_value is not None and inner_value != "":
                return [inner_value]
            return []
        
        # Check for special keywords first
        if template == 'Query' or template == '$Query' or template == '{$Query}':
            return self.original_query
        
        if template == 'PreviousResult' or template == '$PreviousResult':
            return self.previous_result
        
        if template == 'Current' or template == '$Current':
            return self.current_value
        
        # Check for PreviousResult.path or $PreviousResult.path
        prev_match = re.match(r'^(?:\$)?PreviousResult(\.[a-zA-Z0-9_\[\].]+)$', template)
        if prev_match:
            path = prev_match.group(1)
            return self._extract_json_path(self.previous_result, path)
        
        # Check for Current.path or $Current.path
        current_match = re.match(r'^(?:\$)?Current(\.[a-zA-Z0-9_\[\].]+)$', template)
        if current_match:
            path = current_match.group(1)
            return self._extract_json_path(self.current_value, path)
        
        # Check for legacy $Var[Name] or $Var[Name].path syntax
        var_match = re.match(r'^\$Var\[([^\]]+)\](\.[a-zA-Z0-9_\[\].]+)?$', template)
        if var_match:
            var_name = var_match.group(1)
            path = var_match.group(2) or ""
            value = self.variables.get(var_name)
            if path and value is not None:
                return self._extract_json_path(value, path)
            return value
        
        # Check for {$Var[Name]} or {$Var[Name].path} (legacy with braces)
        var_match_brace = re.match(r'^\{\$Var\[([^\]]+)\](\.[^\}]+)?\}$', template)
        if var_match_brace:
            var_name = var_match_brace.group(1)
            path = var_match_brace.group(2) or ""
            value = self.variables.get(var_name)
            if path and value is not None:
                return self._extract_json_path(value, path)
            return value
        
        # NEW: Check for direct variable name (without $Var prefix)
        # Pattern: VarName or VarName.path.to[0].field
        direct_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)(\.[a-zA-Z0-9_\[\].]+)?$', template)
        if direct_match:
            var_name = direct_match.group(1)
            path = direct_match.group(2) or ""
            if var_name in self.variables:
                value = self.variables.get(var_name)
                if path and value is not None:
                    return self._extract_json_path(value, path)
                return value
        
        # Fall back to string interpolation
        return self.interpolate(template)
    
    def set_variable(self, name: str, value: Any):
        """Set a named variable."""
        self.variables[name] = value
        self.previous_result = value
    
    def add_context(self, content: str, label: Optional[str] = None, source: Optional[str] = None):
        """Add content to accumulated context."""
        self.context_items.append({
            "content": content,
            "label": label,
            "source": source,
        })
    
    def get_context_string(self) -> str:
        """Get accumulated context as a formatted string."""
        if not self.context_items:
            return ""
        
        parts = []
        for item in self.context_items:
            if item.get("label"):
                parts.append(f"[{item['label']}]\n{item['content']}")
            else:
                parts.append(item['content'])
        
        return "\n\n---\n\n".join(parts)
    
    def log_step(self, step_id: str, step_type: str, result: Any, error: Optional[str] = None):
        """Log a step execution."""
        self.execution_log.append({
            "step_id": step_id,
            "type": step_type,
            "result": str(result)[:500] if result else None,
            "error": error,
        })


@dataclass
class ExecutionResult:
    """Result from chain execution."""
    content: str
    context: ExecutionContext
    proceed_to_llm: bool = True
    modified: bool = False
    error: Optional[str] = None


class ChainExecutor:
    """
    Executes filter chain definitions.
    
    Supports:
        - All primitive operations
        - Variable resolution and interpolation
        - Conditionals with nested steps
        - Loops (counter and condition-based)
        - Chain composition
        - Error handling
    """
    
    def __init__(
        self,
        llm_func: Optional[LLMFunc] = None,
        tool_func: Optional[ToolFunc] = None,
        chain_loader: Optional[Callable[[str], Optional[dict]]] = None,
        global_max_iterations: int = 100,
        global_debug: bool = False,  # Global debug flag from admin settings
    ):
        self._llm_func = llm_func
        self._tool_func = tool_func
        self._chain_loader = chain_loader  # Function to load chain by name (for call_chain)
        self._global_max_iterations = global_max_iterations
        self._global_debug = global_debug
    
    def set_llm_func(self, func: LLMFunc):
        self._llm_func = func
    
    def set_tool_func(self, func: ToolFunc):
        self._tool_func = func
    
    def set_chain_loader(self, loader: Callable[[str], Optional[dict]]):
        self._chain_loader = loader
    
    def _global_debug_log(self, chain_name: str, message: str, data: Any = None):
        """Log debug message when global debug is enabled (full output, no truncation)."""
        if self._global_debug:
            prefix = f"[FILTER_DEBUG:{chain_name}]"
            if data is not None:
                # Full data output - no truncation for global debug
                data_str = str(data)
                logger.info(f"{prefix} {message}: {data_str}")
            else:
                logger.info(f"{prefix} {message}")
    
    async def execute(
        self,
        chain_def: dict,
        query: str,
        user_id: str,
        chat_id: str,
    ) -> ExecutionResult:
        """Execute a complete chain definition."""
        import time
        chain_start_time = time.time()
        
        chain_name = chain_def.get("name", "unnamed")
        debug = chain_def.get("debug", False)
        
        # Global debug logging
        if self._global_debug:
            logger.info(f"[FILTER_DEBUG:{chain_name}] ════════════════════════════════════════════════════════")
            logger.info(f"[FILTER_DEBUG:{chain_name}] CHAIN EXECUTION START")
            logger.info(f"[FILTER_DEBUG:{chain_name}] ════════════════════════════════════════════════════════")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Input Query: {query}")
        
        ctx = ExecutionContext(
            user_id=user_id,
            chat_id=chat_id,
            chain_name=chain_name,
            debug=debug,
            original_query=query,
            current_value=query,
        )
        
        ctx.debug_log("Starting chain execution")
        ctx.debug_log("Query", query)
        
        # Dump the full chain definition when debug is enabled
        if debug:
            print(f"[FilterChain:{chain_name}] ═══════════════════════════════════════════")
            print(f"[FilterChain:{chain_name}] CHAIN DEFINITION DUMP:")
            print(f"[FilterChain:{chain_name}] ═══════════════════════════════════════════")
            try:
                chain_json = json.dumps(chain_def.get("definition", {}), indent=2)
                for line in chain_json.split('\n'):
                    print(f"[FilterChain:{chain_name}]   {line}")
            except Exception as e:
                print(f"[FilterChain:{chain_name}]   (Failed to dump: {e})")
            print(f"[FilterChain:{chain_name}] ═══════════════════════════════════════════")
        
        max_iter = chain_def.get("max_iterations", self._global_max_iterations)
        steps = chain_def.get("definition", {}).get("steps", [])
        
        ctx.debug_log(f"Steps to execute: {len(steps)}, max_iterations: {max_iter}")
        
        if not steps:
            # No steps, pass through
            ctx.debug_log("No steps defined, passing through")
            if self._global_debug:
                logger.info(f"[FILTER_DEBUG:{chain_name}] No steps defined, passing through")
            return ExecutionResult(
                content=query,
                context=ctx,
                proceed_to_llm=True,
            )
        
        # Execute steps
        ctx = await self._execute_steps(steps, ctx, max_iter, chain_name)
        
        # Determine final content
        if ctx.final_message:
            final_content = ctx.final_message
        elif ctx.context_items:
            # Inject context into query
            context_str = ctx.get_context_string()
            final_content = f"Context:\n{context_str}\n\nQuery: {ctx.original_query}"
        else:
            final_content = ctx.current_value or ctx.original_query
        
        ctx.debug_log("Chain execution complete")
        ctx.debug_log("Final signal", ctx.signal.name)
        ctx.debug_log("Proceed to LLM", ctx.proceed_to_llm)
        ctx.debug_log("Final content", final_content)
        
        # Global debug logging
        if self._global_debug:
            chain_elapsed = time.time() - chain_start_time
            logger.info(f"[FILTER_DEBUG:{chain_name}] ════════════════════════════════════════════════════════")
            logger.info(f"[FILTER_DEBUG:{chain_name}] CHAIN EXECUTION COMPLETE")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Total Time: {chain_elapsed:.3f}s")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Final Signal: {ctx.signal.name}")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Proceed to LLM: {ctx.proceed_to_llm}")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Output Content: {final_content}")
            logger.info(f"[FILTER_DEBUG:{chain_name}] Variables: {dict(ctx.variables)}")
            logger.info(f"[FILTER_DEBUG:{chain_name}] ════════════════════════════════════════════════════════")
        
        return ExecutionResult(
            content=str(final_content),
            context=ctx,
            proceed_to_llm=ctx.proceed_to_llm and ctx.signal != FlowSignal.ERROR,
            modified=(final_content != query),
            error=ctx.error_message,
        )
    
    async def _execute_steps(
        self,
        steps: List[dict],
        ctx: ExecutionContext,
        max_iterations: int,
        chain_name: str = "",
    ) -> ExecutionContext:
        """Execute a list of steps."""
        
        step_index = 0
        step_map = {s.get("id", str(i)): i for i, s in enumerate(steps)}
        
        while step_index < len(steps):
            # Check for signals
            if ctx.signal in (FlowSignal.STOP, FlowSignal.FILTER_COMPLETE, FlowSignal.ERROR):
                break
            
            # Handle jump
            if ctx.signal == FlowSignal.JUMP and ctx.jump_target:
                if ctx.jump_target in step_map:
                    step_index = step_map[ctx.jump_target]
                    ctx.signal = FlowSignal.CONTINUE
                    ctx.jump_target = None
                else:
                    logger.warning(f"Jump target not found: {ctx.jump_target}")
                    ctx.signal = FlowSignal.CONTINUE
                    ctx.jump_target = None
            
            step = steps[step_index]
            
            # Skip disabled steps
            if not step.get("enabled", True):
                step_index += 1
                continue
            
            # Execute step
            ctx = await self._execute_step(step, ctx, max_iterations, chain_name)
            
            step_index += 1
        
        return ctx
    
    async def _execute_step(
        self,
        step: dict,
        ctx: ExecutionContext,
        max_iterations: int,
        chain_name: str = "",
    ) -> ExecutionContext:
        """Execute a single step with optional conditional and loop."""
        import time
        step_start_time = time.time()
        
        step_id = step.get("id", "unknown")
        step_type = step.get("type", "unknown")
        step_name = step.get("name") or step_type
        
        # Capture input for global debug
        input_value = ctx.current_value
        input_vars = dict(ctx.variables)
        
        # Global debug: step start
        if self._global_debug:
            logger.info(f"[FILTER_DEBUG:{chain_name}] ────────────────────────────────────────")
            logger.info(f"[FILTER_DEBUG:{chain_name}] STEP: {step_name} (type={step_type})")
            logger.info(f"[FILTER_DEBUG:{chain_name}]   Input $Current: {input_value}")
            logger.info(f"[FILTER_DEBUG:{chain_name}]   Input $Query: {ctx.original_query}")
            logger.info(f"[FILTER_DEBUG:{chain_name}]   Input $PreviousResult: {ctx.previous_result}")
            logger.info(f"[FILTER_DEBUG:{chain_name}]   Variables: {input_vars}")
        
        ctx.debug_log(f"─── Step: {step_name} (type={step_type}, id={step_id[:8]}...)")
        ctx.debug_log("  Input value", ctx.current_value)
        
        try:
            # Handle loop if present
            loop_config = step.get("loop")
            if loop_config and loop_config.get("enabled"):
                ctx.debug_log("  Executing as LOOP")
                return await self._execute_loop(step, ctx, max_iterations)
            
            # Handle conditional if present
            conditional = step.get("conditional")
            if conditional and conditional.get("enabled"):
                ctx.debug_log("  Has CONDITIONAL, evaluating...")
                return await self._execute_conditional(step, ctx, max_iterations)
            
            # Execute the step directly
            ctx = await self._execute_primitive(step, ctx)
            ctx.debug_log("  Output value", ctx.current_value)
            ctx.debug_log("  Variables", dict(ctx.variables))
            
            # Step-level add_to_context: add the step's output to context
            # This is separate from the primitive's own add_to_context (which is for tool results)
            step_add_to_context = step.get("add_to_context", False)
            if step_add_to_context and ctx.previous_result:
                context_label = step.get("context_label") or step_name
                ctx.debug_log(f"  Adding to context: {context_label}")
                ctx.add_context(str(ctx.previous_result), context_label)
            
            # Global debug: step complete
            if self._global_debug:
                step_elapsed = time.time() - step_start_time
                logger.info(f"[FILTER_DEBUG:{chain_name}]   Output $Current: {ctx.current_value}")
                logger.info(f"[FILTER_DEBUG:{chain_name}]   Output $PreviousResult: {ctx.previous_result}")
                # Show new/changed variables
                new_vars = {k: v for k, v in ctx.variables.items() if k not in input_vars or input_vars[k] != v}
                if new_vars:
                    logger.info(f"[FILTER_DEBUG:{chain_name}]   New/Changed Variables: {new_vars}")
                if step_add_to_context:
                    logger.info(f"[FILTER_DEBUG:{chain_name}]   Added to context: {step.get('context_label') or step_name}")
                logger.info(f"[FILTER_DEBUG:{chain_name}]   Time: {step_elapsed:.3f}s")
                logger.info(f"[FILTER_DEBUG:{chain_name}]   Signal: {ctx.signal.name}")
            
            # Handle jump_to_step for successful completion (if not already jumping)
            # Branch nodes handle their own jumps, so skip for them
            if step_type != "branch" and ctx.signal == FlowSignal.CONTINUE:
                jump_target = step.get("jump_to_step")
                if jump_target:
                    ctx.debug_log(f"  Jump to step: {jump_target}")
                    ctx.signal = FlowSignal.JUMP
                    ctx.jump_target = jump_target
                    if self._global_debug:
                        logger.info(f"[FILTER_DEBUG:{chain_name}]   Jump Target: {jump_target}")
            
            return ctx
            
        except Exception as e:
            logger.error(f"Error in step {step_id} ({step_type}): {e}")
            ctx.debug_log(f"  ERROR: {e}")
            ctx.log_step(step_id, step_type, None, str(e))
            
            # Global debug: error
            if self._global_debug:
                step_elapsed = time.time() - step_start_time
                logger.info(f"[FILTER_DEBUG:{chain_name}]   ERROR: {e}")
                logger.info(f"[FILTER_DEBUG:{chain_name}]   Time: {step_elapsed:.3f}s")
            
            # Handle error based on config
            on_error = step.get("on_error", "skip")
            if on_error == "stop":
                ctx.signal = FlowSignal.ERROR
                ctx.error_message = str(e)
            elif on_error == "jump":
                ctx.signal = FlowSignal.JUMP
                ctx.jump_target = step.get("jump_to_step")
            # else: skip - continue to next step
            
            return ctx
    
    async def _execute_conditional(
        self,
        step: dict,
        ctx: ExecutionContext,
        max_iterations: int,
    ) -> ExecutionContext:
        """Execute a step with conditional branching."""
        
        conditional = step["conditional"]
        comparisons = conditional.get("comparisons", [])
        logic = conditional.get("logic", "and")
        
        # Evaluate comparisons
        results = []
        for comp in comparisons:
            result = self._evaluate_comparison(comp, ctx)
            left_val = ctx.resolve_variable(comp.get("left", ""))
            right_val = ctx.resolve_variable(comp.get("right", ""))
            ctx.debug_log(f"    Condition: {comp.get('left')} ({left_val}) {comp.get('operator')} {comp.get('right')} ({right_val}) = {result}")
            results.append(result)
        
        # Combine results
        if logic == "and":
            condition_met = all(results) if results else False
        else:  # or
            condition_met = any(results) if results else False
        
        ctx.debug_log(f"    Conditional result ({logic}): {condition_met}")
        
        # Execute appropriate branch
        if condition_met:
            branch_steps = conditional.get("on_true", [])
            ctx.debug_log(f"    Executing on_true branch ({len(branch_steps)} steps)")
            # If no on_true steps, execute the step itself
            if not branch_steps:
                ctx.debug_log("    No on_true steps, executing step primitive")
                ctx = await self._execute_primitive(step, ctx)
        else:
            branch_steps = conditional.get("on_false", [])
            ctx.debug_log(f"    Executing on_false branch ({len(branch_steps)} steps)")
            # If no on_false steps, skip this step
            if not branch_steps:
                ctx.debug_log("    No on_false steps, skipping step")
                return ctx
        
        if branch_steps:
            ctx = await self._execute_steps(branch_steps, ctx, max_iterations)
        
        return ctx
    
    async def _execute_loop(
        self,
        step: dict,
        ctx: ExecutionContext,
        max_iterations: int,
    ) -> ExecutionContext:
        """Execute a step in a loop."""
        
        loop_config = step["loop"]
        loop_type = loop_config.get("type", "counter")
        loop_max = min(loop_config.get("max_iterations", max_iterations), max_iterations)
        loop_var = loop_config.get("loop_var", "i")
        
        iteration = 0
        
        while iteration < loop_max:
            # Set loop variable
            ctx.set_variable(loop_var, iteration)
            
            # Check loop condition
            if loop_type == "counter":
                count = loop_config.get("count", 1)
                if iteration >= count:
                    break
            elif loop_type == "condition":
                while_cond = loop_config.get("while")
                if while_cond and not self._evaluate_comparison(while_cond, ctx):
                    break
            
            # Execute step (without loop config to avoid infinite recursion)
            step_copy = copy.deepcopy(step)
            step_copy["loop"] = None
            ctx = await self._execute_step(step_copy, ctx, max_iterations)
            
            # Check for break signals
            if ctx.signal in (FlowSignal.STOP, FlowSignal.FILTER_COMPLETE, FlowSignal.ERROR):
                break
            
            iteration += 1
        
        return ctx
    
    def _evaluate_comparison(self, comp: dict, ctx: ExecutionContext) -> bool:
        """Evaluate a comparison."""
        
        left_ref = comp.get("left", "")
        op = comp.get("operator", "==")
        right_ref = comp.get("right", "")
        
        # Resolve values
        left = ctx.resolve_variable(left_ref)
        right = ctx.resolve_variable(right_ref)
        
        # Also interpolate if strings
        if isinstance(left, str):
            left = ctx.interpolate(left)
        if isinstance(right, str):
            right = ctx.interpolate(right)
        
        # Convert to strings for text operations
        left_str = str(left).lower() if left is not None else ""
        right_str = str(right).lower() if right is not None else ""
        
        try:
            if op == "==" or op == "eq":
                return left_str == right_str
            elif op == "!=" or op == "ne":
                return left_str != right_str
            elif op == "contains":
                return right_str in left_str
            elif op == "not_contains":
                return right_str not in left_str
            elif op == "starts_with":
                return left_str.startswith(right_str)
            elif op == "ends_with":
                return left_str.endswith(right_str)
            elif op == "regex" or op == "matches":
                return bool(re.search(right_str, left_str, re.IGNORECASE))
            elif op == "is_empty":
                return left is None or left_str == "" or left_str == "none"
            elif op == "is_not_empty":
                return left is not None and left_str != "" and left_str != "none"
            elif op in (">", "gt"):
                return float(left) > float(right)
            elif op in ("<", "lt"):
                return float(left) < float(right)
            elif op in (">=", "gte"):
                return float(left) >= float(right)
            elif op in ("<=", "lte"):
                return float(left) <= float(right)
            else:
                logger.warning(f"Unknown operator: {op}")
                return False
        except (ValueError, TypeError) as e:
            logger.warning(f"Comparison error: {e}")
            return False
    
    async def _execute_primitive(
        self,
        step: dict,
        ctx: ExecutionContext,
    ) -> ExecutionContext:
        """Execute a primitive operation."""
        
        step_id = step.get("id", "unknown")
        step_type = step.get("type", "pass")
        config = step.get("config", {})
        
        logger.debug(f"Executing step: {step_id} ({step_type})")
        
        # Dispatch to handler
        handler = getattr(self, f"_prim_{step_type}", None)
        if handler:
            ctx = await handler(config, ctx)
        else:
            logger.warning(f"Unknown step type: {step_type}")
        
        ctx.log_step(step_id, step_type, ctx.previous_result)
        return ctx
    
    # =========================================================================
    # PRIMITIVE HANDLERS
    # =========================================================================
    
    async def _prim_pass(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Pass through unchanged."""
        return ctx
    
    async def _prim_to_llm(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Ask LLM a question."""
        if not self._llm_func:
            raise RuntimeError("LLM function not configured")
        
        prompt_template = config.get("prompt", "")
        system_prompt = config.get("system_prompt")
        output_var = config.get("output_var")
        input_var = config.get("input_var", "$Query")  # What data to use
        
        # Resolve the input variable
        input_value = ctx.resolve_variable(input_var)
        if input_value is None:
            input_value = ctx.original_query
        
        ctx.debug_log("  LLM step - input_var", input_var)
        ctx.debug_log("  LLM step - input_value", input_value)
        ctx.debug_log("  LLM step - prompt_template", prompt_template)
        
        # Build the full prompt
        if prompt_template:
            # If template has variables, interpolate. Otherwise append input.
            if '{$' in prompt_template or '$Query' in prompt_template or '$Var[' in prompt_template or '$PreviousResult' in prompt_template:
                prompt = ctx.interpolate(prompt_template)
            else:
                prompt = f"{prompt_template}\n\n{input_value}"
        else:
            # No template, just use input value
            prompt = str(input_value)
        
        if system_prompt:
            system_prompt = ctx.interpolate(system_prompt)
        
        ctx.debug_log("  LLM step - final prompt", prompt)
        
        # Call LLM
        response = await self._llm_func(prompt, system_prompt)
        
        ctx.debug_log("  LLM step - response", response)
        
        # Store result
        ctx.previous_result = response
        
        # Only update current_value if NOT storing to a named variable
        # This preserves the main flow (original query) when steps are just 
        # computing intermediate values
        if output_var:
            ctx.set_variable(output_var, response)
        else:
            ctx.current_value = response
        
        return ctx
    
    async def _prim_query(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Ask LLM to generate a query (specialized to_llm)."""
        if not self._llm_func:
            raise RuntimeError("LLM function not configured")
        
        prompt_template = config.get("prompt", "Generate a search query for this:")
        output_var = config.get("output_var", "GeneratedQuery")
        input_var = config.get("input_var", "$Query")  # What data to use
        
        # Resolve the input variable
        input_value = ctx.resolve_variable(input_var)
        if input_value is None:
            input_value = ctx.original_query
        
        ctx.debug_log("  Query step - input_var", input_var)
        ctx.debug_log("  Query step - input_value", input_value)
        ctx.debug_log("  Query step - template", prompt_template)
        
        # Build the full prompt: template + input value
        # If template already has variables, interpolate. Otherwise append input.
        if '{$' in prompt_template or '$Query' in prompt_template or '$Var[' in prompt_template or '$PreviousResult' in prompt_template:
            prompt = ctx.interpolate(prompt_template)
        else:
            prompt = f"{prompt_template}\n\n{input_value}"
        
        system = "You are a query generator. Output only the query, nothing else."
        
        ctx.debug_log("  Query step - final prompt", prompt)
        
        response = await self._llm_func(prompt, system)
        
        # Clean up
        query = response.strip().strip('"\'')
        
        ctx.debug_log("  Query step - LLM response", query)
        
        ctx.previous_result = query
        # Query step always stores to a variable, so don't overwrite current_value
        ctx.set_variable(output_var, query)
        
        return ctx
    
    async def _prim_to_tool(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Execute a tool."""
        if not self._tool_func:
            raise RuntimeError("Tool function not configured")
        
        tool_name = config.get("tool_name", config.get("tool"))
        params_template = config.get("params", {})
        output_var = config.get("output_var")
        add_to_context = config.get("add_to_context", False)
        context_label = config.get("context_label", tool_name)
        
        ctx.debug_log("  Tool step - tool_name", tool_name)
        ctx.debug_log("  Tool step - params_template", params_template)
        
        # Build params with interpolation (use interpolate_value for typed results like arrays)
        params = {}
        for key, value in params_template.items():
            if isinstance(value, str):
                # Use interpolate_value to preserve types (arrays, etc.)
                interpolated = ctx.interpolate_value(value)
                ctx.debug_log(f"  Tool step - param '{key}': '{value}' -> {repr(interpolated)}")
                params[key] = interpolated
            else:
                params[key] = value
        
        # If no params but we have a previous result, use it as query
        if not params and ctx.previous_result:
            params = {"query": str(ctx.previous_result)}
            ctx.debug_log("  Tool step - using previous_result as query", params)
        
        ctx.debug_log("  Tool step - final params", params)
        
        # Call tool
        result = await self._tool_func(tool_name, params)
        
        ctx.previous_result = result
        
        # Only update current_value if NOT storing to a named variable
        if output_var:
            ctx.set_variable(output_var, result)
        else:
            ctx.current_value = result
        
        if add_to_context and result:
            ctx.add_context(str(result), context_label, tool_name)
        
        return ctx
    
    async def _prim_from_llm(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Process LLM output (placeholder for bidirectional)."""
        # In bidirectional mode, this receives the LLM response
        # For now, just pass through
        return ctx
    
    async def _prim_from_tool(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Process tool result."""
        # Can be used to transform tool results
        transform = config.get("transform")
        if transform:
            ctx.current_value = ctx.interpolate(transform)
            ctx.previous_result = ctx.current_value
        return ctx
    
    async def _prim_context_insert(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Add content to accumulated context."""
        source_var = config.get("source_var", "$PreviousResult")
        label = config.get("label")
        
        content = ctx.resolve_variable(source_var)
        if isinstance(content, str):
            content = ctx.interpolate(content)
        
        if content:
            ctx.add_context(str(content), label)
        
        return ctx
    
    async def _prim_go_to_llm(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Set up final message for main LLM call."""
        input_var = config.get("input_var", "$Query")  # Default to original query
        include_context = config.get("include_context", True)
        include_vars = config.get("include_vars", [])  # List of variable names to include
        custom_template = config.get("template")  # Optional custom template
        
        ctx.debug_log("  go_to_llm - input_var", input_var)
        ctx.debug_log("  go_to_llm - include_context", include_context)
        ctx.debug_log("  go_to_llm - include_vars", include_vars)
        ctx.debug_log("  go_to_llm - context_items count", len(ctx.context_items))
        
        # Get the content to send to LLM (usually original query)
        content = ctx.resolve_variable(input_var)
        if content is None:
            content = ctx.original_query
        if isinstance(content, str):
            content = ctx.interpolate(content)
        
        ctx.debug_log("  go_to_llm - resolved content", content)
        
        # Build variable section if include_vars specified
        vars_section = ""
        if include_vars:
            var_lines = []
            for var_name in include_vars:
                # Handle both "$Var[name]" format and plain "name"
                if var_name.startswith("$Var[") and var_name.endswith("]"):
                    actual_name = var_name[5:-1]
                else:
                    actual_name = var_name
                value = ctx.variables.get(actual_name, "")
                if value:
                    var_lines.append(f"{actual_name}: {value}")
            if var_lines:
                vars_section = "Additional Information:\n" + "\n".join(var_lines) + "\n\n"
        
        # Build final message
        if custom_template:
            # Use custom template with interpolation
            ctx.final_message = ctx.interpolate(custom_template)
        elif include_context and ctx.context_items:
            context_str = ctx.get_context_string()
            ctx.debug_log("  go_to_llm - context_str length", len(context_str))
            ctx.final_message = f"Context:\n{context_str}\n\n{vars_section}User Question: {content}"
        else:
            # No context items - just use vars and query
            ctx.final_message = f"{vars_section}{content}" if vars_section else str(content)
        
        ctx.debug_log("  go_to_llm - final_message length", len(ctx.final_message))
        ctx.proceed_to_llm = True
        # Stop chain execution - we're done, proceed to main LLM
        ctx.signal = FlowSignal.FILTER_COMPLETE
        return ctx
    
    async def _prim_filter_complete(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Mark filter as complete, skip remaining steps."""
        action = config.get("action", "go_to_llm")
        
        if action == "go_to_llm":
            ctx = await self._prim_go_to_llm(config, ctx)
        elif action == "stop":
            ctx.proceed_to_llm = False
        
        ctx.signal = FlowSignal.FILTER_COMPLETE
        return ctx
    
    async def _prim_set_var(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Set a variable."""
        var_name = config.get("var_name", config.get("name"))
        value = config.get("value", "")
        
        ctx.debug_log("  set_var - config", config)
        ctx.debug_log("  set_var - var_name", var_name)
        ctx.debug_log("  set_var - value template", value)
        
        if not var_name:
            ctx.debug_log("  set_var - SKIPPED (no var_name)")
            return ctx
        
        resolved = ctx.interpolate(str(value))
        ctx.debug_log("  set_var - resolved value", resolved)
        ctx.set_variable(var_name, resolved)
        ctx.debug_log("  set_var - SET", f"{var_name} = {resolved[:100] if len(str(resolved)) > 100 else resolved}")
        
        return ctx
    
    async def _prim_set_array(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Set an array variable from multiple values."""
        var_name = config.get("var_name", config.get("name"))
        values = config.get("values", [])
        
        ctx.debug_log("  set_array - var_name", var_name)
        ctx.debug_log("  set_array - values template", values)
        
        if not var_name:
            ctx.debug_log("  set_array - SKIPPED (no var_name)")
            return ctx
        
        # Resolve each value using interpolate_value to preserve types
        resolved_array = []
        for i, value in enumerate(values):
            if isinstance(value, str) and value.strip():
                resolved = ctx.interpolate_value(value)
                ctx.debug_log(f"  set_array - [{i}]: '{value}' -> {repr(resolved)}")
                if resolved is not None and resolved != "":
                    resolved_array.append(resolved)
            elif value is not None and value != "":
                resolved_array.append(value)
        
        ctx.debug_log("  set_array - final array", resolved_array)
        ctx.set_variable(var_name, resolved_array)
        ctx.previous_result = resolved_array
        # Don't overwrite current_value - set_array stores to a named variable
        
        return ctx
    
    async def _prim_compare(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Perform a comparison and store result."""
        output_var = config.get("output_var", "CompareResult")
        
        result = self._evaluate_comparison(config, ctx)
        ctx.set_variable(output_var, result)
        ctx.previous_result = result
        
        return ctx
    
    async def _prim_branch(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """
        Branch to different outputs based on conditions.
        
        Config format:
        {
            "outputs": [
                {
                    "id": "output_1",
                    "label": "Contains keyword",
                    "condition": {"var": "$Query", "operator": "contains", "value": "help"},
                    "jump_to": "step_id_1"
                },
                {
                    "id": "output_2", 
                    "label": "Is empty",
                    "condition": {"var": "$Var[result]", "operator": "is_empty", "value": ""},
                    "jump_to": "step_id_2"
                },
                {
                    "id": "output_else",
                    "label": "Else",
                    "jump_to": "step_id_3"  # No condition = else/default
                }
            ]
        }
        
        First matching condition wins. Last output without condition is the "else" case.
        """
        outputs = config.get("outputs", [])
        
        if not outputs:
            ctx.debug_log("  Branch: No outputs configured, continuing")
            return ctx
        
        ctx.debug_log(f"  Branch: Evaluating {len(outputs)} outputs")
        
        matched_output = None
        
        for i, output in enumerate(outputs):
            output_id = output.get("id", f"output_{i}")
            output_label = output.get("label", f"Output {i + 1}")
            condition = output.get("condition")
            
            # If no condition, this is the "else" case - only use if it's the last output
            if not condition or not condition.get("var"):
                if i == len(outputs) - 1:
                    ctx.debug_log(f"    Output '{output_label}': No condition (else case) - MATCHED")
                    matched_output = output
                    break
                else:
                    ctx.debug_log(f"    Output '{output_label}': No condition, skipping (not last)")
                    continue
            
            # Evaluate the condition
            # Convert condition format to what _evaluate_comparison expects
            comp = {
                "left": condition.get("var", ""),
                "operator": condition.get("operator", "contains"),
                "right": condition.get("value", ""),
            }
            
            result = self._evaluate_comparison(comp, ctx)
            left_val = ctx.resolve_variable(comp["left"])
            
            ctx.debug_log(f"    Output '{output_label}': {comp['left']} ({left_val}) {comp['operator']} '{comp['right']}' = {result}")
            
            if result:
                ctx.debug_log(f"    Output '{output_label}': MATCHED")
                matched_output = output
                break
        
        # Handle the matched output
        if matched_output:
            jump_to = matched_output.get("jump_to")
            if jump_to:
                ctx.debug_log(f"  Branch: Jumping to {jump_to}")
                ctx.signal = FlowSignal.JUMP
                ctx.jump_target = jump_to
            else:
                ctx.debug_log(f"  Branch: No jump target, continuing to next step")
        else:
            ctx.debug_log(f"  Branch: No output matched, continuing to next step")
        
        return ctx
    
    async def _prim_call_chain(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Call another chain as a subroutine."""
        chain_name = config.get("chain_name")
        
        if not chain_name:
            logger.warning("call_chain: no chain_name specified")
            return ctx
        
        if not self._chain_loader:
            logger.warning("call_chain: no chain loader configured")
            return ctx
        
        # Load the chain
        sub_chain = self._chain_loader(chain_name)
        if not sub_chain:
            logger.warning(f"call_chain: chain '{chain_name}' not found")
            return ctx
        
        # Execute it with current context state
        sub_result = await self.execute(
            sub_chain,
            str(ctx.current_value),
            ctx.user_id,
            ctx.chat_id,
        )
        
        # Merge results back
        ctx.current_value = sub_result.content
        ctx.previous_result = sub_result.content
        ctx.context_items.extend(sub_result.context.context_items)
        ctx.variables.update(sub_result.context.variables)
        
        if sub_result.context.signal == FlowSignal.FILTER_COMPLETE:
            ctx.signal = FlowSignal.FILTER_COMPLETE
            ctx.final_message = sub_result.content
        
        return ctx
    
    async def _prim_stop(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Stop chain execution."""
        response = config.get("response")
        proceed = config.get("proceed_to_llm", False)
        
        if response:
            ctx.final_message = ctx.interpolate(response)
        
        ctx.proceed_to_llm = proceed
        ctx.signal = FlowSignal.STOP
        return ctx
    
    async def _prim_block(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Block the request."""
        reason = config.get("reason", "Blocked by filter")
        
        ctx.error_message = reason
        ctx.proceed_to_llm = False
        ctx.signal = FlowSignal.ERROR
        return ctx
    
    async def _prim_modify(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Modify current content using template."""
        template = config.get("template", "{$Current}")
        output_var = config.get("output_var")
        
        result = ctx.interpolate(template)
        ctx.current_value = result
        ctx.previous_result = result
        
        if output_var:
            ctx.set_variable(output_var, result)
        
        return ctx
    
    async def _prim_log(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """Log a message (for debugging chains)."""
        message = config.get("message", "")
        level = config.get("level", "debug")
        
        resolved = ctx.interpolate(message)
        getattr(logger, level, logger.debug)(f"[ChainLog] {resolved}")
        
        return ctx
    
    async def _prim_call_tool(self, config: dict, ctx: ExecutionContext) -> ExecutionContext:
        """
        Simplified tool call primitive.
        
        Supports two formats:
        1. String format: "tool_name" - calls tool with $Query as default param
        2. Object format: {"name": "tool_name", "query": "...", "param1": "value1"}
        
        Config options:
        - name/tool: Tool name (required)
        - query: Query parameter (defaults to $Query or $PreviousResult)
        - Any other keys become tool parameters
        - output_var: Variable to store result
        - add_to_context: Add result to context for LLM
        - context_label: Label for context
        
        Example usage in chain:
        {"type": "call_tool", "config": "web_search"}
        {"type": "call_tool", "config": {"name": "web_search", "query": "{Query}"}}
        """
        if not self._tool_func:
            raise RuntimeError("Tool function not configured")
        
        # Handle string config (just tool name)
        if isinstance(config, str):
            config = {"name": config}
        
        tool_name = config.get("name") or config.get("tool") or config.get("tool_name")
        if not tool_name:
            ctx.debug_log("  call_tool - SKIPPED (no tool name)")
            return ctx
        
        output_var = config.get("output_var")
        add_to_context = config.get("add_to_context", False)
        context_label = config.get("context_label", tool_name)
        
        ctx.debug_log("  call_tool - tool", tool_name)
        
        # Build params from remaining config keys
        params = {}
        reserved_keys = {"name", "tool", "tool_name", "output_var", "add_to_context", "context_label"}
        
        for key, value in config.items():
            if key not in reserved_keys:
                if isinstance(value, str):
                    params[key] = ctx.interpolate(value)
                else:
                    params[key] = value
        
        # If no query param, use previous result or original query
        if "query" not in params:
            if ctx.previous_result:
                params["query"] = str(ctx.previous_result)
            else:
                params["query"] = ctx.original_query
        
        ctx.debug_log("  call_tool - params", params)
        
        # Call tool
        result = await self._tool_func(tool_name, params)
        
        ctx.previous_result = result
        
        if output_var:
            ctx.set_variable(output_var, result)
        else:
            ctx.current_value = result
        
        if add_to_context and result:
            ctx.add_context(str(result), context_label, tool_name)
        
        ctx.debug_log("  call_tool - result", str(result)[:200] if result else None)
        
        return ctx
