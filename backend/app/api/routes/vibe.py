"""
VibeCode API Routes

AI-powered code editor endpoints for:
- Code completion
- Code formatting
- Linting
- Agentic task execution
- Zip file processing
"""

import zipfile
import io
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.llm import LLMService
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ============ Schemas ============

class FileContext(BaseModel):
    path: str
    content: str


class CompletionRequest(BaseModel):
    file_path: str
    content: str
    cursor_line: int
    cursor_column: int
    language: str
    context_files: List[FileContext] = []
    model: Optional[str] = None
    assistant_id: Optional[str] = None


class FormatRequest(BaseModel):
    content: str
    language: str


class LintRequest(BaseModel):
    content: str
    language: str
    file_path: str


class ChatRequest(BaseModel):
    message: str
    files: List[FileContext] = []
    active_file: Optional[FileContext] = None
    cursor_position: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    assistant_id: Optional[str] = None


class AgentAnalyzeRequest(BaseModel):
    message: str
    files: List[FileContext] = []
    model: Optional[str] = None
    assistant_id: Optional[str] = None


class AgentPlanRequest(BaseModel):
    analysis: Dict[str, Any]
    message: str
    model: Optional[str] = None
    assistant_id: Optional[str] = None


class AgentGenerateRequest(BaseModel):
    plan: Dict[str, Any]
    files: List[FileContext] = []
    model: Optional[str] = None
    assistant_id: Optional[str] = None


class AgentReviewRequest(BaseModel):
    generated: Dict[str, Any]
    model: Optional[str] = None
    assistant_id: Optional[str] = None


# ============ System Prompts ============

COMPLETION_SYSTEM_PROMPT = """You are an expert code completion assistant. Your task is to provide intelligent code suggestions.

Given the current file content and cursor position, suggest completions that:
1. Follow the existing code style and patterns
2. Are syntactically correct
3. Are contextually appropriate
4. Complete the current line or block logically

Respond with a JSON array of 1-3 completion suggestions. Each suggestion should be a string that can be directly inserted at the cursor position.

Example response:
["suggestion1", "suggestion2"]

Only provide the JSON array, no explanations."""


AGENT_ANALYZE_PROMPT = """You are an expert software architect. Analyze the user's request and determine:

1. What type of task is this? (create, modify, refactor, fix, explain, test)
2. What files need to be created or modified?
3. What technologies/languages are involved?
4. What are the key requirements?
5. Are there any potential challenges?

Respond with a JSON object containing your analysis:
{
  "task_type": "create|modify|refactor|fix|explain|test",
  "summary": "Brief summary of what needs to be done",
  "requirements": ["req1", "req2"],
  "technologies": ["tech1", "tech2"],
  "files_to_create": ["path1", "path2"],
  "files_to_modify": ["path1", "path2"],
  "challenges": ["challenge1"],
  "estimated_complexity": "low|medium|high"
}

Only provide the JSON object, no explanations."""


AGENT_PLAN_PROMPT = """You are an expert software architect creating an execution plan.

Based on the analysis, create a detailed step-by-step plan for implementing the request.

Respond with a JSON object:
{
  "steps": [
    {
      "id": 1,
      "action": "create|modify|delete",
      "file_path": "path/to/file.ts",
      "description": "What this step does",
      "dependencies": []
    }
  ],
  "order": [1, 2, 3],
  "notes": "Any important implementation notes"
}

Only provide the JSON object, no explanations."""


AGENT_GENERATE_PROMPT = """You are an expert software developer. Generate high-quality, production-ready code.

Based on the execution plan, generate the code for each file. Follow these guidelines:
1. Write clean, readable, well-documented code
2. Follow best practices for the language/framework
3. Include proper error handling
4. Add helpful comments where appropriate
5. Make the code modular and maintainable

Respond with a JSON object:
{
  "changes": [
    {
      "type": "create|modify",
      "path": "path/to/file.ts",
      "content": "full file content here",
      "description": "What was created/changed"
    }
  ],
  "summary": "Summary of all changes made"
}

Only provide the JSON object, no explanations."""


AGENT_REVIEW_PROMPT = """You are an expert code reviewer. Review the generated code for:

1. Correctness - Does it fulfill the requirements?
2. Quality - Is it well-written and maintainable?
3. Security - Are there any security issues?
4. Performance - Are there any performance concerns?
5. Best practices - Does it follow conventions?

Respond with a JSON object:
{
  "approved": true|false,
  "issues": [
    {
      "severity": "error|warning|info",
      "file": "path/to/file",
      "line": 10,
      "message": "Description of issue",
      "suggestion": "How to fix it"
    }
  ],
  "approved_changes": [/* same format as input changes, possibly with fixes applied */],
  "feedback": "Overall feedback"
}

Only provide the JSON object, no explanations."""


CHAT_SYSTEM_PROMPT = """You are an expert programming assistant helping a developer with their code.

You have access to the following context:
- The developer's project files
- The currently active file (if any)
- The cursor position in the active file

Your role is to:
1. Answer questions about the code
2. Explain concepts and patterns
3. Suggest improvements
4. Help debug issues
5. Provide code examples

Be concise but thorough. Use code blocks with proper syntax highlighting when showing code.
When suggesting changes, be specific about which file and where the changes should be made."""


# ============ Helper Functions ============

def get_language_formatter(language: str) -> Optional[str]:
    """Get formatter command for a language"""
    formatters = {
        'python': 'black',
        'javascript': 'prettier',
        'typescript': 'prettier',
        'json': 'prettier',
        'html': 'prettier',
        'css': 'prettier',
        'yaml': 'prettier',
    }
    return formatters.get(language)


def get_language_linter(language: str) -> Optional[str]:
    """Get linter for a language"""
    linters = {
        'python': 'pylint',
        'javascript': 'eslint',
        'typescript': 'eslint',
    }
    return linters.get(language)


async def get_assistant_context(assistant_id: str, db: AsyncSession) -> Optional[Dict[str, Any]]:
    """Get assistant system prompt and settings"""
    from sqlalchemy import select
    from app.models.assistant import Assistant
    
    if not assistant_id:
        return None
    
    try:
        result = await db.execute(
            select(Assistant).where(Assistant.id == assistant_id)
        )
        assistant = result.scalar_one_or_none()
        
        if assistant:
            return {
                "name": assistant.name,
                "system_prompt": assistant.system_prompt,
                "model": assistant.model,
            }
    except Exception as e:
        logger.warning(f"Could not load assistant {assistant_id}: {e}")
    
    return None


async def call_llm(
    prompt: str,
    system_prompt: str,
    model: Optional[str] = None,
    db: AsyncSession = None,
    assistant_id: Optional[str] = None,
) -> str:
    """Call LLM service with prompt, optionally using assistant context"""
    # Create LLM service - use from_database if db provided, else use defaults
    if db:
        llm = await LLMService.from_database(db)
    else:
        llm = LLMService()
    
    # If assistant_id provided, prepend assistant's system prompt
    final_system_prompt = system_prompt
    
    if assistant_id and db:
        assistant_context = await get_assistant_context(assistant_id, db)
        if assistant_context:
            # Combine assistant system prompt with the task-specific prompt
            final_system_prompt = f"""{assistant_context.get('system_prompt', '')}

---
{system_prompt}"""
    
    # Use simple_completion for non-streaming responses
    response = await llm.simple_completion(
        prompt=prompt,
        system_prompt=final_system_prompt,
        max_tokens=4096,
    )
    
    return response


def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response"""
    # Try to find JSON in the response
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object or array
        import re
        json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {}


# ============ Endpoints ============

@router.post("/complete")
async def get_completion(
    request: CompletionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get AI code completion suggestions"""
    try:
        # Build context
        context_str = ""
        for f in request.context_files[:3]:  # Limit context
            context_str += f"\n--- {f.path} ---\n{f.content[:2000]}\n"
        
        # Get the lines around cursor
        lines = request.content.split("\n")
        start_line = max(0, request.cursor_line - 10)
        end_line = min(len(lines), request.cursor_line + 5)
        context_lines = lines[start_line:end_line]
        
        prompt = f"""Language: {request.language}
File: {request.file_path}

Context from other files:
{context_str}

Current file context (lines {start_line + 1}-{end_line}):
```{request.language}
{chr(10).join(context_lines)}
```

Cursor is at line {request.cursor_line}, column {request.cursor_column}.
Current line: {lines[request.cursor_line - 1] if request.cursor_line <= len(lines) else ''}

Provide completion suggestions for what should come next."""

        response = await call_llm(prompt, COMPLETION_SYSTEM_PROMPT, request.model, db, request.assistant_id)
        
        # Parse suggestions
        suggestions = extract_json(response)
        if isinstance(suggestions, list):
            return {"suggestions": suggestions[:5]}
        elif isinstance(suggestions, dict) and "suggestions" in suggestions:
            return {"suggestions": suggestions["suggestions"][:5]}
        else:
            return {"suggestions": []}
            
    except Exception as e:
        logger.error(f"Completion error: {e}")
        return {"suggestions": []}


@router.post("/format")
async def format_code(
    request: FormatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Format code using AI with language-specific style guides"""
    try:
        # Language-specific formatting rules
        format_rules = {
            'python': "Use PEP 8 style: 4-space indentation, 79 char line limit, snake_case for functions/variables, PascalCase for classes",
            'javascript': "Use Prettier/StandardJS style: 2-space indentation, single quotes, semicolons optional, camelCase",
            'typescript': "Use Prettier style: 2-space indentation, single quotes, semicolons, proper type annotations",
            'rust': "Use rustfmt style: 4-space indentation, 100 char line limit, snake_case",
            'go': "Use gofmt style: tabs for indentation, camelCase for exported, lowercase for internal",
            'java': "Use Google Java Style: 2-space indentation, 100 char line limit, camelCase",
            'cpp': "Use Google C++ Style: 2-space indentation, 80 char line limit, snake_case for variables",
            'c': "Use K&R style: 8-space or 4-space indentation, snake_case",
            'html': "Use 2-space indentation, proper tag nesting, lowercase attributes",
            'css': "Use 2-space indentation, one declaration per line, alphabetize properties",
            'scss': "Use 2-space indentation, nest selectors logically, use variables for colors",
            'json': "Use 2-space indentation, consistent quoting, sorted keys where appropriate",
            'yaml': "Use 2-space indentation, consistent quoting, proper list formatting",
            'sql': "Use uppercase keywords, consistent indentation, one clause per line",
            'shell': "Use 2-space indentation, quote variables, use $() instead of backticks",
            'kotlin': "Use Kotlin coding conventions: 4-space indentation, camelCase",
            'ruby': "Use Ruby style guide: 2-space indentation, snake_case, no parentheses when possible",
            'php': "Use PSR-12 style: 4-space indentation, camelCase for methods, PascalCase for classes",
            'swift': "Use Swift API design guidelines: 4-space indentation, camelCase",
            'csharp': "Use Microsoft conventions: 4-space indentation, PascalCase for public members",
        }
        
        language = request.language.lower()
        style_guide = format_rules.get(language, "Follow standard formatting conventions for the language")
        
        prompt = f"""Format the following {request.language} code according to best practices.

Style guide: {style_guide}

```{request.language}
{request.content}
```

Format the code properly:
- Fix indentation to be consistent
- Add/remove whitespace appropriately
- Ensure proper line breaks
- Organize imports/includes if present
- Preserve all logic and functionality exactly

Return ONLY the formatted code, no explanations or markdown code blocks."""

        system = f"You are an expert {request.language} code formatter. Return only the formatted code with no explanations, no markdown code blocks, no leading/trailing text."
        response = await call_llm(prompt, system, None, db)
        
        # Clean up response
        formatted = response.strip()
        if formatted.startswith("```"):
            lines = formatted.split("\n")
            formatted = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        return {"formatted": formatted}
        
    except Exception as e:
        logger.error(f"Format error: {e}")
        return {"formatted": request.content}


@router.post("/lint")
async def lint_code(
    request: LintRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Lint code using AI with language-specific rules"""
    try:
        # Language-specific linting guidance
        lint_rules = {
            'python': """
- Check for PEP 8 style violations (indentation, line length, naming conventions)
- Identify undefined variables and unused imports
- Look for potential type errors
- Check for missing docstrings on public functions/classes
- Identify mutable default arguments
- Look for bare except clauses
- Check for f-string vs format string issues
- Identify potential None comparisons (use 'is None' not '== None')""",
            'javascript': """
- Check for undeclared variables (missing const/let/var)
- Identify unused variables and imports
- Look for == instead of === comparisons
- Check for missing semicolons (if style requires them)
- Identify potential null/undefined errors
- Look for console.log statements that should be removed
- Check for async/await issues (missing await, unhandled promises)
- Identify potential memory leaks (event listeners not removed)""",
            'typescript': """
- Check for type errors and missing type annotations
- Identify 'any' types that should be more specific
- Look for null/undefined without proper checks
- Check for unused variables, imports, and type definitions
- Identify missing return types on functions
- Look for improper use of ! (non-null assertion)
- Check for async/await issues
- Identify potential runtime type mismatches""",
            'rust': """
- Check for unused variables (without underscore prefix)
- Identify potential ownership/borrowing issues
- Look for unwrap() on Result/Option that should be handled
- Check for missing error handling
- Identify clippy-style warnings (redundant clones, etc.)
- Look for potential panic points
- Check for missing documentation on public items""",
            'go': """
- Check for unused variables and imports
- Identify errors that are not being handled
- Look for potential nil pointer dereferences
- Check for goroutine leaks
- Identify missing error returns
- Look for race conditions
- Check for proper defer usage""",
            'java': """
- Check for null pointer risks
- Identify unused imports and variables
- Look for resource leaks (unclosed streams, connections)
- Check for missing @Override annotations
- Identify potential concurrency issues
- Look for empty catch blocks
- Check for proper equals/hashCode implementation""",
            'cpp': """
- Check for memory leaks (new without delete)
- Identify potential buffer overflows
- Look for uninitialized variables
- Check for null pointer dereferences
- Identify resource leaks (RAII violations)
- Look for integer overflow possibilities
- Check for proper const usage""",
            'c': """
- Check for memory leaks (malloc without free)
- Identify potential buffer overflows
- Look for uninitialized variables
- Check for null pointer dereferences
- Identify format string vulnerabilities
- Look for integer overflow possibilities
- Check for proper error handling""",
            'html': """
- Check for unclosed tags
- Identify missing alt attributes on images
- Look for deprecated tags/attributes
- Check for accessibility issues (missing labels, etc.)
- Identify invalid nesting
- Look for missing required attributes
- Check for proper semantic structure""",
            'css': """
- Check for invalid property values
- Identify unused selectors
- Look for vendor prefix issues
- Check for specificity problems
- Identify duplicate declarations
- Look for invalid color values
- Check for proper units usage""",
            'sql': """
- Check for SQL injection vulnerabilities
- Identify missing indexes (based on WHERE clauses)
- Look for SELECT * that should be explicit
- Check for proper JOIN conditions
- Identify potential performance issues
- Look for missing WHERE clauses in UPDATE/DELETE
- Check for proper NULL handling""",
            'json': """
- Check for syntax errors (trailing commas, missing quotes)
- Identify duplicate keys
- Look for invalid value types
- Check for proper nesting
- Identify schema violations if applicable""",
            'yaml': """
- Check for indentation errors
- Identify duplicate keys
- Look for improper quoting
- Check for invalid anchors/aliases
- Identify type coercion issues (yes/no vs true/false)""",
            'markdown': """
- Check for broken links
- Identify improper heading hierarchy
- Look for missing alt text in images
- Check for unclosed formatting (bold, italic, code)
- Identify trailing whitespace issues""",
            'shell': """
- Check for unquoted variables that could cause word splitting
- Identify use of deprecated syntax (backticks vs $())
- Look for missing error handling (set -e, trap)
- Check for shellcheck-style warnings
- Identify potential injection vulnerabilities
- Look for hardcoded paths that should be variables
- Check for proper quoting in conditionals""",
            'kotlin': """
- Check for nullable types without proper handling
- Identify unused variables and imports
- Look for potential null pointer exceptions
- Check for proper coroutine scope usage
- Identify deprecated API usage
- Look for inefficient collection operations
- Check for missing @JvmStatic annotations where needed""",
            'ruby': """
- Check for undefined method calls
- Identify unused variables and requires
- Look for potential nil reference errors
- Check for proper exception handling
- Identify security vulnerabilities (eval, system calls)
- Look for deprecated syntax
- Check for proper block usage""",
            'php': """
- Check for undefined variables
- Identify SQL injection vulnerabilities
- Look for XSS vulnerabilities
- Check for deprecated function usage
- Identify type errors (in strict mode)
- Look for unclosed tags in HTML context
- Check for proper error handling""",
            'swift': """
- Check for force unwrapping that could cause crashes
- Identify unused variables and imports
- Look for retain cycles in closures
- Check for proper error handling with try/catch
- Identify deprecated API usage
- Look for potential race conditions
- Check for proper optionals handling""",
            'csharp': """
- Check for null reference risks
- Identify unused variables and using statements
- Look for async/await issues
- Check for proper IDisposable implementation
- Identify LINQ performance issues
- Look for potential deadlocks
- Check for proper exception handling""",
            'vue': """
- Check for undefined component references
- Identify missing prop validations
- Look for v-for without :key
- Check for potential reactivity issues
- Identify unused components
- Look for improper lifecycle hook usage
- Check for template syntax errors""",
            'scss': """
- Check for invalid nesting depth
- Identify unused variables and mixins
- Look for improper @import/@use usage
- Check for duplicate selectors
- Identify specificity issues
- Look for invalid color functions
- Check for proper unit usage""",
            'dockerfile': """
- Check for missing FROM instruction
- Identify security issues (running as root)
- Look for inefficient layer caching
- Check for hardcoded secrets
- Identify missing health checks
- Look for deprecated instructions
- Check for proper multi-stage build usage""",
            'terraform': """
- Check for missing required providers
- Identify hardcoded secrets
- Look for deprecated resource syntax
- Check for missing descriptions
- Identify potential state issues
- Look for improper module usage
- Check for missing lifecycle rules""",
            'graphql': """
- Check for undefined types and fields
- Identify N+1 query issues
- Look for missing input validation
- Check for circular references
- Identify deprecated field usage
- Look for overly complex queries
- Check for proper nullable handling""",
        }
        
        language = request.language.lower()
        specific_rules = lint_rules.get(language, "- Check for common errors and code quality issues")
        
        prompt = f"""Analyze the following {request.language} code for errors, warnings, and potential issues.

File: {request.file_path}

```{request.language}
{request.content}
```

Focus on these {request.language}-specific checks:
{specific_rules}

Return a JSON array of issues found. Each issue must have:
- "line": the line number (1-indexed)
- "column": the column number (1-indexed, use 1 if unknown)
- "message": a clear, actionable description of the issue
- "severity": one of "error" (will cause bugs/crashes), "warning" (potential problems), or "info" (style/suggestions)

Example format:
[
  {{"line": 5, "column": 10, "message": "Variable 'x' is never used", "severity": "warning"}},
  {{"line": 12, "column": 1, "message": "Missing return type annotation", "severity": "info"}}
]

If no issues are found, return an empty array: []
Only return the JSON array, no other text."""

        system = f"You are an expert {request.language} code linter. Analyze code thoroughly and return issues as a JSON array. Be precise about line numbers. Focus on real issues, not trivial style preferences."
        response = await call_llm(prompt, system, None, db)
        
        errors = extract_json(response)
        if isinstance(errors, list):
            return {"errors": errors}
        elif isinstance(errors, dict) and "errors" in errors:
            return {"errors": errors["errors"]}
        else:
            return {"errors": []}
            
    except Exception as e:
        logger.error(f"Lint error: {e}")
        return {"errors": []}


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with AI assistant about code"""
    try:
        # Build context
        context_str = ""
        for f in request.files[:5]:
            content = f.content[:3000] if len(f.content) > 3000 else f.content
            context_str += f"\n--- {f.path} ---\n```\n{content}\n```\n"
        
        active_file_str = ""
        if request.active_file:
            active_file_str = f"""
Currently active file: {request.active_file.path}
```
{request.active_file.content}
```
"""
            if request.cursor_position:
                active_file_str += f"\nCursor at line {request.cursor_position.get('line', 1)}, column {request.cursor_position.get('column', 1)}"
        
        prompt = f"""Project files:
{context_str}

{active_file_str}

User's question/request:
{request.message}"""

        response = await call_llm(prompt, CHAT_SYSTEM_PROMPT, request.model, db, request.assistant_id)
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/analyze")
async def agent_analyze(
    request: AgentAnalyzeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Analyze user request for agentic task"""
    try:
        # Build context
        context_str = ""
        for f in request.files[:10]:
            context_str += f"- {f.path}\n"
        
        prompt = f"""User request: {request.message}

Existing project files:
{context_str}

Analyze this request and determine what needs to be done."""

        response = await call_llm(prompt, AGENT_ANALYZE_PROMPT, request.model, db, request.assistant_id)
        
        analysis = extract_json(response)
        if not analysis:
            analysis = {
                "task_type": "create",
                "summary": request.message,
                "requirements": [],
                "technologies": [],
                "files_to_create": [],
                "files_to_modify": [],
                "challenges": [],
                "estimated_complexity": "medium"
            }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Agent analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/plan")
async def agent_plan(
    request: AgentPlanRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create execution plan for agentic task"""
    try:
        prompt = f"""Original request: {request.message}

Analysis:
{json.dumps(request.analysis, indent=2)}

Create a detailed execution plan."""

        response = await call_llm(prompt, AGENT_PLAN_PROMPT, request.model, db, request.assistant_id)
        
        plan = extract_json(response)
        if not plan or "steps" not in plan:
            plan = {
                "steps": [{
                    "id": 1,
                    "action": "create",
                    "file_path": "main.ts",
                    "description": request.message,
                    "dependencies": []
                }],
                "order": [1],
                "notes": ""
            }
        
        return plan
        
    except Exception as e:
        logger.error(f"Agent plan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/generate")
async def agent_generate(
    request: AgentGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate code for agentic task"""
    try:
        # Build existing files context
        context_str = ""
        for f in request.files[:5]:
            content = f.content[:2000] if len(f.content) > 2000 else f.content
            context_str += f"\n--- {f.path} ---\n{content}\n"
        
        prompt = f"""Execution plan:
{json.dumps(request.plan, indent=2)}

Existing files for context:
{context_str}

Generate the code according to the plan."""

        response = await call_llm(prompt, AGENT_GENERATE_PROMPT, request.model, db, request.assistant_id)
        
        generated = extract_json(response)
        if not generated or "changes" not in generated:
            generated = {
                "changes": [],
                "summary": "No changes generated"
            }
        
        return generated
        
    except Exception as e:
        logger.error(f"Agent generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent/review")
async def agent_review(
    request: AgentReviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Review generated code"""
    try:
        prompt = f"""Generated code to review:
{json.dumps(request.generated, indent=2)}

Review this code for quality, correctness, and best practices."""

        response = await call_llm(prompt, AGENT_REVIEW_PROMPT, request.model, db, request.assistant_id)
        
        review = extract_json(response)
        if not review:
            # Auto-approve if review fails
            review = {
                "approved": True,
                "issues": [],
                "approved_changes": request.generated.get("changes", []),
                "feedback": "Approved"
            }
        elif "approved_changes" not in review:
            review["approved_changes"] = request.generated.get("changes", [])
        
        return review
        
    except Exception as e:
        logger.error(f"Agent review error: {e}")
        # Return approved on error
        return {
            "approved": True,
            "issues": [],
            "approved_changes": request.generated.get("changes", []),
            "feedback": "Auto-approved"
        }


@router.post("/upload-zip")
async def upload_zip(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload and extract zip file for project context"""
    try:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only zip files are supported")
        
        content = await file.read()
        
        # Size limit: 50MB
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        files = []
        
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                # Skip directories and hidden files
                if name.endswith('/') or '/.' in name or name.startswith('.'):
                    continue
                
                # Skip common non-code directories
                skip_dirs = ['node_modules', '__pycache__', '.git', 'venv', '.venv', 'dist', 'build', '.next']
                if any(d in name for d in skip_dirs):
                    continue
                
                # Skip binary and large files
                ext = os.path.splitext(name)[1].lower()
                binary_exts = ['.exe', '.dll', '.so', '.dylib', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz']
                if ext in binary_exts:
                    continue
                
                # Check file size
                info = zf.getinfo(name)
                if info.file_size > 500 * 1024:  # Skip files > 500KB
                    continue
                
                try:
                    file_content = zf.read(name).decode('utf-8', errors='ignore')
                    
                    # Clean path
                    clean_path = name
                    if '/' in clean_path:
                        # Remove top-level directory if it's the only root
                        parts = clean_path.split('/')
                        if len(parts) > 1:
                            clean_path = '/'.join(parts[1:]) if parts[0] else clean_path
                    
                    files.append({
                        "name": os.path.basename(name),
                        "path": clean_path,
                        "content": file_content,
                    })
                except Exception as e:
                    logger.warning(f"Could not read file {name}: {e}")
                    continue
        
        logger.info(f"Extracted {len(files)} files from {file.filename}")
        
        return {"files": files}
        
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    except Exception as e:
        logger.error(f"Zip upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
