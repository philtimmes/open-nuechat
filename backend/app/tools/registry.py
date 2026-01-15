"""
Integrated Tool System for LLM
Provides built-in tools that Claude can use
"""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import logging
import math
import re
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag import RAGService
from app.models.models import User

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available tools"""
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._register_builtin_tools()
    
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
    ):
        """Register a tool"""
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": parameters,
                "required": [k for k, v in parameters.items() if v.get("required", False)],
            }
        }
        self._handlers[name] = handler
    
    def get_tool_definitions(self, tool_names: Optional[List[str]] = None) -> List[Dict]:
        """Get tool definitions for API call"""
        if tool_names is None:
            return list(self._tools.values())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """Get handler for a tool"""
        return self._handlers.get(name)
    
    async def execute(self, name: str, arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Any:
        """Execute a tool"""
        handler = self._handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}
        
        try:
            if context:
                return await handler(arguments, context)
            return await handler(arguments)
        except Exception as e:
            return {"error": str(e)}
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        
        # Calculator tool
        self.register(
            name="calculator",
            description="Perform mathematical calculations. Supports basic arithmetic, trigonometry, logarithms, and more.",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'log(100)')",
                    "required": True,
                }
            },
            handler=self._calculator_handler,
        )
        
        # Current time tool
        self.register(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "timezone": {
                    "type": "string",
                    "description": "Timezone (e.g., 'UTC', 'America/New_York'). Defaults to UTC.",
                    "required": False,
                }
            },
            handler=self._time_handler,
        )
        
        # Document search (RAG)
        self.register(
            name="search_documents",
            description="Search through the user's uploaded documents to find relevant information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant document sections",
                    "required": True,
                },
                "document_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of specific document IDs to search. If not provided, searches all documents.",
                    "required": False,
                }
            },
            handler=self._document_search_handler,
        )
        
        # Code execution tool (sandboxed)
        self.register(
            name="execute_python",
            description="Execute Python code in a sandboxed environment. Limited to basic operations, no file system or network access.",
            parameters={
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                    "required": True,
                }
            },
            handler=self._python_executor_handler,
        )
        
        # JSON formatter
        self.register(
            name="format_json",
            description="Format and validate JSON data",
            parameters={
                "data": {
                    "type": "string",
                    "description": "JSON string to format",
                    "required": True,
                },
                "indent": {
                    "type": "integer",
                    "description": "Indentation level (default: 2)",
                    "required": False,
                }
            },
            handler=self._json_formatter_handler,
        )
        
        # Text analysis
        self.register(
            name="analyze_text",
            description="Analyze text for word count, character count, readability metrics",
            parameters={
                "text": {
                    "type": "string",
                    "description": "Text to analyze",
                    "required": True,
                }
            },
            handler=self._text_analyzer_handler,
        )
        
        # NC-0.8.0.7: Image generation tool
        self.register(
            name="generate_image",
            description="Generate an image based on a text prompt. Use this when the user asks you to create, draw, generate, or make an image, picture, illustration, or artwork.",
            parameters={
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate. Be specific about style, colors, composition, and subject matter.",
                    "required": True,
                },
                "width": {
                    "type": "integer",
                    "description": "Width of the image in pixels. Default: 1024. Common sizes: 512, 768, 1024, 1536",
                    "required": False,
                },
                "height": {
                    "type": "integer",
                    "description": "Height of the image in pixels. Default: 1024. Common sizes: 512, 768, 1024, 1536",
                    "required": False,
                }
            },
            handler=self._image_gen_handler,
        )
        
        # Web page fetcher
        self.register(
            name="fetch_webpage",
            description="Fetch and read the content of web pages. Accepts a single URL or an array of URLs. For video URLs (YouTube, Rumble), returns embeddable video players.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "A single URL to fetch (use this OR urls, not both)",
                    "required": False,
                },
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "An array of URLs to fetch (use this OR url, not both)",
                    "required": False,
                },
                "extract_main_content": {
                    "type": "boolean",
                    "description": "If true, attempts to extract just the main content (article text). If false, returns all text. Default: true",
                    "required": False,
                }
            },
            handler=self._webpage_fetch_handler,
        )
        
        # Multi-URL fetch with video embed support (alias for fetch_webpage with urls)
        self.register(
            name="fetch_urls",
            description="Fetch multiple URLs at once. For video URLs (YouTube, Rumble), returns embeddable video players. For regular pages, extracts text content.",
            parameters={
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to fetch",
                    "required": True,
                },
                "extract_main_content": {
                    "type": "boolean",
                    "description": "For non-video URLs, extract just main content (true) or all text (false). Default: true",
                    "required": False,
                }
            },
            handler=self._fetch_urls_handler,
        )
        
        # File viewing tools for uploaded files
        self.register(
            name="view_file_lines",
            description="View specific lines from an uploaded file. Use this to examine portions of large files without loading the entire content.",
            parameters={
                "filename": {
                    "type": "string",
                    "description": "Name of the uploaded file to view",
                    "required": True,
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to view (1-indexed). Default: 1",
                    "required": False,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to view (inclusive). Default: end of file",
                    "required": False,
                }
            },
            handler=FileViewingTools.view_file_lines,
        )
        
        self.register(
            name="search_in_file",
            description="Search for a pattern in an uploaded file and show matching lines with context. Useful for finding specific code or text.",
            parameters={
                "filename": {
                    "type": "string",
                    "description": "Name of the uploaded file to search",
                    "required": True,
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (supports regex or literal text)",
                    "required": True,
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match. Default: 3",
                    "required": False,
                }
            },
            handler=FileViewingTools.search_in_file,
        )
        
        self.register(
            name="list_uploaded_files",
            description="List all files uploaded in the current chat session with basic info (size, line count, preview)",
            parameters={},
            handler=FileViewingTools.list_uploaded_files,
        )
        
        self.register(
            name="view_signature",
            description="View code around a specific function, class, or other code signature. Automatically finds the definition and shows the implementation.",
            parameters={
                "filename": {
                    "type": "string",
                    "description": "Name of the source file",
                    "required": True,
                },
                "signature_name": {
                    "type": "string",
                    "description": "Name of the function, class, method, or label to find",
                    "required": True,
                },
                "lines_after": {
                    "type": "integer",
                    "description": "Number of lines to show after the signature. Default: 20",
                    "required": False,
                }
            },
            handler=FileViewingTools.view_signature,
        )
        
        # Request file content by offset (NC-0.8.0.6)
        self.register(
            name="request_file",
            description="Retrieve content from an uploaded file starting at a specific character offset. Use this when you see truncation notices like '[Use <request_file path=\"...\" offset=\"...\"/>]'.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path/filename of the file to retrieve",
                    "required": True,
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from. Default: 0",
                    "required": False,
                },
                "length": {
                    "type": "integer",
                    "description": "Number of characters to retrieve. Default: 20000",
                    "required": False,
                }
            },
            handler=FileViewingTools.request_file,
        )
        
        # NC-0.8.0.9: Create file tool for artifact creation
        self.register(
            name="create_file",
            description="""Create or overwrite a file with the given content. Use this to save code, documents, data files, or any text content.

USAGE: When you need to create files for the user - code files, config files, documents, etc.

The file will be saved and made available for download. Supports any text-based file type.""",
            parameters={
                "path": {
                    "type": "string",
                    "description": "File path/name to create (e.g., 'script.py', 'data.json', 'src/main.cpp')",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                    "required": True,
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "If true, overwrite existing file. Default: true",
                    "required": False,
                }
            },
            handler=self._create_file_handler,
        )
        
        # =============================================================
        # TASK QUEUE TOOLS - Agentic Multi-Step Task Management
        # =============================================================
        # PURPOSE: Break complex requests into discrete, trackable steps.
        # 
        # WORKFLOW:
        # 1. User makes complex request → LLM plans tasks with add_task/add_tasks_batch
        # 2. System auto-feeds tasks to LLM one at a time
        # 3. LLM works on task → calls complete_task when done
        # 4. System sends next task → repeat until queue empty
        #
        # IMPORTANT: You must ACTUALLY DO THE WORK described in each task!
        # complete_task is called AFTER you've done the work, not instead of it.
        # =============================================================
        
        self.register(
            name="add_task",
            description="""Add a task to the agentic task queue for multi-step workflows.

USE THIS WHEN: User requests something complex that needs multiple steps (research, create files, analyze data, etc.)

WORKFLOW:
1. Plan: Break user's request into discrete tasks using add_task or add_tasks_batch
2. Execute: System will feed you tasks one at a time
3. Work: Actually DO the work described in each task
4. Complete: Call complete_task AFTER you've finished the work

Example - User asks "Research AI trends and write a report":
  → add_tasks_batch with tasks: ["Research recent AI developments", "Analyze key trends", "Write executive summary", "Create full report document"]
  → System feeds you "Research recent AI developments"
  → You actually search/fetch/read about AI trends
  → You call complete_task("Found 5 major trends: ...")
  → System feeds next task, and so on""",
            parameters={
                "description": {
                    "type": "string",
                    "description": "Short task title (shown in UI)",
                    "required": True,
                },
                "instructions": {
                    "type": "string",
                    "description": "Detailed instructions for what to do (up to 512 tokens). Be specific about expected outputs.",
                    "required": True,
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority level (0=normal, higher=more urgent, processed first)",
                    "required": False,
                },
                "auto_continue": {
                    "type": "boolean",
                    "description": "Auto-start next task when this completes (default: true)",
                    "required": False,
                }
            },
            handler=self._add_task_handler,
        )
        
        self.register(
            name="add_tasks_batch",
            description="""Add multiple tasks to the queue at once - use for planning multi-step workflows.

BEST PRACTICE: Plan all steps upfront, then execute them one by one.

Example - User asks "Build me a Python web scraper":
  add_tasks_batch([
    {"description": "Design scraper architecture", "instructions": "Plan the scraper structure, decide on libraries (requests/beautifulsoup/selenium), identify target data"},
    {"description": "Write core scraping logic", "instructions": "Implement the main scraping functions with error handling"},
    {"description": "Add data extraction", "instructions": "Parse HTML and extract the structured data user needs"},
    {"description": "Create output handling", "instructions": "Save results to CSV/JSON, add progress logging"},
    {"description": "Write usage documentation", "instructions": "Create README with installation and usage instructions"}
  ])

After adding tasks, the system will feed them to you one at a time. Do the actual work for each task before calling complete_task.""",
            parameters={
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "Short task title"},
                            "instructions": {"type": "string", "description": "What to do for this task"},
                            "priority": {"type": "integer"},
                            "auto_continue": {"type": "boolean"}
                        }
                    },
                    "description": "Array of task objects with description and instructions",
                    "required": True,
                }
            },
            handler=self._add_tasks_batch_handler,
        )
        
        self.register(
            name="complete_task",
            description="""Mark the current task as COMPLETED - call this AFTER you've done the actual work.

⚠️ IMPORTANT: This does NOT do the work for you! You must:
1. Actually perform the task (search, write code, create files, analyze, etc.)
2. THEN call complete_task to record what you accomplished

WRONG: Receiving task "Research AI" → immediately calling complete_task
RIGHT: Receiving task "Research AI" → search web → read articles → summarize findings → THEN call complete_task

The result_summary should describe what you actually did and any outputs created.""",
            parameters={
                "result_summary": {
                    "type": "string",
                    "description": "Brief summary of what was actually accomplished (files created, findings, etc.)",
                    "required": False,
                }
            },
            handler=self._complete_task_handler,
        )
        
        self.register(
            name="fail_task",
            description="""Mark the current task as FAILED - use when you genuinely cannot complete it.

Use this when:
- Required resources are unavailable (broken links, API errors)
- Task is impossible with available tools
- Task requirements are unclear and user clarification is needed

Don't use this just because a task is difficult - try your best first!""",
            parameters={
                "reason": {
                    "type": "string",
                    "description": "Specific reason why the task could not be completed",
                    "required": False,
                }
            },
            handler=self._fail_task_handler,
        )
        
        self.register(
            name="skip_task",
            description="Skip the current task and move to the next one. Use sparingly - prefer complete_task or fail_task.",
            parameters={},
            handler=self._skip_task_handler,
        )
        
        self.register(
            name="get_task_queue",
            description="View the current task queue status - see pending tasks, completed count, and what's next.",
            parameters={},
            handler=self._get_task_queue_handler,
        )
        
        self.register(
            name="clear_task_queue",
            description="Clear all pending tasks from the queue. Use when user wants to start over or cancel planned work.",
            parameters={},
            handler=self._clear_task_queue_handler,
        )
        
        self.register(
            name="pause_task_queue",
            description="Pause automatic task execution. Tasks won't auto-start until resumed. Use when user needs to review progress.",
            parameters={},
            handler=self._pause_task_queue_handler,
        )
        
        self.register(
            name="resume_task_queue",
            description="Resume task queue execution after pausing. The next pending task will be sent to you.",
            parameters={},
            handler=self._resume_task_queue_handler,
        )
        
        # =============================================================
        # MEMORY TOOLS - Access Archived Conversation History
        # =============================================================
        # PURPOSE: Long conversations get archived into Agent Memory files
        # (Agent0001.md, Agent0002.md, etc.) to manage context size.
        # Use these tools to recall information from earlier in the conversation.
        #
        # WORKFLOW:
        # 1. memory_search("topic") → Find relevant archived content
        # 2. memory_read(memory_number, offset) → Read full content at location
        # =============================================================
        
        self.register(
            name="memory_search",
            description="""Search through archived conversation history stored in Agent Memory files.

USE THIS WHEN: 
- User references something discussed "earlier" or "before"
- You need context from a long conversation that may have been archived
- User asks "what did we decide about X" or "remember when we discussed Y"

Returns: List of matches with memory_number and offset - use memory_read() to get full content.

Example:
  memory_search("database schema design")
  → Returns: [{memory_number: 3, offset: 15420, preview: "...we decided on PostgreSQL..."}]
  → Then call: memory_read(3, 15420) to get full context""",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Keywords or phrases to search for in archived conversation history",
                    "required": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 3, max: 10)",
                    "required": False,
                }
            },
            handler=self._agent_search_handler,
        )
        
        self.register(
            name="memory_read",
            description="""Read content from a specific Agent Memory file at a given offset.

USE THIS AFTER memory_search() to retrieve the full context around a match.

Parameters:
- memory_number: Which file (1 = Agent0001.md, 2 = Agent0002.md, etc.)
- offset: Where to start reading (from memory_search results)
- length: How much to read (default: 4000 chars)

Returns: Content chunk, plus has_more/next_offset if more content available.

Example:
  memory_read(3, 15420)
  → Returns: {content: "...full conversation segment...", has_more: true, next_offset: 19420}
  memory_read(3, 19420)  // Continue reading if needed
  → Returns: {content: "...more content...", has_more: false}""",
            parameters={
                "memory_number": {
                    "type": "integer",
                    "description": "Memory file number (1, 2, 3, etc. - from memory_search results)",
                    "required": True,
                },
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from (default: 0)",
                    "required": False,
                },
                "length": {
                    "type": "integer",
                    "description": "Number of characters to read (default: 4000)",
                    "required": False,
                }
            },
            handler=self._memory_read_handler,
        )
        
        # Legacy aliases for backward compatibility (hidden from typical tool listings)
        self.register(
            name="agent_search",
            description="[DEPRECATED: Use memory_search instead] Search archived conversation history.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default: 3)",
                    "required": False,
                }
            },
            handler=self._agent_search_handler,
        )
        
        self.register(
            name="agent_read",
            description="[DEPRECATED: Use memory_read instead] Read Agent Memory file by filename.",
            parameters={
                "filename": {
                    "type": "string",
                    "description": "The agent memory filename (e.g., 'Agent0001.md')",
                    "required": True,
                }
            },
            handler=self._agent_read_handler,
        )
        
        # NC-0.8.0.7: Temporary MCP server installation tools
        self.register(
            name="install_mcp_server",
            description="Install a temporary MCP (Model Context Protocol) server. The server will be automatically removed after 4 hours of non-use. Use this to add new capabilities on-demand.",
            parameters={
                "name": {
                    "type": "string",
                    "description": "Display name for the MCP server (e.g., 'GitHub Tools', 'Slack Integration')",
                    "required": True,
                },
                "url": {
                    "type": "string",
                    "description": "MCP server URL or npx command (e.g., 'npx -y @modelcontextprotocol/server-github' or 'http://localhost:3000/mcp')",
                    "required": True,
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what tools this server provides",
                    "required": False,
                },
                "api_key": {
                    "type": "string",
                    "description": "API key if the MCP server requires authentication",
                    "required": False,
                }
            },
            handler=self._install_mcp_handler,
        )
        
        self.register(
            name="uninstall_mcp_server",
            description="Uninstall a temporary MCP server that was previously installed.",
            parameters={
                "name": {
                    "type": "string",
                    "description": "Name of the MCP server to uninstall",
                    "required": True,
                }
            },
            handler=self._uninstall_mcp_handler,
        )
        
        self.register(
            name="list_mcp_servers",
            description="List all temporary MCP servers currently installed, including their expiry status.",
            parameters={},
            handler=self._list_mcp_handler,
        )

    
    async def _calculator_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Safe mathematical expression evaluator"""
        expression = args.get("expression", "")
        
        # Allowed functions
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,
            "degrees": math.degrees,
            "radians": math.radians,
        }
        
        try:
            # Basic sanitization
            if any(kw in expression.lower() for kw in ["import", "exec", "eval", "__", "open", "file"]):
                return {"error": "Invalid expression"}
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}
    
    async def _time_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Get current time"""
        try:
            from datetime import timezone
            import pytz
        except ImportError:
            # Fallback to UTC
            now = datetime.now(timezone.utc)
            return {
                "datetime": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "timezone": "UTC",
                "unix_timestamp": int(now.timestamp()),
            }
        
        tz_name = args.get("timezone", "UTC")
        try:
            tz = pytz.timezone(tz_name)
        except:
            tz = pytz.UTC
            tz_name = "UTC"
        
        now = datetime.now(tz)
        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": tz_name,
            "unix_timestamp": int(now.timestamp()),
        }
    
    async def _document_search_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Search documents using RAG - searches all available knowledge sources"""
        if not context or "db" not in context or "user" not in context:
            return {"error": "Document search requires authentication context"}
        
        db = context["db"]
        user = context["user"]
        chat_id = context.get("chat_id")
        query = args.get("query", "")
        document_ids = args.get("document_ids")
        
        rag_service = RAGService()
        all_results = []
        sources_searched = []
        
        # 1. Search Global Knowledge Stores (same as automatic RAG)
        try:
            global_results, global_store_names = await rag_service.search_global_stores(
                db, query, chat_id=chat_id
            )
            if global_results:
                sources_searched.append(f"Global KBs: {', '.join(global_store_names)}")
                for r in global_results:
                    r["source_type"] = "global_kb"
                all_results.extend(global_results)
        except Exception as e:
            logger.warning(f"[TOOL_SEARCH] Global KB search failed: {e}")
        
        # 2. Search Custom GPT Knowledge Stores (if in a GPT conversation)
        if chat_id:
            try:
                from app.models.models import Chat, CustomAssistant
                from app.models.assistant import AssistantConversation
                from sqlalchemy import select
                from sqlalchemy.orm import selectinload
                
                # Check if this chat has an active assistant conversation
                chat_result = await db.execute(
                    select(Chat).where(Chat.id == chat_id)
                )
                chat = chat_result.scalar_one_or_none()
                
                if chat:
                    conv_result = await db.execute(
                        select(AssistantConversation)
                        .where(AssistantConversation.chat_id == chat_id)
                        .order_by(AssistantConversation.created_at.desc())
                        .limit(1)
                    )
                    assistant_conv = conv_result.scalar_one_or_none()
                    
                    if assistant_conv:
                        # Get assistant with knowledge stores
                        assistant_result = await db.execute(
                            select(CustomAssistant)
                            .options(selectinload(CustomAssistant.knowledge_stores))
                            .where(CustomAssistant.id == assistant_conv.assistant_id)
                        )
                        assistant = assistant_result.scalar_one_or_none()
                        
                        if assistant and assistant.knowledge_stores:
                            assistant_ks_ids = [str(ks.id) for ks in assistant.knowledge_stores]
                            ks_names = [ks.name for ks in assistant.knowledge_stores]
                            
                            kb_results = await rag_service.search_knowledge_stores(
                                db=db,
                                user=user,
                                query=query,
                                knowledge_store_ids=assistant_ks_ids,
                                bypass_access_check=True,
                            )
                            
                            if kb_results:
                                sources_searched.append(f"GPT KBs: {', '.join(ks_names)}")
                                for r in kb_results:
                                    r["source_type"] = "assistant_kb"
                                all_results.extend(kb_results)
            except Exception as e:
                logger.warning(f"[TOOL_SEARCH] Assistant KB search failed: {e}")
        
        # 3. Search User's Documents (original behavior)
        try:
            user_results = await rag_service.search(
                db=db,
                user=user,
                query=query,
                document_ids=document_ids,
                top_k=5,
            )
            if user_results:
                sources_searched.append("User Documents")
                for r in user_results:
                    r["source_type"] = "user_docs"
                all_results.extend(user_results)
        except Exception as e:
            logger.warning(f"[TOOL_SEARCH] User doc search failed: {e}")
        
        # Deduplicate by content (same chunk might appear from different sources)
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_key = r.get("content", "")[:200]  # First 200 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)
        
        # Sort by relevance
        unique_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Limit total results
        unique_results = unique_results[:10]
        
        return {
            "query": query,
            "sources_searched": sources_searched,
            "results_count": len(unique_results),
            "results": [
                {
                    "document": r.get("document_name", "Unknown"),
                    "source_type": r.get("source_type", "unknown"),
                    "content": r["content"][:500] + "..." if len(r.get("content", "")) > 500 else r.get("content", ""),
                    "relevance": round(r.get("similarity", 0), 3),
                }
                for r in unique_results
            ]
        }
    
    async def _python_executor_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Execute Python code in a restricted environment"""
        code = args.get("code", "")
        
        # Restricted builtins
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "hex": hex, "int": int, "isinstance": isinstance, "len": len,
            "list": list, "map": map, "max": max, "min": min, "oct": oct,
            "ord": ord, "pow": pow, "print": print, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set, "slice": slice,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip,
        }
        
        # Check for dangerous patterns
        dangerous = ["import", "exec", "eval", "__", "open", "file", "os", "sys", "subprocess"]
        if any(d in code.lower() for d in dangerous):
            return {"error": "Code contains restricted operations"}
        
        # Capture output
        import io
        import sys
        
        output_capture = io.StringIO()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = output_capture
            
            local_vars = {}
            exec(code, {"__builtins__": safe_builtins, "math": math}, local_vars)
            
            sys.stdout = old_stdout
            output = output_capture.getvalue()
            
            # Get returned value if any
            result = local_vars.get("result", None)
            
            return {
                "success": True,
                "output": output if output else None,
                "result": result,
            }
        except Exception as e:
            sys.stdout = old_stdout
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _json_formatter_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Format JSON data"""
        data = args.get("data", "")
        indent = args.get("indent", 2)
        
        try:
            parsed = json.loads(data)
            formatted = json.dumps(parsed, indent=indent, ensure_ascii=False)
            return {
                "valid": True,
                "formatted": formatted,
            }
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": str(e),
            }
    
    async def _text_analyzer_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Analyze text metrics"""
        text = args.get("text", "")
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Basic readability (Flesch-Kincaid approximation)
        syllables = sum(self._count_syllables(word) for word in words)
        
        if len(words) > 0 and len(sentences) > 0:
            fk_grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
        else:
            fk_grade = 0
        
        return {
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
            "average_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0,
            "average_sentence_length": round(len(words) / len(sentences), 2) if sentences else 0,
            "flesch_kincaid_grade": round(fk_grade, 1),
        }
    
    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count"""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith("e"):
            count -= 1
        
        return max(1, count)
    
    async def _image_gen_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """NC-0.8.0.7: Generate an image using the image generation service.
        
        This handler queues an image generation task. The image will be generated
        asynchronously and sent to the frontend via WebSocket when complete.
        The generated image will appear in the chat.
        """
        prompt = args.get("prompt", "")
        
        if not prompt:
            return {"error": "Prompt is required for image generation"}
        
        # Get context info
        chat_id = context.get("chat_id") if context else None
        user_id = context.get("user_id") if context else None
        message_id = context.get("message_id") if context else None
        db = context.get("db") if context else None
        
        if not chat_id or not user_id:
            return {"error": "Missing required context (chat_id, user_id)"}
        
        try:
            from app.services.image_queue import get_image_queue, ensure_queue_started, TaskStatus
            from app.services.websocket import ws_manager
            from app.db.database import async_session_maker
            from app.services.settings_service import SettingsService
            
            # NC-0.8.0.9: Get default dimensions from admin settings using SK keys
            from app.core.settings_keys import SK
            default_width = 1024
            default_height = 1024
            
            if db:
                try:
                    raw_width = await SettingsService.get(db, SK.IMAGE_GEN_DEFAULT_WIDTH)
                    raw_height = await SettingsService.get(db, SK.IMAGE_GEN_DEFAULT_HEIGHT)
                    logger.info(f"[IMAGE_TOOL] Raw settings from DB: width='{raw_width}', height='{raw_height}'")
                    
                    default_width = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_WIDTH)
                    default_height = await SettingsService.get_int(db, SK.IMAGE_GEN_DEFAULT_HEIGHT)
                    logger.info(f"[IMAGE_TOOL] Parsed settings: {default_width}x{default_height}")
                except Exception as e:
                    logger.warning(f"[IMAGE_TOOL] Could not fetch default dimensions: {e}")
            else:
                logger.warning("[IMAGE_TOOL] No db context provided, using defaults 1024x1024")
            
            # Use provided dimensions or fall back to admin defaults
            arg_width = args.get("width")
            arg_height = args.get("height")
            width = arg_width if arg_width else default_width
            height = arg_height if arg_height else default_height
            logger.info(f"[IMAGE_TOOL] Final dimensions: {width}x{height} (args: width={arg_width}, height={arg_height})")
            
            queue = get_image_queue()
            
            # Check queue capacity
            if queue.is_full:
                return {"error": "Image generation queue is full. Please try again in a moment."}
            
            # Create callback to notify frontend when complete
            # Uses same message types as the auto-detect flow for frontend compatibility
            async def notify_completion():
                task = queue.get_task(task_id)
                if not task:
                    return
                
                try:
                    from sqlalchemy import text
                    
                    if task.status == TaskStatus.COMPLETED:
                        image_data = {
                            "url": task.image_url,
                            "width": task.width,
                            "height": task.height,
                            "seed": task.seed,
                            "prompt": task.prompt,
                            "job_id": task.job_id,
                        }
                        
                        # NC-0.8.0.7: Save image data to message metadata for persistence
                        try:
                            async with async_session_maker() as notify_db:
                                # First check if message still exists
                                result = await notify_db.execute(
                                    text("SELECT id FROM messages WHERE id = :id"),
                                    {"id": message_id}
                                )
                                if result.fetchone():
                                    # Build metadata JSON
                                    new_metadata = json.dumps({
                                        "image_generation": {
                                            "status": "completed",
                                            "prompt": task.prompt,
                                            "width": task.width,
                                            "height": task.height,
                                            "seed": task.seed,
                                            "generation_time": task.generation_time,
                                            "job_id": task.job_id,
                                        },
                                        "generated_image": image_data,
                                    })
                                    
                                    # Update message with image metadata
                                    await notify_db.execute(
                                        text("UPDATE messages SET message_metadata = :metadata WHERE id = :id"),
                                        {"metadata": new_metadata, "id": message_id}
                                    )
                                    
                                    # Update chat timestamp
                                    from datetime import datetime, timezone as tz
                                    await notify_db.execute(
                                        text("UPDATE chats SET updated_at = :now WHERE id = :id"),
                                        {"now": datetime.now(tz.utc), "id": chat_id}
                                    )
                                    await notify_db.commit()
                                    logger.info(f"[IMAGE_TOOL] Saved image metadata for message {message_id}")
                        except Exception as db_err:
                            logger.error(f"[IMAGE_TOOL] Failed to save metadata: {db_err}")
                        
                        # Send to user via WebSocket - same message type as auto-detect flow
                        await ws_manager.send_to_user(user_id, {
                            "type": "image_generated",
                            "payload": {
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "image": image_data,
                            },
                        })
                        logger.info(f"[IMAGE_TOOL] Sent image_generated for task {task_id}")
                    else:
                        # Notify frontend of failure
                        await ws_manager.send_to_user(user_id, {
                            "type": "image_generation_failed",
                            "payload": {
                                "chat_id": chat_id,
                                "message_id": message_id,
                                "error": task.error,
                            },
                        })
                        logger.info(f"[IMAGE_TOOL] Sent image_generation_failed for task {task_id}")
                
                except Exception as e:
                    logger.error(f"Error in tool image notify_completion: {e}")
            
            # Ensure queue is started
            await ensure_queue_started()
            
            # Add to queue with callback
            task_id = await queue.add_task(
                prompt=prompt,
                width=width,
                height=height,
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                notify_callback=notify_completion,
            )
            
            logger.info(f"[IMAGE_TOOL] Queued image generation task {task_id} for message {message_id}")
            
            return {
                "status": "queued",
                "task_id": task_id,
                "message": f"Image generation started. The image will appear in the chat when ready.",
                "prompt": prompt,
                "width": width,
                "height": height,
                "queue_position": queue.get_pending_count(),
            }
        except Exception as e:
            logger.error(f"[IMAGE_TOOL] Failed to queue: {e}")
            return {"error": f"Failed to queue image generation: {str(e)}"}
    
    async def _webpage_fetch_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Fetch and extract content from web pages. Supports single URL or array of URLs."""
        import httpx
        import asyncio
        from urllib.parse import urlparse
        
        # Handle both single url and urls array
        url = args.get("url", "")
        urls = args.get("urls", [])
        extract_main = args.get("extract_main_content", True)
        
        # Build list of URLs to fetch
        url_list = []
        if url:
            url_list.append(url)
        if urls:
            if isinstance(urls, str):
                url_list.append(urls)
            else:
                url_list.extend(urls)
        
        if not url_list:
            return {"error": "URL or URLs required"}
        
        # If single URL, return single result (backward compatible)
        if len(url_list) == 1:
            return await self._fetch_single_url(url_list[0], extract_main)
        
        # Multiple URLs - fetch concurrently
        async def fetch_one(u):
            return await self._fetch_single_url(u, extract_main)
        
        tasks = [fetch_one(u) for u in url_list[:10]]  # Limit to 10
        results = await asyncio.gather(*tasks)
        
        return {
            "results": results,
            "fetched_count": len(results),
            "video_count": sum(1 for r in results if r.get("type") == "video"),
            "webpage_count": sum(1 for r in results if r.get("type") == "webpage"),
            "error_count": sum(1 for r in results if "error" in r),
        }
    
    async def _fetch_single_url(self, url: str, extract_main: bool = True) -> Dict[str, Any]:
        """Fetch a single URL with video detection and subtitle extraction."""
        import httpx
        from urllib.parse import urlparse
        
        if not url:
            return {"error": "URL is required"}
        
        # Check if it's a video URL first
        platform, video_id = self._extract_video_id(url)
        if platform and video_id:
            # Use the subtitle-enabled version for videos
            embed = await self._create_video_embed_with_subtitles(platform, video_id, url)
            if embed:
                return embed
        
        # Validate URL
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return {"url": url, "error": "Invalid URL scheme. Must be http or https."}
        except Exception:
            return {"url": url, "error": "Invalid URL format"}
        
        try:
            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; Open-NueChat/1.0; +https://nuechat.ai)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                content_type = response.headers.get("content-type", "")
                
                # Handle non-HTML content
                if "application/json" in content_type:
                    return {
                        "type": "json",
                        "url": str(response.url),
                        "content_type": "json",
                        "content": response.text[:50000],
                    }
                elif "text/plain" in content_type:
                    return {
                        "type": "text",
                        "url": str(response.url),
                        "content_type": "text",
                        "content": response.text[:50000],
                    }
                elif "text/html" not in content_type and "application/xhtml" not in content_type:
                    return {
                        "type": "unknown",
                        "url": str(response.url),
                        "content_type": content_type,
                        "error": "Content type not supported for text extraction",
                    }
                
                html = response.text
                
                # Extract text from HTML
                text_content = self._extract_text_from_html(html, extract_main)
                
                return {
                    "type": "webpage",
                    "url": str(response.url),
                    "title": self._extract_title(html),
                    "content_type": "html",
                    "content": text_content[:50000],
                    "content_length": len(text_content),
                }
                
        except httpx.TimeoutException:
            return {"url": url, "error": "Request timed out"}
        except httpx.HTTPStatusError as e:
            return {"url": url, "error": f"HTTP error {e.response.status_code}"}
        except Exception as e:
            return {"url": url, "error": f"Failed to fetch page: {str(e)}"}
    
    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML"""
        import re
        match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_text_from_html(self, html: str, extract_main: bool = True) -> str:
        """Extract readable text from HTML"""
        try:
            from bs4 import BeautifulSoup
            # Try lxml first (faster), fall back to html.parser
            try:
                soup = BeautifulSoup(html, 'lxml')
            except Exception:
                soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                                'aside', 'noscript', 'iframe', 'svg', 'form']):
                element.decompose()
            
            if extract_main:
                # Try to find main content
                main_content = (
                    soup.find('main') or
                    soup.find('article') or
                    soup.find('div', {'class': re.compile(r'(content|article|post|entry)', re.I)}) or
                    soup.find('div', {'id': re.compile(r'(content|article|post|entry)', re.I)}) or
                    soup.body
                )
                
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Remove excessive newlines
            import re as regex
            text = regex.sub(r'\n{3,}', '\n\n', text)
            
            return text
            
        except ImportError:
            # Fallback: basic regex extraction if bs4 not available
            import re as regex
            # Remove script and style content
            html = regex.sub(r'<script[^>]*>.*?</script>', '', html, flags=regex.DOTALL | regex.IGNORECASE)
            html = regex.sub(r'<style[^>]*>.*?</style>', '', html, flags=regex.DOTALL | regex.IGNORECASE)
            # Remove all tags
            text = regex.sub(r'<[^>]+>', ' ', html)
            # Clean up whitespace
            text = regex.sub(r'\s+', ' ', text).strip()
            # Decode HTML entities
            import html as html_module
            text = html_module.unescape(text)
            return text
    
    def _extract_video_id(self, url: str) -> tuple:
        """
        Extract video ID and platform from a video URL.
        Returns (platform, video_id) or (None, None) if not a recognized video URL.
        """
        import re
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(url)
        
        # YouTube patterns
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            # youtube.com/watch?v=VIDEO_ID
            if 'youtube.com' in parsed.netloc:
                qs = parse_qs(parsed.query)
                if 'v' in qs:
                    return ('youtube', qs['v'][0])
                # youtube.com/embed/VIDEO_ID
                match = re.match(r'/embed/([a-zA-Z0-9_-]+)', parsed.path)
                if match:
                    return ('youtube', match.group(1))
                # youtube.com/shorts/VIDEO_ID
                match = re.match(r'/shorts/([a-zA-Z0-9_-]+)', parsed.path)
                if match:
                    return ('youtube', match.group(1))
            # youtu.be/VIDEO_ID
            elif 'youtu.be' in parsed.netloc:
                video_id = parsed.path.lstrip('/')
                if video_id:
                    return ('youtube', video_id.split('?')[0])
        
        # Rumble patterns
        elif 'rumble.com' in parsed.netloc:
            # rumble.com/v{VIDEO_ID}-title.html or rumble.com/embed/v{VIDEO_ID}
            match = re.match(r'/v([a-zA-Z0-9]+)', parsed.path)
            if match:
                return ('rumble', match.group(1))
            match = re.match(r'/embed/v([a-zA-Z0-9]+)', parsed.path)
            if match:
                return ('rumble', match.group(1))
        
        return (None, None)
    
    async def _fetch_youtube_subtitles(self, video_id: str, lang: str = 'en') -> dict:
        """Fetch YouTube video subtitles/captions."""
        import httpx
        import re
        import json
        import logging
        logger = logging.getLogger(__name__)
        
        # Try youtube-transcript-api first (most reliable)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
            
            logger.info(f"[YOUTUBE] Trying youtube-transcript-api for {video_id}")
            
            try:
                # Try to get transcript in preferred language
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Try manual transcript first, then auto-generated
                transcript = None
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang, 'en'])
                    logger.info(f"[YOUTUBE] Found manual transcript")
                except NoTranscriptFound:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang, 'en'])
                        logger.info(f"[YOUTUBE] Found auto-generated transcript")
                    except NoTranscriptFound:
                        # Try to get any available transcript
                        try:
                            for t in transcript_list:
                                transcript = t
                                logger.info(f"[YOUTUBE] Found transcript in {t.language_code}")
                                break
                        except:
                            pass
                
                if transcript:
                    transcript_data = transcript.fetch()
                    text = " ".join([entry['text'] for entry in transcript_data])
                    return {
                        "success": True,
                        "language": transcript.language_code,
                        "is_auto_generated": transcript.is_generated,
                        "transcript": text,
                        "line_count": len(transcript_data),
                    }
                else:
                    logger.info(f"[YOUTUBE] No transcript found via API")
                    
            except TranscriptsDisabled:
                logger.info(f"[YOUTUBE] Transcripts disabled for {video_id}")
                return {"error": "Transcripts are disabled for this video"}
            except VideoUnavailable:
                logger.info(f"[YOUTUBE] Video unavailable: {video_id}")
                return {"error": "Video is unavailable"}
            except NoTranscriptFound:
                logger.info(f"[YOUTUBE] No transcript found for {video_id}")
            except Exception as e:
                logger.warning(f"[YOUTUBE] API error for {video_id}: {e}")
                
        except ImportError:
            logger.info("[YOUTUBE] youtube-transcript-api not installed, using manual method")
        
        # Fallback: Manual extraction from YouTube page
        logger.info(f"[YOUTUBE] Trying manual extraction for {video_id}")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"https://www.youtube.com/watch?v={video_id}",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    }
                )
                html = response.text
                logger.info(f"[YOUTUBE] Fetched page for {video_id}, length={len(html)}")
                
                # Check if captions exist at all
                if '"captions"' not in html:
                    logger.info(f"[YOUTUBE] No captions section in page for {video_id}")
                    return {"error": "No captions found for this video"}
                
                # Try to extract the baseUrl for captions directly
                # Pattern: "baseUrl":"https://www.youtube.com/api/timedtext?..."
                baseurl_pattern = re.search(r'"baseUrl"\s*:\s*"(https://www\.youtube\.com/api/timedtext[^"]+)"', html)
                if baseurl_pattern:
                    subtitle_url = baseurl_pattern.group(1).replace('\\u0026', '&')
                    logger.info(f"[YOUTUBE] Found baseUrl for subtitles")
                    
                    try:
                        sub_response = await client.get(subtitle_url)
                        subtitle_content = sub_response.text
                        
                        # Parse XML/JSON response
                        if subtitle_content.strip().startswith('<'):
                            # XML format
                            text_pattern = re.compile(r'<text[^>]*>([^<]*)</text>')
                            matches = text_pattern.findall(subtitle_content)
                            if matches:
                                import html as html_module
                                transcript_lines = [html_module.unescape(t).strip() for t in matches if t.strip()]
                                transcript = " ".join(transcript_lines)
                                logger.info(f"[YOUTUBE] Extracted {len(transcript_lines)} lines from XML")
                                return {
                                    "success": True,
                                    "language": "en",
                                    "is_auto_generated": True,
                                    "transcript": transcript,
                                    "line_count": len(transcript_lines),
                                }
                        else:
                            # Try JSON format
                            try:
                                data = json.loads(subtitle_content)
                                events = data.get("events", [])
                                lines = []
                                for event in events:
                                    segs = event.get("segs", [])
                                    for seg in segs:
                                        text = seg.get("utf8", "").strip()
                                        if text and text != "\n":
                                            lines.append(text)
                                if lines:
                                    transcript = " ".join(lines)
                                    logger.info(f"[YOUTUBE] Extracted {len(lines)} segments from JSON")
                                    return {
                                        "success": True,
                                        "language": "en",
                                        "is_auto_generated": True,
                                        "transcript": transcript,
                                        "line_count": len(lines),
                                    }
                            except json.JSONDecodeError:
                                pass
                    except Exception as e:
                        logger.warning(f"[YOUTUBE] Error fetching subtitle URL: {e}")
                
                # Pattern 2: Try to find captionTracks and extract baseUrl
                caption_tracks_match = re.search(r'"captionTracks"\s*:\s*(\[.*?\])', html, re.DOTALL)
                if caption_tracks_match:
                    try:
                        # Clean up the JSON - it might have nested objects
                        tracks_str = caption_tracks_match.group(1)
                        # Find just the array
                        bracket_count = 0
                        end_idx = 0
                        for i, c in enumerate(tracks_str):
                            if c == '[':
                                bracket_count += 1
                            elif c == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_idx = i + 1
                                    break
                        tracks_str = tracks_str[:end_idx]
                        tracks = json.loads(tracks_str)
                        logger.info(f"[YOUTUBE] Found {len(tracks)} caption tracks")
                        
                        for track in tracks:
                            base_url = track.get("baseUrl", "")
                            if base_url:
                                base_url = base_url.replace('\\u0026', '&')
                                sub_response = await client.get(base_url)
                                # ... process response
                                subtitle_content = sub_response.text
                                if '<text' in subtitle_content:
                                    text_pattern = re.compile(r'<text[^>]*>([^<]*)</text>')
                                    matches = text_pattern.findall(subtitle_content)
                                    if matches:
                                        import html as html_module
                                        transcript_lines = [html_module.unescape(t).strip() for t in matches if t.strip()]
                                        transcript = " ".join(transcript_lines)
                                        return {
                                            "success": True,
                                            "language": track.get("languageCode", "en"),
                                            "is_auto_generated": "asr" in track.get("vssId", "").lower(),
                                            "transcript": transcript,
                                            "line_count": len(transcript_lines),
                                        }
                    except Exception as e:
                        logger.warning(f"[YOUTUBE] Error parsing caption tracks: {e}")
                
                logger.info(f"[YOUTUBE] Could not extract subtitles for {video_id}")
                return {"error": "No captions found for this video"}
                
        except httpx.TimeoutException:
            return {"error": "Timeout fetching video page"}
        except Exception as e:
            logger.warning(f"[YOUTUBE] Error in manual extraction: {e}")
            return {"error": f"Failed to fetch subtitles: {str(e)}"}
    
    async def _create_video_embed_with_subtitles(self, platform: str, video_id: str, url: str) -> dict:
        """Create an embeddable video response with subtitles and title if available."""
        import httpx
        import re
        
        # Get basic embed info
        embed = self._create_video_embed(platform, video_id, url)
        if not embed:
            return None
        
        # For YouTube, try to fetch title and subtitles
        if platform == 'youtube':
            # Fetch video page to get title
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"https://www.youtube.com/watch?v={video_id}",
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                            "Accept-Language": "en-US,en;q=0.9",
                        }
                    )
                    html = response.text
                    
                    # Extract title
                    title_match = re.search(r'<title>([^<]+)</title>', html)
                    if title_match:
                        title = title_match.group(1)
                        # Clean up " - YouTube" suffix
                        title = re.sub(r'\s*-\s*YouTube$', '', title)
                        embed["title"] = title
                    
                    # Extract description if available
                    desc_match = re.search(r'"shortDescription":"([^"]{0,500})', html)
                    if desc_match:
                        desc = desc_match.group(1)
                        # Unescape JSON
                        desc = desc.replace('\\n', ' ').replace('\\"', '"')
                        embed["description"] = desc[:500]
                    
                    # Extract channel name
                    channel_match = re.search(r'"ownerChannelName":"([^"]+)"', html)
                    if channel_match:
                        embed["channel"] = channel_match.group(1)
                        
            except Exception as e:
                pass  # Title extraction is optional
            
            # Try to fetch subtitles
            subtitles = await self._fetch_youtube_subtitles(video_id)
            if subtitles.get("success"):
                embed["subtitles"] = {
                    "language": subtitles.get("language"),
                    "is_auto_generated": subtitles.get("is_auto_generated"),
                    "transcript": subtitles.get("transcript"),
                    "line_count": subtitles.get("line_count"),
                }
                # Add transcript as content for context
                embed["content"] = subtitles.get("transcript", "")[:30000]  # Limit size
            else:
                embed["subtitles_error"] = subtitles.get("error")
                # Even without subtitles, provide what we know
                if embed.get("title"):
                    embed["content"] = f"Video Title: {embed['title']}"
                    if embed.get("channel"):
                        embed["content"] += f"\nChannel: {embed['channel']}"
                    if embed.get("description"):
                        embed["content"] += f"\nDescription: {embed['description']}"
        
        return embed
    
    def _create_video_embed(self, platform: str, video_id: str, url: str) -> dict:
        """Create an embeddable video response."""
        if platform == 'youtube':
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            embed_html = f'''<iframe width="560" height="315" src="{embed_url}" frameborder="0" allowfullscreen></iframe>'''
            return {
                "type": "video",
                "platform": "youtube",
                "video_id": video_id,
                "url": url,
                "embed_url": embed_url,
                "embed_html": embed_html,
                "markdown": f"[![YouTube Video](https://img.youtube.com/vi/{video_id}/0.jpg)]({url})",
            }
        elif platform == 'rumble':
            embed_url = f"https://rumble.com/embed/v{video_id}/"
            embed_html = f'''<iframe width="560" height="315" src="{embed_url}" frameborder="0" allowfullscreen></iframe>'''
            return {
                "type": "video",
                "platform": "rumble",
                "video_id": video_id,
                "url": url,
                "embed_url": embed_url,
                "embed_html": embed_html,
            }
        return None
    
    async def _create_file_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Create a file with the given content - NC-0.8.0.9"""
        import os
        import logging
        from pathlib import Path
        
        logger = logging.getLogger(__name__)
        
        path = args.get("path", "").strip()
        content = args.get("content", "")
        overwrite = args.get("overwrite", True)
        
        if not path:
            return {"error": "No path provided"}
        
        if not content:
            return {"error": "No content provided"}
        
        # Sanitize path - prevent directory traversal
        path = path.lstrip("/").lstrip("\\")
        path = path.replace("..", "").replace("~", "")
        
        # Get safe filename
        safe_path = Path(path).name if "/" not in path and "\\" not in path else path
        
        # Determine output directory
        # Use /app/data/artifacts for container, fallback to temp
        artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/app/data/artifacts")
        if not os.path.exists(artifacts_dir):
            try:
                os.makedirs(artifacts_dir, exist_ok=True)
            except:
                artifacts_dir = "/tmp/artifacts"
                os.makedirs(artifacts_dir, exist_ok=True)
        
        # Handle subdirectories in path
        if "/" in path or "\\" in path:
            # Create subdirectory structure
            subdir = os.path.dirname(path)
            full_dir = os.path.join(artifacts_dir, subdir)
            try:
                os.makedirs(full_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create subdirectory {subdir}: {e}")
                # Fall back to flat structure
                safe_path = path.replace("/", "_").replace("\\", "_")
                full_dir = artifacts_dir
            full_path = os.path.join(full_dir, os.path.basename(path))
        else:
            full_path = os.path.join(artifacts_dir, safe_path)
        
        # Check if file exists
        if os.path.exists(full_path) and not overwrite:
            return {
                "error": f"File already exists: {path}",
                "hint": "Set overwrite=true to replace the file"
            }
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_size = os.path.getsize(full_path)
            line_count = content.count('\n') + 1
            
            logger.info(f"[CREATE_FILE] Created: {full_path} ({file_size} bytes, {line_count} lines)")
            
            # Store in context for artifact tracking if available
            if context:
                chat_id = context.get("chat_id")
                message_id = context.get("message_id")
                db = context.get("db")
                
                if chat_id and message_id and db:
                    try:
                        from sqlalchemy import text
                        # Save artifact metadata to message
                        artifact_data = {
                            "type": "file",
                            "path": path,
                            "full_path": full_path,
                            "size": file_size,
                            "lines": line_count,
                        }
                        # This could be enhanced to save to message_metadata
                        logger.debug(f"[CREATE_FILE] Artifact metadata: {artifact_data}")
                    except Exception as e:
                        logger.warning(f"Could not save artifact metadata: {e}")
            
            return {
                "success": True,
                "path": path,
                "full_path": full_path,
                "size_bytes": file_size,
                "line_count": line_count,
                "message": f"File created: {path} ({file_size} bytes)"
            }
            
        except Exception as e:
            logger.error(f"[CREATE_FILE] Failed to create {path}: {e}")
            return {"error": f"Failed to create file: {str(e)}"}
    
    async def _fetch_urls_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Fetch multiple URLs, with special handling for video platforms."""
        import asyncio
        
        urls = args.get("urls", [])
        extract_main = args.get("extract_main_content", True)
        
        if not urls:
            return {"error": "No URLs provided"}
        
        if isinstance(urls, str):
            urls = [urls]
        
        # Fetch all URLs concurrently (with limit)
        async def fetch_one(u):
            return await self._fetch_single_url(u, extract_main)
        
        tasks = [fetch_one(url) for url in urls[:10]]  # Limit to 10 URLs
        results = await asyncio.gather(*tasks)
        
        return {
            "results": results,
            "fetched_count": len(results),
            "video_count": sum(1 for r in results if r.get("type") == "video"),
            "webpage_count": sum(1 for r in results if r.get("type") == "webpage"),
            "error_count": sum(1 for r in results if "error" in r),
        }

    async def _add_task_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Add a single task to the queue"""
        import logging
        logger = logging.getLogger(__name__)
        
        description = args.get("description", "").strip()
        instructions = args.get("instructions", "").strip()
        priority = args.get("priority", 0)
        auto_continue = args.get("auto_continue", True)
        
        if not description:
            return {"error": "Task description is required"}
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService, TaskSource
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            task = await queue_service.add_task(
                description=description,
                instructions=instructions,
                source=TaskSource.LLM,
                priority=priority,
                auto_continue=auto_continue,
            )
            
            status = await queue_service.get_queue_status()
            logger.info(f"[TASK_QUEUE] Added task '{description}' to chat {chat_id}")
            
            return {
                "success": True,
                "task_id": task.id,
                "description": task.description,
                "status": task.status.value,
                "queue_length": status["queue_length"],
                "current_task": status["current_task"]["description"] if status["current_task"] else None,
            }
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error adding task: {e}")
            return {"error": f"Failed to add task: {str(e)}"}
    
    async def _add_tasks_batch_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Add multiple tasks to the queue"""
        import logging
        logger = logging.getLogger(__name__)
        
        tasks_data = args.get("tasks", [])
        
        if not tasks_data:
            return {"error": "No tasks provided"}
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService, TaskSource
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            tasks = await queue_service.add_tasks_batch(
                tasks=tasks_data,
                source=TaskSource.LLM,
            )
            
            status = await queue_service.get_queue_status()
            logger.info(f"[TASK_QUEUE] Added {len(tasks)} tasks to chat {chat_id}")
            
            return {
                "success": True,
                "tasks_added": len(tasks),
                "descriptions": [t.description for t in tasks],
                "queue_length": status["queue_length"],
                "current_task": status["current_task"]["description"] if status["current_task"] else None,
            }
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error adding tasks: {e}")
            return {"error": f"Failed to add tasks: {str(e)}"}
    
    async def _complete_task_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Mark current task as completed"""
        import logging
        logger = logging.getLogger(__name__)
        
        result_summary = args.get("result_summary", "")
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            completed_task, next_task = await queue_service.complete_task(result_summary)
            
            if not completed_task:
                return {"error": "No task in progress to complete"}
            
            logger.info(f"[TASK_QUEUE] Completed '{completed_task.description}' in chat {chat_id}")
            
            result = {
                "success": True,
                "completed": completed_task.description,
                "result_summary": result_summary or "Completed",
            }
            
            if next_task:
                result["next_task"] = {
                    "description": next_task.description,
                    "instructions": next_task.instructions,
                }
                result["message"] = f"Task completed. Now working on: {next_task.description}"
            else:
                status = await queue_service.get_queue_status()
                result["queue_length"] = status["queue_length"]
                result["message"] = "Task completed. No more tasks in queue." if status["queue_length"] == 0 else f"Task completed. {status['queue_length']} tasks remaining (paused or waiting)."
            
            return result
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error completing task: {e}")
            return {"error": f"Failed to complete task: {str(e)}"}
    
    async def _fail_task_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Mark current task as failed"""
        import logging
        logger = logging.getLogger(__name__)
        
        reason = args.get("reason", "")
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            failed_task = await queue_service.fail_task(reason)
            
            if not failed_task:
                return {"error": "No task in progress to fail"}
            
            logger.info(f"[TASK_QUEUE] Failed '{failed_task.description}' in chat {chat_id}: {reason}")
            
            status = await queue_service.get_queue_status()
            return {
                "success": True,
                "failed_task": failed_task.description,
                "reason": reason or "Unspecified",
                "queue_length": status["queue_length"],
            }
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error failing task: {e}")
            return {"error": f"Failed to mark task as failed: {str(e)}"}
    
    async def _skip_task_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Skip current task"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            skipped_task, next_task = await queue_service.skip_task()
            
            if not skipped_task:
                return {"error": "No task in progress to skip"}
            
            logger.info(f"[TASK_QUEUE] Skipped '{skipped_task.description}' in chat {chat_id}")
            
            result = {
                "success": True,
                "skipped": skipped_task.description,
            }
            
            if next_task:
                result["next_task"] = next_task.description
            else:
                status = await queue_service.get_queue_status()
                result["queue_length"] = status["queue_length"]
            
            return result
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error skipping task: {e}")
            return {"error": f"Failed to skip task: {str(e)}"}
    
    async def _get_task_queue_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Get the current task queue status"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            status = await queue_service.get_queue_status()
            
            return status
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error getting queue: {e}")
            return {"error": f"Failed to get queue: {str(e)}"}
    
    async def _clear_task_queue_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Clear all tasks from the queue"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            cleared = await queue_service.clear_queue()
            
            logger.info(f"[TASK_QUEUE] Cleared {cleared} tasks from chat {chat_id}")
            
            return {
                "success": True,
                "tasks_cleared": cleared,
            }
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error clearing queue: {e}")
            return {"error": f"Failed to clear queue: {str(e)}"}
    
    async def _pause_task_queue_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Pause task queue execution"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            await queue_service.pause_queue()
            
            status = await queue_service.get_queue_status()
            logger.info(f"[TASK_QUEUE] Paused queue in chat {chat_id}")
            
            return {
                "success": True,
                "paused": True,
                "queue_length": status["queue_length"],
                "current_task": status["current_task"]["description"] if status["current_task"] else None,
            }
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error pausing queue: {e}")
            return {"error": f"Failed to pause queue: {str(e)}"}
    
    async def _resume_task_queue_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Resume task queue execution"""
        import logging
        logger = logging.getLogger(__name__)
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Missing context - task queue requires active chat"}
        
        try:
            from app.services.task_queue import TaskQueueService
            
            db = context["db"]
            chat_id = context["chat_id"]
            
            queue_service = TaskQueueService(db, chat_id)
            next_task = await queue_service.resume_queue()
            
            status = await queue_service.get_queue_status()
            logger.info(f"[TASK_QUEUE] Resumed queue in chat {chat_id}")
            
            result = {
                "success": True,
                "paused": False,
                "queue_length": status["queue_length"],
            }
            
            if next_task:
                result["started_task"] = next_task.description
            
            return result
        except Exception as e:
            logger.error(f"[TASK_QUEUE] Error resuming queue: {e}")
            return {"error": f"Failed to resume queue: {str(e)}"}
    
    async def _agent_search_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Search through archived conversation history in Agent Memory files."""
        query = args.get("query", "")
        max_results = args.get("max_results", 3)
        
        if not query:
            return {"error": "Query is required"}
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Database context required for agent memory search"}
        
        db = context["db"]
        chat_id = context["chat_id"]
        
        try:
            from app.services.agent_memory import AgentMemoryService
            
            agent_memory = AgentMemoryService()
            results = await agent_memory.search_agent_files(
                db=db,
                chat_id=chat_id,
                query=query,
                max_results=max_results,
            )
            
            if not results:
                return {
                    "found": False,
                    "message": "No relevant archived conversation history found for this query."
                }
            
            # NC-0.8.0.8: Format results with memory_number and offset for memory_read()
            formatted_results = []
            for r in results:
                filename = r.get("filename", "")
                # Extract memory number from filename (e.g., "{Agent0001}.md" -> 1)
                memory_num = 0
                import re
                match = re.search(r'Agent(\d+)', filename)
                if match:
                    memory_num = int(match.group(1))
                
                formatted_results.append({
                    "memory_number": memory_num,
                    "filename": filename,
                    "offset": r.get("offset", 0),
                    "preview": r.get("preview", r.get("snippet", "")),
                    "score": r.get("score", 0),
                })
            
            return {
                "found": True,
                "results_count": len(formatted_results),
                "results": formatted_results,
                "tip": "Use memory_read(memory_number, offset) to get full content."
            }
            
        except Exception as e:
            return {"error": f"Agent memory search failed: {str(e)}"}
    
    async def _agent_read_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Read the contents of a specific Agent Memory file."""
        filename = args.get("filename", "")
        
        if not filename:
            return {"error": "Filename is required"}
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Database context required for agent memory read"}
        
        db = context["db"]
        chat_id = context["chat_id"]
        
        try:
            from app.services.agent_memory import AgentMemoryService, AGENT_FILE_PREFIX
            from sqlalchemy import select
            from app.models.models import UploadedFile
            
            # Normalize filename
            if not filename.startswith(AGENT_FILE_PREFIX):
                filename = f"{AGENT_FILE_PREFIX}{filename}"
            if not filename.endswith(".md"):
                filename = f"{filename}.md"
            
            # Query for the file
            result = await db.execute(
                select(UploadedFile)
                .where(UploadedFile.chat_id == chat_id)
                .where(UploadedFile.filepath == filename)
            )
            agent_file = result.scalar_one_or_none()
            
            if not agent_file:
                # List available agent files
                all_result = await db.execute(
                    select(UploadedFile.filepath)
                    .where(UploadedFile.chat_id == chat_id)
                    .where(UploadedFile.filepath.like(f"{AGENT_FILE_PREFIX}%"))
                )
                available = [r[0] for r in all_result.fetchall()]
                
                return {
                    "error": f"Agent memory file '{filename}' not found",
                    "available_files": available if available else "No agent memory files exist for this chat"
                }
            
            return {
                "filename": filename,
                "content": agent_file.content,
                "size": agent_file.size,
            }
            
        except Exception as e:
            return {"error": f"Agent memory read failed: {str(e)}"}
    
    # NC-0.8.0.8: New memory_read handler with offset support
    async def _memory_read_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Read contents from a specific Agent Memory file at a given offset."""
        memory_number = args.get("memory_number", 0)
        offset = args.get("offset", 0)
        length = args.get("length", 4000)
        
        if not memory_number:
            return {"error": "memory_number is required (e.g., 1 for Agent0001.md)"}
        
        if not context or "db" not in context or "chat_id" not in context:
            return {"error": "Database context required for memory read"}
        
        db = context["db"]
        chat_id = context["chat_id"]
        
        try:
            from app.services.agent_memory import AgentMemoryService, AGENT_FILE_PREFIX
            from sqlalchemy import select
            from app.models.models import UploadedFile
            
            # Build filename from memory number (e.g., 1 -> "{Agent0001}.md")
            filename = f"{AGENT_FILE_PREFIX}{memory_number:04d}}}.md"
            
            # Query for the file
            result = await db.execute(
                select(UploadedFile)
                .where(UploadedFile.chat_id == chat_id)
                .where(UploadedFile.filepath == filename)
            )
            agent_file = result.scalar_one_or_none()
            
            if not agent_file:
                # List available agent files
                all_result = await db.execute(
                    select(UploadedFile.filepath)
                    .where(UploadedFile.chat_id == chat_id)
                    .where(UploadedFile.filepath.like(f"{AGENT_FILE_PREFIX}%"))
                )
                available = [r[0] for r in all_result.fetchall()]
                
                # Extract memory numbers from available files
                available_nums = []
                import re
                for f in available:
                    match = re.search(r'Agent(\d+)', f)
                    if match:
                        available_nums.append(int(match.group(1)))
                
                return {
                    "error": f"Memory file {memory_number} (Agent{memory_number:04d}.md) not found",
                    "available_memory_numbers": sorted(available_nums) if available_nums else "No agent memory files exist for this chat"
                }
            
            # Extract content at offset
            content = agent_file.content or ""
            total_size = len(content)
            
            # Clamp offset and length
            offset = max(0, min(offset, total_size))
            end_offset = min(offset + length, total_size)
            
            extracted_content = content[offset:end_offset]
            
            return {
                "memory_number": memory_number,
                "filename": filename,
                "offset": offset,
                "length": len(extracted_content),
                "total_size": total_size,
                "has_more": end_offset < total_size,
                "next_offset": end_offset if end_offset < total_size else None,
                "content": extracted_content,
            }
            
        except Exception as e:
            return {"error": f"Memory read failed: {str(e)}"}
    
    # NC-0.8.0.7: Temporary MCP server management handlers
    async def _install_mcp_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Install a temporary MCP server."""
        name = args.get("name", "").strip()
        url = args.get("url", "").strip()
        description = args.get("description", "")
        api_key = args.get("api_key", "")
        
        if not name:
            return {"error": "Server name is required"}
        if not url:
            return {"error": "Server URL is required"}
        
        if not context or "db" not in context or "user" not in context:
            return {"error": "Authentication context required for MCP installation"}
        
        db = context["db"]
        user = context["user"]
        
        try:
            from app.services.temp_mcp_manager import get_temp_mcp_manager
            
            manager = get_temp_mcp_manager()
            if not manager:
                return {"error": "MCP installation service not available"}
            
            result = await manager.install_temp_server(
                db=db,
                user_id=user.id,
                name=name,
                url=url,
                description=description,
                api_key=api_key
            )
            
            return result
            
        except Exception as e:
            return {"error": f"MCP installation failed: {str(e)}"}
    
    async def _uninstall_mcp_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Uninstall a temporary MCP server."""
        name = args.get("name", "").strip()
        
        if not name:
            return {"error": "Server name is required"}
        
        if not context or "db" not in context or "user" not in context:
            return {"error": "Authentication context required for MCP uninstallation"}
        
        db = context["db"]
        user = context["user"]
        
        try:
            from app.services.temp_mcp_manager import get_temp_mcp_manager
            
            manager = get_temp_mcp_manager()
            if not manager:
                return {"error": "MCP service not available"}
            
            result = await manager.uninstall_temp_server(
                db=db,
                user_id=user.id,
                tool_name=name
            )
            
            return result
            
        except Exception as e:
            return {"error": f"MCP uninstallation failed: {str(e)}"}
    
    async def _list_mcp_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """List all temporary MCP servers."""
        if not context or "db" not in context or "user" not in context:
            return {"error": "Authentication context required to list MCP servers"}
        
        db = context["db"]
        user = context["user"]
        
        try:
            from app.services.temp_mcp_manager import get_temp_mcp_manager
            
            manager = get_temp_mcp_manager()
            if not manager:
                return {"error": "MCP service not available"}
            
            servers = await manager.list_temp_servers(db=db, user_id=user.id)
            
            if not servers:
                return {
                    "message": "No temporary MCP servers installed",
                    "servers": []
                }
            
            return {
                "count": len(servers),
                "servers": servers,
                "note": "Servers are automatically removed after 4 hours of non-use"
            }
            
        except Exception as e:
            return {"error": f"Failed to list MCP servers: {str(e)}"}


# Session-based file store for uploaded files
# Key: chat_id, Value: Dict[filename, content]
_session_files: Dict[str, Dict[str, str]] = {}


def store_session_file(chat_id: str, filename: str, content: str):
    """Store a file in the session for tool access"""
    if chat_id not in _session_files:
        _session_files[chat_id] = {}
    _session_files[chat_id][filename] = content


def get_session_file(chat_id: str, filename: str) -> Optional[str]:
    """Get a file from the session"""
    return _session_files.get(chat_id, {}).get(filename)


def get_session_files(chat_id: str) -> Dict[str, str]:
    """Get all files for a chat session"""
    return _session_files.get(chat_id, {})


async def get_session_file_with_db_fallback(chat_id: str, filename: str, db) -> Optional[str]:
    """Get a file from session, falling back to database if not in memory"""
    from sqlalchemy import select
    from app.models.upload import UploadedFile
    
    # Try in-memory first
    content = get_session_file(chat_id, filename)
    if content:
        return content
    
    # Fallback to database
    try:
        result = await db.execute(
            select(UploadedFile)
            .where(UploadedFile.chat_id == chat_id)
            .where(UploadedFile.filepath == filename)
        )
        uploaded = result.scalar_one_or_none()
        if uploaded and uploaded.content:
            # Cache it for future use
            store_session_file(chat_id, filename, uploaded.content)
            return uploaded.content
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load file from DB: {e}")
    
    return None


async def get_session_files_with_db_fallback(chat_id: str, db) -> Dict[str, str]:
    """Get all files for a chat session, including from database"""
    from sqlalchemy import select
    from app.models.upload import UploadedFile
    
    # Start with in-memory files
    files = dict(get_session_files(chat_id))
    
    # Add any database files not in memory
    try:
        result = await db.execute(
            select(UploadedFile)
            .where(UploadedFile.chat_id == chat_id)
            .where(UploadedFile.content.isnot(None))
        )
        uploaded_files = result.scalars().all()
        for uf in uploaded_files:
            if uf.filepath not in files and uf.content:
                files[uf.filepath] = uf.content
                # Cache it
                store_session_file(chat_id, uf.filepath, uf.content)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load files from DB: {e}")
    
    return files


def clear_session_files(chat_id: str):
    """Clear files for a chat session"""
    if chat_id in _session_files:
        del _session_files[chat_id]


class FileViewingTools:
    """Tools for viewing uploaded files partially"""
    
    @staticmethod
    async def view_file_lines(arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """View specific lines from an uploaded file"""
        filename = arguments.get("filename")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        
        if not filename:
            return {"error": "filename is required"}
        
        chat_id = context.get("chat_id") if context else None
        db = context.get("db") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Try with database fallback
        if db:
            content = await get_session_file_with_db_fallback(chat_id, filename, db)
        else:
            content = get_session_file(chat_id, filename)
            
        if not content:
            # List available files (also with db fallback)
            if db:
                all_files = await get_session_files_with_db_fallback(chat_id, db)
            else:
                all_files = get_session_files(chat_id)
            available = list(all_files.keys())
            return {
                "error": f"File '{filename}' not found",
                "available_files": available if available else "No files uploaded in this session"
            }
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Adjust indices (1-based to 0-based)
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line) if end_line else total_lines
        
        selected_lines = lines[start_idx:end_idx]
        
        # Format with line numbers
        numbered_lines = [
            f"{start_idx + i + 1}: {line}"
            for i, line in enumerate(selected_lines)
        ]
        
        return {
            "filename": filename,
            "total_lines": total_lines,
            "showing": f"lines {start_idx + 1}-{end_idx}",
            "content": "\n".join(numbered_lines)
        }
    
    @staticmethod
    async def search_in_file(arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for a pattern in an uploaded file and show context"""
        filename = arguments.get("filename")
        pattern = arguments.get("pattern")
        context_lines = arguments.get("context_lines", 3)
        
        if not filename or not pattern:
            return {"error": "filename and pattern are required"}
        
        chat_id = context.get("chat_id") if context else None
        db = context.get("db") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Try with database fallback
        if db:
            content = await get_session_file_with_db_fallback(chat_id, filename, db)
        else:
            content = get_session_file(chat_id, filename)
            
        if not content:
            if db:
                all_files = await get_session_files_with_db_fallback(chat_id, db)
            else:
                all_files = get_session_files(chat_id)
            available = list(all_files.keys())
            return {
                "error": f"File '{filename}' not found",
                "available_files": available if available else "No files uploaded"
            }
        
        lines = content.split('\n')
        results = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Treat as literal string
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
        
        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                context_block = []
                for j in range(start, end):
                    marker = ">" if j == i else " "
                    context_block.append(f"{j + 1}{marker} {lines[j]}")
                
                results.append({
                    "line_number": i + 1,
                    "match": line.strip(),
                    "context": "\n".join(context_block)
                })
                
                # Limit results
                if len(results) >= 10:
                    break
        
        return {
            "filename": filename,
            "pattern": pattern,
            "total_matches": len(results),
            "results": results
        }
    
    @staticmethod
    async def list_uploaded_files(arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """List all uploaded files in the current session"""
        chat_id = context.get("chat_id") if context else None
        db = context.get("db") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Use database fallback
        if db:
            files = await get_session_files_with_db_fallback(chat_id, db)
        else:
            files = get_session_files(chat_id)
            
        if not files:
            return {"message": "No files uploaded in this session"}
        
        file_info = []
        for filename, content in files.items():
            lines = content.split('\n')
            file_info.append({
                "filename": filename,
                "lines": len(lines),
                "size": len(content),
                "preview": lines[0][:100] + "..." if lines and len(lines[0]) > 100 else (lines[0] if lines else "")
            })
        
        return {
            "file_count": len(files),
            "files": file_info
        }
    
    @staticmethod
    async def view_signature(arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """View code around a function/class signature"""
        filename = arguments.get("filename")
        signature_name = arguments.get("signature_name")
        lines_after = arguments.get("lines_after", 20)
        
        if not filename or not signature_name:
            return {"error": "filename and signature_name are required"}
        
        chat_id = context.get("chat_id") if context else None
        db = context.get("db") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Try with database fallback
        if db:
            content = await get_session_file_with_db_fallback(chat_id, filename, db)
        else:
            content = get_session_file(chat_id, filename)
            
        if not content:
            if db:
                all_files = await get_session_files_with_db_fallback(chat_id, db)
            else:
                all_files = get_session_files(chat_id)
            available = list(all_files.keys())
            return {
                "error": f"File '{filename}' not found",
                "available_files": available if available else "No files uploaded"
            }
        
        lines = content.split('\n')
        
        # Search for the signature name in common patterns
        patterns = [
            rf"^\s*(?:async\s+)?def\s+{re.escape(signature_name)}\s*\(",  # Python function
            rf"^\s*class\s+{re.escape(signature_name)}[\s:(]",  # Python/JS class
            rf"^\s*(?:export\s+)?(?:async\s+)?function\s+{re.escape(signature_name)}\s*\(",  # JS function
            rf"^\s*(?:export\s+)?const\s+{re.escape(signature_name)}\s*=",  # JS const
            rf"^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+{re.escape(signature_name)}\s*\(",  # Java/C# method
            rf"^{re.escape(signature_name)}:",  # Assembly label
            rf"^\s*fn\s+{re.escape(signature_name)}\s*[<(]",  # Rust function
            rf"^\s*func\s+{re.escape(signature_name)}\s*\(",  # Go function
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    start = i
                    end = min(len(lines), i + lines_after + 1)
                    
                    numbered = [f"{start + j + 1}: {lines[start + j]}" for j in range(end - start)]
                    
                    return {
                        "filename": filename,
                        "signature": signature_name,
                        "start_line": start + 1,
                        "content": "\n".join(numbered)
                    }
        
        return {
            "error": f"Signature '{signature_name}' not found in {filename}",
            "tip": "Try using search_in_file to find the exact location"
        }
    
    @staticmethod
    async def request_file(arguments: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retrieve file content starting at a specific offset.
        NC-0.8.0.6: Handles truncated file requests from LLM.
        """
        path = arguments.get("path")
        offset = arguments.get("offset", 0)
        length = arguments.get("length", 20000)  # Default 20KB chunk
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        db = context.get("db") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Try to get file content
        content = None
        if db:
            content = await get_session_file_with_db_fallback(chat_id, path, db)
        else:
            content = get_session_file(chat_id, path)
        
        if not content:
            # List available files
            if db:
                all_files = await get_session_files_with_db_fallback(chat_id, db)
            else:
                all_files = get_session_files(chat_id)
            available = list(all_files.keys())
            return {
                "error": f"File '{path}' not found",
                "available_files": available if available else "No files uploaded"
            }
        
        total_size = len(content)
        
        # Validate offset
        if offset >= total_size:
            return {
                "error": f"Offset {offset} exceeds file size {total_size}",
                "total_size": total_size
            }
        
        # Extract chunk
        end_offset = min(offset + length, total_size)
        chunk = content[offset:end_offset]
        
        # Try to break at line boundary if not at end
        if end_offset < total_size:
            last_newline = chunk.rfind('\n')
            if last_newline > length // 2:
                chunk = chunk[:last_newline + 1]
                end_offset = offset + len(chunk)
        
        result = {
            "path": path,
            "offset": offset,
            "end_offset": end_offset,
            "total_size": total_size,
            "content": chunk,
        }
        
        # Add continuation hint if more content available
        if end_offset < total_size:
            remaining = total_size - end_offset
            result["more_available"] = True
            result["remaining_chars"] = remaining
            result["next_request"] = f'<request_file path="{path}" offset="{end_offset}"/>'
        else:
            result["more_available"] = False
            result["message"] = "End of file reached"
        
        return result


# Global tool registry instance
tool_registry = ToolRegistry()
