"""
Integrated Tool System for LLM
Provides built-in tools that Claude can use
"""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
import math
import re
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.rag import RAGService
from app.models.models import User


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
        
        # Web page fetcher
        self.register(
            name="fetch_webpage",
            description="Fetch and read the content of a web page. Use this to read articles, documentation, or any web content when given a URL.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "The URL of the web page to fetch",
                    "required": True,
                },
                "extract_main_content": {
                    "type": "boolean",
                    "description": "If true, attempts to extract just the main content (article text). If false, returns all text. Default: true",
                    "required": False,
                }
            },
            handler=self._webpage_fetch_handler,
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
    
    async def _webpage_fetch_handler(self, args: Dict[str, Any], context: Dict = None) -> Dict[str, Any]:
        """Fetch and extract content from a web page"""
        import httpx
        from urllib.parse import urlparse
        
        url = args.get("url", "")
        extract_main = args.get("extract_main_content", True)
        
        if not url:
            return {"error": "URL is required"}
        
        # Validate URL
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return {"error": "Invalid URL scheme. Must be http or https."}
        except Exception:
            return {"error": "Invalid URL format"}
        
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
                        "url": str(response.url),
                        "content_type": "json",
                        "content": response.text[:50000],  # Limit size
                    }
                elif "text/plain" in content_type:
                    return {
                        "url": str(response.url),
                        "content_type": "text",
                        "content": response.text[:50000],
                    }
                elif "text/html" not in content_type and "application/xhtml" not in content_type:
                    return {
                        "url": str(response.url),
                        "content_type": content_type,
                        "error": "Content type not supported for text extraction",
                    }
                
                html = response.text
                
                # Extract text from HTML
                text_content = self._extract_text_from_html(html, extract_main)
                
                return {
                    "url": str(response.url),
                    "title": self._extract_title(html),
                    "content_type": "html",
                    "content": text_content[:50000],  # Limit to ~50k chars
                    "content_length": len(text_content),
                }
                
        except httpx.TimeoutException:
            return {"error": "Request timed out"}
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Failed to fetch page: {str(e)}"}
    
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


# Global tool registry instance
tool_registry = ToolRegistry()
