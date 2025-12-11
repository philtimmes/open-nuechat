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
        """Search documents using RAG"""
        if not context or "db" not in context or "user" not in context:
            return {"error": "Document search requires authentication context"}
        
        rag_service = RAGService()
        results = await rag_service.search(
            db=context["db"],
            user=context["user"],
            query=args.get("query", ""),
            document_ids=args.get("document_ids"),
            top_k=5,
        )
        
        return {
            "query": args.get("query"),
            "results_count": len(results),
            "results": [
                {
                    "document": r["document_name"],
                    "content": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
                    "relevance": round(r["similarity"], 3),
                }
                for r in results
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


# Global tool registry instance
tool_registry = ToolRegistry()
