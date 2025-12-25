"""
Artifact/File Editing Tools for LLM

Provides tools for creating and editing artifacts (code files, configs, etc.)
with proper state tracking to prevent confusion when search_replace fails.

Key features:
- State tracking: Maintains current content of all artifacts per chat session
- Smart error messages: Shows actual content when search fails
- Duplicate detection: Prevents repeated identical failed operations
- Clear feedback: Explicit success/failure with reasons
"""
import re
import logging
import difflib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ArtifactState:
    """Tracks the current state of an artifact"""
    path: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    history: List[str] = field(default_factory=list)  # Previous versions
    
    def update(self, new_content: str):
        """Update content and track history"""
        # Keep last 5 versions
        if len(self.history) >= 5:
            self.history.pop(0)
        self.history.append(self.content)
        
        self.content = new_content
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1


@dataclass  
class FailedOperation:
    """Tracks a failed operation to prevent identical retries"""
    operation: str  # 'search_replace'
    params_hash: str
    timestamp: datetime
    error: str


class ArtifactStateManager:
    """
    Manages artifact state per chat session.
    
    Prevents the confusion seen in search_replace operations by:
    1. Tracking actual current content
    2. Returning current content when search fails
    3. Detecting repeated identical failures
    """
    
    def __init__(self):
        # chat_id -> path -> ArtifactState
        self._artifacts: Dict[str, Dict[str, ArtifactState]] = defaultdict(dict)
        # chat_id -> list of recent failed operations
        self._failed_ops: Dict[str, List[FailedOperation]] = defaultdict(list)
        # Max failed ops to track per session
        self._max_failed_ops = 10
    
    def create_artifact(
        self,
        chat_id: str,
        path: str,
        content: str,
    ) -> ArtifactState:
        """Create a new artifact or overwrite existing"""
        artifact = ArtifactState(path=path, content=content)
        self._artifacts[chat_id][path] = artifact
        logger.info(f"[ARTIFACT] Created: {path} ({len(content)} chars) in chat {chat_id[:8]}")
        return artifact
    
    def get_artifact(self, chat_id: str, path: str) -> Optional[ArtifactState]:
        """Get current state of an artifact"""
        return self._artifacts.get(chat_id, {}).get(path)
    
    def list_artifacts(self, chat_id: str) -> List[Dict[str, Any]]:
        """List all artifacts in a chat session"""
        artifacts = self._artifacts.get(chat_id, {})
        return [
            {
                "path": path,
                "size": len(state.content),
                "lines": state.content.count('\n') + 1,
                "version": state.version,
                "updated_at": state.updated_at.isoformat(),
            }
            for path, state in artifacts.items()
        ]
    
    def update_artifact(
        self,
        chat_id: str,
        path: str,
        new_content: str,
    ) -> Optional[ArtifactState]:
        """Update an existing artifact"""
        artifact = self.get_artifact(chat_id, path)
        if artifact:
            artifact.update(new_content)
            logger.info(f"[ARTIFACT] Updated: {path} v{artifact.version} ({len(new_content)} chars)")
            return artifact
        return None
    
    def search_replace(
        self,
        chat_id: str,
        path: str,
        search_text: str,
        replace_text: str,
    ) -> Dict[str, Any]:
        """
        Perform search and replace on an artifact.
        
        Returns detailed information including:
        - Success/failure status
        - Actual current content on failure
        - Similar matches if exact match not found
        - Duplicate operation detection
        """
        artifact = self.get_artifact(chat_id, path)
        
        if not artifact:
            return {
                "success": False,
                "error": f"Artifact not found: {path}",
                "available_artifacts": self.list_artifacts(chat_id),
                "hint": "Use create_artifact to create the file first, or check the path.",
            }
        
        current_content = artifact.content
        
        # Check for duplicate failed operation
        params_hash = f"{path}:{hash(search_text)}"
        duplicate = self._check_duplicate_failure(chat_id, params_hash)
        if duplicate:
            return {
                "success": False,
                "error": "This exact search_replace was already attempted and failed",
                "previous_error": duplicate.error,
                "hint": "The search text doesn't match the current file content. Read the 'actual_content' below to see the current state.",
                "actual_content": self._truncate_content(current_content),
                "actual_lines": current_content.count('\n') + 1,
            }
        
        # Try exact match
        if search_text in current_content:
            # Count occurrences
            count = current_content.count(search_text)
            if count > 1:
                return {
                    "success": False,
                    "error": f"Search text found {count} times. Must be unique for safe replacement.",
                    "hint": "Add more context to make the search text unique.",
                    "matches_preview": self._find_matches_with_context(current_content, search_text),
                }
            
            # Perform replacement
            new_content = current_content.replace(search_text, replace_text, 1)
            artifact.update(new_content)
            
            # Clear failed ops for this file since we succeeded
            self._failed_ops[chat_id] = [
                op for op in self._failed_ops[chat_id]
                if not op.params_hash.startswith(f"{path}:")
            ]
            
            return {
                "success": True,
                "path": path,
                "version": artifact.version,
                "message": f"Successfully replaced text in {path}",
                "diff_preview": self._generate_diff_preview(search_text, replace_text),
            }
        
        # Search text not found - provide helpful information
        error_info = {
            "success": False,
            "error": "Search text not found in file",
            "path": path,
            "search_text_preview": self._truncate_content(search_text, 200),
        }
        
        # Find similar matches
        similar = self._find_similar_matches(current_content, search_text)
        if similar:
            error_info["similar_matches"] = similar
            error_info["hint"] = "Found similar text. The file may have been modified. Review similar_matches and actual_content."
        else:
            error_info["hint"] = "No similar text found. The file content may be completely different from expected."
        
        # Always include actual content for recovery
        error_info["actual_content"] = self._truncate_content(current_content)
        error_info["actual_lines"] = current_content.count('\n') + 1
        
        # Track this failed operation
        self._track_failure(chat_id, "search_replace", params_hash, error_info["error"])
        
        return error_info
    
    def _check_duplicate_failure(
        self,
        chat_id: str,
        params_hash: str,
    ) -> Optional[FailedOperation]:
        """Check if this exact operation already failed"""
        for op in self._failed_ops.get(chat_id, []):
            if op.params_hash == params_hash:
                return op
        return None
    
    def _track_failure(
        self,
        chat_id: str,
        operation: str,
        params_hash: str,
        error: str,
    ):
        """Track a failed operation"""
        ops = self._failed_ops[chat_id]
        
        # Remove old ops if at limit
        if len(ops) >= self._max_failed_ops:
            ops.pop(0)
        
        ops.append(FailedOperation(
            operation=operation,
            params_hash=params_hash,
            timestamp=datetime.now(timezone.utc),
            error=error,
        ))
    
    def _truncate_content(self, content: str, max_length: int = 2000) -> str:
        """Truncate content with indicator"""
        if len(content) <= max_length:
            return content
        
        half = max_length // 2
        return f"{content[:half]}\n\n... [truncated {len(content) - max_length} chars] ...\n\n{content[-half:]}"
    
    def _find_similar_matches(
        self,
        content: str,
        search_text: str,
        threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """Find text in content similar to search_text"""
        search_lines = search_text.strip().split('\n')
        content_lines = content.split('\n')
        
        matches = []
        
        # For single-line search
        if len(search_lines) == 1:
            search_line = search_lines[0].strip()
            for i, line in enumerate(content_lines):
                ratio = difflib.SequenceMatcher(None, search_line, line.strip()).ratio()
                if ratio >= threshold:
                    matches.append({
                        "line_number": i + 1,
                        "similarity": f"{ratio:.0%}",
                        "content": line,
                    })
                if len(matches) >= 5:
                    break
        else:
            # Multi-line search - look for similar blocks
            search_block = '\n'.join(search_lines)
            for i in range(len(content_lines) - len(search_lines) + 1):
                block = '\n'.join(content_lines[i:i + len(search_lines)])
                ratio = difflib.SequenceMatcher(None, search_block, block).ratio()
                if ratio >= threshold:
                    matches.append({
                        "start_line": i + 1,
                        "end_line": i + len(search_lines),
                        "similarity": f"{ratio:.0%}",
                        "content": block,
                    })
                if len(matches) >= 3:
                    break
        
        return matches
    
    def _find_matches_with_context(
        self,
        content: str,
        search_text: str,
        context_lines: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find all matches with surrounding context"""
        lines = content.split('\n')
        search_lines = search_text.split('\n')
        matches = []
        
        # Find start positions of matches
        idx = 0
        while True:
            pos = content.find(search_text, idx)
            if pos == -1:
                break
            
            # Convert position to line number
            line_num = content[:pos].count('\n')
            start = max(0, line_num - context_lines)
            end = min(len(lines), line_num + len(search_lines) + context_lines)
            
            context_block = '\n'.join(
                f"{i+1}: {lines[i]}" for i in range(start, end)
            )
            
            matches.append({
                "match_number": len(matches) + 1,
                "line": line_num + 1,
                "context": context_block,
            })
            
            idx = pos + 1
            if len(matches) >= 5:
                break
        
        return matches
    
    def _generate_diff_preview(
        self,
        old_text: str,
        new_text: str,
        max_lines: int = 10,
    ) -> str:
        """Generate a simple diff preview"""
        old_lines = old_text.split('\n')[:max_lines]
        new_lines = new_text.split('\n')[:max_lines]
        
        diff = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='before',
            tofile='after',
            lineterm='',
        ))
        
        if len(diff) > max_lines * 2:
            diff = diff[:max_lines * 2] + ['... (diff truncated)']
        
        return '\n'.join(diff)
    
    def clear_session(self, chat_id: str):
        """Clear all artifacts for a chat session"""
        if chat_id in self._artifacts:
            del self._artifacts[chat_id]
        if chat_id in self._failed_ops:
            del self._failed_ops[chat_id]


# Global instance
artifact_manager = ArtifactStateManager()


class ArtifactTools:
    """Tool handlers for artifact operations"""
    
    @staticmethod
    async def create_artifact(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create a new artifact (file)"""
        path = arguments.get("path")
        content = arguments.get("content", "")
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        # Check if exists
        existing = artifact_manager.get_artifact(chat_id, path)
        if existing and not arguments.get("overwrite", False):
            return {
                "success": False,
                "error": f"Artifact already exists: {path}",
                "hint": "Set overwrite=true to replace, or use search_replace to modify.",
                "existing_preview": artifact_manager._truncate_content(existing.content, 500),
            }
        
        artifact = artifact_manager.create_artifact(chat_id, path, content)
        
        return {
            "success": True,
            "path": path,
            "size": len(content),
            "lines": content.count('\n') + 1,
            "message": f"Created artifact: {path}",
        }
    
    @staticmethod
    async def read_artifact(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Read the current content of an artifact"""
        path = arguments.get("path")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        artifact = artifact_manager.get_artifact(chat_id, path)
        
        if not artifact:
            return {
                "success": False,
                "error": f"Artifact not found: {path}",
                "available_artifacts": artifact_manager.list_artifacts(chat_id),
            }
        
        lines = artifact.content.split('\n')
        total_lines = len(lines)
        
        # Handle line range
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line) if end_line else total_lines
        
        selected = lines[start_idx:end_idx]
        numbered = [f"{start_idx + i + 1}: {line}" for i, line in enumerate(selected)]
        
        return {
            "success": True,
            "path": path,
            "version": artifact.version,
            "total_lines": total_lines,
            "showing": f"lines {start_idx + 1}-{end_idx}",
            "content": '\n'.join(numbered),
        }
    
    @staticmethod
    async def search_replace(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Search and replace in an artifact.
        
        Key features:
        - Returns actual file content when search fails
        - Detects duplicate failed operations
        - Finds similar matches to help locate the right text
        """
        path = arguments.get("path")
        search_text = arguments.get("search")
        replace_text = arguments.get("replace", "")
        
        if not path:
            return {"error": "path is required"}
        if not search_text:
            return {"error": "search text is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        return artifact_manager.search_replace(
            chat_id=chat_id,
            path=path,
            search_text=search_text,
            replace_text=replace_text,
        )
    
    @staticmethod
    async def list_artifacts(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """List all artifacts in the current session"""
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        artifacts = artifact_manager.list_artifacts(chat_id)
        
        if not artifacts:
            return {"message": "No artifacts created in this session"}
        
        return {
            "count": len(artifacts),
            "artifacts": artifacts,
        }
    
    @staticmethod
    async def append_to_artifact(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Append content to an existing artifact"""
        path = arguments.get("path")
        content = arguments.get("content", "")
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        artifact = artifact_manager.get_artifact(chat_id, path)
        
        if not artifact:
            return {
                "success": False,
                "error": f"Artifact not found: {path}",
                "available_artifacts": artifact_manager.list_artifacts(chat_id),
            }
        
        new_content = artifact.content + content
        artifact_manager.update_artifact(chat_id, path, new_content)
        
        return {
            "success": True,
            "path": path,
            "version": artifact.version,
            "new_size": len(new_content),
            "appended_size": len(content),
        }
    
    @staticmethod
    async def insert_at_line(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Insert content at a specific line number"""
        path = arguments.get("path")
        line_number = arguments.get("line_number", 1)
        content = arguments.get("content", "")
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        artifact = artifact_manager.get_artifact(chat_id, path)
        
        if not artifact:
            return {
                "success": False,
                "error": f"Artifact not found: {path}",
                "available_artifacts": artifact_manager.list_artifacts(chat_id),
            }
        
        lines = artifact.content.split('\n')
        insert_idx = max(0, min(len(lines), line_number - 1))
        
        # Insert the new content
        new_lines = content.split('\n')
        lines[insert_idx:insert_idx] = new_lines
        
        new_content = '\n'.join(lines)
        artifact_manager.update_artifact(chat_id, path, new_content)
        
        return {
            "success": True,
            "path": path,
            "version": artifact.version,
            "inserted_at_line": insert_idx + 1,
            "lines_inserted": len(new_lines),
            "new_total_lines": len(lines),
        }
    
    @staticmethod
    async def delete_lines(
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Delete a range of lines from an artifact"""
        path = arguments.get("path")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line", start_line)
        
        if not path:
            return {"error": "path is required"}
        
        chat_id = context.get("chat_id") if context else None
        if not chat_id:
            return {"error": "No chat context available"}
        
        artifact = artifact_manager.get_artifact(chat_id, path)
        
        if not artifact:
            return {
                "success": False,
                "error": f"Artifact not found: {path}",
                "available_artifacts": artifact_manager.list_artifacts(chat_id),
            }
        
        lines = artifact.content.split('\n')
        total_lines = len(lines)
        
        # Validate range
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        
        if start_idx >= end_idx:
            return {
                "success": False,
                "error": "Invalid line range",
                "total_lines": total_lines,
            }
        
        # Delete the lines
        deleted = lines[start_idx:end_idx]
        del lines[start_idx:end_idx]
        
        new_content = '\n'.join(lines)
        artifact_manager.update_artifact(chat_id, path, new_content)
        
        return {
            "success": True,
            "path": path,
            "version": artifact.version,
            "deleted_lines": f"{start_line}-{end_line}",
            "lines_deleted": len(deleted),
            "new_total_lines": len(lines),
            "deleted_preview": '\n'.join(deleted[:5]) + ('...' if len(deleted) > 5 else ''),
        }


def register_artifact_tools(registry):
    """Register artifact tools with the tool registry"""
    
    registry.register(
        name="create_artifact",
        description="Create a new artifact (code file, config, etc.). The artifact is tracked in the session and can be edited with search_replace.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path/name for the artifact (e.g., 'src/main.py', 'config.json')",
                "required": True,
            },
            "content": {
                "type": "string",
                "description": "Initial content of the artifact",
                "required": True,
            },
            "overwrite": {
                "type": "boolean",
                "description": "If true, overwrite existing artifact. Default: false",
                "required": False,
            },
        },
        handler=ArtifactTools.create_artifact,
    )
    
    registry.register(
        name="read_artifact",
        description="Read the current content of an artifact. Use this to see the actual state of a file before attempting edits.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path of the artifact to read",
                "required": True,
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed). Default: 1",
                "required": False,
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive). Default: end of file",
                "required": False,
            },
        },
        handler=ArtifactTools.read_artifact,
    )
    
    registry.register(
        name="search_replace",
        description="""Search and replace text in an artifact. 
IMPORTANT: If this fails, it returns the actual current content of the file. 
Always check the returned content before retrying with different search text.
The search text must be unique in the file and match EXACTLY.""",
        parameters={
            "path": {
                "type": "string",
                "description": "Path of the artifact to edit",
                "required": True,
            },
            "search": {
                "type": "string",
                "description": "Exact text to search for (must be unique in file)",
                "required": True,
            },
            "replace": {
                "type": "string",
                "description": "Text to replace with. Use empty string to delete.",
                "required": False,
            },
        },
        handler=ArtifactTools.search_replace,
    )
    
    registry.register(
        name="list_artifacts",
        description="List all artifacts created in the current chat session",
        parameters={},
        handler=ArtifactTools.list_artifacts,
    )
    
    registry.register(
        name="append_to_artifact",
        description="Append content to the end of an existing artifact",
        parameters={
            "path": {
                "type": "string",
                "description": "Path of the artifact",
                "required": True,
            },
            "content": {
                "type": "string",
                "description": "Content to append",
                "required": True,
            },
        },
        handler=ArtifactTools.append_to_artifact,
    )
    
    registry.register(
        name="insert_at_line",
        description="Insert content at a specific line number in an artifact",
        parameters={
            "path": {
                "type": "string",
                "description": "Path of the artifact",
                "required": True,
            },
            "line_number": {
                "type": "integer",
                "description": "Line number to insert at (1-indexed)",
                "required": True,
            },
            "content": {
                "type": "string",
                "description": "Content to insert (can be multiple lines)",
                "required": True,
            },
        },
        handler=ArtifactTools.insert_at_line,
    )
    
    registry.register(
        name="delete_lines",
        description="Delete a range of lines from an artifact",
        parameters={
            "path": {
                "type": "string",
                "description": "Path of the artifact",
                "required": True,
            },
            "start_line": {
                "type": "integer",
                "description": "First line to delete (1-indexed)",
                "required": True,
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to delete (inclusive). Default: same as start_line",
                "required": False,
            },
        },
        handler=ArtifactTools.delete_lines,
    )
