"""
Agent Memory Service

Handles context overflow by storing older conversation history in {AgentNNNN}.md files.
These files are:
- Stored as UploadedFile artifacts with special prefix
- Hidden from the UI artifacts viewer
- Excluded from zip exports
- Automatically searched when context is needed
- Deleted when chat is deleted (via cascade)

Each file contains:
- Summary header (max 50 lines) with key topics, timestamps, and searchable keywords
- Full conversation content below the header
"""
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import json
import re

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.models import UploadedFile, Chat
from app.core.config import settings

logger = logging.getLogger(__name__)

# Prefix for agent memory files - special character to sort first and identify
AGENT_FILE_PREFIX = "{Agent"
AGENT_FILE_SUFFIX = "}.md"

# Token thresholds
DEFAULT_TOKEN_THRESHOLD_PERCENT = 50  # Compress when history > 50% of model context
CHARS_PER_TOKEN = 4  # Rough estimate


def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens in message list"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 85
        total += 4  # Role and formatting
    return total


def is_agent_file(filename: str) -> bool:
    """Check if a filename is an agent memory file"""
    return filename.startswith(AGENT_FILE_PREFIX) and filename.endswith(AGENT_FILE_SUFFIX)


class AgentMemoryService:
    """Service for managing agent memory files"""
    
    def __init__(self, model_context_size: int = 128000):
        self.model_context_size = model_context_size
        self.threshold_tokens = int(model_context_size * DEFAULT_TOKEN_THRESHOLD_PERCENT / 100)
    
    async def get_next_agent_number(self, db: AsyncSession, chat_id: str) -> int:
        """Get the next available agent file number for a chat"""
        result = await db.execute(
            select(UploadedFile.filepath)
            .where(UploadedFile.chat_id == chat_id)
            .where(UploadedFile.filepath.like(f"{AGENT_FILE_PREFIX}%"))
        )
        existing_files = result.scalars().all()
        
        max_num = 0
        for filepath in existing_files:
            # Extract number from {AgentNNNN}.md
            match = re.search(r'\{Agent(\d+)\}\.md', filepath)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)
        
        return max_num + 1
    
    def should_compress(self, messages: List[Dict]) -> bool:
        """Check if messages exceed threshold and should be compressed"""
        total_tokens = estimate_message_tokens(messages)
        return total_tokens > self.threshold_tokens
    
    async def generate_summary_header(
        self,
        messages: List[Dict],
        llm_client,
        model: str,
        api_base: str,
        api_key: str,
    ) -> str:
        """Generate a summary header for the agent file (max 50 lines)"""
        import httpx
        
        # Format messages for summarization
        conversation_text = self._format_messages(messages)
        
        prompt = f"""Create a concise summary header for archived conversation content. This header will be used to search and retrieve relevant context.

CONVERSATION TO SUMMARIZE:
{conversation_text[:15000]}  # Limit input

Generate a header with these sections (total max 50 lines):
1. **Date Range**: When this conversation occurred
2. **Key Topics** (bullet points): Main subjects discussed
3. **Important Entities**: Names, projects, technical terms mentioned
4. **Decisions Made**: Any conclusions or choices
5. **Search Keywords**: Comma-separated terms for finding this content

Format as markdown. Be concise but comprehensive."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a precise conversation archivist. Create searchable summaries."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 1500,
                        "temperature": 0.3,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    summary = data["choices"][0]["message"]["content"]
                    # Ensure max 50 lines
                    lines = summary.split('\n')[:50]
                    return '\n'.join(lines)
                else:
                    logger.error(f"Summary generation failed: {response.status_code}")
                    return self._fallback_summary(messages)
                    
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return self._fallback_summary(messages)
    
    def _fallback_summary(self, messages: List[Dict]) -> str:
        """Generate basic summary without LLM"""
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')
        topics = set()
        
        for msg in messages:
            content = str(msg.get("content", ""))[:500]
            # Extract potential keywords
            words = re.findall(r'\b[A-Za-z][a-z]{4,}\b', content)
            topics.update(words[:20])
        
        return f"""# Agent Memory Archive
**Archived**: {now}
**Messages**: {len(messages)}
**Topics**: {', '.join(list(topics)[:15])}

---
"""
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages into readable text"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            
            lines.append(f"**{role}**: {content}")
        
        return "\n\n".join(lines)
    
    async def compress_to_agent_file(
        self,
        db: AsyncSession,
        chat_id: str,
        messages_to_compress: List[Dict],
        llm_client,
        model: str,
        api_base: str,
        api_key: str,
    ) -> str:
        """
        Compress messages into an agent memory file.
        Returns the filename of the created file.
        """
        # Get next file number
        file_num = await self.get_next_agent_number(db, chat_id)
        filename = f"{AGENT_FILE_PREFIX}{file_num:04d}{AGENT_FILE_SUFFIX}"
        
        # Generate summary header
        summary = await self.generate_summary_header(
            messages_to_compress, llm_client, model, api_base, api_key
        )
        
        # Format full content
        full_content = self._format_messages(messages_to_compress)
        
        # Build file content
        file_content = f"""{summary}

---
# Full Conversation Archive
---

{full_content}
"""
        
        # Store as UploadedFile
        agent_file = UploadedFile(
            chat_id=chat_id,
            archive_name=None,
            filepath=filename,
            filename=filename,
            extension=".md",
            language="markdown",
            size=len(file_content),
            is_binary=False,
            content=file_content,
            signatures=None,
        )
        db.add(agent_file)
        await db.commit()
        
        logger.info(f"[AGENT_MEMORY] Created {filename} for chat {chat_id} ({len(messages_to_compress)} messages, {len(file_content)} chars)")
        
        return filename
    
    async def search_agent_files(
        self,
        db: AsyncSession,
        chat_id: str,
        query: str,
        max_results: int = 3,
    ) -> List[Dict]:
        """
        Search agent memory files for relevant context.
        Returns list of relevant excerpts with their sources.
        """
        # Get all agent files for this chat
        result = await db.execute(
            select(UploadedFile)
            .where(UploadedFile.chat_id == chat_id)
            .where(UploadedFile.filepath.like(f"{AGENT_FILE_PREFIX}%"))
            .order_by(UploadedFile.created_at.desc())
        )
        agent_files = result.scalars().all()
        
        if not agent_files:
            return []
        
        # Simple keyword matching for now
        # Could be enhanced with embeddings/RAG later
        query_terms = set(query.lower().split())
        
        scored_results = []
        for af in agent_files:
            content = af.content or ""
            content_lower = content.lower()
            
            # Score based on keyword matches
            score = sum(1 for term in query_terms if term in content_lower)
            
            if score > 0:
                # Extract relevant section (around first match)
                excerpt = self._extract_relevant_excerpt(content, query_terms)
                scored_results.append({
                    "filename": af.filepath,
                    "score": score,
                    "excerpt": excerpt,
                    "created_at": af.created_at.isoformat() if af.created_at else None,
                })
        
        # Sort by score and return top results
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:max_results]
    
    def _extract_relevant_excerpt(self, content: str, query_terms: set, context_chars: int = 500) -> str:
        """Extract relevant excerpt around query matches"""
        content_lower = content.lower()
        
        # Find first matching term
        first_pos = len(content)
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos
        
        if first_pos == len(content):
            # No match found, return header
            return content[:context_chars]
        
        # Extract context around match
        start = max(0, first_pos - context_chars // 2)
        end = min(len(content), first_pos + context_chars // 2)
        
        excerpt = content[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
        
        return excerpt
    
    async def get_agent_context_for_query(
        self,
        db: AsyncSession,
        chat_id: str,
        query: str,
    ) -> Optional[str]:
        """
        Get relevant context from agent files for a query.
        Returns formatted context string or None if no relevant content found.
        """
        results = await self.search_agent_files(db, chat_id, query)
        
        if not results:
            return None
        
        context_parts = []
        for r in results:
            context_parts.append(f"[From archived history: {r['filename']}]\n{r['excerpt']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""
<archived_context>
The following is relevant context from earlier in this conversation that was archived:

{context}
</archived_context>
"""


async def compress_chat_history(
    db: AsyncSession,
    chat_id: str,
    messages: List[Dict],
    model: str,
    api_base: str,
    api_key: str,
    model_context_size: int = 128000,
    keep_recent: int = 10,
) -> Tuple[List[Dict], bool]:
    """
    Main entry point for compressing chat history.
    
    Returns:
        Tuple of (compressed_messages, was_compressed)
    """
    service = AgentMemoryService(model_context_size)
    
    if not service.should_compress(messages):
        return messages, False
    
    # Split messages: system, to_compress, to_keep
    system_msg = None
    conversation_msgs = []
    
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg
        else:
            conversation_msgs.append(msg)
    
    # Keep recent messages
    keep_count = min(keep_recent * 2, len(conversation_msgs))  # *2 for pairs
    
    if len(conversation_msgs) <= keep_count:
        return messages, False
    
    to_compress = conversation_msgs[:-keep_count]
    to_keep = conversation_msgs[-keep_count:]
    
    logger.info(f"[AGENT_MEMORY] Compressing {len(to_compress)} messages, keeping {len(to_keep)}")
    
    # Create agent file with compressed content
    filename = await service.compress_to_agent_file(
        db=db,
        chat_id=chat_id,
        messages_to_compress=to_compress,
        llm_client=None,
        model=model,
        api_base=api_base,
        api_key=api_key,
    )
    
    # Search for any relevant context from agent files for the current query
    current_query = ""
    for msg in reversed(to_keep):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                current_query = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        current_query = part.get("text", "")
                        break
            break
    
    agent_context = await service.get_agent_context_for_query(db, chat_id, current_query)
    
    # Build compressed message list
    compressed = []
    
    # Enhanced system message with reference to archived content
    system_content = system_msg.get("content", "You are a helpful AI assistant.") if system_msg else "You are a helpful AI assistant."
    
    archive_notice = f"""

--- CONVERSATION ARCHIVE NOTICE ---
Earlier parts of this conversation have been archived to {filename}.
The system will automatically retrieve relevant context when needed.
If the user references something from earlier that you don't see in recent messages, 
the relevant archived content will be provided below.
--- END NOTICE ---
"""
    
    if agent_context:
        archive_notice += f"\n{agent_context}"
    
    compressed.append({
        "role": "system",
        "content": system_content + archive_notice,
    })
    
    # Add recent messages
    compressed.extend(to_keep)
    
    original_tokens = estimate_message_tokens(messages)
    compressed_tokens = estimate_message_tokens(compressed)
    
    logger.info(
        f"[AGENT_MEMORY] Compression complete: {original_tokens} -> {compressed_tokens} tokens "
        f"({100 - (compressed_tokens/original_tokens*100):.1f}% reduction)"
    )
    
    return compressed, True


# Export for use in other modules
__all__ = [
    'AgentMemoryService',
    'compress_chat_history',
    'is_agent_file',
    'AGENT_FILE_PREFIX',
    'estimate_message_tokens',
]
