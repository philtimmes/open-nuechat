"""
Chat History Compression Service

Compresses older chat history into a summary while preserving:
- Key talking points and topics
- User preferences and constraints
- Important decisions and conclusions
- Original intent/premise of the conversation

This helps manage context window limits while maintaining conversation coherence.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import tiktoken

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Approximate tokens per character (for estimation without tiktoken)
CHARS_PER_TOKEN = 4


@dataclass
class CompressionConfig:
    """Configuration for history compression"""
    enabled: bool = True
    threshold_messages: int = 20  # Compress when history exceeds this many messages
    keep_recent: int = 6  # Keep this many recent message pairs intact
    max_summary_tokens: int = 1000  # Max tokens for the summary
    target_total_tokens: int = 8000  # Target total context tokens after compression


def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    try:
        # Try to use tiktoken for accurate count
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback to character-based estimation
        return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(messages: List[Dict]) -> int:
    """Estimate total tokens in a message list"""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multi-modal content
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    total += 85  # Base tokens for image
        total += 4  # Role and formatting tokens
    return total


COMPRESSION_PROMPT = """You are a conversation summarizer. Your task is to compress the following conversation history into a concise summary that preserves:

1. **Key Topics**: What subjects were discussed
2. **User Intent**: What the user is trying to accomplish
3. **Preferences & Constraints**: Any requirements, preferences, or limitations mentioned
4. **Important Decisions**: Any conclusions or choices made
5. **Context Needed**: Information necessary to continue the conversation coherently

Rules:
- Be concise but comprehensive
- Use bullet points for clarity
- Preserve specific details (names, numbers, technical terms)
- Focus on information needed to continue the conversation
- Do NOT include pleasantries or filler

CONVERSATION TO SUMMARIZE:
{conversation}

Provide a structured summary:"""


class HistoryCompressionService:
    """Service for compressing chat history"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
    
    def should_compress(self, messages: List[Dict]) -> bool:
        """Check if compression is needed based on message count"""
        if not self.config.enabled:
            return False
        
        # Count user/assistant message pairs (exclude system)
        pair_count = sum(1 for m in messages if m.get("role") in ("user", "assistant"))
        
        return pair_count > self.config.threshold_messages
    
    def should_compress_by_tokens(self, messages: List[Dict]) -> bool:
        """Check if compression is needed based on token count"""
        if not self.config.enabled:
            return False
        
        total_tokens = estimate_message_tokens(messages)
        return total_tokens > self.config.target_total_tokens
    
    def split_messages_for_compression(
        self, 
        messages: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Optional[Dict]]:
        """
        Split messages into:
        - system message (if present)
        - messages to compress (older)
        - messages to keep (recent)
        
        Returns: (to_compress, to_keep, system_message)
        """
        system_msg = None
        conversation_msgs = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                conversation_msgs.append(msg)
        
        # Keep the most recent messages intact
        keep_count = min(self.config.keep_recent * 2, len(conversation_msgs))  # *2 for pairs
        
        if len(conversation_msgs) <= keep_count:
            # Not enough to compress
            return [], conversation_msgs, system_msg
        
        to_compress = conversation_msgs[:-keep_count]
        to_keep = conversation_msgs[-keep_count:]
        
        return to_compress, to_keep, system_msg
    
    def format_messages_for_summary(self, messages: List[Dict]) -> str:
        """Format messages into a readable string for summarization"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            if isinstance(content, list):
                # Multi-modal - extract text parts
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "... [truncated]"
            
            lines.append(f"{role}: {content}")
        
        return "\n\n".join(lines)
    
    async def generate_summary(
        self,
        messages_to_compress: List[Dict],
        llm_client,  # The HTTP client for LLM API
        model: str,
        api_base: str,
        api_key: str,
    ) -> str:
        """Generate a summary of the messages using the LLM"""
        import httpx
        
        conversation_text = self.format_messages_for_summary(messages_to_compress)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)
        
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
                            {"role": "system", "content": "You are a precise conversation summarizer."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": self.config.max_summary_tokens,
                        "temperature": 0.3,  # Lower temperature for factual summarization
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    summary = data["choices"][0]["message"]["content"]
                    logger.info(f"Generated history summary ({len(summary)} chars)")
                    return summary
                else:
                    logger.error(f"Summary generation failed: {response.status_code}")
                    return self._fallback_summary(messages_to_compress)
                    
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return self._fallback_summary(messages_to_compress)
    
    def _fallback_summary(self, messages: List[Dict]) -> str:
        """Generate a basic summary without LLM (fallback)"""
        # Extract key information manually
        topics = set()
        user_requests = []
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            
            # Extract potential topics (simple heuristic)
            words = content.lower().split()
            for word in words:
                if len(word) > 5 and word.isalpha():
                    topics.add(word)
            
            if msg.get("role") == "user" and len(content) < 200:
                user_requests.append(content[:100])
        
        summary_parts = ["[Previous conversation summary]"]
        
        if user_requests:
            summary_parts.append(f"User discussed: {'; '.join(user_requests[:5])}")
        
        if topics:
            top_topics = list(topics)[:10]
            summary_parts.append(f"Topics mentioned: {', '.join(top_topics)}")
        
        summary_parts.append(f"({len(messages)} messages compressed)")
        
        return "\n".join(summary_parts)
    
    async def compress_history(
        self,
        messages: List[Dict],
        llm_client,
        model: str,
        api_base: str,
        api_key: str,
    ) -> List[Dict]:
        """
        Compress chat history, returning a new message list with:
        - Original system message
        - Summary of older messages as a system context addition
        - Recent messages intact
        """
        to_compress, to_keep, system_msg = self.split_messages_for_compression(messages)
        
        if not to_compress:
            # Nothing to compress
            return messages
        
        logger.info(f"Compressing {len(to_compress)} messages, keeping {len(to_keep)} recent")
        
        # Generate summary
        summary = await self.generate_summary(
            to_compress, llm_client, model, api_base, api_key
        )
        
        # Build new message list
        compressed_messages = []
        
        # Add system message with summary context
        system_content = system_msg.get("content", "You are a helpful AI assistant.") if system_msg else "You are a helpful AI assistant."
        
        enhanced_system = f"""{system_content}

--- CONVERSATION CONTEXT ---
The following is a summary of the earlier part of this conversation. Use this context to maintain continuity:

{summary}
--- END CONTEXT ---

Continue the conversation naturally, referencing the above context when relevant."""
        
        compressed_messages.append({
            "role": "system",
            "content": enhanced_system
        })
        
        # Add recent messages
        compressed_messages.extend(to_keep)
        
        original_tokens = estimate_message_tokens(messages)
        compressed_tokens = estimate_message_tokens(compressed_messages)
        
        logger.info(
            f"History compression: {original_tokens} -> {compressed_tokens} tokens "
            f"({100 - (compressed_tokens/original_tokens*100):.1f}% reduction)"
        )
        
        return compressed_messages


# Singleton instance
_compression_service: Optional[HistoryCompressionService] = None


def get_compression_service(config: Optional[CompressionConfig] = None) -> HistoryCompressionService:
    """Get or create compression service singleton"""
    global _compression_service
    if _compression_service is None or config is not None:
        _compression_service = HistoryCompressionService(config)
    return _compression_service
