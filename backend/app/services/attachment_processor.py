"""
Large File Attachment Processor

When files attached to a message exceed a threshold, instead of including
the full content in the LLM message, this service:

1. Stores them as UploadedFile artifacts
2. Generates signatures (function/class definitions with line numbers)
3. Creates embeddings for RAG search
4. Generates a compact manifest for the LLM
5. Returns the manifest instead of full content

The LLM can then:
- See the file structure and signatures
- Request specific file content via <request_file path="..."/>
- Search for relevant code sections

This prevents context window overflow while maintaining full access to file contents.
"""
import logging
import os
import json
import uuid
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.upload import UploadedFile, UploadedArchive
from app.services.zip_processor import (
    detect_language,
    extract_signatures,
    EXT_TO_LANG,
)

logger = logging.getLogger(__name__)

# Threshold for "large" files - files over this size get processed as artifacts
# These can be overridden by settings
LARGE_FILE_THRESHOLD = 100_000  # 100KB per file
TOTAL_ATTACHMENT_THRESHOLD = 200_000  # 200KB total for all attachments

# What percentage of model context to allow for attachments before switching to artifacts
# Remaining context is for history, system prompt, RAG context, and response
ATTACHMENT_CONTEXT_PERCENT = 30  # 30% of context for attachments

# Maximum content to include inline (for small files)
MAX_INLINE_CONTENT = 20_000  # 20KB


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)"""
    return len(text) // 4


class AttachmentProcessor:
    """Process file attachments into searchable artifacts"""
    
    def __init__(self, db: AsyncSession, chat_id: str, user_id: str):
        self.db = db
        self.chat_id = chat_id
        self.user_id = user_id
    
    async def process_attachments(
        self,
        attachments: List[Dict],
        model_context_size: int = 128000,
    ) -> Tuple[List[Dict], Optional[str], bool]:
        """
        Process attachments, storing large files as artifacts.
        
        Args:
            attachments: List of attachment dicts with type, filename, content
            model_context_size: Model's context window size
            
        Returns:
            Tuple of:
            - Modified attachments list (large files have content removed)
            - Manifest string for system prompt (or None if no large files)
            - Boolean: True if any files were stored as artifacts
        """
        if not attachments:
            return attachments, None, False
        
        # Calculate total size of file attachments
        total_size = 0
        file_attachments = []
        other_attachments = []
        
        for att in attachments:
            if att.get("type") == "file":
                content = att.get("content", "")
                total_size += len(content)
                file_attachments.append(att)
            else:
                other_attachments.append(att)
        
        # Check if we need to process as artifacts
        # Tokens ≈ chars/4, so context in chars ≈ model_context_size * 4
        # But we only use ATTACHMENT_CONTEXT_PERCENT for attachments
        context_chars = model_context_size * 4  # Approximate chars for full context
        context_threshold = context_chars * ATTACHMENT_CONTEXT_PERCENT // 100
        
        needs_artifact_processing = (
            total_size > TOTAL_ATTACHMENT_THRESHOLD or
            total_size > context_threshold or
            any(len(att.get("content", "")) > LARGE_FILE_THRESHOLD for att in file_attachments)
        )
        
        if not needs_artifact_processing:
            logger.debug(f"[ATTACH_PROC] Files small enough for inline ({total_size} bytes, threshold={context_threshold})")
            return attachments, None, False
        
        logger.info(f"[ATTACH_PROC] Processing {len(file_attachments)} files as artifacts ({total_size:,} bytes > threshold {context_threshold:,})")
        
        # Process and store files
        stored_files = []
        manifest_data = {
            "total_files": len(file_attachments),
            "total_size": total_size,
            "languages": {},
            "files": [],
            "signatures": {},
        }
        
        for att in file_attachments:
            filename = att.get("filename", "unnamed")
            content = att.get("content", "")
            
            if not content:
                continue
            
            # Detect language and extract signatures
            ext = os.path.splitext(filename)[1].lower()
            language = detect_language(filename) or EXT_TO_LANG.get(ext, "text")
            signatures = extract_signatures(content, language) if language else []
            
            # Update language counts
            manifest_data["languages"][language] = manifest_data["languages"].get(language, 0) + 1
            
            # Store file info
            file_info = {
                "filename": filename,
                "size": len(content),
                "language": language,
                "lines": content.count('\n') + 1,
            }
            manifest_data["files"].append(file_info)
            
            # Store signatures
            if signatures:
                manifest_data["signatures"][filename] = signatures
            
            # Store as UploadedFile
            await self._store_file(filename, content, language, signatures)
            stored_files.append(filename)
        
        # Generate manifest for LLM
        manifest = self._generate_manifest(manifest_data)
        
        # Create modified attachments - remove content from large files
        modified_attachments = other_attachments.copy()
        for att in file_attachments:
            filename = att.get("filename", "")
            content = att.get("content", "")
            
            # Include small excerpt for context, or just filename
            if len(content) <= MAX_INLINE_CONTENT:
                # Small enough to include
                modified_attachments.append(att)
            else:
                # Include only a summary reference
                modified_attachments.append({
                    "type": "file",
                    "filename": filename,
                    "content": f"[File stored as artifact - {len(content):,} bytes - use <request_file path=\"{filename}\"/> to view]",
                })
        
        logger.info(f"[ATTACH_PROC] Stored {len(stored_files)} files as artifacts, generated manifest")
        
        return modified_attachments, manifest, True
    
    async def _store_file(
        self,
        filename: str,
        content: str,
        language: str,
        signatures: List[Dict],
    ) -> str:
        """Store a file as UploadedFile artifact"""
        ext = os.path.splitext(filename)[1].lower()
        
        # Check if file already exists
        result = await self.db.execute(
            select(UploadedFile)
            .where(UploadedFile.chat_id == self.chat_id)
            .where(UploadedFile.filepath == filename)
            .order_by(UploadedFile.created_at.desc())
            .limit(1)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing file
            existing.content = content
            existing.language = language
            existing.size = len(content)
            existing.signatures = json.dumps(signatures) if signatures else None
            logger.debug(f"[ATTACH_PROC] Updated existing file: {filename}")
        else:
            # Create new file
            uploaded_file = UploadedFile(
                chat_id=self.chat_id,
                archive_name=None,
                filepath=filename,
                filename=os.path.basename(filename),
                extension=ext,
                language=language,
                size=len(content),
                is_binary=False,
                content=content,
                signatures=json.dumps(signatures) if signatures else None,
            )
            self.db.add(uploaded_file)
            logger.debug(f"[ATTACH_PROC] Created new file: {filename}")
        
        return filename
    
    def _generate_manifest(self, data: Dict) -> str:
        """Generate a compact manifest for LLM context"""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        lines = [
            f"[ATTACHED_FILES | {timestamp}]",
            f"FILES: {data['total_files']} | SIZE: {data['total_size']:,} bytes",
            f"LANGUAGES: {', '.join(f'{lang}({count})' for lang, count in sorted(data['languages'].items(), key=lambda x: -x[1]))}",
            "",
            "FILE LIST:",
        ]
        
        # File list with sizes
        for f in sorted(data['files'], key=lambda x: x['filename']):
            lines.append(f"  {f['filename']} ({f['size']:,}b, {f['lines']} lines, {f['language']})")
        
        lines.append("")
        
        # Signatures section
        if data['signatures']:
            lines.append("CODE SIGNATURES (for searching):")
            for filepath, sigs in sorted(data['signatures'].items()):
                if sigs:
                    sig_summary = []
                    for sig in sigs[:15]:  # Limit signatures shown
                        kind = sig.get('kind', '?')
                        name = sig.get('name', '?')
                        line = sig.get('line', 0)
                        kind_short = {
                            'function': 'fn', 'method': 'fn', 'class': 'cls',
                            'interface': 'iface', 'type': 'type', 'struct': 'struct',
                            'variable': 'var', 'import': 'imp', 'export': 'exp',
                            'constant': 'const', 'enum': 'enum',
                        }.get(kind, kind[:3])
                        sig_summary.append(f"{kind_short}:{name}@L{line}")
                    
                    more = f" +{len(sigs)-15} more" if len(sigs) > 15 else ""
                    lines.append(f"  {filepath}: {', '.join(sig_summary)}{more}")
            lines.append("")
        
        # Instructions
        lines.extend([
            "─" * 50,
            "FILE ACCESS INSTRUCTIONS:",
            "• To view a file: Include <request_file path=\"filename\"/> in your response",
            "• To search code: Reference function/class names from signatures above",
            "• To modify: Create an artifact with COMPLETE file content",
            "• Files are available for the duration of this chat",
            "[END_ATTACHED_FILES]",
        ])
        
        return "\n".join(lines)


async def process_large_attachments(
    db: AsyncSession,
    chat_id: str,
    user_id: str,
    attachments: List[Dict],
    model_context_size: int = 128000,
) -> Tuple[List[Dict], Optional[str], bool]:
    """
    Convenience function to process attachments.
    
    Returns:
        Tuple of (modified_attachments, manifest_string, was_processed)
    """
    processor = AttachmentProcessor(db, chat_id, user_id)
    return await processor.process_attachments(attachments, model_context_size)


# Export
__all__ = [
    'AttachmentProcessor',
    'process_large_attachments',
    'LARGE_FILE_THRESHOLD',
    'TOTAL_ATTACHMENT_THRESHOLD',
]
