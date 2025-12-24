"""
RAG Service with FAISS GPU (ROCm) and sentence-transformers

Uses:
- sentence-transformers for embeddings (ROCm/GPU accelerated via PyTorch)
- FAISS GPU for vector indexing and search (ROCm)
- IVF-PQ index for efficient large dataset handling

Index Strategy:
- IVF (Inverted File) for fast approximate search
- PQ (Product Quantization) for memory efficiency
- Supports millions of vectors efficiently
"""
import os
import pickle
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
import logging

import faiss

from app.core.config import settings
from app.models.models import Document, DocumentChunk, User, KnowledgeStore, KnowledgeStoreShare

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=4)

# Cache for debug_rag setting (refreshed on each search)
_debug_rag_cache: Dict[str, bool] = {"enabled": False}


async def _is_debug_rag_enabled(db: AsyncSession) -> bool:
    """Check if debug RAG logging is enabled."""
    from app.models.models import SystemSetting
    try:
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == "debug_rag")
        )
        setting = result.scalar_one_or_none()
        enabled = setting and setting.value == "true"
        _debug_rag_cache["enabled"] = enabled
        return enabled
    except Exception:
        return _debug_rag_cache.get("enabled", False)


def _log_rag_debug(message: str, **kwargs):
    """Log RAG debug message with optional structured data."""
    if kwargs:
        data_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"[RAG DEBUG] {message} | {data_str}")
    else:
        logger.info(f"[RAG DEBUG] {message}")


class FAISSIndexManager:
    """
    Manages FAISS GPU indexes with IVF-PQ for efficient large-scale search.
    
    Index types by size:
    - < 10K vectors: Flat (exact search)
    - 10K - 1M vectors: IVF-Flat
    - > 1M vectors: IVF-PQ
    """
    
    # Index configuration
    EMBEDDING_DIM = 384  # Default for all-MiniLM-L6-v2
    NLIST_FACTOR = 4  # nlist = NLIST_FACTOR * sqrt(n)
    NPROBE = 32  # Number of clusters to search (tradeoff: speed vs accuracy)
    PQ_M = 48  # Number of subquantizers for PQ (must divide EMBEDDING_DIM)
    PQ_NBITS = 8  # Bits per subquantizer
    
    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir or settings.FAISS_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU resources
        self.gpu_resources = None
        self.gpu_available = False
        self._init_gpu()
        
        # In-memory index cache
        self._indexes: Dict[str, faiss.Index] = {}
        self._id_maps: Dict[str, Dict[int, str]] = {}  # faiss_id -> chunk_id
    
    def _init_gpu(self):
        """Initialize FAISS GPU resources"""
        try:
            self.gpu_resources = faiss.StandardGpuResources()
            self.gpu_available = True
            logger.info("FAISS GPU initialized successfully")
        except Exception as e:
            logger.warning(f"FAISS GPU not available, using CPU: {e}")
            self.gpu_available = False
    
    def _get_index_path(self, index_id: str) -> Tuple[Path, Path]:
        """Get paths for index and ID map files"""
        index_path = self.index_dir / f"{index_id}.faiss"
        map_path = self.index_dir / f"{index_id}.map"
        return index_path, map_path
    
    def _create_index(self, n_vectors: int) -> faiss.Index:
        """
        Create appropriate FAISS index based on dataset size.
        
        Strategy:
        - Flat: Best accuracy, O(n) search, good for < 10K
        - IVF-Flat: Good accuracy, faster search, 10K - 1M
        - IVF-PQ: Memory efficient, fast, 1M+
        """
        dim = self.EMBEDDING_DIM
        
        if n_vectors < 10_000:
            # Small dataset: exact search
            logger.info(f"Creating Flat index for {n_vectors} vectors")
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
        
        elif n_vectors < 1_000_000:
            # Medium dataset: IVF with flat quantizer
            nlist = max(16, int(self.NLIST_FACTOR * np.sqrt(n_vectors)))
            logger.info(f"Creating IVF-Flat index: {n_vectors} vectors, nlist={nlist}")
            
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        else:
            # Large dataset: IVF with product quantization
            nlist = max(256, int(self.NLIST_FACTOR * np.sqrt(n_vectors)))
            logger.info(f"Creating IVF-PQ index: {n_vectors} vectors, nlist={nlist}, m={self.PQ_M}")
            
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(
                quantizer, dim, nlist, self.PQ_M, self.PQ_NBITS,
                faiss.METRIC_INNER_PRODUCT
            )
        
        return index
    
    def _to_gpu(self, index: faiss.Index) -> faiss.Index:
        """Move index to GPU if available"""
        if self.gpu_available and self.gpu_resources:
            try:
                return faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
        return index
    
    def _to_cpu(self, index: faiss.Index) -> faiss.Index:
        """Move index to CPU for saving"""
        if self.gpu_available:
            try:
                return faiss.index_gpu_to_cpu(index)
            except:
                pass
        return index
    
    def build_index(
        self,
        index_id: str,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        """
        Build a new FAISS index from embeddings.
        
        Args:
            index_id: Unique identifier for the index (e.g., knowledge_store_id)
            embeddings: numpy array of shape (n, dim) with normalized embeddings
            chunk_ids: List of chunk IDs corresponding to each embedding
        """
        logger.info(f"[FAISS_BUILD] Building index for: {index_id} (type: {type(index_id).__name__})")
        n_vectors = len(embeddings)
        
        if n_vectors == 0:
            logger.warning(f"No vectors to index for {index_id}")
            return
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = self._create_index(n_vectors)
        
        # Train if required (IVF indexes need training)
        if hasattr(index, 'train') and not index.is_trained:
            logger.info(f"Training index {index_id} with {n_vectors} vectors")
            # For training, use CPU then move to GPU
            index.train(embeddings)
        
        # Move to GPU for adding vectors
        index = self._to_gpu(index)
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = self.NPROBE
        
        # Add vectors
        index.add(embeddings)
        
        # Create ID mapping
        id_map = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}
        
        # Save to disk (CPU version)
        cpu_index = self._to_cpu(index)
        index_path, map_path = self._get_index_path(index_id)
        
        logger.info(f"[FAISS_BUILD] Saving index to: {index_path}")
        faiss.write_index(cpu_index, str(index_path))
        with open(map_path, 'wb') as f:
            pickle.dump(id_map, f)
        logger.info(f"[FAISS_BUILD] Index saved successfully: {index_path.exists()}")
        
        # Cache GPU version
        self._indexes[index_id] = index
        self._id_maps[index_id] = id_map
        
        logger.info(f"Built index {index_id}: {n_vectors} vectors, saved to {index_path}")
    
    def load_index(self, index_id: str) -> bool:
        """Load index from disk into memory/GPU"""
        logger.info(f"[FAISS_LOAD] Attempting to load index: {index_id}")
        
        if index_id in self._indexes:
            logger.info(f"[FAISS_LOAD] Index {index_id} already cached in memory")
            return True
        
        index_path, map_path = self._get_index_path(index_id)
        logger.info(f"[FAISS_LOAD] Looking for index file: {index_path}")
        
        if not index_path.exists():
            logger.warning(f"[FAISS_LOAD] Index file NOT FOUND: {index_path}")
            # List what files ARE in the directory
            try:
                if self.index_dir.exists():
                    files = list(self.index_dir.glob("*.faiss"))
                    logger.warning(f"[FAISS_LOAD] Available index files: {[f.name for f in files]}")
                else:
                    logger.warning(f"[FAISS_LOAD] Index directory does not exist: {self.index_dir}")
            except Exception as e:
                logger.warning(f"[FAISS_LOAD] Error listing index dir: {e}")
            return False
        
        try:
            # Load from disk
            logger.info(f"[FAISS_LOAD] Loading index from disk: {index_path}")
            index = faiss.read_index(str(index_path))
            
            # Set search parameters
            if hasattr(index, 'nprobe'):
                index.nprobe = self.NPROBE
            
            # Move to GPU
            index = self._to_gpu(index)
            
            # Load ID map
            with open(map_path, 'rb') as f:
                id_map = pickle.load(f)
            
            self._indexes[index_id] = index
            self._id_maps[index_id] = id_map
            
            logger.info(f"Loaded index {index_id}: {index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index {index_id}: {e}")
            return False
    
    def search(
        self,
        index_id: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search index for nearest neighbors.
        
        Args:
            index_id: Index to search
            query_embedding: Query vector (will be normalized)
            top_k: Number of results
            
        Returns:
            List of (chunk_id, score) tuples
        """
        # Ensure index is loaded
        if not self.load_index(index_id):
            logger.warning(f"[FAISS] Index {index_id} not found or failed to load")
            return []
        
        index = self._indexes[index_id]
        id_map = self._id_maps[index_id]
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Log query info
        logger.debug(f"[FAISS] Search: index={index_id}, index_vectors={index.ntotal}, query_dim={query.shape[1]}, query_norm_before={np.linalg.norm(query):.4f}")
        
        faiss.normalize_L2(query)
        
        logger.debug(f"[FAISS] Query norm after normalize_L2={np.linalg.norm(query):.4f}")
        
        # Search
        scores, indices = index.search(query, min(top_k, index.ntotal))
        
        logger.debug(f"[FAISS] Raw results: indices={indices[0][:5].tolist()}, scores={scores[0][:5].tolist()}")
        
        # Map back to chunk IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in id_map:
                results.append((id_map[idx], float(score)))
        
        logger.debug(f"[FAISS] Mapped {len(results)} results, top_scores={[r[1] for r in results[:3]]}")
        
        return results
    
    def delete_index(self, index_id: str) -> None:
        """Delete index from memory and disk"""
        # Remove from cache
        self._indexes.pop(index_id, None)
        self._id_maps.pop(index_id, None)
        
        # Remove from disk
        index_path, map_path = self._get_index_path(index_id)
        index_path.unlink(missing_ok=True)
        map_path.unlink(missing_ok=True)
        
        logger.info(f"Deleted index {index_id}")
    
    def add_vectors(
        self,
        index_id: str,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> None:
        """Add vectors to existing index (rebuilds for IVF indexes)"""
        # For IVF indexes, we need to rebuild
        # This is because IVF centroids are fixed after training
        # For incremental updates, you'd need a more complex strategy
        
        if not self.load_index(index_id):
            # No existing index, build new one
            self.build_index(index_id, embeddings, chunk_ids)
            return
        
        # Get existing vectors (need to reload from original data)
        # For now, just rebuild - in production, you'd store vectors separately
        logger.info(f"Rebuilding index {index_id} with new vectors")
        self.build_index(index_id, embeddings, chunk_ids)


class RAGService:
    """
    Retrieval Augmented Generation service using FAISS GPU and sentence-transformers.
    
    Features:
    - GPU-accelerated embeddings via PyTorch ROCm
    - FAISS GPU for vector search
    - IVF-PQ indexing for large datasets
    - Async-friendly with thread pool for blocking ops
    """
    
    _model = None
    _model_loaded = False
    _model_load_failed = False  # Track if model loading permanently failed
    _model_load_last_attempt = 0  # Timestamp of last load attempt
    _model_retry_delay = 60  # Retry after 60 seconds on failure
    _index_manager = None
    
    @classmethod
    def reset_model(cls):
        """Reset model state to allow retry of loading"""
        logger.info("[RAG] Resetting model state for retry...")
        cls._model = None
        cls._model_loaded = False
        cls._model_load_failed = False
        cls._model_load_last_attempt = 0
    
    @classmethod
    def get_model_status(cls) -> dict:
        """Get current model status for debugging"""
        import time
        retry_in = 0
        if cls._model_load_failed:
            elapsed = time.time() - cls._model_load_last_attempt
            retry_in = max(0, cls._model_retry_delay - elapsed)
        return {
            "loaded": cls._model_loaded,
            "failed": cls._model_load_failed,
            "model_exists": cls._model is not None,
            "retry_in_seconds": round(retry_in, 1) if cls._model_load_failed else None,
        }
    
    @classmethod
    def get_model(cls):
        """Lazy load the embedding model with GPU support"""
        import time
        
        # If model loading already failed, check if we should retry
        if cls._model_load_failed:
            elapsed = time.time() - cls._model_load_last_attempt
            if elapsed < cls._model_retry_delay:
                # Too soon to retry
                return None
            else:
                # Enough time has passed, reset and retry
                logger.info(f"[RAG] Retrying model load after {elapsed:.0f}s...")
                cls._model_load_failed = False
        
        if not cls._model_loaded:
            try:
                import torch
                import os
                
                # CRITICAL: Disable accelerate's device_map which causes meta tensor issues
                # Must be set BEFORE importing sentence_transformers/transformers
                os.environ["ACCELERATE_DISABLE_RICH"] = "1"
                os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
                os.environ["ACCELERATE_USE_CPU"] = "1"
                # Disable low_cpu_mem_usage which triggers meta tensors
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                
                # Force reload transformers modules to pick up env vars
                import importlib
                import sys
                for mod_name in list(sys.modules.keys()):
                    if 'transformers' in mod_name or 'accelerate' in mod_name:
                        try:
                            del sys.modules[mod_name]
                        except:
                            pass
                
                from sentence_transformers import SentenceTransformer
                
                # ROCm/HIP uses torch.cuda API - same interface, different backend
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL}', target device: {target_device}")
                
                model = None
                
                # Try multiple loading strategies
                loading_strategies = [
                    # Strategy 1: Force no accelerate features
                    lambda: SentenceTransformer(
                        settings.EMBEDDING_MODEL, 
                        device="cpu",
                        model_kwargs={"low_cpu_mem_usage": False},
                    ),
                    # Strategy 2: Basic load
                    lambda: SentenceTransformer(
                        settings.EMBEDDING_MODEL, 
                        device="cpu",
                    ),
                    # Strategy 3: Trust remote code
                    lambda: SentenceTransformer(
                        settings.EMBEDDING_MODEL, 
                        device="cpu",
                        trust_remote_code=True,
                    ),
                    # Strategy 4: Direct from local cache if available
                    lambda: SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        device="cpu",
                        cache_folder="/root/.cache/huggingface/hub",
                    ),
                ]
                
                for i, strategy in enumerate(loading_strategies):
                    try:
                        logger.info(f"Trying loading strategy {i+1}/{len(loading_strategies)}...")
                        model = strategy()
                        logger.info(f"Strategy {i+1} succeeded!")
                        break
                    except Exception as e:
                        logger.warning(f"Strategy {i+1} failed: {e}")
                        model = None
                
                if model is None:
                    logger.error("All loading strategies failed")
                    cls._model_load_failed = True
                    cls._model_load_last_attempt = time.time()
                    cls._model = None
                    return None
                
                logger.info(f"Model loaded, current device: {model.device}")
                
                # Move to GPU if available and model is on CPU
                if target_device == "cuda" and str(model.device) == "cpu":
                    logger.info("Moving embedding model to GPU...")
                    try:
                        model = model.to(target_device)
                        logger.info(f"Model moved to GPU successfully")
                    except Exception as e:
                        logger.warning(f"Failed to move to GPU ({e}), keeping on CPU")
                
                cls._model = model
                cls._model_loaded = True
                
                # Update embedding dimension in index manager
                dim = cls._model.get_sentence_embedding_dimension()
                FAISSIndexManager.EMBEDDING_DIM = dim
                logger.info(f"Embedding model ready: dim={dim}, device={cls._model.device}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                cls._model = None
                cls._model_load_failed = True
                cls._model_load_last_attempt = time.time()
        return cls._model
    
    @classmethod
    def get_index_manager(cls) -> FAISSIndexManager:
        """Get or create FAISS index manager"""
        if cls._index_manager is None:
            cls._index_manager = FAISSIndexManager()
        return cls._index_manager
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.top_k = settings.TOP_K_RESULTS
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using GPU"""
        model = self.get_model()
        if model is None:
            logger.warning("[EMBED] Model not loaded!")
            return None
        
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        
        # Debug logging
        logger.debug(f"[EMBED] text='{text[:50]}...', shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}, first5={embedding[:5].tolist()}")
        
        return embedding
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Async wrapper for embed_text that returns a list of floats.
        Used by the OpenAI-compatible /v1/embeddings endpoint.
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.embed_text, text)
        
        if embedding is None:
            raise RuntimeError("Embedding model not available")
        
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts with batching"""
        model = self.get_model()
        if model is None:
            return None
        
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100
        )
        return embeddings
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [{"content": text, "index": 0}]
        
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "content": chunk_text,
                "index": chunk_index,
            })
            
            chunk_index += 1
            start = end - self.chunk_overlap
            
            if start >= len(words) - self.chunk_overlap:
                break
        
        return chunks
    
    async def process_document(
        self,
        db: AsyncSession,
        document: Document,
        text_content: str,
    ) -> int:
        """Process a document: chunk, embed, and store"""
        
        # Delete existing chunks
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        
        # Chunk the text
        chunks = self.chunk_text(text_content)
        
        if not chunks:
            return 0
        
        # Generate embeddings (run in thread pool)
        chunk_texts = [c["content"] for c in chunks]
        embeddings = await asyncio.get_event_loop().run_in_executor(
            _executor,
            self.embed_texts,
            chunk_texts
        )
        
        # Store chunks in database
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            embedding_bytes = None
            if embeddings is not None:
                embedding_bytes = embeddings[i].astype(np.float32).tobytes()
            
            db_chunk = DocumentChunk(
                document_id=document.id,
                content=chunk["content"],
                chunk_index=chunk["index"],
                embedding=embedding_bytes,
                metadata={"word_count": len(chunk["content"].split())},
            )
            db.add(db_chunk)
            await db.flush()
            chunk_ids.append(str(db_chunk.id))
        
        # Update document status
        document.is_processed = True
        document.chunk_count = len(chunks)
        
        await db.flush()
        
        # Update FAISS index if document belongs to a knowledge store
        if document.knowledge_store_id and embeddings is not None:
            await self._update_knowledge_store_index(
                db, document.knowledge_store_id
            )
        
        return len(chunks)
    
    async def delete_document(
        self,
        db: AsyncSession,
        document_id: str,
    ) -> None:
        """Delete a document and its chunks, update FAISS indexes"""
        
        # Get document to check knowledge store
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        
        if not document:
            return  # Document already deleted
        
        knowledge_store_id = document.knowledge_store_id
        
        # Delete chunks first (explicit delete for reliability)
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        
        # Delete document using ORM to trigger cascades
        await db.delete(document)
        
        # Rebuild FAISS index if document was in a knowledge store
        if knowledge_store_id:
            try:
                await self._update_knowledge_store_index(db, knowledge_store_id)
            except Exception as e:
                logger.warning(f"Failed to update FAISS index after document deletion: {e}")
    
    async def _update_knowledge_store_index(
        self,
        db: AsyncSession,
        knowledge_store_id: str,
    ) -> None:
        """Rebuild FAISS index for a knowledge store"""
        
        logger.info(f"[FAISS_BUILD] Starting index rebuild for knowledge store: {knowledge_store_id}")
        
        # Get all chunks for this knowledge store
        result = await db.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.knowledge_store_id == knowledge_store_id)
            .where(Document.is_processed == True)
            .where(DocumentChunk.embedding.isnot(None))
        )
        rows = result.all()
        
        if not rows:
            logger.warning(f"[FAISS_BUILD] No chunks found for knowledge store {knowledge_store_id}")
            self.get_index_manager().delete_index(knowledge_store_id)
            return
        
        logger.info(f"[FAISS_BUILD] Found {len(rows)} chunks to index")
        
        # Collect embeddings and IDs
        embeddings = []
        chunk_ids = []
        
        for chunk, document in rows:
            embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
            embeddings.append(embedding)
            chunk_ids.append(str(chunk.id))
        
        embeddings_array = np.vstack(embeddings)
        
        logger.info(f"[FAISS_BUILD] Embedding array shape: {embeddings_array.shape}, first embedding norm: {np.linalg.norm(embeddings_array[0]):.4f}")
        
        # Build index in thread pool
        await asyncio.get_event_loop().run_in_executor(
            _executor,
            self.get_index_manager().build_index,
            knowledge_store_id,
            embeddings_array,
            chunk_ids
        )
        
        logger.info(f"[FAISS_BUILD] Index build complete for {knowledge_store_id}")
    
    async def search(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using FAISS.
        
        Args:
            document_ids: If provided, search only in these specific documents.
                         If None, search user's unitemized documents (not in any knowledge store).
        
        Note: Documents in knowledge stores are only searched via search_knowledge_stores(),
        which requires explicit knowledge store IDs.
        """
        top_k = top_k or self.top_k
        
        # Check if debug logging is enabled
        debug_enabled = await _is_debug_rag_enabled(db)
        
        if debug_enabled:
            _log_rag_debug(
                "Search started",
                user_id=user.id,
                query=query[:100] + "..." if len(query) > 100 else query,
                document_ids=document_ids,
                top_k=top_k
            )
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, self.embed_text, query
        )
        
        if query_embedding is None:
            if debug_enabled:
                _log_rag_debug("Falling back to keyword search (embedding failed)")
            return await self._keyword_search(db, user, query, document_ids, top_k)
        
        # If document_ids specified, do direct search
        if document_ids:
            results = await self._direct_search(db, query_embedding, document_ids, top_k)
            if debug_enabled:
                _log_rag_debug(
                    "Direct search completed",
                    results_count=len(results),
                    top_scores=[r.get("similarity", 0) for r in results[:3]]
                )
            return results
        
        # Otherwise search user's documents
        results = await self._user_document_search(db, user, query_embedding, top_k)
        if debug_enabled:
            _log_rag_debug(
                "User document search completed",
                results_count=len(results),
                top_scores=[r.get("similarity", 0) for r in results[:3]]
            )
        return results
    
    async def _direct_search(
        self,
        db: AsyncSession,
        query_embedding: np.ndarray,
        document_ids: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Direct search in specific documents (no FAISS index)"""
        query_builder = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.id.in_(document_ids))
            .where(DocumentChunk.embedding.isnot(None))
        )
        
        result = await db.execute(query_builder)
        rows = result.all()
        
        if not rows:
            logger.debug(f"[DIRECT_SEARCH] No chunks found for documents: {document_ids}")
            return []
        
        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_normalized = query_embedding / query_norm
        else:
            logger.warning("[DIRECT_SEARCH] Query embedding has zero norm!")
            return []
        
        logger.debug(f"[DIRECT_SEARCH] Found {len(rows)} chunks, query_norm_before={query_norm:.4f}, query_dim={len(query_embedding)}")
        
        # Calculate similarities
        results = []
        for chunk, document in rows:
            chunk_embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
            
            # Check dimensions match
            if len(chunk_embedding) != len(query_normalized):
                logger.warning(f"[DIRECT_SEARCH] Dimension mismatch! chunk_dim={len(chunk_embedding)}, query_dim={len(query_normalized)}")
                continue
            
            # Normalize chunk embedding
            chunk_norm = np.linalg.norm(chunk_embedding)
            if chunk_norm > 0:
                chunk_normalized = chunk_embedding / chunk_norm
            else:
                logger.warning(f"[DIRECT_SEARCH] Chunk has zero norm: {chunk.id}")
                continue
            
            # Cosine similarity
            similarity = float(np.dot(query_normalized, chunk_normalized))
            
            if len(results) < 3:  # Log first 3
                logger.debug(f"[DIRECT_SEARCH] Chunk '{chunk.content[:50]}...' norm={chunk_norm:.4f}, similarity={similarity:.4f}")
            
            results.append({
                "chunk_id": str(chunk.id),
                "document_id": str(document.id),
                "document_name": document.name,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "similarity": similarity,
                "metadata": chunk.chunk_metadata,
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.debug(f"[DIRECT_SEARCH] Top similarities: {[round(r['similarity'], 4) for r in results[:5]]}")
        return results[:top_k]
    
    async def _user_document_search(
        self,
        db: AsyncSession,
        user: User,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Search user's unitemized documents (not in any knowledge store).
        
        Documents in knowledge stores are only searched when:
        - A custom GPT with that knowledge store is selected
        - The user explicitly searches a knowledge store
        
        This prevents searching subscribed GPT knowledge stores when no GPT is selected.
        """
        # Get user's documents that are NOT in any knowledge store
        result = await db.execute(
            select(Document)
            .where(Document.owner_id == user.id)
            .where(Document.is_processed == True)
            .where(Document.knowledge_store_id.is_(None))  # Only unitemized documents
        )
        documents = result.scalars().all()
        
        if not documents:
            logger.debug(f"[USER_DOC_SEARCH] No unitemized documents found for user {user.id}")
            return []
        
        logger.debug(f"[USER_DOC_SEARCH] Found {len(documents)} unitemized documents for user {user.id}")
        
        doc_ids = [str(d.id) for d in documents]
        return await self._direct_search(db, query_embedding, doc_ids, top_k)
    
    async def search_knowledge_stores(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        knowledge_store_ids: List[str],
        top_k: Optional[int] = None,
        bypass_access_check: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search across knowledge stores using FAISS indexes.
        
        Args:
            bypass_access_check: If True, skip ownership/permission checks.
                                Used when accessing KB through a public GPT.
        """
        if not knowledge_store_ids:
            return []
        
        top_k = top_k or self.top_k
        
        # Check if debug logging is enabled
        debug_enabled = await _is_debug_rag_enabled(db)
        
        if debug_enabled:
            # Get knowledge store names for logging
            kb_result = await db.execute(
                select(KnowledgeStore).where(KnowledgeStore.id.in_(knowledge_store_ids))
            )
            kb_names = {str(kb.id): kb.name for kb in kb_result.scalars().all()}
            
            _log_rag_debug(
                "Knowledge store search started",
                user_id=user.id,
                query=query[:100] + "..." if len(query) > 100 else query,
                knowledge_stores=[kb_names.get(kid, kid) for kid in knowledge_store_ids],
                top_k=top_k,
                bypass_access_check=bypass_access_check
            )
        
        # Verify access (unless bypassed for assistant access)
        if bypass_access_check:
            accessible_stores = knowledge_store_ids
        else:
            accessible_stores = await self._get_accessible_store_ids(db, user, knowledge_store_ids)
        
        if not accessible_stores:
            if debug_enabled:
                _log_rag_debug("No accessible knowledge stores found")
            return []
        
        if debug_enabled and len(accessible_stores) != len(knowledge_store_ids):
            _log_rag_debug(
                "Access filtered stores",
                requested=len(knowledge_store_ids),
                accessible=len(accessible_stores)
            )
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, self.embed_text, query
        )
        
        if query_embedding is None:
            if debug_enabled:
                _log_rag_debug("Falling back to keyword search (embedding failed)")
            # Fallback to keyword search
            doc_ids = await self._get_store_document_ids(db, accessible_stores)
            return await self._keyword_search_documents(db, doc_ids, query, top_k)
        
        # Search each knowledge store's FAISS index
        all_results = []
        index_manager = self.get_index_manager()
        
        logger.info(f"[KB_SEARCH] Searching {len(accessible_stores)} knowledge stores: {accessible_stores}")
        
        for store_id in accessible_stores:
            logger.info(f"[KB_SEARCH] Searching store: {store_id} (type: {type(store_id).__name__})")
            results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                index_manager.search,
                store_id,
                query_embedding,
                top_k
            )
            logger.info(f"[KB_SEARCH] Store {store_id} returned {len(results)} results")
            
            if debug_enabled and results:
                _log_rag_debug(
                    f"FAISS index search for store",
                    store_id=store_id,
                    results_count=len(results),
                    top_score=results[0][1] if results else 0
                )
            
            for chunk_id, score in results:
                all_results.append((chunk_id, score, store_id))
        
        if not all_results:
            if debug_enabled:
                _log_rag_debug("No FAISS results, falling back to direct search")
            # Fallback to direct search
            doc_ids = await self._get_store_document_ids(db, accessible_stores)
            return await self._direct_search(db, query_embedding, doc_ids, top_k)
        
        # Sort by score and get top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:top_k]
        
        if debug_enabled:
            _log_rag_debug(
                "Knowledge store search completed",
                total_results=len(all_results),
                returned_results=len(top_results),
                top_scores=[round(r[1], 4) for r in top_results[:5]]
            )
        
        # Fetch chunk details
        chunk_ids = [r[0] for r in top_results]
        final_results = await self._get_chunks_by_ids(db, chunk_ids, {r[0]: r[1] for r in top_results})
        
        if debug_enabled and final_results:
            _log_rag_debug(
                "Retrieved context chunks",
                chunks=[{
                    "doc": r.get("document_name", "?"),
                    "score": round(r.get("similarity", 0), 4),
                    "preview": r.get("content", "")[:80] + "..."
                } for r in final_results[:3]]
            )
        
        return final_results
    
    async def search_global_stores(
        self,
        db: AsyncSession,
        query: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Search all global knowledge stores for relevant context.
        
        Returns:
            Tuple of (results, store_names) where results are filtered by 
            each store's global_min_score threshold.
            
        This function is called automatically on every chat message when
        the global_knowledge_store_enabled setting is true.
        """
        import time
        from app.models.models import SystemSetting
        
        total_start = time.time()
        
        # Check if global knowledge stores are enabled
        setting_start = time.time()
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == "global_knowledge_store_enabled")
        )
        setting = result.scalar_one_or_none()
        setting_time = time.time() - setting_start
        logger.debug(f"[GLOBAL_RAG_TIMING] Setting check took {setting_time:.3f}s")
        
        if setting and setting.value.lower() != "true":
            logger.debug("[GLOBAL_RAG] Global knowledge stores disabled")
            return [], []
        
        # Find all global knowledge stores
        stores_start = time.time()
        result = await db.execute(
            select(KnowledgeStore).where(KnowledgeStore.is_global == True)
        )
        global_stores = result.scalars().all()
        stores_time = time.time() - stores_start
        logger.debug(f"[GLOBAL_RAG_TIMING] Store query took {stores_time:.3f}s, found {len(global_stores)} stores")
        
        if not global_stores:
            logger.debug("[GLOBAL_RAG] No global knowledge stores configured")
            return [], []
        
        logger.info(f"[GLOBAL_RAG] Searching {len(global_stores)} global stores for query: {query[:50]}...")
        
        # Generate query embedding once
        embed_start = time.time()
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, self.embed_text, query
        )
        embed_time = time.time() - embed_start
        logger.debug(f"[GLOBAL_RAG_TIMING] Embedding generation took {embed_time:.3f}s")
        
        if query_embedding is None:
            logger.warning("[GLOBAL_RAG] Failed to generate embedding for query")
            return [], []
        
        all_results = []
        matched_store_names = []
        index_manager = self.get_index_manager()
        
        for store in global_stores:
            store_start = time.time()
            store_id = str(store.id)
            min_score = store.global_min_score or 0.7
            max_results = store.global_max_results or 3
            
            # Search this store's FAISS index
            results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                index_manager.search,
                store_id,
                query_embedding,
                max_results * 2  # Get extra to filter by score
            )
            store_search_time = time.time() - store_start
            
            # Filter by minimum score threshold
            filtered_results = [(chunk_id, score) for chunk_id, score in results if score >= min_score]
            
            logger.debug(f"[GLOBAL_RAG_TIMING] Store '{store.name}' search took {store_search_time:.3f}s - raw={len(results)}, filtered={len(filtered_results)}")
            
            if filtered_results:
                logger.info(f"[GLOBAL_RAG] Store '{store.name}' matched with {len(filtered_results)} results (min_score={min_score})")
                for chunk_id, score in filtered_results[:max_results]:
                    logger.debug(f"[GLOBAL_RAG_DETAIL] Store '{store.name}' - chunk_id={chunk_id}, score={score:.4f}")
                matched_store_names.append(store.name)
                
                # Limit to max_results
                for chunk_id, score in filtered_results[:max_results]:
                    all_results.append((chunk_id, score, store_id))
            else:
                logger.debug(f"[GLOBAL_RAG] Store '{store.name}' had no matches above threshold {min_score}")
        
        if not all_results:
            total_time = time.time() - total_start
            logger.debug(f"[GLOBAL_RAG_TIMING] No relevant results found in any global store (total time: {total_time:.3f}s)")
            return [], []
        
        # Sort by score and fetch details
        all_results.sort(key=lambda x: x[1], reverse=True)
        chunk_ids = [r[0] for r in all_results]
        scores_map = {r[0]: r[1] for r in all_results}
        
        fetch_start = time.time()
        final_results = await self._get_chunks_by_ids(db, chunk_ids, scores_map)
        fetch_time = time.time() - fetch_start
        
        total_time = time.time() - total_start
        logger.info(f"[GLOBAL_RAG_TIMING] Total search time: {total_time:.3f}s (embed={embed_time:.3f}s, search={total_time - embed_time - fetch_time:.3f}s, fetch={fetch_time:.3f}s)")
        logger.info(f"[GLOBAL_RAG] Returning {len(final_results)} results from stores: {matched_store_names}")
        
        return final_results, matched_store_names
    
    async def _get_chunks_by_ids(
        self,
        db: AsyncSession,
        chunk_ids: List[str],
        scores: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Fetch chunk details by IDs"""
        result = await db.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
        rows = result.all()
        
        results = []
        for chunk, document in rows:
            results.append({
                "chunk_id": str(chunk.id),
                "document_id": str(document.id),
                "document_name": document.name,
                "knowledge_store_id": str(document.knowledge_store_id) if document.knowledge_store_id else None,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "similarity": scores.get(str(chunk.id), 0.0),
                "metadata": chunk.chunk_metadata,
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
    
    async def _get_store_document_ids(
        self,
        db: AsyncSession,
        store_ids: List[str],
    ) -> List[str]:
        """Get all document IDs for knowledge stores"""
        result = await db.execute(
            select(Document.id)
            .where(Document.knowledge_store_id.in_(store_ids))
            .where(Document.is_processed == True)
        )
        return [str(row[0]) for row in result.all()]
    
    async def _get_accessible_store_ids(
        self,
        db: AsyncSession,
        user: User,
        store_ids: List[str],
    ) -> List[str]:
        """Get list of knowledge store IDs the user has access to"""
        accessible = []
        
        for store_id in store_ids:
            result = await db.execute(
                select(KnowledgeStore).where(KnowledgeStore.id == store_id)
            )
            store = result.scalar_one_or_none()
            
            if not store:
                continue
            
            if store.owner_id == user.id or store.is_public:
                accessible.append(store_id)
                continue
            
            # Check for explicit share
            result = await db.execute(
                select(KnowledgeStoreShare).where(
                    KnowledgeStoreShare.knowledge_store_id == store_id,
                    KnowledgeStoreShare.shared_with_user_id == user.id
                )
            )
            if result.scalar_one_or_none():
                accessible.append(store_id)
        
        return accessible
    
    async def _keyword_search(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        document_ids: Optional[List[str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search.
        
        When document_ids is not specified, only searches unitemized documents
        (documents not in any knowledge store).
        """
        keywords = query.lower().split()
        
        query_builder = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.owner_id == user.id)
            .where(Document.is_processed == True)
        )
        
        if document_ids:
            query_builder = query_builder.where(Document.id.in_(document_ids))
        else:
            # Only search unitemized documents (not in any knowledge store)
            query_builder = query_builder.where(Document.knowledge_store_id.is_(None))
        
        result = await db.execute(query_builder)
        rows = result.all()
        
        results = []
        for chunk, document in rows:
            content_lower = chunk.content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            
            if score > 0:
                results.append({
                    "chunk_id": str(chunk.id),
                    "document_id": str(document.id),
                    "document_name": document.name,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "similarity": score / len(keywords),
                    "metadata": chunk.chunk_metadata,
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    async def _keyword_search_documents(
        self,
        db: AsyncSession,
        document_ids: List[str],
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Keyword search across specific documents"""
        keywords = query.lower().split()
        
        result = await db.execute(
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.id.in_(document_ids))
        )
        rows = result.all()
        
        results = []
        for chunk, document in rows:
            content_lower = chunk.content.lower()
            score = sum(1 for kw in keywords if kw in content_lower)
            
            if score > 0:
                results.append({
                    "chunk_id": str(chunk.id),
                    "document_id": str(document.id),
                    "document_name": document.name,
                    "knowledge_store_id": str(document.knowledge_store_id) if document.knowledge_store_id else None,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "similarity": score / len(keywords),
                    "metadata": chunk.chunk_metadata,
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    async def get_context_for_query(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """
        Get formatted context from user's UNITEMIZED documents for LLM prompt.
        This is the main entry point for RAG context retrieval.
        
        Only searches documents that are NOT in any knowledge store.
        Documents in knowledge stores are only accessed when:
        - A custom GPT with that knowledge store is selected (uses get_knowledge_store_context)
        - The user explicitly provides document_ids
        
        This prevents searching subscribed GPTs' knowledge stores when no GPT is selected.
        """
        # If specific document IDs provided, search only those
        if document_ids:
            results = await self.search(db, user, query, document_ids, top_k)
        else:
            # Search user's unitemized documents (not in any knowledge store)
            results = await self.search(db, user, query, None, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['document_name']}]\n{result['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    async def get_knowledge_store_context(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        knowledge_store_ids: List[str],
        top_k: int = 5,
        bypass_access_check: bool = False,
    ) -> str:
        """
        Get formatted context from knowledge stores for LLM prompt.
        
        Args:
            bypass_access_check: If True, skip ownership/permission checks.
                                Used when accessing KB through a public GPT.
        """
        results = await self.search_knowledge_stores(
            db, user, query, knowledge_store_ids, top_k,
            bypass_access_check=bypass_access_check,
        )
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['document_name']}]\n{result['content']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Log final context length if debug enabled
        debug_enabled = await _is_debug_rag_enabled(db)
        if debug_enabled:
            _log_rag_debug(
                "Context built for LLM prompt",
                sources=len(results),
                context_length=len(context),
                context_preview=context[:200] + "..." if len(context) > 200 else context
            )
        
        return context
    
    async def get_chat_knowledge_context(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        chat_knowledge_store_id: str,
        current_assistant_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """
        Get formatted context from user's chat history knowledge store.
        
        Filters results to match current assistant context:
        - If current_assistant_id is None: only return results from chats without an assistant
        - If current_assistant_id is set: only return results from chats with that specific assistant
        
        This prevents knowledge leakage between different assistant contexts.
        """
        # First get all results from the chat knowledge store
        results = await self.search_knowledge_stores(
            db, user, query, [chat_knowledge_store_id], top_k * 3,  # Get more to filter
            bypass_access_check=True,  # User owns this KB
        )
        
        if not results:
            return ""
        
        # Filter by assistant context
        filtered_results = []
        for result in results:
            # Note: _get_chunks_by_ids returns 'metadata' not 'chunk_metadata'
            chunk_metadata = result.get("metadata", {}) or {}
            result_assistant_id = chunk_metadata.get("assistant_id")
            
            # Match assistant context:
            # - Both None: chat without assistant matches query without assistant
            # - Both same ID: chat with assistant X matches query with assistant X
            if result_assistant_id == current_assistant_id:
                filtered_results.append(result)
                if len(filtered_results) >= top_k:
                    break
        
        if not filtered_results:
            logger.debug(f"[CHAT_KB] No results after assistant filter (current_assistant_id={current_assistant_id})")
            return ""
        
        logger.info(f"[CHAT_KB] Filtered {len(results)} -> {len(filtered_results)} results for assistant_id={current_assistant_id}")
        
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
            chat_title = (result.get("metadata") or {}).get("chat_title", "Previous Chat")
            context_parts.append(
                f"[From: {chat_title}]\n{result['content']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    async def rebuild_all_indexes(self, db: AsyncSession) -> Dict[str, int]:
        """Rebuild all knowledge store FAISS indexes"""
        result = await db.execute(select(KnowledgeStore))
        stores = result.scalars().all()
        
        rebuilt = {}
        for store in stores:
            await self._update_knowledge_store_index(db, str(store.id))
            
            # Count vectors in index
            index_manager = self.get_index_manager()
            if str(store.id) in index_manager._indexes:
                rebuilt[str(store.id)] = index_manager._indexes[str(store.id)].ntotal
            else:
                rebuilt[str(store.id)] = 0
        
        return rebuilt


class DocumentProcessor:
    """Process various document types for RAG"""
    
    @staticmethod
    async def extract_text_from_pdf_tika(file_path: str) -> str:
        """Extract text from PDF using Apache Tika"""
        import httpx
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(file_path, 'rb') as f:
                    pdf_content = f.read()
                
                response = await client.put(
                    settings.TIKA_URL,
                    content=pdf_content,
                    headers={
                        'Content-Type': 'application/pdf',
                        'Accept': 'text/plain',
                    }
                )
                
                if response.status_code == 200:
                    text = response.text.strip()
                    if text:
                        logger.info(f"Tika extracted {len(text)} chars from PDF")
                        return text
                else:
                    logger.warning(f"Tika returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Tika extraction failed: {e}")
        
        return ""
    
    @staticmethod
    async def extract_text(file_path: str, mime_type: str) -> str:
        """Extract text content from various file types"""
        import json
        path = Path(file_path)
        
        if mime_type in ["text/plain", "text/markdown"]:
            return path.read_text(encoding="utf-8")
        
        elif mime_type == "application/json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        
        elif mime_type == "text/csv":
            import csv
            text_parts = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    text_parts.append(" | ".join(row))
            return "\n".join(text_parts)
        
        elif mime_type == "application/pdf":
            # Try Tika first (better extraction)
            tika_text = await DocumentProcessor.extract_text_from_pdf_tika(file_path)
            if tika_text:
                return tika_text
            
            # Fallback to PyMuPDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                return "\n\n".join(text_parts)
            except ImportError:
                logger.warning("PyMuPDF not available for PDF fallback")
                return ""
        
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or path.suffix.lower() == ".docx":
            # Word document (.docx)
            return await DocumentProcessor.extract_text_from_docx(file_path)
        
        elif mime_type == "application/msword" or path.suffix.lower() == ".doc":
            # Legacy Word document (.doc) - try with docx library (may not work for all)
            # For full .doc support, would need antiword or libreoffice
            logger.warning(".doc files may not extract properly - consider converting to .docx")
            return await DocumentProcessor.extract_text_from_docx(file_path)
        
        elif mime_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"] or path.suffix.lower() in [".xlsx", ".xls"]:
            # Excel spreadsheet
            return await DocumentProcessor.extract_text_from_xlsx(file_path)
        
        elif mime_type in ["application/rtf", "text/rtf"] or path.suffix.lower() == ".rtf":
            # RTF document
            return await DocumentProcessor.extract_text_from_rtf(file_path)
        
        else:
            try:
                return path.read_text(encoding="utf-8")
            except:
                return ""
    
    @staticmethod
    async def extract_text_from_docx(file_path: str) -> str:
        """Extract text from Word document (.docx)"""
        try:
            from docx import Document
            doc = Document(file_path)
            
            text_parts = []
            
            # Extract paragraphs
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(text)} chars from DOCX")
            return text
            
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return ""
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    @staticmethod
    async def extract_text_from_xlsx(file_path: str) -> str:
        """Extract text from Excel spreadsheet (.xlsx/.xls)"""
        try:
            from openpyxl import load_workbook
            
            wb = load_workbook(file_path, data_only=True)
            text_parts = []
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"=== Sheet: {sheet_name} ===")
                
                for row in sheet.iter_rows():
                    row_values = []
                    for cell in row:
                        if cell.value is not None:
                            row_values.append(str(cell.value))
                    if row_values:
                        text_parts.append(" | ".join(row_values))
            
            wb.close()
            text = "\n".join(text_parts)
            logger.info(f"Extracted {len(text)} chars from XLSX")
            return text
            
        except ImportError:
            logger.error("openpyxl not installed. Run: pip install openpyxl")
            return ""
        except Exception as e:
            logger.error(f"XLSX extraction failed: {e}")
            # Try with xlrd for .xls files
            if file_path.lower().endswith('.xls'):
                return await DocumentProcessor._extract_text_from_xls_legacy(file_path)
            return ""
    
    @staticmethod
    async def _extract_text_from_xls_legacy(file_path: str) -> str:
        """Extract text from legacy Excel (.xls) using xlrd"""
        try:
            import xlrd
            
            wb = xlrd.open_workbook(file_path)
            text_parts = []
            
            for sheet_idx in range(wb.nsheets):
                sheet = wb.sheet_by_index(sheet_idx)
                text_parts.append(f"=== Sheet: {sheet.name} ===")
                
                for row_idx in range(sheet.nrows):
                    row_values = [str(cell.value) for cell in sheet.row(row_idx) if cell.value]
                    if row_values:
                        text_parts.append(" | ".join(row_values))
            
            text = "\n".join(text_parts)
            logger.info(f"Extracted {len(text)} chars from XLS (legacy)")
            return text
            
        except ImportError:
            logger.error("xlrd not installed. Run: pip install xlrd")
            return ""
        except Exception as e:
            logger.error(f"XLS extraction failed: {e}")
            return ""
    
    @staticmethod
    async def extract_text_from_rtf(file_path: str) -> str:
        """Extract text from RTF document"""
        try:
            from striprtf.striprtf import rtf_to_text
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            
            text = rtf_to_text(rtf_content)
            logger.info(f"Extracted {len(text)} chars from RTF")
            return text
            
        except ImportError:
            logger.error("striprtf not installed. Run: pip install striprtf")
            return ""
        except Exception as e:
            logger.error(f"RTF extraction failed: {e}")
            return ""
