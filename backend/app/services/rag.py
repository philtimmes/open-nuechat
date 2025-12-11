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
        
        faiss.write_index(cpu_index, str(index_path))
        with open(map_path, 'wb') as f:
            pickle.dump(id_map, f)
        
        # Cache GPU version
        self._indexes[index_id] = index
        self._id_maps[index_id] = id_map
        
        logger.info(f"Built index {index_id}: {n_vectors} vectors, saved to {index_path}")
    
    def load_index(self, index_id: str) -> bool:
        """Load index from disk into memory/GPU"""
        if index_id in self._indexes:
            return True
        
        index_path, map_path = self._get_index_path(index_id)
        
        if not index_path.exists():
            return False
        
        try:
            # Load from disk
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
            return []
        
        index = self._indexes[index_id]
        id_map = self._id_maps[index_id]
        
        # Prepare query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = index.search(query, min(top_k, index.ntotal))
        
        # Map back to chunk IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in id_map:
                results.append((id_map[idx], float(score)))
        
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
    _index_manager = None
    
    @classmethod
    def get_model(cls):
        """Lazy load the embedding model with GPU support"""
        if not cls._model_loaded:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                
                # ROCm/HIP uses torch.cuda API - same interface, different backend
                # torch.cuda.is_available() returns True for ROCm GPUs
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading embedding model on device: {device} (ROCm/HIP)")
                
                cls._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=device
                )
                cls._model_loaded = True
                
                # Update embedding dimension in index manager
                dim = cls._model.get_sentence_embedding_dimension()
                FAISSIndexManager.EMBEDDING_DIM = dim
                logger.info(f"Embedding model loaded: dim={dim}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                cls._model = None
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
            return None
        
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
    
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
            self.get_index_manager().delete_index(knowledge_store_id)
            return
        
        # Collect embeddings and IDs
        embeddings = []
        chunk_ids = []
        
        for chunk, document in rows:
            embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
            embeddings.append(embedding)
            chunk_ids.append(str(chunk.id))
        
        embeddings_array = np.vstack(embeddings)
        
        # Build index in thread pool
        await asyncio.get_event_loop().run_in_executor(
            _executor,
            self.get_index_manager().build_index,
            knowledge_store_id,
            embeddings_array,
            chunk_ids
        )
    
    async def search(
        self,
        db: AsyncSession,
        user: User,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks using FAISS"""
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, self.embed_text, query
        )
        
        if query_embedding is None:
            return await self._keyword_search(db, user, query, document_ids, top_k)
        
        # If document_ids specified, do direct search
        if document_ids:
            return await self._direct_search(db, query_embedding, document_ids, top_k)
        
        # Otherwise search user's documents
        return await self._user_document_search(db, user, query_embedding, top_k)
    
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
            return []
        
        # Calculate similarities
        results = []
        for chunk, document in rows:
            chunk_embedding = np.frombuffer(chunk.embedding, dtype=np.float32)
            # Cosine similarity (vectors should be normalized)
            similarity = float(np.dot(query_embedding, chunk_embedding))
            
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
        return results[:top_k]
    
    async def _user_document_search(
        self,
        db: AsyncSession,
        user: User,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search all user's documents"""
        # Get user's documents
        result = await db.execute(
            select(Document)
            .where(Document.owner_id == user.id)
            .where(Document.is_processed == True)
        )
        documents = result.scalars().all()
        
        if not documents:
            return []
        
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
        
        # Verify access (unless bypassed for assistant access)
        if bypass_access_check:
            accessible_stores = knowledge_store_ids
        else:
            accessible_stores = await self._get_accessible_store_ids(db, user, knowledge_store_ids)
        
        if not accessible_stores:
            return []
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            _executor, self.embed_text, query
        )
        
        if query_embedding is None:
            # Fallback to keyword search
            doc_ids = await self._get_store_document_ids(db, accessible_stores)
            return await self._keyword_search_documents(db, doc_ids, query, top_k)
        
        # Search each knowledge store's FAISS index
        all_results = []
        index_manager = self.get_index_manager()
        
        for store_id in accessible_stores:
            results = await asyncio.get_event_loop().run_in_executor(
                _executor,
                index_manager.search,
                store_id,
                query_embedding,
                top_k
            )
            
            for chunk_id, score in results:
                all_results.append((chunk_id, score, store_id))
        
        if not all_results:
            # Fallback to direct search
            doc_ids = await self._get_store_document_ids(db, accessible_stores)
            return await self._direct_search(db, query_embedding, doc_ids, top_k)
        
        # Sort by score and get top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:top_k]
        
        # Fetch chunk details
        chunk_ids = [r[0] for r in top_results]
        return await self._get_chunks_by_ids(db, chunk_ids, {r[0]: r[1] for r in top_results})
    
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
        """Fallback keyword-based search"""
        keywords = query.lower().split()
        
        query_builder = (
            select(DocumentChunk, Document)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.owner_id == user.id)
            .where(Document.is_processed == True)
        )
        
        if document_ids:
            query_builder = query_builder.where(Document.id.in_(document_ids))
        
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
        Get formatted context from user's documents for LLM prompt.
        This is the main entry point for RAG context retrieval.
        """
        # If specific document IDs provided, use those
        if document_ids:
            results = await self._user_document_search(
                db, user, query, document_ids, top_k
            )
        else:
            # Search across all user's documents
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
        
        else:
            try:
                return path.read_text(encoding="utf-8")
            except:
                return ""
