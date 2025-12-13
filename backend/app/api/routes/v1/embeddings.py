"""
OpenAI-compatible /v1/embeddings endpoint

Creates embeddings using the configured embedding model.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models import User, APIKey
from app.api.routes.api_keys import get_api_key_user, require_scope
from app.services.rag import RAGService
from app.services.billing import BillingService

logger = logging.getLogger(__name__)

router = APIRouter()


# === Schemas (OpenAI-compatible) ===

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default="text-embedding-ada-002", description="Model to use")
    encoding_format: Optional[str] = Field(
        default="float",
        description="Encoding format: 'float' or 'base64'"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Dimensions for the output (if model supports)"
    )
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingData(BaseModel):
    """Single embedding result"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage for embeddings"""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# === Endpoints ===

@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    auth: tuple[User, APIKey] = Depends(require_scope("embeddings")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create embeddings for the given input.
    
    Compatible with OpenAI's /v1/embeddings endpoint.
    
    **Model:**
    Uses the locally configured embedding model (sentence-transformers).
    The model parameter is accepted for compatibility but the local model is used.
    
    **Input:**
    - Single string: Returns one embedding
    - List of strings: Returns embeddings for each string
    
    **Dimensions:**
    Output dimensions depend on the configured embedding model.
    Typical: 384 (MiniLM), 768 (base), 1024 (large)
    """
    user, api_key = auth
    
    # Normalize input to list
    inputs: List[str] = []
    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = request.input
    
    if not inputs:
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    if len(inputs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 inputs per request")
    
    # Get RAG service (which has the embedding model)
    rag_service = RAGService(db)
    
    # Generate embeddings
    embeddings: List[EmbeddingData] = []
    total_tokens = 0
    
    try:
        for i, text in enumerate(inputs):
            if not text or not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Input at index {i} is empty"
                )
            
            # Count tokens (approximate)
            tokens = len(text.split()) * 1.3
            total_tokens += int(tokens)
            
            # Generate embedding
            embedding = await rag_service.get_embedding(text)
            
            embeddings.append(EmbeddingData(
                embedding=embedding,
                index=i,
            ))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    
    # Bill for embedding tokens
    billing = BillingService(db)
    await billing.record_usage(
        user_id=user.id,
        input_tokens=total_tokens,
        output_tokens=0,
        model="embedding",
        source="v1_api",
    )
    
    return EmbeddingResponse(
        data=embeddings,
        model=request.model,  # Return requested model name for compatibility
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


@router.get("/embeddings/models")
async def list_embedding_models(
    auth: tuple[User, APIKey] = Depends(require_scope("embeddings")),
):
    """
    List available embedding models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "text-embedding-ada-002",
                "object": "model",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "owned_by": "system",
                "description": "Local embedding model (sentence-transformers compatible)",
                "dimensions": 384,  # Typical for MiniLM models
            },
            {
                "id": "text-embedding-3-small",
                "object": "model",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "owned_by": "system",
                "description": "Alias for local embedding model",
                "dimensions": 384,
            },
        ]
    }
