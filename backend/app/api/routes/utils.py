"""
Utility API routes
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from markdown_pdf import MarkdownPdf, Section
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.models.models import User
from app.db.database import get_db
from app.api.routes.admin import get_system_setting


router = APIRouter(tags=["Utils"])


class MarkdownToPdfRequest(BaseModel):
    content: str
    filename: str | None = None
    title: str | None = None


class ThinkingTokensResponse(BaseModel):
    think_begin_token: str
    think_end_token: str


@router.get("/thinking-tokens", response_model=ThinkingTokensResponse)
async def get_thinking_tokens(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get thinking tokens configuration for rendering thinking blocks"""
    return ThinkingTokensResponse(
        think_begin_token=await get_system_setting(db, "think_begin_token") or "",
        think_end_token=await get_system_setting(db, "think_end_token") or "",
    )


@router.post("/markdown-to-pdf")
async def convert_markdown_to_pdf(
    request: MarkdownToPdfRequest,
    user: User = Depends(get_current_user),
):
    """Convert markdown content to PDF"""
    try:
        # Create PDF from markdown
        pdf = MarkdownPdf(toc_level=2)
        pdf.add_section(Section(request.content))
        
        # Generate PDF bytes
        pdf_bytes = pdf.out_pdf
        
        # Determine filename
        filename = request.filename or request.title or "document"
        if not filename.endswith('.pdf'):
            filename = filename.replace('.md', '') + '.pdf'
        
        # Return PDF as downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert to PDF: {str(e)}")
