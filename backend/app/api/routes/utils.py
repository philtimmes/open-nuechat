"""
Utility API routes
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from markdown_pdf import MarkdownPdf, Section

from app.api.dependencies import get_current_user
from app.models.models import User


router = APIRouter(tags=["Utils"])


class MarkdownToPdfRequest(BaseModel):
    content: str
    filename: str | None = None
    title: str | None = None


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
