"""
Centralized exception handlers for consistent error responses

This module provides FastAPI exception handlers that ensure all errors
are returned in a consistent JSON format.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError
import logging

logger = logging.getLogger(__name__)


def setup_exception_handlers(app: FastAPI):
    """
    Register all exception handlers on the FastAPI app.
    
    Call this after creating the FastAPI app instance:
        app = FastAPI()
        setup_exception_handlers(app)
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors with detailed field info"""
        errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"]
            })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": errors
            }
        )
    
    @app.exception_handler(IntegrityError)
    async def integrity_exception_handler(request: Request, exc: IntegrityError):
        """Handle database integrity errors (unique constraints, foreign keys)"""
        logger.error(f"Database integrity error: {exc}")
        
        # Parse common constraint violations
        error_str = str(exc.orig) if exc.orig else str(exc)
        
        if "UNIQUE constraint failed" in error_str:
            # Extract field name from error
            field = error_str.split(".")[-1] if "." in error_str else "field"
            return JSONResponse(
                status_code=409,
                content={
                    "error": "duplicate_entry",
                    "message": f"A record with this {field} already exists",
                    "field": field
                }
            )
        
        if "FOREIGN KEY constraint failed" in error_str:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_reference",
                    "message": "Referenced resource does not exist"
                }
            )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "database_error",
                "message": "A database error occurred"
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors as bad requests"""
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_value",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all handler for unhandled exceptions"""
        logger.exception(f"Unhandled exception: {exc}")
        
        # Don't expose internal details in production
        from app.core.config import settings
        
        if settings.DEBUG:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": str(exc),
                    "type": type(exc).__name__
                }
            )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An unexpected error occurred"
            }
        )
