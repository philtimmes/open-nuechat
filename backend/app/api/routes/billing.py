"""
Billing API routes
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.api.schemas import UsageSummary, UsageHistory, InvoiceResponse
from app.models.models import User, UserTier
from app.services.billing import BillingService


router = APIRouter(tags=["Billing"])


@router.get("/usage", response_model=UsageSummary)
async def get_usage(
    year: Optional[int] = None,
    month: Optional[int] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get token usage summary for current or specified month"""
    
    billing = BillingService()
    summary = await billing.get_usage_summary(db, user, year, month)
    
    return UsageSummary(**summary)


@router.get("/usage/history", response_model=List[UsageHistory])
async def get_usage_history(
    days: int = Query(30, ge=1, le=90),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get daily usage history"""
    
    billing = BillingService()
    history = await billing.get_usage_history(db, user, days)
    
    return [UsageHistory(**h) for h in history]


@router.get("/usage/by-chat")
async def get_usage_by_chat(
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get usage breakdown by chat"""
    
    billing = BillingService()
    usage = await billing.get_usage_by_chat(db, user, limit)
    
    return usage


@router.get("/check-limit")
async def check_limit(
    estimated_tokens: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check if user has remaining quota"""
    
    billing = BillingService()
    result = await billing.check_usage_limit(db, user, estimated_tokens)
    
    return result


@router.get("/invoice/{year}/{month}", response_model=InvoiceResponse)
async def get_invoice(
    year: int,
    month: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get invoice data for a specific month"""
    
    if month < 1 or month > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid month",
        )
    
    billing = BillingService()
    invoice = await billing.get_invoice_data(db, user, year, month)
    
    return InvoiceResponse(**invoice)


@router.get("/tiers")
async def get_tiers():
    """Get available subscription tiers"""
    
    return {
        "tiers": [
            {
                "id": "free",
                "name": "Free",
                "price": 0,
                "tokens_limit": 100_000,
                "features": [
                    "100K tokens/month",
                    "Basic chat",
                    "Single user",
                ]
            },
            {
                "id": "pro",
                "name": "Pro",
                "price": 20,
                "tokens_limit": 1_000_000,
                "features": [
                    "1M tokens/month",
                    "All chat features",
                    "RAG with 10 documents",
                    "Tool calling",
                    "Priority support",
                ]
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": 100,
                "tokens_limit": 10_000_000,
                "features": [
                    "10M tokens/month",
                    "All Pro features",
                    "Unlimited documents",
                    "Custom integrations",
                    "Dedicated support",
                    "SSO/SAML",
                ]
            }
        ]
    }


@router.post("/upgrade/{tier}")
async def upgrade_tier(
    tier: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upgrade subscription tier (placeholder for Stripe integration)"""
    
    valid_tiers = {"free", "pro", "enterprise"}
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Must be one of: {valid_tiers}",
        )
    
    new_tier = UserTier(tier)
    
    # In production, this would initiate Stripe checkout
    # For now, just update the tier directly
    billing = BillingService()
    await billing.upgrade_tier(db, user, new_tier)
    
    await db.commit()
    
    return {
        "status": "upgraded",
        "tier": tier,
        "tokens_limit": user.tokens_limit,
    }
