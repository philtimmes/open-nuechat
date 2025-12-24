"""
Billing API routes

Provides:
- Usage tracking and summaries
- Subscription management
- Payment processing (Stripe, PayPal, Google Pay)
- Invoice generation
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime

from app.db.database import get_db
from app.api.dependencies import get_current_user
from app.api.schemas import UsageSummary, UsageHistory, InvoiceResponse
from app.models.models import User, UserTier
from app.services.billing import BillingService
from app.services.payments import PaymentService, TIER_CONFIG
from app.core.config import settings


router = APIRouter(tags=["Billing"])
payment_service = PaymentService()


# =============================================================================
# USAGE ENDPOINTS
# =============================================================================

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


# =============================================================================
# SUBSCRIPTION ENDPOINTS
# =============================================================================

@router.get("/subscription")
async def get_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current subscription status"""
    
    subscription = await payment_service.get_subscription(db, user)
    
    return {
        "tier": user.tier.value,
        "tokens_limit": user.tokens_limit,
        "subscription": subscription,
    }


@router.post("/subscribe/{tier}")
async def create_subscription(
    tier: str,
    provider: str = Query("stripe", regex="^(stripe|paypal|google_pay)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a subscription checkout session"""
    
    if tier not in TIER_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {tier}. Valid tiers: {list(TIER_CONFIG.keys())}",
        )
    
    if TIER_CONFIG[tier]["price"] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot subscribe to free tier",
        )
    
    available = payment_service.get_available_providers()
    if provider not in available and provider != "google_pay":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment provider '{provider}' not configured. Available: {available}",
        )
    
    try:
        result = await payment_service.create_checkout(
            db=db,
            user=user,
            tier=tier,
            provider=provider,
        )
        await db.commit()
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/cancel-subscription")
async def cancel_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel current subscription (will remain active until period end)"""
    
    success = await payment_service.cancel_subscription(db, user)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to cancel",
        )
    
    await db.commit()
    
    return {
        "status": "cancelled",
        "message": "Subscription will remain active until the end of the current billing period",
    }


# =============================================================================
# PAYMENT METHOD ENDPOINTS
# =============================================================================

@router.get("/payment-methods")
async def get_payment_methods(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get stored payment methods"""
    
    methods = await payment_service.get_payment_methods(db, user)
    return {"payment_methods": methods}


@router.get("/transactions")
async def get_transactions(
    limit: int = Query(10, ge=1, le=50),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get transaction history"""
    
    transactions = await payment_service.get_transactions(db, user, limit)
    return {"transactions": transactions}


# =============================================================================
# PAYMENT PROVIDER CONFIG
# =============================================================================

@router.get("/providers")
async def get_payment_providers():
    """Get available payment providers and configuration"""
    
    providers = payment_service.get_available_providers()
    
    config = {
        "available_providers": providers,
        "currency": settings.PAYMENT_CURRENCY,
    }
    
    # Add Google Pay config if Stripe is available
    if "google_pay" in providers:
        config["google_pay"] = payment_service.google_pay.get_config()
    
    # Add Stripe publishable key if available
    if "stripe" in providers and settings.STRIPE_PUBLISHABLE_KEY:
        config["stripe_publishable_key"] = settings.STRIPE_PUBLISHABLE_KEY
    
    return config


# =============================================================================
# TIERS & PRICING
# =============================================================================

@router.get("/tiers")
async def get_tiers():
    """Get available subscription tiers"""
    
    return {
        "tiers": [
            {
                "id": "free",
                "name": "Free",
                "price": 0,
                "tokens_limit": settings.FREE_TIER_TOKENS,
                "features": [
                    f"{settings.FREE_TIER_TOKENS:,} tokens/month",
                    "Basic chat",
                    "Single user",
                ],
                "popular": False,
            },
            {
                "id": "pro",
                "name": "Pro",
                "price": settings.PRO_TIER_PRICE,
                "tokens_limit": settings.PRO_TIER_TOKENS,
                "features": [
                    f"{settings.PRO_TIER_TOKENS:,} tokens/month",
                    "All chat features",
                    "RAG with 10 documents",
                    "Tool calling",
                    "Priority support",
                ],
                "popular": True,
            },
            {
                "id": "enterprise",
                "name": "Enterprise",
                "price": settings.ENTERPRISE_TIER_PRICE,
                "tokens_limit": settings.ENTERPRISE_TIER_TOKENS,
                "features": [
                    f"{settings.ENTERPRISE_TIER_TOKENS:,} tokens/month",
                    "All Pro features",
                    "Unlimited documents",
                    "Custom integrations",
                    "Dedicated support",
                    "SSO/SAML",
                ],
                "popular": False,
            }
        ]
    }


# =============================================================================
# INVOICE
# =============================================================================

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


# =============================================================================
# WEBHOOKS
# =============================================================================

@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle Stripe webhooks"""
    
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    
    try:
        result = await payment_service.handle_webhook(db, "stripe", payload, signature)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/webhooks/paypal")
async def paypal_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle PayPal webhooks"""
    
    payload = await request.body()
    signature = request.headers.get("paypal-transmission-sig", "")
    
    try:
        result = await payment_service.handle_webhook(db, "paypal", payload, signature)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# =============================================================================
# LEGACY UPGRADE ENDPOINT (for backward compatibility)
# =============================================================================

@router.post("/upgrade/{tier}")
async def upgrade_tier(
    tier: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Legacy upgrade endpoint.
    
    For free tier: directly updates user tier
    For paid tiers: redirects to subscription flow
    """
    
    valid_tiers = {"free", "pro", "enterprise"}
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Must be one of: {valid_tiers}",
        )
    
    # Free tier upgrade is instant (downgrade)
    if tier == "free":
        new_tier = UserTier.FREE
        billing = BillingService()
        await billing.upgrade_tier(db, user, new_tier)
        await db.commit()
        
        return {
            "status": "upgraded",
            "tier": tier,
            "tokens_limit": user.tokens_limit,
        }
    
    # Paid tiers require payment
    return {
        "status": "payment_required",
        "tier": tier,
        "message": "Use /billing/subscribe/{tier} endpoint with payment provider",
        "available_providers": payment_service.get_available_providers(),
    }
