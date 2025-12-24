"""
Payment service for handling Stripe, PayPal, and Google Pay integrations.

Provides:
- Subscription management
- One-time payments
- Payment method storage
- Webhook handling
"""
import logging
import json
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from abc import ABC, abstractmethod

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import settings
from app.models.billing import (
    Subscription, PaymentMethod, Transaction,
    PaymentProvider, PaymentStatus, SubscriptionStatus
)
from app.models.models import User, UserTier

logger = logging.getLogger(__name__)


# =============================================================================
# TIER CONFIGURATION
# =============================================================================

TIER_CONFIG = {
    "free": {
        "name": "Free",
        "price": 0,
        "tokens": settings.FREE_TIER_TOKENS,
        "stripe_price_id": None,
        "paypal_plan_id": None,
    },
    "pro": {
        "name": "Pro",
        "price": settings.PRO_TIER_PRICE,
        "tokens": settings.PRO_TIER_TOKENS,
        "stripe_price_id": None,  # Set via env: STRIPE_PRO_PRICE_ID
        "paypal_plan_id": None,   # Set via env: PAYPAL_PRO_PLAN_ID
    },
    "enterprise": {
        "name": "Enterprise",
        "price": settings.ENTERPRISE_TIER_PRICE,
        "tokens": settings.ENTERPRISE_TIER_TOKENS,
        "stripe_price_id": None,  # Set via env: STRIPE_ENTERPRISE_PRICE_ID
        "paypal_plan_id": None,   # Set via env: PAYPAL_ENTERPRISE_PLAN_ID
    },
}


# =============================================================================
# ABSTRACT PAYMENT PROVIDER
# =============================================================================

class PaymentProviderBase(ABC):
    """Abstract base class for payment providers"""
    
    @abstractmethod
    async def create_checkout_session(
        self,
        user: User,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """Create a checkout session for subscription"""
        pass
    
    @abstractmethod
    async def create_customer(self, user: User) -> str:
        """Create a customer in the payment provider"""
        pass
    
    @abstractmethod
    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription"""
        pass
    
    @abstractmethod
    async def get_payment_methods(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get stored payment methods for a customer"""
        pass
    
    @abstractmethod
    async def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """Handle webhook from payment provider"""
        pass


# =============================================================================
# STRIPE PROVIDER
# =============================================================================

class StripeProvider(PaymentProviderBase):
    """Stripe payment provider implementation"""
    
    def __init__(self):
        self.api_key = settings.STRIPE_API_KEY
        self.webhook_secret = settings.STRIPE_WEBHOOK_SECRET
        self._stripe = None
    
    @property
    def stripe(self):
        if self._stripe is None:
            try:
                import stripe
                stripe.api_key = self.api_key
                self._stripe = stripe
            except ImportError:
                logger.error("Stripe library not installed. Run: pip install stripe")
                raise
        return self._stripe
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    async def create_customer(self, user: User) -> str:
        """Create a Stripe customer"""
        if not self.is_configured():
            raise ValueError("Stripe not configured")
        
        customer = self.stripe.Customer.create(
            email=user.email,
            name=user.full_name or user.username,
            metadata={"user_id": user.id}
        )
        return customer.id
    
    async def create_checkout_session(
        self,
        user: User,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout session"""
        if not self.is_configured():
            raise ValueError("Stripe not configured")
        
        tier_config = TIER_CONFIG.get(tier)
        if not tier_config or tier_config["price"] == 0:
            raise ValueError(f"Invalid tier for checkout: {tier}")
        
        # Get or create customer
        customer_id = user.stripe_customer_id
        if not customer_id:
            customer_id = await self.create_customer(user)
        
        # Create checkout session
        session = self.stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": settings.PAYMENT_CURRENCY.lower(),
                    "product_data": {
                        "name": f"NueChat {tier_config['name']} Plan",
                        "description": f"{tier_config['tokens']:,} tokens/month",
                    },
                    "unit_amount": int(tier_config["price"] * 100),  # Cents
                    "recurring": {"interval": "month"},
                },
                "quantity": 1,
            }],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": user.id,
                "tier": tier,
            },
            # Enable Google Pay / Apple Pay
            payment_method_options={
                "card": {
                    "request_three_d_secure": "automatic",
                },
            },
        )
        
        return {
            "session_id": session.id,
            "url": session.url,
            "provider": "stripe",
        }
    
    async def create_payment_intent(
        self,
        amount: float,
        currency: str,
        customer_id: str,
        metadata: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Create a payment intent for one-time payment"""
        if not self.is_configured():
            raise ValueError("Stripe not configured")
        
        intent = self.stripe.PaymentIntent.create(
            amount=int(amount * 100),  # Cents
            currency=currency.lower(),
            customer=customer_id,
            metadata=metadata or {},
            automatic_payment_methods={"enabled": True},
        )
        
        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
        }
    
    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a Stripe subscription"""
        if not self.is_configured():
            raise ValueError("Stripe not configured")
        
        try:
            self.stripe.Subscription.modify(
                subscription_id,
                cancel_at_period_end=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cancel Stripe subscription: {e}")
            return False
    
    async def get_payment_methods(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get stored payment methods"""
        if not self.is_configured():
            return []
        
        try:
            methods = self.stripe.PaymentMethod.list(
                customer=customer_id,
                type="card"
            )
            return [
                {
                    "id": pm.id,
                    "type": "card",
                    "brand": pm.card.brand,
                    "last_four": pm.card.last4,
                    "exp_month": pm.card.exp_month,
                    "exp_year": pm.card.exp_year,
                }
                for pm in methods.data
            ]
        except Exception as e:
            logger.error(f"Failed to get Stripe payment methods: {e}")
            return []
    
    async def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """Handle Stripe webhook"""
        if not self.is_configured() or not self.webhook_secret:
            raise ValueError("Stripe webhooks not configured")
        
        try:
            event = self.stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
        except Exception as e:
            logger.error(f"Stripe webhook verification failed: {e}")
            raise ValueError("Invalid webhook signature")
        
        return {
            "type": event.type,
            "data": event.data.object,
            "provider": "stripe",
        }


# =============================================================================
# PAYPAL PROVIDER
# =============================================================================

class PayPalProvider(PaymentProviderBase):
    """PayPal payment provider implementation"""
    
    def __init__(self):
        self.client_id = settings.PAYPAL_CLIENT_ID
        self.client_secret = settings.PAYPAL_CLIENT_SECRET
        self.mode = settings.PAYPAL_MODE
        self._access_token = None
        self._token_expires = None
    
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)
    
    @property
    def base_url(self) -> str:
        if self.mode == "live":
            return "https://api-m.paypal.com"
        return "https://api-m.sandbox.paypal.com"
    
    async def _get_access_token(self) -> str:
        """Get PayPal OAuth access token"""
        import aiohttp
        import base64
        
        if self._access_token and self._token_expires and datetime.now(timezone.utc) < self._token_expires:
            return self._access_token
        
        auth = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/oauth2/token",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data="grant_type=client_credentials",
            ) as resp:
                if resp.status != 200:
                    raise ValueError("Failed to get PayPal access token")
                data = await resp.json()
                self._access_token = data["access_token"]
                # Token expires in seconds, subtract 60 for safety
                from datetime import timedelta
                self._token_expires = datetime.now(timezone.utc) + timedelta(seconds=data["expires_in"] - 60)
                return self._access_token
    
    async def create_customer(self, user: User) -> str:
        """PayPal doesn't have persistent customers, return email as ID"""
        return user.email
    
    async def create_checkout_session(
        self,
        user: User,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> Dict[str, Any]:
        """Create a PayPal subscription"""
        import aiohttp
        
        if not self.is_configured():
            raise ValueError("PayPal not configured")
        
        tier_config = TIER_CONFIG.get(tier)
        if not tier_config or tier_config["price"] == 0:
            raise ValueError(f"Invalid tier for checkout: {tier}")
        
        token = await self._get_access_token()
        
        # Create subscription
        async with aiohttp.ClientSession() as session:
            # First create a product if needed (in production, these would be pre-created)
            product_data = {
                "name": f"NueChat {tier_config['name']} Plan",
                "description": f"{tier_config['tokens']:,} tokens/month",
                "type": "SERVICE",
                "category": "SOFTWARE",
            }
            
            async with session.post(
                f"{self.base_url}/v1/catalogs/products",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=product_data,
            ) as resp:
                if resp.status not in (200, 201):
                    # Product might already exist, continue
                    logger.warning("Could not create PayPal product, may already exist")
                    product_id = f"nuechat-{tier}"
                else:
                    product = await resp.json()
                    product_id = product["id"]
            
            # Create billing plan
            plan_data = {
                "product_id": product_id,
                "name": f"NueChat {tier_config['name']} Monthly",
                "billing_cycles": [
                    {
                        "frequency": {"interval_unit": "MONTH", "interval_count": 1},
                        "tenure_type": "REGULAR",
                        "sequence": 1,
                        "total_cycles": 0,  # Infinite
                        "pricing_scheme": {
                            "fixed_price": {
                                "value": str(tier_config["price"]),
                                "currency_code": settings.PAYMENT_CURRENCY,
                            }
                        }
                    }
                ],
                "payment_preferences": {
                    "auto_bill_outstanding": True,
                    "payment_failure_threshold": 3,
                }
            }
            
            async with session.post(
                f"{self.base_url}/v1/billing/plans",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=plan_data,
            ) as resp:
                if resp.status not in (200, 201):
                    error = await resp.text()
                    logger.error(f"PayPal plan creation failed: {error}")
                    raise ValueError("Failed to create PayPal plan")
                plan = await resp.json()
                plan_id = plan["id"]
            
            # Create subscription
            subscription_data = {
                "plan_id": plan_id,
                "subscriber": {
                    "email_address": user.email,
                },
                "application_context": {
                    "return_url": success_url,
                    "cancel_url": cancel_url,
                    "user_action": "SUBSCRIBE_NOW",
                },
                "custom_id": json.dumps({"user_id": user.id, "tier": tier}),
            }
            
            async with session.post(
                f"{self.base_url}/v1/billing/subscriptions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=subscription_data,
            ) as resp:
                if resp.status not in (200, 201):
                    error = await resp.text()
                    logger.error(f"PayPal subscription creation failed: {error}")
                    raise ValueError("Failed to create PayPal subscription")
                subscription = await resp.json()
        
        # Find approval link
        approval_url = None
        for link in subscription.get("links", []):
            if link["rel"] == "approve":
                approval_url = link["href"]
                break
        
        return {
            "subscription_id": subscription["id"],
            "url": approval_url,
            "provider": "paypal",
        }
    
    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a PayPal subscription"""
        import aiohttp
        
        if not self.is_configured():
            raise ValueError("PayPal not configured")
        
        token = await self._get_access_token()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/billing/subscriptions/{subscription_id}/cancel",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"reason": "User requested cancellation"},
            ) as resp:
                return resp.status == 204
    
    async def get_payment_methods(self, customer_id: str) -> List[Dict[str, Any]]:
        """PayPal doesn't store payment methods the same way"""
        return []
    
    async def handle_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """Handle PayPal webhook"""
        import aiohttp
        
        if not self.is_configured():
            raise ValueError("PayPal webhooks not configured")
        
        # Verify webhook (simplified - in production use PayPal's verification API)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError("Invalid webhook payload")
        
        return {
            "type": data.get("event_type"),
            "data": data.get("resource", {}),
            "provider": "paypal",
        }


# =============================================================================
# GOOGLE PAY PROVIDER (Uses Stripe as processor)
# =============================================================================

class GooglePayProvider:
    """
    Google Pay integration.
    
    Note: Google Pay is a payment method, not a payment processor.
    It uses Stripe as the underlying processor.
    """
    
    def __init__(self, stripe_provider: StripeProvider):
        self.stripe = stripe_provider
        self.merchant_id = settings.GOOGLE_PAY_MERCHANT_ID
        self.merchant_name = settings.GOOGLE_PAY_MERCHANT_NAME
    
    def is_configured(self) -> bool:
        return self.stripe.is_configured()
    
    def get_config(self) -> Dict[str, Any]:
        """Get Google Pay client configuration"""
        return {
            "environment": "TEST" if settings.PAYPAL_MODE == "sandbox" else "PRODUCTION",
            "merchantInfo": {
                "merchantId": self.merchant_id,
                "merchantName": self.merchant_name,
            },
            "allowedPaymentMethods": [{
                "type": "CARD",
                "parameters": {
                    "allowedAuthMethods": ["PAN_ONLY", "CRYPTOGRAM_3DS"],
                    "allowedCardNetworks": ["AMEX", "DISCOVER", "MASTERCARD", "VISA"],
                },
                "tokenizationSpecification": {
                    "type": "PAYMENT_GATEWAY",
                    "parameters": {
                        "gateway": "stripe",
                        "stripe:version": "2023-10-16",
                        "stripe:publishableKey": settings.STRIPE_PUBLISHABLE_KEY,
                    }
                }
            }],
        }


# =============================================================================
# PAYMENT SERVICE
# =============================================================================

class PaymentService:
    """
    Unified payment service that manages all payment providers.
    """
    
    def __init__(self):
        self.stripe = StripeProvider()
        self.paypal = PayPalProvider()
        self.google_pay = GooglePayProvider(self.stripe)
    
    def get_available_providers(self) -> List[str]:
        """Get list of configured payment providers"""
        providers = []
        if self.stripe.is_configured():
            providers.append("stripe")
            providers.append("google_pay")  # Google Pay uses Stripe
        if self.paypal.is_configured():
            providers.append("paypal")
        return providers
    
    def get_provider(self, provider: str) -> PaymentProviderBase:
        """Get a specific payment provider"""
        if provider == "stripe":
            return self.stripe
        elif provider == "paypal":
            return self.paypal
        elif provider == "google_pay":
            return self.stripe  # Google Pay uses Stripe
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def create_checkout(
        self,
        db: AsyncSession,
        user: User,
        tier: str,
        provider: str,
        success_url: str = None,
        cancel_url: str = None,
    ) -> Dict[str, Any]:
        """Create a checkout session with the specified provider"""
        
        if tier not in TIER_CONFIG:
            raise ValueError(f"Invalid tier: {tier}")
        
        if TIER_CONFIG[tier]["price"] == 0:
            raise ValueError("Cannot create checkout for free tier")
        
        success_url = success_url or settings.PAYMENT_SUCCESS_URL
        cancel_url = cancel_url or settings.PAYMENT_CANCEL_URL
        
        payment_provider = self.get_provider(provider)
        result = await payment_provider.create_checkout_session(
            user=user,
            tier=tier,
            success_url=success_url,
            cancel_url=cancel_url,
        )
        
        # Create pending transaction
        transaction = Transaction(
            user_id=user.id,
            type="subscription",
            amount=TIER_CONFIG[tier]["price"],
            currency=settings.PAYMENT_CURRENCY,
            status=PaymentStatus.PENDING,
            provider=PaymentProvider(provider if provider != "google_pay" else "stripe"),
            description=f"Subscription to {tier} plan",
            payment_metadata=json.dumps({"tier": tier, "session_id": result.get("session_id")}),
        )
        db.add(transaction)
        await db.flush()
        
        result["transaction_id"] = transaction.id
        return result
    
    async def handle_successful_payment(
        self,
        db: AsyncSession,
        user_id: str,
        tier: str,
        provider: str,
        provider_subscription_id: str,
        provider_customer_id: str = None,
    ) -> Subscription:
        """Handle successful subscription payment"""
        
        # Get user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User not found: {user_id}")
        
        # Update user tier
        tier_enum = UserTier(tier)
        user.tier = tier_enum
        user.tokens_limit = TIER_CONFIG[tier]["tokens"]
        
        if provider_customer_id:
            user.stripe_customer_id = provider_customer_id
        
        # Create or update subscription
        result = await db.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        subscription = result.scalar_one_or_none()
        
        now = datetime.now(timezone.utc)
        
        if subscription:
            subscription.tier = tier
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.provider = PaymentProvider(provider)
            subscription.provider_subscription_id = provider_subscription_id
            subscription.provider_customer_id = provider_customer_id
            subscription.current_period_start = now
            subscription.cancel_at_period_end = False
            subscription.cancelled_at = None
        else:
            subscription = Subscription(
                user_id=user_id,
                tier=tier,
                status=SubscriptionStatus.ACTIVE,
                provider=PaymentProvider(provider),
                provider_subscription_id=provider_subscription_id,
                provider_customer_id=provider_customer_id,
                current_period_start=now,
            )
            db.add(subscription)
        
        await db.flush()
        return subscription
    
    async def cancel_subscription(
        self,
        db: AsyncSession,
        user: User,
    ) -> bool:
        """Cancel user's subscription"""
        
        result = await db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription or subscription.status != SubscriptionStatus.ACTIVE:
            return False
        
        # Cancel with provider
        provider = self.get_provider(subscription.provider.value)
        cancelled = await provider.cancel_subscription(subscription.provider_subscription_id)
        
        if cancelled:
            subscription.cancel_at_period_end = True
            subscription.cancelled_at = datetime.now(timezone.utc)
            await db.flush()
        
        return cancelled
    
    async def get_subscription(
        self,
        db: AsyncSession,
        user: User,
    ) -> Optional[Dict[str, Any]]:
        """Get user's current subscription"""
        
        result = await db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
        subscription = result.scalar_one_or_none()
        
        if not subscription:
            return None
        
        return {
            "id": subscription.id,
            "tier": subscription.tier,
            "status": subscription.status.value,
            "provider": subscription.provider.value if subscription.provider else None,
            "current_period_start": subscription.current_period_start.isoformat() if subscription.current_period_start else None,
            "current_period_end": subscription.current_period_end.isoformat() if subscription.current_period_end else None,
            "cancel_at_period_end": subscription.cancel_at_period_end,
        }
    
    async def get_payment_methods(
        self,
        db: AsyncSession,
        user: User,
    ) -> List[Dict[str, Any]]:
        """Get user's stored payment methods"""
        
        result = await db.execute(
            select(PaymentMethod).where(
                PaymentMethod.user_id == user.id,
                PaymentMethod.is_active == True
            )
        )
        methods = result.scalars().all()
        
        return [
            {
                "id": m.id,
                "provider": m.provider.value,
                "type": m.type,
                "last_four": m.last_four,
                "brand": m.brand,
                "exp_month": m.exp_month,
                "exp_year": m.exp_year,
                "email": m.email,
                "is_default": m.is_default,
            }
            for m in methods
        ]
    
    async def get_transactions(
        self,
        db: AsyncSession,
        user: User,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get user's transaction history"""
        
        result = await db.execute(
            select(Transaction)
            .where(Transaction.user_id == user.id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
        )
        transactions = result.scalars().all()
        
        return [
            {
                "id": t.id,
                "type": t.type,
                "amount": t.amount,
                "currency": t.currency,
                "status": t.status.value,
                "provider": t.provider.value,
                "description": t.description,
                "created_at": t.created_at.isoformat(),
            }
            for t in transactions
        ]
    
    async def handle_webhook(
        self,
        db: AsyncSession,
        provider: str,
        payload: bytes,
        signature: str,
    ) -> Dict[str, Any]:
        """Handle webhook from payment provider"""
        
        payment_provider = self.get_provider(provider)
        event = await payment_provider.handle_webhook(payload, signature)
        
        # Process based on event type
        event_type = event["type"]
        data = event["data"]
        
        if provider == "stripe":
            await self._handle_stripe_event(db, event_type, data)
        elif provider == "paypal":
            await self._handle_paypal_event(db, event_type, data)
        
        await db.commit()
        return {"status": "processed", "event_type": event_type}
    
    async def _handle_stripe_event(
        self,
        db: AsyncSession,
        event_type: str,
        data: Dict[str, Any],
    ):
        """Handle Stripe-specific webhook events"""
        
        if event_type == "checkout.session.completed":
            # Subscription created
            metadata = data.get("metadata", {})
            user_id = metadata.get("user_id")
            tier = metadata.get("tier")
            subscription_id = data.get("subscription")
            customer_id = data.get("customer")
            
            if user_id and tier and subscription_id:
                await self.handle_successful_payment(
                    db, user_id, tier, "stripe", subscription_id, customer_id
                )
        
        elif event_type == "customer.subscription.deleted":
            # Subscription cancelled/expired
            subscription_id = data.get("id")
            result = await db.execute(
                select(Subscription).where(
                    Subscription.provider_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()
            if subscription:
                subscription.status = SubscriptionStatus.CANCELLED
                
                # Downgrade user to free
                user_result = await db.execute(
                    select(User).where(User.id == subscription.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user:
                    user.tier = UserTier.FREE
                    user.tokens_limit = settings.FREE_TIER_TOKENS
        
        elif event_type == "invoice.payment_failed":
            # Payment failed
            subscription_id = data.get("subscription")
            result = await db.execute(
                select(Subscription).where(
                    Subscription.provider_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()
            if subscription:
                subscription.status = SubscriptionStatus.PAST_DUE
    
    async def _handle_paypal_event(
        self,
        db: AsyncSession,
        event_type: str,
        data: Dict[str, Any],
    ):
        """Handle PayPal-specific webhook events"""
        
        if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
            # Subscription activated
            custom_id = data.get("custom_id", "{}")
            try:
                metadata = json.loads(custom_id)
            except json.JSONDecodeError:
                metadata = {}
            
            user_id = metadata.get("user_id")
            tier = metadata.get("tier")
            subscription_id = data.get("id")
            
            if user_id and tier and subscription_id:
                await self.handle_successful_payment(
                    db, user_id, tier, "paypal", subscription_id
                )
        
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            subscription_id = data.get("id")
            result = await db.execute(
                select(Subscription).where(
                    Subscription.provider_subscription_id == subscription_id
                )
            )
            subscription = result.scalar_one_or_none()
            if subscription:
                subscription.status = SubscriptionStatus.CANCELLED
                
                # Downgrade user to free
                user_result = await db.execute(
                    select(User).where(User.id == subscription.user_id)
                )
                user = user_result.scalar_one_or_none()
                if user:
                    user.tier = UserTier.FREE
                    user.tokens_limit = settings.FREE_TIER_TOKENS
