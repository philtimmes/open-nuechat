"""
Billing service for token tracking and subscription management
"""
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

from app.core.config import settings
from app.models.models import User, TokenUsage, UserTier, SystemSetting

logger = logging.getLogger(__name__)


class BillingService:
    """Handle billing, token tracking, and usage limits"""
    
    TIER_LIMITS = {
        UserTier.FREE: settings.FREE_TIER_TOKENS,
        UserTier.PRO: settings.PRO_TIER_TOKENS,
        UserTier.ENTERPRISE: settings.ENTERPRISE_TIER_TOKENS,
    }
    
    TIER_PRICES = {
        UserTier.FREE: 0,
        UserTier.PRO: 20.00,  # $20/month
        UserTier.ENTERPRISE: 100.00,  # $100/month
    }
    
    async def check_token_reset(self, db: AsyncSession) -> Dict[str, Any]:
        """
        Check if we need to reset token counters based on configurable interval.
        Uses the 'token_refill_interval_hours' system setting.
        Called from background task and health check.
        Returns info about what was done.
        """
        now = datetime.now(timezone.utc)
        
        # Check if debug logging is enabled
        debug_result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == "debug_token_resets")
        )
        debug_setting = debug_result.scalar_one_or_none()
        debug_enabled = debug_setting and debug_setting.value == "true"
        
        # Get the refill interval from settings (default 720 hours = 30 days)
        interval_result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == "token_refill_interval_hours")
        )
        interval_setting = interval_result.scalar_one_or_none()
        refill_hours = int(interval_setting.value) if interval_setting else 720
        
        # Get last reset timestamp from system settings
        result = await db.execute(
            select(SystemSetting).where(SystemSetting.key == "last_token_reset_timestamp")
        )
        setting = result.scalar_one_or_none()
        
        last_reset = None
        if setting and setting.value:
            try:
                last_reset = datetime.fromisoformat(setting.value.replace('Z', '+00:00'))
                # Ensure timezone aware
                if last_reset.tzinfo is None:
                    last_reset = last_reset.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                last_reset = None
        
        # Debug: Log user token counts before check
        if debug_enabled:
            users_result = await db.execute(
                select(User.id, User.email, User.tokens_used_this_month, User.tier)
            )
            users_data = users_result.all()
            logger.info("=" * 60)
            logger.info("[TOKEN_RESET_DEBUG] Token reset check triggered")
            logger.info(f"[TOKEN_RESET_DEBUG] Current time (UTC): {now.isoformat()}")
            logger.info(f"[TOKEN_RESET_DEBUG] Last reset: {last_reset.isoformat() if last_reset else 'Never'}")
            logger.info(f"[TOKEN_RESET_DEBUG] Refill interval: {refill_hours} hours")
            logger.info(f"[TOKEN_RESET_DEBUG] User token counts ({len(users_data)} users):")
            for user_id, email, tokens, tier in users_data:
                logger.info(f"[TOKEN_RESET_DEBUG]   {email}: {tokens:,} tokens (tier: {tier})")
            logger.info("=" * 60)
        
        # Check if we need to reset (first time or interval has passed)
        if last_reset:
            time_since_reset = now - last_reset
            hours_since_reset = time_since_reset.total_seconds() / 3600
            
            if hours_since_reset < refill_hours:
                if debug_enabled:
                    logger.info("[Token Reset not necessary]")
                    logger.info(f"[TOKEN_RESET_DEBUG] {round(refill_hours - hours_since_reset, 1)} hours until next reset.")
                return {
                    "action": "none",
                    "reason": "interval_not_elapsed",
                    "hours_since_reset": round(hours_since_reset, 1),
                    "refill_interval_hours": refill_hours,
                    "next_reset_in_hours": round(refill_hours - hours_since_reset, 1)
                }
        
        if debug_enabled:
            logger.info("[Token Reset Queued]")
            logger.info("[TOKEN_RESET_DEBUG] *** RESETTING ALL USER TOKEN COUNTS TO 0 ***")
        
        # Reset all users' monthly token counter
        await db.execute(
            update(User).values(tokens_used_this_month=0)
        )
        
        # Update the last reset timestamp
        now_str = now.isoformat()
        if setting:
            setting.value = now_str
        else:
            db.add(SystemSetting(key="last_token_reset_timestamp", value=now_str))
        
        await db.commit()
        
        if debug_enabled:
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"[Token Reset Completed {formatted_time}]")
        
        return {
            "action": "reset",
            "previous_reset": last_reset.isoformat() if last_reset else None,
            "new_reset": now_str,
            "refill_interval_hours": refill_hours,
            "users_reset": "all"
        }
    
    # Keep old method name as alias for backwards compatibility
    async def check_monthly_token_reset(self, db: AsyncSession) -> Dict[str, Any]:
        """Alias for check_token_reset for backwards compatibility."""
        return await self.check_token_reset(db)
    
    async def get_usage_summary(
        self,
        db: AsyncSession,
        user: User,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get token usage summary for a user"""
        now = datetime.now(timezone.utc)
        year = year or now.year
        month = month or now.month
        
        # Get aggregated usage
        result = await db.execute(
            select(
                func.sum(TokenUsage.input_tokens).label("total_input"),
                func.sum(TokenUsage.output_tokens).label("total_output"),
                func.sum(TokenUsage.total_cost).label("total_cost"),
                func.count(TokenUsage.id).label("request_count"),
            )
            .where(
                TokenUsage.user_id == user.id,
                TokenUsage.year == year,
                TokenUsage.month == month,
            )
        )
        row = result.one()
        
        total_input = row.total_input or 0
        total_output = row.total_output or 0
        total_tokens = total_input + total_output
        
        return {
            "user_id": user.id,
            "year": year,
            "month": month,
            "tier": user.tier.value,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "tokens_limit": user.tokens_limit,
            "tokens_remaining": max(0, user.tokens_limit - total_tokens),
            "usage_percentage": round((total_tokens / user.tokens_limit) * 100, 2) if user.tokens_limit > 0 else 0,
            "total_cost": round(row.total_cost or 0, 4),
            "request_count": row.request_count or 0,
        }
    
    async def get_usage_history(
        self,
        db: AsyncSession,
        user: User,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get daily usage breakdown"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        result = await db.execute(
            select(
                func.date(TokenUsage.created_at).label("date"),
                func.sum(TokenUsage.input_tokens).label("input_tokens"),
                func.sum(TokenUsage.output_tokens).label("output_tokens"),
                func.sum(TokenUsage.total_cost).label("cost"),
                func.count(TokenUsage.id).label("requests"),
            )
            .where(
                TokenUsage.user_id == user.id,
                TokenUsage.created_at >= start_date,
            )
            .group_by(func.date(TokenUsage.created_at))
            .order_by(func.date(TokenUsage.created_at))
        )
        
        return [
            {
                "date": str(row.date),
                "input_tokens": row.input_tokens or 0,
                "output_tokens": row.output_tokens or 0,
                "total_tokens": (row.input_tokens or 0) + (row.output_tokens or 0),
                "cost": round(row.cost or 0, 4),
                "requests": row.requests or 0,
            }
            for row in result.all()
        ]
    
    async def get_usage_by_chat(
        self,
        db: AsyncSession,
        user: User,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get usage breakdown by chat"""
        from app.models.models import Chat
        
        result = await db.execute(
            select(
                Chat.id,
                Chat.title,
                func.sum(TokenUsage.input_tokens).label("input_tokens"),
                func.sum(TokenUsage.output_tokens).label("output_tokens"),
                func.sum(TokenUsage.total_cost).label("cost"),
            )
            .join(TokenUsage, TokenUsage.chat_id == Chat.id)
            .where(Chat.owner_id == user.id)
            .group_by(Chat.id, Chat.title)
            .order_by(func.sum(TokenUsage.total_cost).desc())
            .limit(limit)
        )
        
        return [
            {
                "chat_id": row.id,
                "title": row.title,
                "input_tokens": row.input_tokens or 0,
                "output_tokens": row.output_tokens or 0,
                "total_tokens": (row.input_tokens or 0) + (row.output_tokens or 0),
                "cost": round(row.cost or 0, 4),
            }
            for row in result.all()
        ]
    
    async def check_usage_limit(
        self,
        db: AsyncSession,
        user: User,
        estimated_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Check if user has remaining quota
        
        Always allows:
        - Admins (unlimited tokens)
        - When FREEFORALL=True (no limits enforced) - checks database first
        """
        from app.services.settings_service import SettingsService
        
        # Use a very large number for "unlimited" (JSON doesn't support infinity)
        UNLIMITED = 999_999_999_999
        
        # Admin always has unlimited access
        if user.is_admin:
            return {
                "can_proceed": True,
                "tokens_remaining": UNLIMITED,
                "tokens_limit": UNLIMITED,
                "tier": "admin",
                "upgrade_available": False,
                "is_unlimited": True,
            }
        
        # Check FREEFORALL mode from database first, then config
        is_freeforall = await SettingsService.is_freeforall(db) if db else settings.FREEFORALL
        
        if is_freeforall:
            summary = await self.get_usage_summary(db, user)
            return {
                "can_proceed": True,
                "tokens_remaining": UNLIMITED,
                "tokens_limit": summary["tokens_limit"],
                "tier": summary["tier"],
                "upgrade_available": False,
                "is_unlimited": True,
                "freeforall": True,
            }
        
        summary = await self.get_usage_summary(db, user)
        
        can_proceed = summary["tokens_remaining"] > estimated_tokens
        
        return {
            "can_proceed": can_proceed,
            "tokens_remaining": summary["tokens_remaining"],
            "tokens_limit": summary["tokens_limit"],
            "tier": summary["tier"],
            "upgrade_available": user.tier != UserTier.ENTERPRISE,
            "is_unlimited": False,
        }
    
    async def reset_user_tokens(
        self,
        db: AsyncSession,
        user: User,
    ) -> Dict[str, Any]:
        """Reset a user's token usage for the current month (admin action)"""
        now = datetime.now(timezone.utc)
        
        # Delete all token usage records for current month
        from sqlalchemy import delete
        await db.execute(
            delete(TokenUsage).where(
                TokenUsage.user_id == user.id,
                TokenUsage.year == now.year,
                TokenUsage.month == now.month,
            )
        )
        await db.commit()
        
        # Return new usage summary
        return await self.get_usage_summary(db, user)
    
    async def upgrade_tier(
        self,
        db: AsyncSession,
        user: User,
        new_tier: UserTier,
    ) -> User:
        """Upgrade user's subscription tier"""
        user.tier = new_tier
        user.tokens_limit = self.TIER_LIMITS[new_tier]
        await db.flush()
        return user
    
    async def reset_monthly_usage(
        self,
        db: AsyncSession,
        user: User,
    ):
        """Reset user's monthly token counter (called at billing cycle)"""
        user.tokens_used_this_month = 0
        await db.flush()
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4
    
    async def get_invoice_data(
        self,
        db: AsyncSession,
        user: User,
        year: int,
        month: int,
    ) -> Dict[str, Any]:
        """Generate invoice data for a month"""
        summary = await self.get_usage_summary(db, user, year, month)
        history = await self.get_usage_by_chat(db, user, limit=50)
        
        base_price = self.TIER_PRICES[user.tier]
        overage_tokens = max(0, summary["total_tokens"] - self.TIER_LIMITS[user.tier])
        overage_cost = (overage_tokens / 1_000_000) * (settings.INPUT_TOKEN_PRICE + settings.OUTPUT_TOKEN_PRICE) / 2
        
        return {
            "user_id": user.id,
            "email": user.email,
            "period": f"{year}-{month:02d}",
            "tier": user.tier.value,
            "base_price": base_price,
            "usage_summary": summary,
            "chat_breakdown": history,
            "overage_tokens": overage_tokens,
            "overage_cost": round(overage_cost, 2),
            "total_amount": round(base_price + overage_cost, 2),
        }
    
    async def record_usage(
        self,
        db: AsyncSession,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        source: str = "api",
        chat_id: str = None,
        message_id: str = None,
    ) -> None:
        """
        Record token usage for billing.
        
        Args:
            db: Database session
            user_id: User ID
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            model: Model name used
            source: Source of the request (e.g., 'v1_api', 'chat')
            chat_id: Optional chat ID
            message_id: Optional message ID
        """
        from datetime import datetime, timezone
        from app.models.billing import TokenUsage
        
        now = datetime.now(timezone.utc)
        
        usage = TokenUsage(
            user_id=user_id,
            chat_id=chat_id,
            message_id=message_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            year=now.year,
            month=now.month,
        )
        
        db.add(usage)
        # Don't commit here - let caller control transaction
