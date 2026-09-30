"""Database-backed LLM accounting: usage rows, the monthly token budget and org AI policy.

* :class:`DbUsageRecorder` writes one ``llm_usage`` row per model call in its own short
  transaction under the organisation's RLS context. Accounting must never take a request
  down, so write failures are logged and swallowed.
* :class:`DbBudgetGuard` sums the organisation's tokens for the current UTC calendar month
  and raises :class:`QuotaExceeded` (HTTP 429) once the budget is used up. The budget is
  ``llm.monthly_token_budget_per_org`` (``0`` = unlimited); an organisation may *lower* it
  with ``organizations.settings["llm"]["monthly_token_budget"]`` (a positive integer;
  absent, zero or invalid values leave the deployment budget in force). The sum is cached
  for 60 seconds per organisation (encrypted, org-scoped key), so enforcement may lag a
  burst by up to that window.
* :class:`DbOrgLlmPolicy` exposes ``organizations.settings["llm"]["external_max_classification"]``
  to the gateway, which only ever uses it to *lower* the deployment's external ceiling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from docassist.cache.redis import Cache
from docassist.core.config import Settings
from docassist.core.context import current_request_id, utcnow
from docassist.core.enums import Classification
from docassist.core.errors import QuotaExceeded
from docassist.core.logging import get_logger
from docassist.db.models import LlmUsage, Organization
from docassist.db.session import Database, DbContext

log = get_logger(__name__)

BUDGET_CACHE_SECONDS = 60
POLICY_CACHE_SECONDS = 60


def month_start(now: datetime) -> datetime:
    return datetime(now.year, now.month, 1, tzinfo=UTC)


class DbUsageRecorder:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        *,
        org_id: uuid.UUID,
        user_id: uuid.UUID | None,
        task: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        status: str,
    ) -> None:
        row = LlmUsage(
            organization_id=org_id,
            user_id=user_id,
            request_id=current_request_id(),
            task=task[:32],
            provider=provider[:32],
            model=model[:100],
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
            cost_usd=Decimal(str(round(cost_usd, 6))),
            latency_ms=max(0, int(latency_ms)),
            status=status[:24],
        )
        try:
            async with self._db.transaction(DbContext(org_id=org_id, user_id=user_id)) as session:
                session.add(row)
        except Exception:  # accounting must never fail the user's request
            log.exception("llm_usage_write_failed", task=task, provider=provider)


async def _org_llm_settings(db: Database, org_id: uuid.UUID) -> dict[str, Any]:
    async with db.session(DbContext(org_id=org_id)) as session:
        raw = (
            await session.execute(select(Organization.settings).where(Organization.id == org_id))
        ).scalar_one_or_none()
    section = raw.get("llm") if isinstance(raw, dict) else None
    return section if isinstance(section, dict) else {}


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    limit: int | None  # None = unlimited
    used: int
    remaining: int | None

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self.used >= self.limit


class DbBudgetGuard:
    def __init__(self, db: Database, settings: Settings, cache: Cache) -> None:
        self._db = db
        self._deployment_budget = settings.llm.monthly_token_budget_per_org
        self._cache = cache

    def _effective_limit(self, org_settings: dict[str, Any]) -> int | None:
        limit: int | None = self._deployment_budget or None
        override = org_settings.get("monthly_token_budget")
        if isinstance(override, int) and not isinstance(override, bool) and override > 0:
            limit = override if limit is None else min(limit, override)
        return limit

    async def status(self, org_id: uuid.UUID, *, now: datetime | None = None) -> BudgetStatus:
        now = now or utcnow()
        start = month_start(now)
        key = self._cache.key("llmbudget", str(org_id), start.strftime("%Y-%m"))
        cached = await self._cache.get_json(key)
        if isinstance(cached, dict) and isinstance(cached.get("used"), int):
            limit = cached.get("limit")
            used = int(cached["used"])
            limit_value = int(limit) if isinstance(limit, int) else None
        else:
            limit_value = self._effective_limit(await _org_llm_settings(self._db, org_id))
            async with self._db.session(DbContext(org_id=org_id)) as session:
                used = int(
                    (
                        await session.execute(
                            select(
                                func.coalesce(
                                    func.sum(LlmUsage.input_tokens + LlmUsage.output_tokens), 0
                                )
                            ).where(
                                LlmUsage.organization_id == org_id, LlmUsage.created_at >= start
                            )
                        )
                    ).scalar_one()
                )
            await self._cache.set_json(
                key, {"used": used, "limit": limit_value}, BUDGET_CACHE_SECONDS
            )
        remaining = None if limit_value is None else max(0, limit_value - used)
        return BudgetStatus(limit=limit_value, used=used, remaining=remaining)

    async def check(self, org_id: uuid.UUID) -> None:
        status = await self.status(org_id)
        if status.exceeded:
            raise QuotaExceeded(
                internal_detail=f"monthly token budget used ({status.used}/{status.limit})"
            )


def policy_cache_key(cache: Cache, org_id: uuid.UUID) -> str:
    """Cache key of an organisation's AI policy (shared with the admin service)."""
    return cache.key("llmpolicy", str(org_id), "external_max_classification")


class DbOrgLlmPolicy:
    def __init__(self, db: Database, cache: Cache) -> None:
        self._db = db
        self._cache = cache

    async def external_ceiling(self, org_id: uuid.UUID) -> Classification | None:
        key = policy_cache_key(self._cache, org_id)
        cached = await self._cache.get_json(key)
        if isinstance(cached, dict) and "value" in cached:
            value = cached["value"]
        else:
            value = (await _org_llm_settings(self._db, org_id)).get("external_max_classification")
            await self._cache.set_json(key, {"value": value}, POLICY_CACHE_SECONDS)
        try:
            return Classification(value) if isinstance(value, str) else None
        except ValueError:
            log.warning("org_llm_policy_invalid", org_id=str(org_id))
            return None
