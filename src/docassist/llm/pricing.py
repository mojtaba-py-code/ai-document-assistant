"""Cost estimation from ``settings.llm.pricing`` (USD per million tokens).

Prompt-cache accounting follows the provider's published multipliers: cache writes are
billed at 1.25x the input rate and cache reads at 0.1x. Models without a price entry
(local models, the offline provider) cost 0.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal

from docassist.core.config import ModelPrice

CACHE_WRITE_MULTIPLIER = Decimal("1.25")
CACHE_READ_MULTIPLIER = Decimal("0.1")
_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.000001")


def estimate_cost(
    pricing: Mapping[str, ModelPrice],
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> Decimal:
    price = pricing.get(model)
    if price is None:
        return Decimal(0)
    input_rate = Decimal(str(price.input_per_mtok))
    output_rate = Decimal(str(price.output_per_mtok))
    billed_input = (
        Decimal(max(0, input_tokens))
        + Decimal(max(0, cache_creation_input_tokens)) * CACHE_WRITE_MULTIPLIER
        + Decimal(max(0, cache_read_input_tokens)) * CACHE_READ_MULTIPLIER
    )
    cost = (
        billed_input * input_rate / _MILLION
        + Decimal(max(0, output_tokens)) * output_rate / _MILLION
    )
    return cost.quantize(_QUANTUM, rounding=ROUND_HALF_UP)
