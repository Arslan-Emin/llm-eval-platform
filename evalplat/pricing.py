"""Token prices for the judge model, used to estimate and record what judging costs.

USD per 1M tokens, from api-docs.deepseek.com/quick_start/pricing (checked
2026-11-21). DeepSeek charges half outside peak hours; the peak price is used, so
every cost here is an upper bound. Same table as the agent project's pricing.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input: float  # cache miss
    cached_input: float  # cache hit
    output: float


PRICES: dict[str, ModelPrice] = {
    "deepseek-flash": ModelPrice(input=0.30, cached_input=0.006, output=1.20),
    "deepseek-v4-pro": ModelPrice(input=1.32, cached_input=0.044, output=3.96),
}


def cost_usd(
    model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached * price.input
        + cached_input_tokens * price.cached_input
        + output_tokens * price.output
    ) / 1_000_000
