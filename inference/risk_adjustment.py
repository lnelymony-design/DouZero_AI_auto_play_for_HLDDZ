"""Transparent risk adjustment for DouZero candidate actions.

The raw DouZero score and hidden-hand response risk are different quantities.
This module never treats either as a calibrated win probability. It only builds
an optional relative ranking within the current candidate set.

Formula:
    model_component = min-max normalized raw model score in [0, 1]
    safety_component = 1 - response_risk
    adjusted = 100 * ((1 - weight) * model_component
                      + weight * safety_component)

For Pass, response risk is undefined, so safety is neutral (0.5).
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AdjustedCandidate:
    action: str
    model_score: float
    response_risk: float | None
    model_rank: int
    adjusted_score: float
    adjusted_rank: int = 0


def _to_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def adjust_candidates(candidates, weight=0.30):
    """Return candidates with a transparent risk-adjusted relative ranking."""
    weight = max(0.0, min(0.8, float(weight)))

    parsed = []
    for model_rank, item in enumerate(candidates, start=1):
        if len(item) < 3:
            continue
        action, score_value, risk_value = item[:3]
        score = _to_float(score_value)
        if score is None:
            continue
        risk = _to_float(risk_value) if risk_value is not None else None
        if risk is not None:
            risk = max(0.0, min(1.0, risk))
        parsed.append((str(action), score, risk, model_rank))

    if not parsed:
        return []

    scores = [item[1] for item in parsed]
    low = min(scores)
    high = max(scores)
    span = high - low

    provisional = []
    for action, score, risk, model_rank in parsed:
        model_component = 0.5 if span <= 1e-12 else (score - low) / span
        safety_component = 0.5 if risk is None else 1.0 - risk
        adjusted = 100.0 * (
            (1.0 - weight) * model_component
            + weight * safety_component
        )
        provisional.append(
            AdjustedCandidate(
                action=action,
                model_score=score,
                response_risk=risk,
                model_rank=model_rank,
                adjusted_score=adjusted,
            )
        )

    order = sorted(
        range(len(provisional)),
        key=lambda i: (
            provisional[i].adjusted_score,
            provisional[i].model_score,
            -provisional[i].model_rank,
        ),
        reverse=True,
    )
    rank_by_index = {index: rank + 1 for rank, index in enumerate(order)}

    return [
        AdjustedCandidate(
            action=item.action,
            model_score=item.model_score,
            response_risk=item.response_risk,
            model_rank=item.model_rank,
            adjusted_score=item.adjusted_score,
            adjusted_rank=rank_by_index[index],
        )
        for index, item in enumerate(provisional)
    ]
