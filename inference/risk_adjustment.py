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


def adjust_candidates(candidates, weight=0.45):
    """Return candidates with a transparent risk-adjusted relative ranking."""
    # Keep the heuristic strictly below half the combined score so safety
    # alone can never promote the worst model candidate above the model top.
    weight = max(0.0, min(0.49, float(weight)))

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


@dataclass(frozen=True)
class SaferAlternative:
    raw_action: str
    raw_model_score: float
    raw_response_risk: float
    action: str
    model_rank: int
    model_score: float
    response_risk: float
    risk_gain: float
    model_gap: float
    model_gap_fraction: float


def select_safer_alternative(
    candidates,
    min_risk_gain=0.20,
    max_model_gap_fraction=0.35,
    max_model_rank=3,
):
    """Find a safer alternative without replacing DouZero's top action.

    candidates are expected in raw model order as:
        (action, model_score, response_risk)

    A candidate qualifies only when:
    - the raw top and candidate both have a measurable response risk;
    - response risk improves by at least min_risk_gain;
    - the candidate is within max_model_rank;
    - its model-score loss is not more than the configured fraction of the
      score span among the considered candidates.

    This answers a narrower question than adjust_candidates:
    is there a near-model-equivalent move that is materially harder to beat?
    It never claims the alternative is strategically better overall.
    """
    parsed = []
    for model_rank, item in enumerate(candidates, start=1):
        if model_rank > max(1, int(max_model_rank)):
            break
        if len(item) < 3:
            continue
        action, score_value, risk_value = item[:3]
        score = _to_float(score_value)
        risk = _to_float(risk_value) if risk_value is not None else None
        if score is None:
            continue
        if risk is not None:
            risk = max(0.0, min(1.0, risk))
        parsed.append((str(action), score, risk, model_rank))

    if len(parsed) < 2:
        return None

    raw_action, raw_score, raw_risk, _ = parsed[0]
    if raw_risk is None:
        return None

    scores = [item[1] for item in parsed]
    span = max(scores) - min(scores)
    min_risk_gain = max(0.0, min(1.0, float(min_risk_gain)))
    max_model_gap_fraction = max(
        0.0, min(1.0, float(max_model_gap_fraction))
    )

    eligible = []
    for action, score, risk, model_rank in parsed[1:]:
        if risk is None:
            continue
        risk_gain = raw_risk - risk
        if risk_gain + 1e-12 < min_risk_gain:
            continue

        model_gap = raw_score - score
        if model_gap < -1e-12:
            continue
        gap_fraction = 0.0 if span <= 1e-12 else model_gap / span
        if gap_fraction > max_model_gap_fraction + 1e-12:
            continue

        eligible.append(
            SaferAlternative(
                raw_action=raw_action,
                raw_model_score=raw_score,
                raw_response_risk=raw_risk,
                action=action,
                model_rank=model_rank,
                model_score=score,
                response_risk=risk,
                risk_gain=risk_gain,
                model_gap=model_gap,
                model_gap_fraction=gap_fraction,
            )
        )

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda item: (
            item.risk_gain,
            -item.model_gap_fraction,
            item.model_score,
            -item.model_rank,
        ),
    )
