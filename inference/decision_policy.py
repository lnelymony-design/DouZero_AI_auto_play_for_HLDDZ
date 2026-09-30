"""Conservative policy for combining DouZero with hidden-hand belief.

The policy deliberately does not replace DouZero with a hand-written score.
It may change the top recommendation only in two situations:

1. game_new has already selected a deterministic direct-finish/path action;
2. a near-model-equivalent candidate is materially safer according to the
   hidden-hand posterior and the posterior has enough effective samples.

Pass is never overridden in this first version, and a non-bomb model choice is
never promoted to a bomb purely because the bomb is harder to beat.
"""

from dataclasses import dataclass

from .risk_adjustment import select_safer_alternative


@dataclass(frozen=True)
class RecommendationDecision:
    action: str
    source: str
    changed_from_model_top: bool
    model_rank: int
    reason: str
    risk_gain: float | None = None
    model_gap_fraction: float | None = None


def choose_recommendation(
    candidates,
    *,
    ess_ratio,
    enabled,
    min_ess_ratio,
    min_risk_gain,
    max_model_gap_fraction,
    max_model_rank,
    locked_action=None,
    locked_model_rank=None,
    is_bomb_action=None,
):
    """Choose one top action without turning belief risk into a fake win rate.

    candidates must be in raw DouZero model order and contain:
        (action, model_score, posterior_pressure_or_none)
    """
    if not candidates:
        return None

    raw_action = str(candidates[0][0])

    if locked_action:
        locked_action = str(locked_action)
        return RecommendationDecision(
            action=locked_action,
            source="env_override",
            changed_from_model_top=locked_action != raw_action,
            model_rank=int(locked_model_rank or 1),
            reason="game_new直接出完/路径选择",
        )

    if not enabled:
        return RecommendationDecision(
            action=raw_action,
            source="douzero",
            changed_from_model_top=False,
            model_rank=1,
            reason="概率重排未启用",
        )

    if float(ess_ratio or 0.0) < float(min_ess_ratio):
        return RecommendationDecision(
            action=raw_action,
            source="douzero",
            changed_from_model_top=False,
            model_rank=1,
            reason="推断有效样本不足",
        )

    if raw_action == "Pass":
        return RecommendationDecision(
            action=raw_action,
            source="douzero",
            changed_from_model_top=False,
            model_rank=1,
            reason="首版不覆盖DouZero的不出判断",
        )

    alternative = select_safer_alternative(
        candidates,
        min_risk_gain=min_risk_gain,
        max_model_gap_fraction=max_model_gap_fraction,
        max_model_rank=max_model_rank,
    )
    if alternative is None:
        return RecommendationDecision(
            action=raw_action,
            source="douzero",
            changed_from_model_top=False,
            model_rank=1,
            reason="没有同时满足模型接近与风险明显下降的候选",
        )

    if is_bomb_action is not None:
        raw_is_bomb = bool(is_bomb_action(raw_action))
        alt_is_bomb = bool(is_bomb_action(alternative.action))
        if alt_is_bomb and not raw_is_bomb:
            return RecommendationDecision(
                action=raw_action,
                source="douzero",
                changed_from_model_top=False,
                model_rank=1,
                reason="禁止仅因响应风险较低而主动升级为炸弹",
            )

    return RecommendationDecision(
        action=alternative.action,
        source="belief_safer",
        changed_from_model_top=True,
        model_rank=alternative.model_rank,
        reason="后验压力明显下降且模型分差受控",
        risk_gain=alternative.risk_gain,
        model_gap_fraction=alternative.model_gap_fraction,
    )
