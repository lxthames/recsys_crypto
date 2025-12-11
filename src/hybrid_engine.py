from typing import Dict
from .user_model import UserProfile


def compute_final_scores(
    ml_conf: Dict[str, float],
    llm_score: Dict[str, float],
    pattern_support: Dict[str, float],
    alpha: float,
    beta: float,
    gamma: float,
    user: UserProfile,
) -> Dict[str, float]:
    actions = ["Buy", "Hold", "Sell"]
    scores: Dict[str, float] = {}

    for act in actions:
        base = (
            alpha * ml_conf.get(act, 0.0) +
            beta * llm_score.get(act, 0.0) +
            gamma * pattern_support.get(act, 0.0)
        )

        # user risk adjustment
        if user.risk_level == "low":
            if act == "Buy":
                base *= 0.7
            elif act == "Sell":
                base *= 1.1
        elif user.risk_level == "high":
            if act == "Buy":
                base *= 1.2
            elif act == "Sell":
                base *= 1.1
            elif act == "Hold":
                base *= 0.8

        scores[act] = base

    total = sum(scores.values()) or 1.0
    for k in scores:
        scores[k] /= total
    return scores


def rank_actions(scores: Dict[str, float]):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
