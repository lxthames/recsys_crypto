# src/backtest.py

# src/backtest.py

from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .ml_models import (
    load_labeled_data,
    get_feature_columns,
    train_classifier,
)
from .user_model import UserProfile
from .hybrid_engine import compute_final_scores, rank_actions
from .recsys_metrics import (
    ndcg_at_k,
    map_at_k,
    diversity,
    coverage,
    serendipity,
)
from . import ollama_client as oc


# -------------------------
# Train / test time split
# -------------------------


def time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split to avoid information leakage.
    """
    df_sorted = df.sort_values(["timestamp", "asset"])
    n = len(df_sorted)
    split_idx = int(n * train_ratio)
    df_train = df_sorted.iloc[:split_idx].copy()
    df_test = df_sorted.iloc[split_idx:].copy()
    return df_train, df_test


# -------------
# PnL & equity
# -------------


def compute_pnl(top_action: str, future_return: float) -> float:
    """
    Simple PnL without transaction cost:
    - Buy  -> +future_return
    - Sell -> -future_return (assume perfect shorting)
    - Hold -> 0
    """
    if top_action == "Buy":
        return float(future_return)
    if top_action == "Sell":
        return float(-future_return)
    return 0.0


def compute_pnl_with_costs(
    top_action: str,
    future_return: float,
    fee_rate: float = 0.001,
) -> float:
    """
    PnL with simple transaction costs:
    - Buy:  profit ≈ future_return - 2 * fee_rate (entry + exit)
    - Sell: profit ≈ -future_return - 2 * fee_rate
    - Hold: 0
    """
    if top_action == "Buy":
        gross = float(future_return)
    elif top_action == "Sell":
        gross = float(-future_return)
    else:
        return 0.0

    cost = 2.0 * fee_rate
    return gross - cost


def equity_curve(
    pnls: List[float],
    initial_capital: float = 1.0,
) -> List[float]:
    """
    Convert per-trade returns into an equity curve,
    assuming full capital allocated on each trade.
    """
    equity = float(initial_capital)
    curve = [equity]
    for r in pnls:
        equity *= (1.0 + r)
        curve.append(equity)
    return curve


def equity_stats(curve: List[float]) -> Dict[str, float]:
    """
    Compute simple equity stats:
    - final: final capital
    - max_drawdown: minimum (equity / running_max - 1)
    """
    if not curve:
        return {"final": 0.0, "max_drawdown": 0.0}

    arr = np.array(curve, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdown = (arr - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "final": float(arr[-1]),
        "max_drawdown": max_dd,
    }


# -------------------
# LLM parallel helper
# -------------------


def _llm_call_single(model_name: str, row_dict: Dict) -> Dict:
    """
    Simple wrapper to call Ollama for one row_dict.
    """
    return oc.ask_ollama(model_name, row_dict)


def call_llm_parallel(
    model_name: str,
    rows: List[Dict],
    max_workers: int = 4,
) -> Tuple[List[Optional[Dict]], int]:
    """
    Call LLM in parallel for a list of row dicts.

    Returns:
        outputs: list of LLM outputs (or None on error) aligned with rows
        error_count: how many calls failed
    """
    outputs: List[Optional[Dict]] = [None] * len(rows)
    error_count = 0

    if max_workers <= 1 or len(rows) <= 1:
        # Fallback to sequential for small cases / debugging
        for i, rd in enumerate(rows):
            try:
                outputs[i] = _llm_call_single(model_name, rd)
            except Exception:
                error_count += 1
        return outputs, error_count

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_llm_call_single, model_name, rd): i
            for i, rd in enumerate(rows)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                outputs[i] = fut.result()
            except Exception:
                error_count += 1

    return outputs, error_count


# ---------------------------
# Simple pattern heuristics
# ---------------------------


def heuristic_pattern_support(row: pd.Series) -> Dict[str, float]:
    """
    Simple rule-based pattern support based on RSI and MACD.

    This is a placeholder for real SPMF-based pattern mining,
    but already injects pattern information into the hybrid engine.
    """
    rsi = float(row.get("rsi", 50.0))
    macd = float(row.get("macd", 0.0))

    # Start from neutral
    support = {"Buy": 1.0, "Hold": 1.0, "Sell": 1.0}

    # RSI-based pattern
    if rsi < 30:
        # Oversold -> strongly support Buy
        support["Buy"] += 3.0
        support["Hold"] += 1.0
    elif rsi > 70:
        # Overbought -> strongly support Sell
        support["Sell"] += 3.0
        support["Hold"] += 1.0
    else:
        # Neutral RSI -> light bias to Hold
        support["Hold"] += 0.5

    # MACD-based pattern
    if macd > 0:
        support["Buy"] += 1.5
    elif macd < 0:
        support["Sell"] += 1.5

    # Normalize to [0,1]
    total = sum(support.values()) or 1.0
    for k in support:
        support[k] /= total

    return support


# ----------------
# ML-only backtest
# ----------------


def backtest_ml_only(
    model,
    feature_cols: List[str],
    df_test: pd.DataFrame,
) -> Dict:
    """
    Baseline: ML-only ranking from RandomForest predicted probabilities.
    """
    records: List[Dict] = []
    pnls: List[float] = []
    top1_actions: List[str] = []

    df_sorted = df_test.sort_values(["timestamp", "asset"])
    X = df_sorted[feature_cols]
    y_true = df_sorted["true_action"].tolist()
    fut_ret = df_sorted["future_return"].tolist()

    probas = model.predict_proba(X)
    classes = list(model.classes_)

    for i in range(len(df_sorted)):
        proba = probas[i]
        ml_conf = {cls: float(p) for cls, p in zip(classes, proba)}
        for a in ["Buy", "Hold", "Sell"]:
            ml_conf.setdefault(a, 0.0)

        # For ML-only, we can just treat ml_conf as final scores
        scores = ml_conf
        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        true_action = y_true[i]

        records.append({"ranking": ranking, "true": true_action})
        top1_actions.append(ranking[0])

        # Use PnL with costs for more realism
        pnl = compute_pnl_with_costs(ranking[0], fut_ret[i], fee_rate=0.001)
        pnls.append(pnl)

    k = 3
    ndcg_val = ndcg_at_k(records, k)
    map_val = map_at_k(records, k)
    div = diversity(records)
    cov = coverage(records)
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0

    curve = equity_curve(pnls, initial_capital=1.0)
    eq = equity_stats(curve)

    metrics = {
        "mode": "ml_only",
        "ndcg": ndcg_val,
        "map": map_val,
        "diversity": div,
        "coverage": cov,
        "serendipity": None,
        "avg_pnl": avg_pnl,
        "equity_final": eq["final"],
        "equity_max_drawdown": eq["max_drawdown"],
        "n": len(records),
        "baseline_top1": top1_actions,
        "records": records,
    }
    return metrics


# -----------------
# LLM-only backtest
# -----------------


def backtest_llm_only(
    model_name: str,
    df_test: pd.DataFrame,
    user: UserProfile,
    max_workers: int = 4,
) -> Dict:
    """
    LLM-only:
    - Ignore ML scores, only take LLM 'action_scores'.
    - If Ollama fails for a row, fall back to neutral scores.
    - Uses parallel calls to the LLM for efficiency.
    """
    records: List[Dict] = []
    pnls: List[float] = []
    top1_actions: List[str] = []

    # Materialize rows once
    rows = list(df_test.iterrows())
    row_dicts = [row.to_dict() for _, row in rows]

    # Parallel LLM calls
    llm_outputs, error_count = call_llm_parallel(
        model_name=model_name,
        rows=row_dicts,
        max_workers=max_workers,
    )

    for idx, (pandas_idx, row) in enumerate(rows):
        out = llm_outputs[idx]

        if out is None:
            # LLM failed for this row
            scores = {"Buy": 1.0 / 3.0, "Hold": 1.0 / 3.0, "Sell": 1.0 / 3.0}
        else:
            scores = out.get("action_scores", {})
        for a in ["Buy", "Hold", "Sell"]:
            scores.setdefault(a, 0.0)

        final_scores = compute_final_scores(
            ml_conf={"Buy": 0.0, "Hold": 0.0, "Sell": 0.0},
            llm_score=scores,
            pattern_support={"Buy": 0.0, "Hold": 0.0, "Sell": 0.0},
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
            user=user,
        )
        ranking = [a for a, _ in rank_actions(final_scores)]
        true_action = row["true_action"]

        records.append({"ranking": ranking, "true": true_action})
        top1_actions.append(ranking[0])

        pnl = compute_pnl_with_costs(ranking[0], row["future_return"], fee_rate=0.001)
        pnls.append(pnl)

    k = 3
    ndcg_val = ndcg_at_k(records, k)
    map_val = map_at_k(records, k)
    div = diversity(records)
    cov = coverage(records)
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0

    curve = equity_curve(pnls, initial_capital=1.0)
    eq = equity_stats(curve)

    metrics = {
        "mode": f"llm_only_{model_name}",
        "ndcg": ndcg_val,
        "map": map_val,
        "diversity": div,
        "coverage": cov,
        "serendipity": None,
        "avg_pnl": avg_pnl,
        "equity_final": eq["final"],
        "equity_max_drawdown": eq["max_drawdown"],
        "n": len(records),
        "baseline_top1": top1_actions,
        "records": records,
        "llm_errors": error_count,
    }
    return metrics


# -------------------
# Hybrid backtest
# -------------------


def backtest_hybrid(
    model,
    feature_cols: List[str],
    df_test: pd.DataFrame,
    model_name: str,
    user: UserProfile,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    ml_baseline_top1: Optional[List[str]] = None,
    max_workers: int = 4,
) -> Dict:
    """
    Hybrid:
    - Combine ML probabilities, LLM scores and pattern support.
    - If LLM fails, fall back to ML-only for that row.
    - Uses parallel LLM calls for efficiency.
    """
    records: List[Dict] = []
    pnls: List[float] = []
    hybrid_top1: List[str] = []

    # Materialize rows once
    rows = list(df_test.iterrows())
    row_dicts = [row.to_dict() for _, row in rows]

    # Parallel LLM calls
    llm_outputs, error_count = call_llm_parallel(
        model_name=model_name,
        rows=row_dicts,
        max_workers=max_workers,
    )

    for idx, (pandas_idx, row) in enumerate(rows):
        # ML confidences
        X_row = row[feature_cols].to_frame().T
        proba = model.predict_proba(X_row)[0]
        classes = model.classes_
        ml_conf = {cls: float(p) for cls, p in zip(classes, proba)}
        for a in ["Buy", "Hold", "Sell"]:
            ml_conf.setdefault(a, 0.0)

        # LLM scores
        out = llm_outputs[idx]
        if out is None:
            llm_scores = {"Buy": 0.0, "Hold": 0.0, "Sell": 0.0}
        else:
            llm_scores = out.get("action_scores", {})
        for a in ["Buy", "Hold", "Sell"]:
            llm_scores.setdefault(a, 0.0)

        # Pattern support (heuristic for now)
        pattern_support = heuristic_pattern_support(row)

        final_scores = compute_final_scores(
            ml_conf=ml_conf,
            llm_score=llm_scores,
            pattern_support=pattern_support,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            user=user,
        )
        ranking = [a for a, _ in rank_actions(final_scores)]
        true_action = row["true_action"]

        records.append({"ranking": ranking, "true": true_action})
        hybrid_top1.append(ranking[0])

        pnl = compute_pnl_with_costs(ranking[0], row["future_return"], fee_rate=0.001)
        pnls.append(pnl)

    k = 3
    ndcg_val = ndcg_at_k(records, k)
    map_val = map_at_k(records, k)
    div = diversity(records)
    cov = coverage(records)
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0

    curve = equity_curve(pnls, initial_capital=1.0)
    eq = equity_stats(curve)

    ser = None
    if ml_baseline_top1 is not None:
        ser = serendipity(records, ml_baseline_top1)

    metrics = {
        "mode": f"hybrid_{model_name}",
        "ndcg": ndcg_val,
        "map": map_val,
        "diversity": div,
        "coverage": cov,
        "serendipity": ser,
        "avg_pnl": avg_pnl,
        "equity_final": eq["final"],
        "equity_max_drawdown": eq["max_drawdown"],
        "n": len(records),
        "hybrid_top1": hybrid_top1,
        "records": records,
        "llm_errors": error_count,
    }
    return metrics


# --------------
# Top-level run
# --------------


def run_all(
    model_name: str = "gemma3:4b",
    user_type: str = "moderate",
    max_eval_rows: int = 300,
    llm_max_workers: int = 4,
) -> Dict[str, Dict]:
    """
    End-to-end:
    - Load labeled data
    - Time-based split
    - Train RF classifier
    - Sample subset of test period
    - Run ML-only, LLM-only, Hybrid backtests

    Returns a dict with three metrics dicts.
    """
    df = load_labeled_data()
    df_train, df_test = time_split(df, train_ratio=0.7)

    # train_classifier returns (clf, feature_cols)
    clf, feature_cols = train_classifier(df_train)

    # Choose user profile
    user = UserProfile.from_type(user_type)

    # Sample subset of test period for evaluation
    df_eval = df_test.sample(
        n=min(max_eval_rows, len(df_test)), random_state=42
    ).sort_values(["timestamp", "asset"])

    ml_m = backtest_ml_only(clf, feature_cols, df_eval)
    llm_m = backtest_llm_only(
        model_name,
        df_eval,
        user=user,
        max_workers=llm_max_workers,
    )
    hyb_m = backtest_hybrid(
        clf,
        feature_cols,
        df_eval,
        model_name,
        user=user,
        alpha=0.4,
        beta=0.4,
        gamma=0.2,
        ml_baseline_top1=ml_m["baseline_top1"],
        max_workers=llm_max_workers,
    )

    # Simple printout (optional)
    print("=== ML-only ===")
    print(
        f"NDCG={ml_m['ndcg']:.4f}, MAP={ml_m['map']:.4f}, "
        f"Div={ml_m['diversity']:.4f}, Cov={ml_m['coverage']:.4f}, "
        f"AvgPnL={ml_m['avg_pnl']:.4f}, FinalEq={ml_m['equity_final']:.4f}, "
        f"MaxDD={ml_m['equity_max_drawdown']:.4f}"
    )
    print("=== LLM-only ===")
    print(
        f"NDCG={llm_m['ndcg']:.4f}, MAP={llm_m['map']:.4f}, "
        f"Div={llm_m['diversity']:.4f}, Cov={llm_m['coverage']:.4f}, "
        f"AvgPnL={llm_m['avg_pnl']:.4f}, FinalEq={llm_m['equity_final']:.4f}, "
        f"MaxDD={llm_m['equity_max_drawdown']:.4f}, LLM_errors={llm_m['llm_errors']}"
    )
    print("=== Hybrid ===")
    print(
        f"NDCG={hyb_m['ndcg']:.4f}, MAP={hyb_m['map']:.4f}, "
        f"Div={hyb_m['diversity']:.4f}, Cov={hyb_m['coverage']:.4f}, "
        f"Ser={hyb_m['serendipity']}, AvgPnL={hyb_m['avg_pnl']:.4f}, "
        f"FinalEq={hyb_m['equity_final']:.4f}, MaxDD={hyb_m['equity_max_drawdown']:.4f}, "
        f"LLM_errors={hyb_m['llm_errors']}"
    )

    return {"ml": ml_m, "llm": llm_m, "hybrid": hyb_m}
