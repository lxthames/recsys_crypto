from typing import List, Dict
import math
from collections import Counter


def _relevance(action: str, true_action: str) -> int:
    return int(action == true_action)


def ndcg_at_k(records: List[Dict], k: int) -> float:
    scores = []
    for rec in records:
        ranking = rec["ranking"][:k]
        true = rec["true"]
        dcg = 0.0
        for i, a in enumerate(ranking):
            rel = _relevance(a, true)
            dcg += (2**rel - 1) / math.log2(i + 2)
        idcg = (2**1 - 1) / math.log2(1 + 1)
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def map_at_k(records: List[Dict], k: int) -> float:
    aps = []
    for rec in records:
        ranking = rec["ranking"][:k]
        true = rec["true"]
        if true in ranking:
            idx = ranking.index(true)
            aps.append(1.0 / (idx + 1))
        else:
            aps.append(0.0)
    return sum(aps) / len(aps) if aps else 0.0


def diversity(records: List[Dict]) -> float:
    top1 = [r["ranking"][0] for r in records if r["ranking"]]
    counts = Counter(top1)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c/total) * math.log2(c/total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if counts else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def coverage(records: List[Dict]) -> float:
    """
    Very simple coverage: fraction of queries where we produced a non-empty ranking.
    If you add asset info into each record, you can refine this to 'fraction of assets covered'.
    """
    if not records:
        return 0.0
    non_empty = sum(1 for r in records if r.get("ranking"))
    return non_empty / len(records)


def serendipity(records: List[Dict], baseline_top1: List[str]) -> float:
    """
    Simple serendipity definition:
    - 'Serendipitous' if our top-1 action differs from baseline's top-1,
      AND it is actually correct (matches true_action).
    - Serendipity = (# serendipitous cases) / (# total cases).
    
    baseline_top1 should be a list of same length as records,
    where each entry is the baseline's top-1 action for that query.
    """
    if not records or not baseline_top1 or len(records) != len(baseline_top1):
        return 0.0

    serend = 0
    total = 0

    for rec, base_action in zip(records, baseline_top1):
        if not rec.get("ranking"):
            continue
        top_action = rec["ranking"][0]
        true = rec["true"]
        total += 1
        if top_action != base_action and top_action == true:
            serend += 1

    return serend / total if total else 0.0
