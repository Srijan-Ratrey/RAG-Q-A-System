"""Pure information-retrieval metric functions (unit-tested)."""

import math
from typing import List, Sequence


def dedupe_preserve_order(items) -> List[str]:
    """Return items with duplicates removed, keeping first-occurrence order."""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """Fraction of relevant items found within the top-k retrieved items."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for r in set(top) if r in relevant)
    return found / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set) -> float:
    """1 / rank of the first relevant item (0 if none retrieved)."""
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: set, k: int) -> float:
    """Normalized DCG@k with binary relevance gains."""
    if not relevant:
        return 0.0
    gains = [1.0 if item in relevant else 0.0 for item in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denom = dcg(ideal)
    return dcg(gains) / denom if denom else 0.0
