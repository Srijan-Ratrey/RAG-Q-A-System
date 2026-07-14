#!/usr/bin/env python3
"""
Information-retrieval evaluation for the RAG Q&A System.

Reports proper rank metrics — Recall@k, MRR, and nDCG@k — for each retrieval
mode (baseline / hybrid / rrf / cross_encoder), plus warmup-corrected latency.
This replaces the self-referential confidence heuristics in evaluation.py with
metrics that measure whether the *right* documents are actually retrieved.

Relevance judgements ("qrels") are annotated at the source-file level, which is
practical for this corpus: a retrieved chunk counts as relevant if its source
file is in the query's relevant-files set. See examples/qrels.example.json.

Usage:
    python -m rag_qa.evaluation.ir                     # uses evaluation/qrels.json
    python -m rag_qa.evaluation.ir --qrels path/to.json --k 5
    rag-ir-eval --k 5                                  # console-script entry point
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

from rag_qa.evaluation.metrics import (
    dedupe_preserve_order, ndcg_at_k, recall_at_k, reciprocal_rank,
)

MODES = ["baseline", "hybrid", "rrf"]  # add "cross_encoder" with --cross-encoder


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #

class IREvaluator:
    def __init__(self, qa_service, k: int = 5, relevance_field: str = "relevant_files"):
        self.qa_service = qa_service
        self.k = k
        self.relevance_field = relevance_field

    def _retrieved_keys(self, result) -> str:
        # Relevance is judged at the source-file level.
        return result.source_file

    def _warmup(self, queries: List[str], mode: str) -> None:
        for q in queries[:2]:
            self.qa_service.search_system.search(q, k=self.k, mode=mode)

    def evaluate(self, qrels: List[Dict], modes: List[str]) -> Dict:
        report = {"k": self.k, "num_queries": len(qrels), "modes": {}}

        for mode in modes:
            self._warmup([q["query"] for q in qrels], mode)

            recalls, rrs, ndcgs, latencies = [], [], [], []
            for item in qrels:
                relevant = set(item.get(self.relevance_field, []))

                start = time.perf_counter()
                results = self.qa_service.search_system.search(
                    item["query"], k=self.k, mode=mode)
                latencies.append(time.perf_counter() - start)

                # Relevance is judged per source file, but several retrieved
                # chunks can share a file. Collapse to a ranked list of unique
                # files (first occurrence wins) so metrics stay well-defined
                # (otherwise gains can exceed |relevant| and nDCG > 1).
                retrieved = dedupe_preserve_order(
                    self._retrieved_keys(r) for r in results)
                recalls.append(recall_at_k(retrieved, relevant, self.k))
                rrs.append(reciprocal_rank(retrieved, relevant))
                ndcgs.append(ndcg_at_k(retrieved, relevant, self.k))

            n = len(qrels) or 1
            report["modes"][mode] = {
                f"recall@{self.k}": round(sum(recalls) / n, 4),
                "mrr": round(sum(rrs) / n, 4),
                f"ndcg@{self.k}": round(sum(ndcgs) / n, 4),
                "latency_ms": {
                    "mean": round(1000 * sum(latencies) / n, 2),
                    "p50": round(1000 * sorted(latencies)[n // 2], 2),
                    "max": round(1000 * max(latencies), 2),
                },
            }
        return report

    @staticmethod
    def print_report(report: Dict) -> None:
        print(f"\n📊 IR Evaluation  (k={report['k']}, {report['num_queries']} queries)")
        print("=" * 68)
        header = f"{'mode':<15}{'recall':>10}{'mrr':>10}{'ndcg':>10}{'lat(ms)':>12}"
        print(header)
        print("-" * 68)
        k = report["k"]
        for mode, m in report["modes"].items():
            print(f"{mode:<15}{m[f'recall@{k}']:>10.3f}{m['mrr']:>10.3f}"
                  f"{m[f'ndcg@{k}']:>10.3f}{m['latency_ms']['mean']:>12.1f}")
        print("=" * 68)
        print("Note: latency excludes 2 warmup queries per mode.")


def main() -> int:
    parser = argparse.ArgumentParser(description="IR evaluation for RAG Q&A.")
    parser.add_argument("--qrels", default="qrels.json")
    parser.add_argument("--db", default="data/rag_database.db")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cross-encoder", action="store_true",
                        help="Also evaluate the cross_encoder mode (slower)")
    parser.add_argument("--out", default="ir_evaluation_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.qrels):
        print(f"❌ qrels file not found: {args.qrels}\n"
              f"   Copy examples/qrels.example.json to {args.qrels} and fill "
              f"in relevant_files for each query.", file=sys.stderr)
        return 1

    with open(args.qrels, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    from rag_qa.api import QAService
    qa_service = QAService(db_path=args.db)

    modes = list(MODES)
    if args.cross_encoder:
        modes.append("cross_encoder")

    evaluator = IREvaluator(qa_service, k=args.k)
    report = evaluator.evaluate(qrels, modes)
    evaluator.print_report(report)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Saved report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
