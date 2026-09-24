"""Runs a pipeline config against an eval split and writes a markdown report.

Usage:
    python -m eval.run_eval --config configs/experiments/b0_naive.yaml --split aapl_dev
    python -m eval.run_eval --config configs/experiments/section_hybrid.yaml --split aapl_dev
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.judges.correctness import judge_answer
from eval.metrics import doc_recall_at_k, faithfulness_score, latency_stats, mrr
from sec_rag.config import load_experiment_config
from sec_rag.openai_utils import RequestTooLargeError
from sec_rag.pipeline import build_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "raw" / "aapl"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"


def run(config_path: str, split: str, limit: int | None, unlock_test: bool, skip_judge: bool) -> Path:
    if "test" in split.lower() and not unlock_test:
        raise SystemExit(f"Refusing to run on a split named {split!r} without --unlock-test.")

    config = load_experiment_config(config_path)
    pipeline = build_pipeline(config, DOCS_DIR)

    df = pd.read_csv(REPO_ROOT / "data" / "eval" / f"{split}.csv")
    if limit:
        df = df.head(limit)

    rows = []
    skipped = []
    for _, row in df.iterrows():
        gold_doc_ids = row["doc_ids"].split(";")
        try:
            answer, retrieved = pipeline.answer(row["Question"], gold_doc_ids)
        except RequestTooLargeError as e:
            skipped.append({"question": row["Question"], "doc_ids": gold_doc_ids, "reason": str(e)[:200]})
            print(f"SKIPPED (request too large): {row['Question'][:70]!r}")
            continue

        context_text = " ".join(getattr(c, "text", "") for c in retrieved) if retrieved else ""
        retrieved_doc_ids = [c.doc_id for c in retrieved] if retrieved else gold_doc_ids

        judge_score = None
        judge_reason = ""
        if not skip_judge:
            result = judge_answer(row["Question"], row["Answer"], answer.text)
            judge_score, judge_reason = result.score, result.reason

        rows.append(
            {
                "question": row["Question"],
                "question_type": row["Question Type"],
                "chunk_type": row["Source Chunk Type"],
                "answer_text": answer.text,
                "judge_score": judge_score,
                "judge_reason": judge_reason,
                "faithfulness": faithfulness_score(answer.text, context_text) if context_text else None,
                "doc_recall": doc_recall_at_k(retrieved_doc_ids, gold_doc_ids) if retrieved else None,
                "mrr": mrr(retrieved_doc_ids, gold_doc_ids) if retrieved else None,
                "latency_ms": answer.latency_ms,
                "cost_usd": answer.cost_usd,
                "model": answer.model,
            }
        )

    return _write_report(config, split, rows, skip_judge, skipped)


def _slice_table(rows: list[dict], key: str) -> str:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["judge_score"] is not None:
            groups[r[key]].append(r["judge_score"])
    lines = [f"| {key} | n | mean score |", "|---|---|---|"]
    for group, scores in sorted(groups.items()):
        lines.append(f"| {group} | {len(scores)} | {statistics.mean(scores):.2f} |")
    return "\n".join(lines)


def _write_report(config: dict, split: str, rows: list[dict], skip_judge: bool, skipped: list[dict] | None = None) -> Path:
    skipped = skipped or []
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = config.get("name", "unnamed")
    report_path = REPORTS_DIR / f"{name}_{split}_{timestamp}.md"

    scores = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    faithfulness = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    recalls = [r["doc_recall"] for r in rows if r["doc_recall"] is not None]
    mrrs = [r["mrr"] for r in rows if r["mrr"] is not None]
    latencies = [r["latency_ms"] for r in rows]
    total_cost = sum(r["cost_usd"] for r in rows)
    lat_stats = latency_stats(latencies)

    lines = [
        f"# Eval report: {name} ({split} split, {len(rows)} answered / {len(rows) + len(skipped)} attempted)",
        "",
        f"Config: `{config.get('_config_path', '?')}`",
        f"Generated: {timestamp}",
        "",
    ]

    if skipped:
        lines.extend([f"## Skipped ({len(skipped)})", ""])
        for s in skipped:
            lines.append(f"- {s['question'][:100]!r} (docs: {', '.join(s['doc_ids'])})")
        lines.append("")

    lines.extend(
        [
            "## Overall",
            "",
            f"- Mean correctness (judge): {statistics.mean(scores):.3f}" if scores else "- Correctness: skipped (--skip-judge)",
            f"- Mean faithfulness (numeric): {statistics.mean(faithfulness):.3f}" if faithfulness else "- Faithfulness: n/a",
            f"- Mean doc recall: {statistics.mean(recalls):.3f}" if recalls else "- Doc recall: n/a",
            f"- Mean MRR: {statistics.mean(mrrs):.3f}" if mrrs else "- MRR: n/a",
            f"- Latency p50/p95/mean (ms): {lat_stats.p50_ms:.0f} / {lat_stats.p95_ms:.0f} / {lat_stats.mean_ms:.0f}",
            f"- Total cost: ${total_cost:.4f} (${total_cost / len(rows):.4f} / query)" if rows else "- Total cost: $0",
            "",
            "## By question type" if scores else "",
            _slice_table(rows, "question_type") if scores else "",
            "",
            "## By source chunk type" if scores else "",
            _slice_table(rows, "chunk_type") if scores else "",
            "",
            "## Failure examples (score < 1)" if scores else "",
        ]
    )

    if scores:
        failures = [r for r in rows if r["judge_score"] is not None and r["judge_score"] < 1.0][:5]
        for r in failures:
            lines.extend(
                [
                    f"### {r['question']}",
                    f"- Score: {r['judge_score']} -- {r['judge_reason']}",
                    f"- Answer: {r['answer_text'][:500]}",
                    "",
                ]
            )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="aapl_dev", help="basename of data/eval/<split>.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    report_path = run(args.config, args.split, args.limit, args.unlock_test, args.skip_judge)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
