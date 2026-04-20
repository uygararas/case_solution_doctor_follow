"""
Part 2D - Retrieval evaluation utilities.

Provides a reproducible evaluation path for the 5 required assessment queries
using cached relevance labels, with an optional live Google-backed fallback
when a cached judgement is missing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .retrieval import build_retrievers, compute_metrics, load_articles

DATA_DIR = Path(__file__).parent.parent / "data"
JUDGE_CACHE_PATH = DATA_DIR / "judge_cache.json"
DEFAULT_RESULTS_PATH = DATA_DIR / "evaluation_results.json"

EVAL_QUERIES = [
    "What are the latest guidelines for managing type 2 diabetes?",
    "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    "Iron supplementation dosing for anemia during pregnancy",
    "Çölyak hastalığı tanı kriterleri nelerdir?",
    "Antibiotic resistance patterns in community acquired pneumonia",
]

JUDGE_PROMPT = """You are an expert medical librarian evaluating PubMed abstract relevance.

Query: {query}

Abstract:
Title: {title}
{abstract}

Rate the relevance of this abstract to the query on a 0-2 scale:
0 = Not relevant
1 = Partially relevant
2 = Highly relevant

Respond with ONLY the number 0, 1, or 2."""


def _load_judge_cache(path: Path = JUDGE_CACHE_PATH) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_judge_cache(cache: dict[str, int], path: Path = JUDGE_CACHE_PATH) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _pick_default_google_model(client) -> str:
    for model in client.models.list():
        name = getattr(model, "name", "")
        methods = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", []) or []
        if "flash" in name.lower() and any("generate" in str(method).lower() for method in methods):
            return name.split("/")[-1]
    raise RuntimeError("No compatible Google scoring model was found for this account.")


def _score_with_google_service(query: str, title: str, abstract: str, model: Optional[str] = None) -> int:
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    client = genai.Client(api_key=api_key)
    resolved_model = model or os.getenv("GOOGLE_MODEL") or _pick_default_google_model(client)
    prompt = JUDGE_PROMPT.format(query=query, title=title, abstract=abstract[:1200])
    response = client.models.generate_content(
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0),
    )

    text = (response.text or "").strip()
    if not text or text[0] not in "012":
        raise ValueError(f"Unexpected judge response: {text!r}")
    return int(text[0])


def get_relevance_score(
    query: str,
    doc: dict,
    judge_cache: dict[str, int],
    allow_live_scoring: bool = False,
) -> int:
    cache_key = f"{query}|||{doc['pmid']}"
    if cache_key in judge_cache:
        return int(judge_cache[cache_key])

    if not allow_live_scoring:
        raise KeyError(
            f"Missing cached judgement for query={query!r}, pmid={doc['pmid']}. "
            "Re-run with allow_live_scoring=True and GOOGLE_API_KEY set."
        )

    score = _score_with_google_service(query, doc.get("title", ""), doc.get("abstract", ""))
    judge_cache[cache_key] = score
    _save_judge_cache(judge_cache)
    time.sleep(0.5)
    return score


def evaluate_retrievers(
    articles: Optional[list[dict]] = None,
    allow_live_scoring: bool = False,
    use_cache: bool = True,
) -> dict:
    """Evaluate BM25, Semantic, and Hybrid RRF on the required 5 queries."""
    articles = articles or load_articles()
    retrievers = build_retrievers(articles, use_cache=use_cache)
    judge_cache = _load_judge_cache()

    methods = {
        "BM25": retrievers["bm25"],
        "Semantic": retrievers["semantic"],
        "Hybrid RRF": retrievers["hybrid"],
    }

    all_results: dict[str, dict] = {}

    for method_name, retriever in methods.items():
        per_query = []
        for query in EVAL_QUERIES:
            docs = retriever.search(query, top_k=5)
            relevance_scores = [
                get_relevance_score(
                    query=query,
                    doc=doc,
                    judge_cache=judge_cache,
                    allow_live_scoring=allow_live_scoring,
                )
                for doc in docs
            ]
            metrics = compute_metrics(relevance_scores)
            per_query.append(
                {
                    "query": query,
                    "pmids": [doc["pmid"] for doc in docs],
                    "relevance_scores": relevance_scores,
                    "metrics": metrics,
                }
            )

        average = {
            "P@5": round(float(np.mean([row["metrics"]["P@5"] for row in per_query])), 4),
            "MRR": round(float(np.mean([row["metrics"]["MRR"] for row in per_query])), 4),
            "nDCG@5": round(float(np.mean([row["metrics"]["nDCG@5"] for row in per_query])), 4),
        }
        all_results[method_name] = {"per_query": per_query, "average": average}

    winner_by_metric = {}
    for metric in ("P@5", "MRR", "nDCG@5"):
        best_value = max(payload["average"][metric] for payload in all_results.values())
        winner_by_metric[metric] = [
            method_name
            for method_name, payload in all_results.items()
            if payload["average"][metric] == best_value
        ]

    return {
        "queries": EVAL_QUERIES,
        "methods": all_results,
        "winner_by_metric": winner_by_metric,
    }


def format_evaluation_report(results: dict) -> str:
    lines = []
    lines.append("=" * 65)
    lines.append("EVALUATION RESULTS - All Methods Comparison")
    lines.append("=" * 65)
    lines.append(f"{'Method':<15} {'P@5':>8} {'MRR':>8} {'nDCG@5':>8}")
    lines.append("-" * 45)
    for method_name, payload in results["methods"].items():
        avg = payload["average"]
        lines.append(f"{method_name:<15} {avg['P@5']:>8.4f} {avg['MRR']:>8.4f} {avg['nDCG@5']:>8.4f}")
    lines.append("=" * 65)
    lines.append("")
    lines.append("Winners by metric:")
    for metric, winners in results["winner_by_metric"].items():
        lines.append(f"  {metric}: {', '.join(winners)}")
    lines.append("")
    lines.append("Per-query breakdown:")
    for method_name, payload in results["methods"].items():
        lines.append(f"  [{method_name}]")
        for row in payload["per_query"]:
            metrics = row["metrics"]
            lines.append(
                f"    {row['query'][:50]:50s} "
                f"P@5={metrics['P@5']:.2f} MRR={metrics['MRR']:.2f} nDCG@5={metrics['nDCG@5']:.2f}"
            )
    return "\n".join(lines)


def save_evaluation_results(results: dict, path: Path = DEFAULT_RESULTS_PATH) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
