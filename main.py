#!/usr/bin/env python3
"""
DoctorFollow — Medical Retrieval System
CLI entry point.

Usage:
  python main.py fetch                           # Part 1: fetch PubMed articles
  python main.py retrieve --query "diabetes"     # Part 2: search with all methods
  python main.py evaluate                        # Part 2D: metric-based comparison
  python main.py rag --query "..."               # Part 3: full RAG with cited answer
  python main.py demo                            # Run full demo (all 3 parts)
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
ARTICLES_FILE = DATA_DIR / "pubmed_articles.json"


def cmd_fetch(args):
    from src.pipeline import run_pipeline
    run_pipeline(
        retmax=args.retmax,
        batch_size=args.batch_size,
        candidate_retmax=args.candidate_retmax,
    )


def cmd_retrieve(args):
    if not ARTICLES_FILE.exists():
        print("No articles found. Run: python main.py fetch")
        return

    with open(ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    method = args.method
    if method == "all":
        methods_to_run = ["bm25", "semantic", "hybrid"]
        methods_to_build = set(methods_to_run)
    else:
        methods_to_run = [method]
        methods_to_build = {method}

    from src.retrieval import build_retrievers
    retrievers = build_retrievers(articles, methods=methods_to_build)

    top_k = args.top_k
    print(f"\nQuery: {args.query!r}\n")

    for m in methods_to_run:
        print(f"{'-'*60}")
        print(f"Method: {m.upper()}")
        print(f"{'-'*60}")
        results = retrievers[m].search(args.query, top_k=top_k)
        for r in results:
            pmid = r.get("pmid", "?")
            title = r.get("title", "No title")[:72]
            score = r.get("score", 0)
            year = r.get("year", "")
            print(f"  [{r['rank']}] ({score:.4f}) [{pmid}] {title}... {year}")
        print()


def cmd_rag(args):
    if not ARTICLES_FILE.exists():
        print("No articles found. Run: python main.py fetch")
        return

    with open(ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    from src.retrieval import build_retrievers
    from src.rag import generate_answer

    print("Building retrievers...")
    retrievers = build_retrievers(articles, methods={"hybrid"})
    hybrid = retrievers["hybrid"]

    docs = hybrid.search(args.query, top_k=args.top_k)
    try:
        generate_answer(
            query=args.query,
            retrieved_docs=docs,
            provider=args.provider,
            verbose=True,
        )
    except ValueError as exc:
        print(f"\nRAG configuration error: {exc}")


def cmd_evaluate(args):
    from src.evaluation import evaluate_retrievers, format_evaluation_report, save_evaluation_results

    if not ARTICLES_FILE.exists():
        print("No articles found. Run: python main.py fetch")
        return

    with open(ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    results = evaluate_retrievers(
        articles=articles,
        allow_live_scoring=args.allow_live_scoring,
        use_cache=not args.no_embedding_cache,
    )
    print(format_evaluation_report(results))
    save_evaluation_results(results)
    print(f"\nSaved evaluation report to {Path('data') / 'evaluation_results.json'}")


def cmd_demo(args):
    """Full end-to-end demo: fetch (if needed) → retrieve → RAG."""
    # Step 1: Fetch if needed
    if not ARTICLES_FILE.exists():
        print("Step 1: Fetching PubMed articles...")
        from src.pipeline import run_pipeline
        run_pipeline()
    else:
        with open(ARTICLES_FILE, encoding="utf-8") as f:
            articles = json.load(f)
        print(f"Step 1: Using cached corpus ({len(articles)} articles)")

    with open(ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    # Step 2: Build retrievers
    print("\nStep 2: Building retrieval indices...")
    from src.retrieval import build_retrievers
    retrievers = build_retrievers(articles)

    # Step 3: Evaluate 5 queries
    print("\nStep 3: Evaluating 5 queries across all retrieval methods...")
    eval_queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
        "Iron supplementation dosing for anemia during pregnancy",
        "Çölyak hastalığı tanı kriterleri nelerdir?",
        "Antibiotic resistance patterns in community acquired pneumonia",
    ]

    for query in eval_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        for method_name in ["bm25", "semantic", "hybrid"]:
            results = retrievers[method_name].search(query, top_k=5)
            print(f"\n  [{method_name.upper()}]")
            for r in results:
                print(f"    [{r['rank']}] [{r['pmid']}] {r['title'][:65]}...")

    # Step 4: RAG demo
    print("\n\nStep 4: RAG Generation Demo")
    from src.rag import generate_answer

    rag_queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    ]

    for query in rag_queries:
        docs = retrievers["hybrid"].search(query, top_k=5)
        try:
            generate_answer(query, docs, provider=args.provider, verbose=True)
        except ValueError as exc:
            print(f"\nRAG configuration error: {exc}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="DoctorFollow — Medical Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch PubMed articles (Part 1)")
    fetch_parser.add_argument("--retmax", type=int, default=5, help="Articles kept per medical term after reranking")
    fetch_parser.add_argument(
        "--candidate-retmax",
        type=int,
        default=12,
        help="Broader recent PubMed candidate pool fetched per term before filtering",
    )
    fetch_parser.add_argument("--batch-size", type=int, default=20, help="PMID batch size for efetch")

    # retrieve
    ret_parser = subparsers.add_parser("retrieve", help="Search articles (Part 2)")
    ret_parser.add_argument("--query", "-q", required=True, help="Search query")
    ret_parser.add_argument(
        "--method", "-m",
        choices=["bm25", "semantic", "hybrid", "all"],
        default="all",
        help="Retrieval method (default: all)",
    )
    ret_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")

    # rag
    rag_parser = subparsers.add_parser("rag", help="RAG generation (Part 3)")
    rag_parser.add_argument("--query", "-q", required=True, help="Medical question")
    rag_parser.add_argument("--top-k", "-k", type=int, default=5, help="Docs to retrieve")
    rag_parser.add_argument(
        "--provider", "-p",
        choices=["google", "groq"],
        default="google",
        help="Answer generation provider (default: google)",
    )

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate retrieval methods (Part 2D)")
    eval_parser.add_argument(
        "--allow-live-scoring",
        action="store_true",
        help="Score uncached query-document pairs through the configured Google service and extend judge_cache.json",
    )
    eval_parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Rebuild semantic embeddings instead of using cached embeddings/",
    )

    # demo
    demo_parser = subparsers.add_parser("demo", help="Full end-to-end demo")
    demo_parser.add_argument(
        "--provider", "-p",
        choices=["google", "groq"],
        default="google",
        help="Answer generation provider (default: google)",
    )

    args = parser.parse_args()

    commands = {
        "fetch": cmd_fetch,
        "retrieve": cmd_retrieve,
        "evaluate": cmd_evaluate,
        "rag": cmd_rag,
        "demo": cmd_demo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
