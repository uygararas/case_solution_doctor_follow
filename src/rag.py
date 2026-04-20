"""
Part 3 — RAG Generation
Retrieves top articles using Hybrid RRF and generates cited answers.
"""

import io
import os
import re
import sys
import time
from typing import Optional

# Ensure stdout handles Unicode (important on Windows with cp1252 console)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SYSTEM_PROMPT = """You are a medical literature assistant for DoctorFollow, helping \
Turkish-speaking physicians understand English PubMed research.

Instructions:
1. Answer the question based ONLY on the provided PubMed abstracts below.
2. Cite every factual claim using [PMID: XXXXXXXX] inline citations.
3. If the provided context does not contain enough information to answer, explicitly state:
   "The retrieved articles do not provide sufficient information to answer this question."
4. Respond in the SAME LANGUAGE as the question:
   - If the question is in Turkish → answer in Turkish
   - If the question is in English → answer in English
5. Do NOT hallucinate drug names, dosages, diagnostic criteria, or clinical recommendations
   beyond what is explicitly stated in the abstracts.
6. Be concise but complete. Structure your answer with bullet points when listing multiple
   findings or recommendations."""


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower())


def _infer_query_terms(query: str) -> list[str]:
    normalized = _normalize_text(query)
    term_map = {
        "type 2 diabetes mellitus": ("type 2 diabetes", "diabetes mellitus", "diyabet"),
        "acute otitis media": ("acute otitis media", "akut otitis media", "otitis media"),
        "iron deficiency anemia": ("iron deficiency anemia", "iron supplementation", "anemia"),
        "celiac disease diagnosis": ("celiac disease", "coeliac disease", "colyak", "çölyak"),
        "community acquired pneumonia": ("community acquired pneumonia", "pneumonia", "pnomoni", "pnömoni"),
    }
    matches = []
    for term, triggers in term_map.items():
        if any(trigger in normalized for trigger in triggers):
            matches.append(term)
    return matches


def _query_tokens(query: str) -> list[str]:
    return [tok for tok in _normalize_text(query).split() if len(tok) > 3]


def select_context_docs(query: str, retrieved_docs: list[dict], max_docs: int = 5) -> list[dict]:
    """
    Re-rank retrieved docs for answer grounding.

    This helps the RAG step prefer documents that are clearly on-topic even when
    the retriever returns noisy documents from the tiny seeded corpus.
    """
    expected_terms = set(_infer_query_terms(query))
    query_tokens = _query_tokens(query)
    rescored = []

    for position, doc in enumerate(retrieved_docs):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        matched_terms = set(doc.get("matched_terms", []))
        normalized_title = _normalize_text(title)
        normalized_abstract = _normalize_text(abstract)

        score = float(doc.get("score", 0.0))
        score += 0.15 * len(expected_terms & matched_terms)
        score += 0.02 * sum(token in normalized_title for token in query_tokens)
        score += 0.005 * sum(token in normalized_abstract for token in query_tokens)

        if len(title.strip()) < 12:
            score -= 0.08
        if title.strip() == "":
            score -= 0.12

        # Small preference for earlier-ranked docs when scores are otherwise close.
        score -= position * 0.0005
        rescored.append((score, doc))

    rescored.sort(key=lambda item: item[0], reverse=True)

    selected = []
    deferred = []
    for _, doc in rescored:
        if len(doc.get("title", "").strip()) < 12:
            deferred.append(doc)
        else:
            selected.append(doc)
        if len(selected) >= max_docs:
            break

    if len(selected) < max_docs:
        selected.extend(deferred[: max_docs - len(selected)])

    return selected[:max_docs]


def _format_context(docs: list[dict]) -> str:
    """Format retrieved articles into a context block for the answer generator."""
    parts = []
    for i, doc in enumerate(docs, 1):
        pmid = doc.get("pmid", "unknown")
        title = doc.get("title", "No title")
        authors = doc.get("authors", "")
        journal = doc.get("journal", "")
        year = doc.get("year", "")
        abstract = doc.get("abstract", "No abstract available.")

        header = f"[{i}] PMID: {pmid} | {title}"
        if authors:
            header += f" | {authors}"
        if journal and year:
            header += f" | {journal} ({year})"
        parts.append(f"{header}\n{abstract}")

    return "\n\n---\n\n".join(parts)


def _pick_default_google_model(client) -> str:
    for model in client.models.list():
        name = getattr(model, "name", "")
        methods = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", []) or []
        if "flash" in name.lower() and any("generate" in str(method).lower() for method in methods):
            return name.split("/")[-1]
    raise RuntimeError("No compatible Google text generation model was found for this account.")


def _call_google(system_prompt: str, user_message: str, model: Optional[str] = None) -> str:
    """Call the Google text generation API. Retries once on rate limit."""
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey\n"
            "Then add GOOGLE_API_KEY=your_key to your .env file"
        )

    client = genai.Client(api_key=api_key)
    resolved_model = model or os.getenv("GOOGLE_MODEL") or _pick_default_google_model(client)

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "resource" in err_str or "429" in err_str or "rate" in err_str:
                if attempt == 0:
                    print("  [Google] Rate limit hit, waiting 60s...")
                    time.sleep(60)
                    continue
            raise

    raise RuntimeError("Google text generation request failed after retry")


def _call_groq(system_prompt: str, user_message: str, model: str = "llama-3.1-8b-instant") -> str:
    """Fallback: Groq API. Requires GROQ_API_KEY env var."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


def generate_answer(
    query: str,
    retrieved_docs: list[dict],
    provider: str = "google",
    model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Generate a cited answer for a medical query using retrieved PubMed abstracts.

    Parameters
    ----------
    query : str
        The medical question (English or Turkish).
    retrieved_docs : list[dict]
        Top-k articles from the retrieval system.
    provider : str
        'google' (default) or 'groq'.
    model : str, optional
        Override the default model name.
    verbose : bool
        Print query, sources, and answer to stdout.

    Returns
    -------
    dict with keys: query, answer, sources (list of PMID strings), num_docs
    """
    if not retrieved_docs:
        return {
            "query": query,
            "answer": "No articles were retrieved to answer this question.",
            "sources": [],
            "num_docs": 0,
        }

    context_docs = select_context_docs(query, retrieved_docs)
    context = _format_context(context_docs)
    user_message = f"Context (PubMed Abstracts):\n\n{context}\n\n---\n\nQuestion: {query}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")
        print(f"Retrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            score = doc.get("score", 0)
            print(f"  * [{doc['pmid']}] {doc['title'][:70]}... (score={score:.4f})")
        if [doc["pmid"] for doc in context_docs] != [doc["pmid"] for doc in retrieved_docs[: len(context_docs)]]:
            print("\nContext reranked for answer grounding:")
            for doc in context_docs:
                print(f"  -> [{doc['pmid']}] {doc['title'][:70]}...")
        print(f"\nGenerating answer with {provider}...\n")

    if provider == "google":
        answer = _call_google(SYSTEM_PROMPT, user_message, model=model)
    elif provider == "groq":
        _model = model or "llama-3.1-8b-instant"
        answer = _call_groq(SYSTEM_PROMPT, user_message, model=_model)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'google' or 'groq'.")

    # Extract cited PMIDs from the answer
    import re
    cited_pmids = re.findall(r"PMID:\s*(\d+)", answer)

    if verbose:
        print(f"ANSWER:\n{answer}")
        print(f"\nCited PMIDs: {cited_pmids}")
        print(f"{'='*70}\n")

    return {
        "query": query,
        "answer": answer,
        "sources": [doc["pmid"] for doc in context_docs],
        "cited_pmids": cited_pmids,
        "num_docs": len(context_docs),
    }


def run_rag_demo(articles: list[dict], retriever, provider: str = "google"):
    """
    Run the RAG demo on the required queries.
    Returns list of result dicts.
    """
    demo_queries = [
        "What are the latest guidelines for managing type 2 diabetes?",
        "Çocuklarda akut otitis media tedavisi nasıl yapılır?",
    ]

    results = []
    for query in demo_queries:
        docs = retriever.search(query, top_k=5)
        result = generate_answer(query, docs, provider=provider)
        results.append(result)

    return results


if __name__ == "__main__":
    from pathlib import Path
    import json

    # Load articles and build retrievers
    data_path = Path(__file__).parent.parent / "data" / "pubmed_articles.json"
    if not data_path.exists():
        print("Run 'python main.py fetch' first to build the article corpus.")
        exit(1)

    with open(data_path) as f:
        articles = json.load(f)

    from src.retrieval import build_retrievers
    retrievers = build_retrievers(articles)
    hybrid = retrievers["hybrid"]

    run_rag_demo(articles, hybrid)
