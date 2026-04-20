"""
Part 2 — Retrieval System
Three retrieval methods over PubMed corpus:
  A. BM25 (rank_bm25)
  B. Semantic (intfloat/multilingual-e5-small)
  C. Hybrid — Reciprocal Rank Fusion (RRF)
"""

import json
import math
import os
import re
import unicodedata
import time
from pathlib import Path
from typing import Optional

import numpy as np

EMBEDDINGS_DIR = Path(__file__).parent.parent / "embeddings"
DATA_DIR = Path(__file__).parent.parent / "data"

QUERY_TERM_EXPANSIONS = {
    "type 2 diabetes mellitus": {
        "triggers": (
            "type 2 diabetes",
            "type 2 diabetes mellitus",
            "diabetes mellitus",
            "t2dm",
            "diyabet",
        ),
        "expansions": (
            "type 2 diabetes mellitus management guidelines",
            "management of type 2 diabetes mellitus",
        ),
    },
    "acute otitis media": {
        "triggers": (
            "acute otitis media",
            "akut otitis media",
            "otitis media",
            "ear infection",
            "kulak enfeksiyonu",
        ),
        "expansions": (
            "acute otitis media treatment in children",
            "pediatric acute otitis media management",
        ),
    },
    "iron deficiency anemia": {
        "triggers": (
            "iron deficiency anemia",
            "iron deficiency",
            "iron supplementation",
            "anemia during pregnancy",
            "demir eksikligi",
            "anemi",
        ),
        "expansions": (
            "iron deficiency anemia treatment and supplementation",
            "anemia in pregnancy iron treatment",
        ),
    },
    "celiac disease diagnosis": {
        "triggers": (
            "celiac disease",
            "coeliac disease",
            "celiac diagnosis",
            "coeliac diagnosis",
            "çölyak",
            "colyak",
        ),
        "expansions": (
            "celiac disease diagnosis",
            "diagnostic criteria for celiac disease",
            "coeliac disease diagnosis and testing",
        ),
    },
    "community acquired pneumonia": {
        "triggers": (
            "community acquired pneumonia",
            "pneumonia",
            "cap",
            "pnömoni",
            "antibiotic resistance",
        ),
        "expansions": (
            "community acquired pneumonia antibiotic resistance",
            "community acquired pneumonia diagnosis and severity",
        ),
    },
}

IMPORTANT_QUERY_TOKENS = {
    "guidelines",
    "management",
    "treatment",
    "diagnosis",
    "diagnostic",
    "criteria",
    "pregnancy",
    "children",
    "pediatric",
    "antibiotic",
    "resistance",
    "severity",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric. Simple but effective for medical English."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize_text(text: str) -> str:
    """Lowercase and strip accents to make Turkish/English keyword matching more robust."""
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def infer_query_terms(query: str) -> list[str]:
    """Map a free-text query to the seeded medical terms present in the small corpus."""
    normalized_query = normalize_text(query)
    matched_terms = []
    for canonical_term, config in QUERY_TERM_EXPANSIONS.items():
        if any(trigger in normalized_query for trigger in config["triggers"]):
            matched_terms.append(canonical_term)
    return matched_terms


def expand_query_variants(query: str) -> list[str]:
    """
    Create deterministic English-friendly query variants.

    The corpus is tiny and seeded from 10 fixed terms, so expanding toward those
    canonical medical terms improves recall without requiring another API call.
    """
    variants = [query.strip()]
    for canonical_term in infer_query_terms(query):
        for expansion in QUERY_TERM_EXPANSIONS[canonical_term]["expansions"]:
            if expansion not in variants:
                variants.append(expansion)
    return variants


# ---------------------------------------------------------------------------
# A. BM25 Retrieval
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 retrieval over title + abstract.
    Title is repeated `title_boost` times to upweight title term matches.

    Parameters
    ----------
    k1 : float
        Term frequency saturation. Controls how much repeated occurrences of a
        term in a document contribute. At k1=0, TF has no effect (pure IDF).
        At k1=2.0, repeated terms are heavily rewarded. Typical range: 1.2–2.0.
    b : float
        Document length normalization. At b=0, no length penalty applied.
        At b=1, full normalization against average document length. For PubMed
        abstracts (constrained length), b=0.5–0.75 works well.
    title_boost : int
        Number of times to repeat the title tokens during indexing, to upweight
        title-term matches without reimplementing BM25.
    """

    def __init__(self, articles: list[dict], k1: float = 1.5, b: float = 0.75, title_boost: int = 3):
        from rank_bm25 import BM25Okapi

        self.articles = articles
        self.k1 = k1
        self.b = b

        # Build corpus: title (boosted) + abstract
        corpus = []
        for art in articles:
            title_tokens = tokenize(art.get("title", "")) * title_boost
            abstract_tokens = tokenize(art.get("abstract", ""))
            corpus.append(title_tokens + abstract_tokens)

        self.bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k articles with BM25 scores."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices):
            art = self.articles[idx].copy()
            art["score"] = float(scores[idx])
            art["rank"] = rank + 1
            art["method"] = "bm25"
            results.append(art)
        return results

    def get_ranked_list(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return (pmid, score) pairs for RRF fusion."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.articles[i]["pmid"], float(scores[i])) for i in ranked_indices]


# ---------------------------------------------------------------------------
# B. Semantic Retrieval
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """
    Semantic retrieval using intfloat/multilingual-e5-small.

    Model choice rationale:
    - multilingual-e5-small (~470MB) vs BAAI/bge-m3 (~2.3GB):
      At N≈50 documents, quality difference is negligible. The multilingual model
      handles Turkish queries natively (trained on 100+ languages via mC4/CC-100),
      which is critical for DoctorFollow's Turkish-speaking users.
    - The model requires explicit "query: " and "passage: " prefixes per the
      E5 paper (Wang et al., 2022). Missing these degrades retrieval significantly.
    - Cosine similarity computed in numpy — no FAISS needed at this scale.

    Embeddings are cached to disk to avoid re-encoding on every run.
    """

    MODEL_NAME = "intfloat/multilingual-e5-small"
    CACHE_FILE = EMBEDDINGS_DIR / "doc_embeddings.npy"
    PMID_CACHE_FILE = EMBEDDINGS_DIR / "doc_pmids.json"

    def __init__(self, articles: list[dict], use_cache: bool = True):
        self.articles = articles
        self.pmids = [art["pmid"] for art in articles]

        # Build or load cached embeddings
        cache_valid = (
            use_cache
            and self.CACHE_FILE.exists()
            and self.PMID_CACHE_FILE.exists()
            and json.loads(self.PMID_CACHE_FILE.read_text()) == self.pmids
        )

        model_kwargs = {}
        if cache_valid:
            # Avoid unnecessary Hugging Face network checks when the model and
            # embedding cache are already present locally.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            model_kwargs["local_files_only"] = True

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL_NAME, **model_kwargs)

        if cache_valid:
            print(f"  [SemanticRetriever] Loading cached embeddings from {self.CACHE_FILE}")
            self.doc_embeddings = np.load(str(self.CACHE_FILE))
        else:
            print(f"  [SemanticRetriever] Encoding {len(articles)} documents with {self.MODEL_NAME}...")
            t0 = time.time()
            passages = [
                f"passage: {art.get('title', '')} {art.get('abstract', '')}"
                for art in articles
            ]
            self.doc_embeddings = self.model.encode(
                passages, normalize_embeddings=True, show_progress_bar=True, batch_size=32
            )
            EMBEDDINGS_DIR.mkdir(exist_ok=True)
            np.save(str(self.CACHE_FILE), self.doc_embeddings)
            self.PMID_CACHE_FILE.write_text(json.dumps(self.pmids))
            print(f"  [SemanticRetriever] Done in {time.time()-t0:.1f}s. Cached to {self.CACHE_FILE}")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a query with required 'query: ' prefix."""
        return self.model.encode(
            f"query: {query}", normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k articles by cosine similarity."""
        q_emb = self._encode_query(query)
        # Since embeddings are L2-normalized, dot product == cosine similarity
        scores = self.doc_embeddings @ q_emb

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices):
            art = self.articles[idx].copy()
            art["score"] = float(scores[idx])
            art["rank"] = rank + 1
            art["method"] = "semantic"
            results.append(art)
        return results

    def get_ranked_list(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return (pmid, score) pairs for RRF fusion."""
        q_emb = self._encode_query(query)
        scores = self.doc_embeddings @ q_emb
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.articles[i]["pmid"], float(scores[i])) for i in ranked_indices]


# ---------------------------------------------------------------------------
# C. Hybrid — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke & Butt, SIGIR 2009).

    Formula:
        RRF_score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

    where rank_r(d) is the 1-based position of document d in ranker r's list.
    Documents absent from a ranker's list contribute 0 (rank treated as +∞).

    Parameters
    ----------
    ranked_lists : list of lists of (doc_id, score) pairs, each sorted by desc score
    k : int
        Smoothing constant (default 60, from the original paper).
        k=0 → scores diverge for rank-1 documents (extreme sensitivity to top rank).
        k=1000 → 1/(1001) ≈ 1/(1002), scores become nearly uniform (no rank signal).
        k=60 provides a good balance: rank differences matter, but outliers are dampened.

    Why ranks instead of raw scores?
        BM25 scores (unbounded, depends on corpus statistics) and cosine similarity
        scores (bounded [-1, 1]) live on completely different scales. Normalizing raw
        scores is heuristic and brittle. Using rank positions makes the combination
        scale-invariant and robust — rank 1 always means "best in this list" regardless
        of the underlying scoring function.

    Returns
    -------
    list of (doc_id, rrf_score) sorted by descending RRF score
    """
    rrf_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank_0based, (doc_id, _) in enumerate(ranked_list):
            rank_1based = rank_0based + 1
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank_1based)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Hybrid retrieval: BM25 + Semantic fused via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        articles: list[dict],
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        k: int = 60,
        candidate_pool: int = 50,
    ):
        self.articles = articles
        self.pmid_to_article = {art["pmid"]: art for art in articles}
        self.bm25 = bm25_retriever or BM25Retriever(articles)
        self.semantic = semantic_retriever or SemanticRetriever(articles)
        self.k = k
        self.candidate_pool = candidate_pool
        self._normalized_titles = {
            art["pmid"]: normalize_text(art.get("title", ""))
            for art in articles
        }
        self._normalized_abstracts = {
            art["pmid"]: normalize_text(art.get("abstract", ""))
            for art in articles
        }

    def _query_tokens(self, query: str) -> list[str]:
        tokens = tokenize(normalize_text(query))
        return [token for token in tokens if len(token) > 3 or token in IMPORTANT_QUERY_TOKENS]

    def _article_boost(self, pmid: str, query: str, expected_terms: list[str]) -> float:
        article = self.pmid_to_article[pmid]
        matched_terms = article.get("matched_terms", [])
        boost = 0.0

        normalized_title = self._normalized_titles.get(pmid, "")
        normalized_abstract = self._normalized_abstracts.get(pmid, "")
        title_token_hits = 0
        abstract_token_hits = 0
        for token in self._query_tokens(query):
            if token in normalized_title:
                title_token_hits += 1
                boost += 0.02
            elif token in normalized_abstract:
                abstract_token_hits += 1
                boost += 0.008

        if matched_terms:
            title_term_hits = sum(1 for term in matched_terms if any(t in normalized_title for t in tokenize(term)))
            boost += min(0.03, 0.01 * title_term_hits)
        else:
            title_term_hits = 0

        overlap_count = len(set(matched_terms) & set(expected_terms))
        if overlap_count:
            confidence_factor = 0.25 + min(1.0, 0.35 * title_term_hits + 0.2 * title_token_hits + 0.05 * abstract_token_hits)
            boost += 0.08 * overlap_count * confidence_factor

        raw_title = article.get("title", "").strip()
        if raw_title == "":
            boost -= 0.12
        elif len(raw_title) < 12:
            boost -= 0.08

        return boost

    def _fused_candidates(self, query: str) -> list[tuple[str, float]]:
        expected_terms = infer_query_terms(query)
        ranked_lists = []

        for query_variant in expand_query_variants(query):
            ranked_lists.append(self.bm25.get_ranked_list(query_variant, top_k=self.candidate_pool))
            ranked_lists.append(self.semantic.get_ranked_list(query_variant, top_k=self.candidate_pool))

        fused = reciprocal_rank_fusion(ranked_lists, k=self.k)
        rescored = []
        for pmid, base_score in fused:
            rescored.append((pmid, base_score + self._article_boost(pmid, query, expected_terms)))

        rescored.sort(key=lambda item: item[1], reverse=True)
        return rescored

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Fuse BM25 and Semantic results via RRF, then apply term-aware reranking."""
        fused = self._fused_candidates(query)

        results = []
        for rank, (pmid, rrf_score) in enumerate(fused[:top_k]):
            if pmid in self.pmid_to_article:
                art = self.pmid_to_article[pmid].copy()
                art["score"] = rrf_score
                art["rank"] = rank + 1
                art["method"] = "hybrid_rrf"
                results.append(art)
        return results

    def get_ranked_list(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return (pmid, reranked_score) pairs for chained fusion or evaluation."""
        return self._fused_candidates(query)[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retrievers(
    articles: list[dict],
    use_cache: bool = True,
    methods: Optional[set[str]] = None,
) -> dict:
    """
    Initialize the requested retrievers from a list of article dicts.
    Returns a dict that may include keys: 'bm25', 'semantic', 'hybrid'.
    """
    requested = methods or {"bm25", "semantic", "hybrid"}
    retrievers = {}

    needs_bm25 = bool({"bm25", "hybrid"} & requested)
    needs_semantic = bool({"semantic", "hybrid"} & requested)

    bm25 = None
    semantic = None

    if needs_bm25:
        print("Building BM25 index...")
        bm25 = BM25Retriever(articles)
        retrievers["bm25"] = bm25

    if needs_semantic:
        print("Building Semantic index...")
        semantic = SemanticRetriever(articles, use_cache=use_cache)
        retrievers["semantic"] = semantic

    if "hybrid" in requested:
        print("Building Hybrid (RRF) retriever...")
        hybrid = HybridRetriever(articles, bm25_retriever=bm25, semantic_retriever=semantic)
        retrievers["hybrid"] = hybrid

    return retrievers


def load_articles(path: Path = None) -> list[dict]:
    """Load articles from the pipeline JSON output."""
    path = path or (DATA_DIR / "pubmed_articles.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Articles file not found at {path}. Run the pipeline first:\n"
            "  python main.py fetch"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation Utilities
# ---------------------------------------------------------------------------

def precision_at_k(relevant: list[int], k: int = 5) -> float:
    """P@k: fraction of top-k results that are relevant."""
    return sum(relevant[:k]) / k


def reciprocal_rank(relevant: list[int]) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""
    for i, r in enumerate(relevant):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(gains: list[float], k: int = 5) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(gains: list[float], k: int = 5) -> float:
    """Normalized DCG at k. Gains should be relevance scores (0/1/2)."""
    actual = dcg_at_k(gains, k)
    ideal = dcg_at_k(sorted(gains, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def compute_metrics(relevance_scores: list[int], k: int = 5) -> dict:
    """
    Compute all retrieval metrics given a list of relevance scores
    (0=not relevant, 1=partially relevant, 2=highly relevant) for top-k results.
    """
    binary = [1 if r > 0 else 0 for r in relevance_scores]
    return {
        "P@5": round(precision_at_k(binary, k), 4),
        "MRR": round(reciprocal_rank(binary), 4),
        "nDCG@5": round(ndcg_at_k([float(r) for r in relevance_scores], k), 4),
    }


if __name__ == "__main__":
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")
    retrievers = build_retrievers(articles)

    test_query = "type 2 diabetes management guidelines"
    print(f"\nQuery: {test_query!r}\n")
    for method, retriever in retrievers.items():
        print(f"--- {method.upper()} ---")
        for r in retriever.search(test_query, top_k=3):
            print(f"  [{r['rank']}] {r['title'][:80]}... (score={r['score']:.4f})")
        print()
