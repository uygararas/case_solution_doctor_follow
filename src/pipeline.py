"""
Part 1 — PubMed Data Pipeline
Fetches 5 most recent abstracts per medical term via E-utilities API.
Handles rate limiting (3 req/sec), deduplication, and missing fields.
"""

import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import requests

# E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# With an NCBI_API_KEY, rate limit rises to 10 req/sec
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
# 3 req/sec without key, 10 with key
REQUEST_DELAY = 0.11 if NCBI_API_KEY else 0.34

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "pubmed_articles.json"
SUMMARY_FILE = DATA_DIR / "corpus_summary.json"

TERM_SEARCH_CONFIG = {
    "atrial fibrillation": {
        "query": '("atrial fibrillation"[Title/Abstract] OR "AF"[Title/Abstract])',
        "keywords": ("atrial", "fibrillation", "af"),
        "core_keywords": ("fibrillation", "atrial fibrillation"),
    },
    "type 2 diabetes mellitus": {
        "query": '("type 2 diabetes"[Title/Abstract] OR "type 2 diabetes mellitus"[Title/Abstract] OR T2DM[Title/Abstract])',
        "keywords": ("type 2 diabetes", "diabetes mellitus", "t2dm", "glycemic"),
        "core_keywords": ("diabetes", "t2dm"),
    },
    "pediatric asthma management": {
        "query": '((asthma[Title/Abstract]) AND (pediatric[Title/Abstract] OR children[Title/Abstract] OR child[Title/Abstract]))',
        "keywords": ("asthma", "pediatric", "children", "child"),
        "core_keywords": ("asthma",),
    },
    "acute otitis media": {
        "query": '("acute otitis media"[Title/Abstract] OR ((otitis media[Title/Abstract]) AND (acute[Title/Abstract] OR pediatric[Title/Abstract] OR child*[Title/Abstract])))',
        "keywords": ("acute otitis media", "otitis media", "middle ear", "pediatric"),
        "core_keywords": ("otitis", "otitis media"),
    },
    "chronic kidney disease": {
        "query": '("chronic kidney disease"[Title/Abstract] OR CKD[Title/Abstract])',
        "keywords": ("chronic kidney disease", "ckd", "kidney disease", "renal"),
        "core_keywords": ("kidney", "ckd", "renal"),
    },
    "iron deficiency anemia": {
        "query": '("iron deficiency anemia"[Title/Abstract] OR ((iron deficiency[Title/Abstract]) AND anemia[Title/Abstract]))',
        "keywords": ("iron deficiency anemia", "iron deficiency", "anemia", "ferrous"),
        "core_keywords": ("iron", "anemia", "ferrous"),
    },
    "community acquired pneumonia": {
        "query": '("community-acquired pneumonia"[Title/Abstract] OR "community acquired pneumonia"[Title/Abstract] OR CAP[Title/Abstract])',
        "keywords": ("community-acquired pneumonia", "community acquired pneumonia", "pneumonia", "cap"),
        "core_keywords": ("pneumonia",),
    },
    "gestational diabetes": {
        "query": '("gestational diabetes"[Title/Abstract] OR GDM[Title/Abstract] OR "gestational diabetes mellitus"[Title/Abstract])',
        "keywords": ("gestational diabetes", "gdm", "pregnancy", "maternal"),
        "core_keywords": ("gestational", "gdm"),
    },
    "celiac disease diagnosis": {
        "query": '(("celiac disease"[Title/Abstract] OR "coeliac disease"[Title/Abstract]) AND (diagnos*[Title/Abstract] OR criteri*[Title/Abstract] OR serolog*[Title/Abstract] OR biopsy[Title/Abstract]))',
        "keywords": ("celiac disease", "coeliac disease", "diagnos", "criteria", "serolog", "biopsy"),
        "core_keywords": ("celiac", "coeliac"),
    },
    "allergic rhinitis treatment": {
        "query": '((allergic rhinitis[Title/Abstract]) AND (treat*[Title/Abstract] OR therap*[Title/Abstract] OR management[Title/Abstract]))',
        "keywords": ("allergic rhinitis", "rhinitis", "treat", "therapy", "management"),
        "core_keywords": ("rhinitis",),
    },
}


class RateLimiter:
    """Token-bucket style rate limiter to respect PubMed's 3 req/sec limit."""

    def __init__(self, min_interval: float = REQUEST_DELAY):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


_rate_limiter = RateLimiter()


def _get(url: str, params: dict, retries: int = 3) -> requests.Response:
    """GET with rate limiting and exponential backoff on failures."""
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    for attempt in range(retries):
        _rate_limiter.wait()
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429 or resp.status_code == 503:
                wait_time = 2 ** attempt
                print(f"  Rate limited (HTTP {resp.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Network error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed after {retries} retries")


def _tokenize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower())


def build_pubmed_query(term: str) -> str:
    config = TERM_SEARCH_CONFIG.get(term)
    if config:
        return config["query"]
    return f'"{term}"[Title/Abstract]'


def esearch(term: str, retmax: int = 5) -> list[str]:
    """Search PubMed for a term, return list of PMIDs (most recent first)."""
    params = {
        "db": "pubmed",
        "term": build_pubmed_query(term),
        "retmax": retmax,
        "sort": "pub_date",
        "retmode": "json",
    }
    resp = _get(ESEARCH_URL, params)
    data = resp.json()
    result = data.get("esearchresult", {})
    count = int(result.get("count", 0))
    pmids = result.get("idlist", [])
    return pmids, count


def _extract_text(node: Optional[ET.Element], path: str, default: str = "") -> str:
    """Safe text extraction from XML node."""
    if node is None:
        return default
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else default


def _parse_article(medline_citation: ET.Element, pubmed_data: ET.Element) -> dict:
    """Parse a single MedlineCitation XML element into a structured dict."""
    article = medline_citation.find("Article")
    if article is None:
        return {}

    # PMID
    pmid_el = medline_citation.find("PMID")
    pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""

    # Title
    title = _extract_text(article, "ArticleTitle")

    # Abstract — may have multiple structured sections
    abstract_el = article.find("Abstract")
    if abstract_el is not None:
        parts = []
        for ab_text in abstract_el.findall("AbstractText"):
            label = ab_text.get("Label", "")
            text = ab_text.text or ""
            # Collect tail text too (some parsers need this)
            tail = ab_text.tail or ""
            if label:
                parts.append(f"{label}: {text.strip()}")
            elif text.strip():
                parts.append(text.strip())
        abstract = " ".join(parts)
    else:
        abstract = ""

    # First author
    author_list = article.find("AuthorList")
    first_author = ""
    if author_list is not None:
        author = author_list.find("Author")
        if author is not None:
            last = _extract_text(author, "LastName")
            fore = _extract_text(author, "ForeName")
            first_author = f"{last}, {fore}".strip(", ")

    # Journal
    journal_el = article.find("Journal")
    journal = ""
    if journal_el is not None:
        journal = _extract_text(journal_el, "Title") or _extract_text(journal_el, "ISOAbbreviation")

    # Year — try multiple paths
    year = ""
    if journal_el is not None:
        pub_date = journal_el.find("JournalIssue/PubDate")
        if pub_date is not None:
            year = _extract_text(pub_date, "Year")
            if not year:
                medline_date = _extract_text(pub_date, "MedlineDate")
                year = medline_date[:4] if medline_date else ""

    # DOI — from ArticleIdList in PubmedData
    doi = None
    if pubmed_data is not None:
        for article_id in pubmed_data.findall("ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text.strip() if article_id.text else None
                break

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": first_author,
        "journal": journal,
        "year": year,
        "doi": doi,
    }


def efetch_batch(pmids: list[str]) -> list[dict]:
    """Fetch full records for a batch of PMIDs. Returns list of article dicts."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = _get(EFETCH_URL, params)

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return []

    articles = []
    for pubmed_article in root.findall("PubmedArticle"):
        medline = pubmed_article.find("MedlineCitation")
        pubmed_data = pubmed_article.find("PubmedData")
        if medline is None:
            continue
        article = _parse_article(medline, pubmed_data)
        if article.get("pmid"):
            articles.append(article)

    return articles


def _article_relevance_score(article: dict, term: str) -> float:
    """Prefer recent candidates that actually talk about the requested concept."""
    config = TERM_SEARCH_CONFIG.get(term, {})
    keywords = config.get("keywords", tuple(tokenize_term(term)))
    core_keywords = config.get("core_keywords", tuple())
    title = _tokenize_for_match(article.get("title", ""))
    abstract = _tokenize_for_match(article.get("abstract", ""))

    if core_keywords and not any(keyword.lower() in title or keyword.lower() in abstract for keyword in core_keywords):
        return -1.0

    score = 0.0
    for keyword in keywords:
        normalized_keyword = keyword.lower()
        if normalized_keyword in title:
            score += 3.0
        if normalized_keyword in abstract:
            score += 1.0

    if article.get("abstract", "").strip():
        score += 0.5
    if len(article.get("title", "").strip()) >= 12:
        score += 0.2

    year = article.get("year", "")
    if year.isdigit():
        score += int(year) / 10000.0

    return score


def tokenize_term(term: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", term.lower()) if len(tok) > 3]


def load_terms(csv_path: Path) -> list[str]:
    """Read medical terms from CSV. Supports 'term' column or bare lines."""
    terms = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "term" in (reader.fieldnames or []):
            for row in reader:
                t = row["term"].strip()
                if t:
                    terms.append(t)
        else:
            f.seek(0)
            for line in f:
                t = line.strip()
                if t and not t.lower() == "term":
                    terms.append(t)
    return terms


def run_pipeline(
    csv_path: Path = None,
    output_path: Path = None,
    summary_path: Path = None,
    retmax: int = 5,
    batch_size: int = 20,
    candidate_retmax: int = 12,
) -> list[dict]:
    """
    Full pipeline: fetch → deduplicate → save.
    Returns the list of deduplicated article dicts.
    """
    csv_path = csv_path or (Path(__file__).parent.parent / "medical_terms.csv")
    output_path = output_path or OUTPUT_FILE
    summary_path = summary_path or SUMMARY_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    terms = load_terms(csv_path)
    print(f"\n{'='*60}")
    print(f"DoctorFollow -- PubMed Data Pipeline")
    print(f"{'='*60}")
    print(f"Terms to process : {len(terms)}")
    print(f"Articles per term: {retmax}")
    print(f"NCBI API key     : {'present' if NCBI_API_KEY else 'not set (3 req/sec)'}")
    print(f"{'='*60}\n")

    # Step 1: esearch for all terms, collect PMIDs
    term_to_candidate_pmids: dict[str, list[str]] = {}
    terms_processed = 0
    errors = []

    for term in terms:
        print(f"[esearch] {term!r}...", end=" ", flush=True)
        try:
            pmids, count = esearch(term, retmax=candidate_retmax)
            print(f"found {count} total, fetched {len(pmids)} candidates")
            term_to_candidate_pmids[term] = pmids
            terms_processed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"term": term, "error": str(e)})

    all_pmids = sorted({pmid for pmids in term_to_candidate_pmids.values() for pmid in pmids})
    unique_before_dedup = len(all_pmids)
    print(f"\nUnique PMIDs across all terms: {unique_before_dedup}")

    # Step 2: efetch in batches
    print(f"\nFetching full records in batches of {batch_size}...")
    fetched_articles = []
    for i in range(0, len(all_pmids), batch_size):
        batch = all_pmids[i : i + batch_size]
        print(f"  Batch {i//batch_size + 1}: PMIDs {i+1}-{min(i+batch_size, len(all_pmids))}", end=" ")
        try:
            articles = efetch_batch(batch)
            fetched_articles.extend(articles)
            print(f"-> {len(articles)} parsed")
        except Exception as e:
            print(f"-> ERROR: {e}")
            errors.append({"batch_start": i, "error": str(e)})

    # Step 3: Rank candidates per term, keep top-N per term, then deduplicate by PMID
    pmid_to_article = {article["pmid"]: article for article in fetched_articles}
    pmid_to_terms: dict[str, list[str]] = defaultdict(list)

    term_selected_pmids: dict[str, list[str]] = {}
    for term, candidate_pmids in term_to_candidate_pmids.items():
        ranked_candidates = []
        for pmid in candidate_pmids:
            article = pmid_to_article.get(pmid)
            if not article:
                continue
            ranked_candidates.append((_article_relevance_score(article, term), pmid))
        ranked_candidates.sort(reverse=True)
        selected_for_term = []
        for _, pmid in ranked_candidates[:retmax]:
            pmid_to_terms[pmid].append(term)
            selected_for_term.append(pmid)
        term_selected_pmids[term] = selected_for_term

    seen_pmids = set()
    deduplicated = []
    duplicates_removed = 0

    for article in fetched_articles:
        pmid = article["pmid"]
        if pmid not in pmid_to_terms:
            continue
        if pmid in seen_pmids:
            duplicates_removed += 1
            continue
        seen_pmids.add(pmid)
        article["matched_terms"] = pmid_to_terms.get(pmid, [])
        deduplicated.append(article)

    # Step 4: Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)

    journal_counts: dict[str, int] = defaultdict(int)
    year_counts: dict[str, int] = defaultdict(int)
    missing_abstracts = 0
    for article in deduplicated:
        if article.get("journal"):
            journal_counts[article["journal"]] += 1
        if article.get("year"):
            year_counts[article["year"]] += 1
        if not article.get("abstract", "").strip():
            missing_abstracts += 1

    summary = {
        "terms_processed": terms_processed,
        "term_count": len(terms),
        "candidate_retmax": candidate_retmax,
        "retmax": retmax,
        "candidate_pmids_before_filtering": unique_before_dedup,
        "unique_articles": len(deduplicated),
        "duplicates_removed": duplicates_removed,
        "errors": errors,
        "missing_abstracts": missing_abstracts,
        "selected_pmids_by_term": term_selected_pmids,
        "top_journals": sorted(journal_counts.items(), key=lambda item: (-item[1], item[0]))[:10],
        "year_distribution": dict(sorted(year_counts.items())),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Step 5: Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Terms processed  : {terms_processed}/{len(terms)}")
    print(f"Unique articles  : {len(deduplicated)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Errors           : {len(errors)}")
    if errors:
        for e in errors:
            print(f"  - {e}")
    print(f"Output saved to  : {output_path}")
    print(f"Summary saved to : {summary_path}")
    print(f"{'='*60}\n")

    return deduplicated


if __name__ == "__main__":
    run_pipeline()
