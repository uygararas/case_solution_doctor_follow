# DoctorFollow — Medical Retrieval System

A medical retrieval-augmented generation system that fetches PubMed articles, retrieves relevant documents via BM25, Semantic, and Hybrid RRF methods, and generates cited answers through a hosted text-generation endpoint. Built for Turkish-speaking physicians querying English medical literature.

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users**: PyTorch's long file paths can exceed Windows' 260-character limit. If you hit an `OSError` during `torch` install, create a venv with a short path:
> ```bash
> python -m venv C:\venv311
> C:\venv311\Scripts\pip install -r requirements.txt
> # Then run all commands with: C:\venv311\Scripts\python main.py ...
> ```

> **Note on sentence-transformers model**: `intfloat/multilingual-e5-small` (~470MB) downloads automatically on first run and caches to `~/.cache/huggingface/`.

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required:
- `GOOGLE_API_KEY` — for answer generation (Part 3). Free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

Optional but recommended:
- `NCBI_API_KEY` — raises PubMed rate limit from 3 to 10 req/sec. Free at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/)

### 3. Run

```bash
# Part 1: Fetch PubMed articles
python main.py fetch
python main.py fetch --candidate-retmax 12 --retmax 5

# Part 2: Search with all retrieval methods
python main.py retrieve --query "type 2 diabetes management"
python main.py retrieve --query "Çocuklarda akut otitis media" --method hybrid

# Part 2D: Reproduce metric-based evaluation from cached relevance judgements
python main.py evaluate

# Part 3: Full RAG with cited answer
python main.py rag --query "What are the latest guidelines for managing type 2 diabetes?"
python main.py rag --query "Çölyak hastalığı tanı kriterleri nelerdir?"

# Full demo (all parts)
python main.py demo
```

### 4. Notebooks (for analysis and exploration)

```bash
cd notebooks
jupyter notebook
```

- `01_data_pipeline.ipynb` — PubMed API exploration, pipeline walkthrough, corpus statistics
- `02_retrieval_analysis.ipynb` — BM25 parameter sweep, semantic analysis, RRF walkthrough, evaluation
- `03_rag_demo.ipynb` — Full RAG demo with query expansion bonus

All three notebooks are already executed in the repository and include saved outputs/results.

---

## Approach

### Data Collection Strategy

The PubMed pipeline now uses a two-stage collection flow instead of taking the first 5 recent hits blindly:

1. For each medical term, submit a term-specific PubMed query template tuned to the concept.
2. Fetch a broader recent candidate pool per term.
3. Score the fetched abstracts by title/abstract keyword match and keep the best 5 per term.
4. Deduplicate the final corpus across terms and preserve `matched_terms` for downstream retrieval.

This helps reduce obviously off-topic recent hits and gives the retrieval layer a cleaner starting corpus.

The pipeline also writes `data/corpus_summary.json`, which records the selected PMIDs per term and a lightweight summary of the final corpus.

### Model Choice: `intfloat/multilingual-e5-small`

Chosen over `BAAI/bge-m3` for three reasons:

1. **Size**: 470MB vs 2.3GB. Practical for demo machines without GPU.
2. **Turkish support**: Trained on 100+ languages including Turkish (mC4, CC-100). Handles cross-lingual retrieval natively — a Turkish query retrieves relevant English abstracts.
3. **Quality at this scale**: With ~50 documents, quality differences between models are negligible. bge-m3's advantages appear on large heterogeneous corpora.

**Critical**: multilingual-e5 requires `"query: "` prefix for queries and `"passage: "` prefix for documents. Missing these degrades retrieval significantly (Wang et al., 2022).

### Generation Service Choice: Google Flash

- Free tier: 15 RPM, 1M tokens/day — sufficient for demo and evaluation
- 1M context window: all 5 retrieved abstracts fit without truncation
- Native Turkish/English support — language mirroring for our users
- Simple SDK: single `pip install`, single env var
- Fallback: Groq `llama-3.1-8b-instant` (also free, faster inference)

### What I'd change with more time

1. **Chunking**: Split long abstracts into overlapping ~200-word chunks for finer-grained retrieval. Currently retrieving full abstracts which may dilute relevance signals.
2. **Re-ranking**: Add a cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a 3rd stage after RRF — much slower but significantly more accurate.
3. **FAISS**: At >10K documents, replace numpy dot product with FAISS ANN index for sub-linear search.
4. **Structured metadata filtering**: Filter by year (post-2020 guidelines only), journal impact factor.
5. **Answer faithfulness evaluation**: Measure citation accuracy (does the cited PMID actually support the claim?).

---

## BM25 Analysis

### Formula

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}$$

### k1 — Term Frequency Saturation

`k1` controls how much repeated occurrences of a query term in a document contribute to the score. The denominator ensures the TF contribution plateaus as term frequency grows.

| k1 value | Behavior |
|----------|----------|
| k1 = 0 | TF has no effect — pure IDF scoring. All documents with the term score equally regardless of how many times it appears. |
| k1 = 1.0 | Moderate saturation. A term appearing 5× gets about 83% of the score of infinite repetitions. |
| k1 = 1.5 | Our default. Medical abstracts often repeat key terms. Gives meaningful reward for repetition without over-rewarding. |
| k1 = 2.0 | Slow saturation. Heavily rewards documents where query terms appear many times. Can promote verbose documents. |

**Example**: Query = `"atrial fibrillation treatment"`
- k1=0.5: An abstract mentioning "atrial fibrillation" once scores nearly the same as one mentioning it 10 times
- k1=2.0: The abstract mentioning it 10 times scores significantly higher

**Choice**: k1=1.5 for PubMed abstracts. Key terms ("diabetes", "atrial fibrillation") appear 3-8 times in relevant abstracts — worth rewarding, but not excessively.

### b — Document Length Normalization

`b` controls how much to penalize long documents (which naturally accumulate more term matches).

| b value | Behavior |
|---------|----------|
| b = 0 | No length penalty. Long documents score higher just by having more text. |
| b = 0.5 | Partial normalization — moderate length penalty. |
| b = 0.75 | Standard default. Full normalization toward avgdl. |
| b = 1.0 | Full normalization. Completely levels the field between short and long documents. |

**Example**: Query = `"iron deficiency anemia pregnancy"`
- b=0: A 400-word abstract with 2 term matches may outscore a 200-word abstract with 2 matches
- b=1.0: Both score identically (length-normalized)

**Choice**: b=0.75. PubMed abstracts have constrained length (150-350 words), so length normalization matters less than in open-domain corpora. We keep the standard default.

### Title Boosting

Titles are repeated 3× during index construction as a simple upweighting mechanism. This is equivalent to giving title term matches 3× the raw term frequency contribution. Example:

- Query: `"celiac disease diagnosis"` 
- Without boost: An abstract mentioning "celiac" 5× in the body beats a title containing "celiac disease" 1×
- With 3× title boost: The title match counts as 3 occurrences → stronger signal for exact title matches

---

## RRF Analysis

### Formula

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ = {BM25, Semantic}, $\text{rank}_r(d)$ = 1-based position of document $d$ in ranker $r$'s list.

Documents absent from a ranker's list contribute 0 (rank treated as $+\infty$).

### What does k=60 do?

The constant `k` prevents rank-1 documents from dominating by smoothing the score distribution.

| k | Score for rank #1 | Score for rank #2 | Gap | Effect |
|---|---|---|---|---|
| k=0 | 1.000 | 0.500 | 0.500 | Extreme: rank-1 is 2× rank-2 |
| k=1 | 0.500 | 0.333 | 0.167 | Large gap |
| k=10 | 0.0909 | 0.0833 | 0.0076 | Moderate |
| **k=60** | **0.01639** | **0.01613** | **0.00026** | **Gentle: small differences** |
| k=1000 | 0.000999 | 0.000998 | 0.000001 | Flat: nearly uniform scores |

- **k=0**: Unstable — the rank-1 document in any list dominates completely. A single highly-ranked document in one list will push all others down.
- **k=1000**: Over-smoothed — all ranks are treated nearly equally. The fusion loses all rank information.
- **k=60** (Cormack et al. 2009 default): Balances both. Documents consistently ranked at the top get meaningfully higher scores, but no single ranker's rank-1 completely overrides the other. Empirically the best constant across diverse IR benchmarks.

### Why rank positions instead of raw scores?

**Scale incompatibility**: BM25 scores are unbounded and depend on corpus statistics (IDF values change as the corpus grows). Cosine similarity is bounded in [-1, 1] but clusters near 0.7-0.95 for semantic models. These two score distributions are not comparable.

**Normalization is brittle**: You could normalize both to [0, 1], but the normalization itself depends on the min/max scores in the current corpus — making it corpus-size-dependent and sensitive to outliers.

**Rank is universal**: Rank position 1 always means "this ranker considers this document best" regardless of the underlying scoring function. This is scale-invariant and robust. Whether BM25 gives a score of 12.3 or 0.85, rank 1 means the same thing.

---

## Evaluation

### Metric Choice

Since no ground-truth relevance labels exist, we use a cached **model-based judge**: a Google-hosted scoring pass assigns each (query, abstract) pair a score of 0 (not relevant), 1 (partially relevant), or 2 (highly relevant), creating pseudo-labels.

**Why MRR + nDCG@5 + P@5?**

| Metric | What it measures | Why relevant |
|--------|-----------------|--------------|
| **MRR** | Position of first relevant result | Doctors want the first result to be useful — clinical time pressure |
| **nDCG@5** | Quality of full ranked list, graded relevance, position-weighted | A highly relevant doc at rank 1 is better than rank 5 |
| **P@5** | Fraction of top-5 that are relevant (binary) | Simple baseline; comparable across systems |

**Limitations**: These scores reflect model-estimated relevance, not clinical utility. Results are relative comparisons between methods, not absolute quality scores.

### Measured Results

Using the committed `data/judge_cache.json` and the reproducible `python main.py evaluate` command:

| Method | P@5 | MRR | nDCG@5 |
|--------|-----|-----|--------|
| BM25 | 0.2800 | 0.5500 | 0.5960 |
| Semantic | 0.2800 | 0.5667 | 0.5901 |
| Hybrid RRF | 0.4000 | 0.7067 | 0.7702 |

Interpretation:

- **Hybrid RRF** is now best on all three offline metrics on the refreshed judged set.
- **P@5** improves most visibly with Hybrid, which means the top-5 list contains more relevant articles on average.
- **MRR** also improves with Hybrid, so the first useful article tends to appear earlier.
- **nDCG@5** is highest for Hybrid as well, which suggests the overall ranking quality improved rather than only the first result.

The main takeaway is that the improved corpus and term-aware hybrid retrieval work better together than either BM25 or semantic retrieval alone, especially on the harder multilingual and treatment-oriented queries.

### Evaluation Queries

| # | Query | Language | Challenge |
|---|-------|----------|-----------|
| Q1 | "What are the latest guidelines for managing type 2 diabetes?" | English | Broad, guideline-focused |
| Q2 | "Çocuklarda akut otitis media tedavisi nasıl yapılır?" | Turkish | Cross-lingual |
| Q3 | "Iron supplementation dosing for anemia during pregnancy" | English | Multi-concept, dosing specificity |
| Q4 | "Çölyak hastalığı tanı kriterleri nelerdir?" | Turkish | Cross-lingual, diagnostic |
| Q5 | "Antibiotic resistance patterns in community acquired pneumonia" | English | Exact corpus term match |

---

## Hardest Problem

**Turkish cross-lingual retrieval with BM25**

The core challenge: Turkish queries have zero lexical overlap with English PubMed abstracts. BM25 is purely lexical — it will return empty or random results for `"Çocuklarda akut otitis media tedavisi nasıl yapılır?"` because none of those tokens appear in the English corpus.

**How I solved it**:

1. **Primary fix**: `intfloat/multilingual-e5-small` handles this natively — the model's joint multilingual embedding space places Turkish "akut otitis media" and English "acute otitis media" near each other. Semantic retrieval works out of the box.

2. **Bonus improvement** (Notebook 03): A `QueryExpandingHybridRetriever` that:
   - Detects if the query is likely Turkish
   - Uses an automatic translation step to convert it to English
   - Runs BM25 on BOTH the original (Turkish) and translated (English) query
   - Fuses three ranked lists via RRF: BM25(TR), BM25(EN), Semantic(TR)
   
   This improves BM25 recall for Turkish queries without sacrificing semantic coverage.

3. **The RRF safety net**: Even if BM25 fails completely for a Turkish query (returns irrelevant results), RRF gracefully degrades to the semantic results — documents not in BM25's top-k get 0 contribution from BM25 and are ranked purely by their semantic score.

## Bonus Improvement

The bonus improvement is a query-expanding hybrid retriever demonstrated in `notebooks/03_rag_demo.ipynb`. It uses an automatic translation step to translate Turkish queries into English, then fuses:

- BM25 over the original Turkish query
- BM25 over the translated English query
- Semantic retrieval over the original Turkish query

This is helpful because BM25 depends on lexical overlap with English PubMed abstracts, while semantic retrieval handles Turkish naturally. The bonus path improves recall for Turkish queries when the translation step succeeds.

---

## Scenario Question

> Your team needs to benchmark a 70B open-source medical QA model. Your usual GPU provider doesn't have L40S available today. Your manager is busy all day. Results are needed by end of week.

**What I'd do:**

**Step 1 — Identify alternatives immediately (first 30 min)**

Check these GPU cloud providers in order of setup speed:
- **RunPod**: Spot GPU marketplace. An 8×A100 pod for a 70B model costs about $3-4/hr. Can be provisioned in minutes. Set max bid to 2× spot price to avoid preemption.
- **Vast**: Another flexible GPU marketplace worth checking if the first option is constrained.
- **Replicate**: Serverless inference with per-prediction billing.
- **Modal**: Managed GPU execution for short-lived benchmark runs.

**Step 2 — Choose based on benchmark requirements**

For *benchmarking* (not production), I need:
- Reproducibility (same hardware, same settings across runs)
- Control over quantization (FP16 vs INT8 vs INT4 — this matters for medical QA accuracy)
- Ability to run the same evaluation set multiple times

**Best choice**: RunPod with a manually provisioned `llama.cpp` or another tensor-parallel inference server on 2×A100 80GB. Gives full control. 70B in FP16 requires ~140GB VRAM, so 2×A100 80GB works with tensor parallelism.

**Alternative if budget is tight**: A serverless endpoint serving `Meta-Llama-3.1-70B-Instruct`. Quick to set up, but less control over quantization.

**Step 3 — Set up benchmark framework while provisioning**

While waiting for GPU allocation (5-15 min), prepare:
- Evaluation dataset (MedQA, PubMedQA, or internal QA pairs)
- Evaluation script with `lm-evaluation-harness` or custom metrics
- Run parameters: temperature=0, max_tokens=256, system prompt

**Step 4 — Document decision without blocking on manager**

Leave a Slack message for the manager: "L40S unavailable, using RunPod 2×A100 80GB at ~$3.50/hr. Estimated cost for full benchmark: ~$20. Proceeding unless you message back by noon."

This respects their time while keeping the project on track and creating an audit trail.

**Trade-offs**:
- L40S vs A100 for benchmarking: L40S is faster for inference (Ada Lovelace vs Ampere), but benchmark *accuracy* metrics don't depend on hardware. Latency metrics will differ.
- Managed inference (Together/Replicate) vs self-managed (RunPod): Managed is faster to set up but opaque about quantization. For medical QA benchmarking, I'd prefer self-managed to control quantization precision.
- Cost: A full benchmark run of a 70B model on 500 QA pairs takes ~2-3 hrs → $10-15 total on RunPod.

---

## Project Structure

```
doctorfollowcase/
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example               # API key template
├── .gitignore
├── medical_terms.csv          # 10 medical terms for pipeline
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Part 1: PubMed fetcher
│   ├── retrieval.py           # Part 2: BM25 + Semantic + RRF
│   ├── evaluation.py          # Reproducible retrieval evaluation
│   └── rag.py                 # Part 3: RAG generation
├── data/
│   ├── pubmed_articles.json   # Pipeline output (committed)
│   ├── corpus_summary.json    # Term-level corpus summary and selected PMIDs
│   ├── judge_cache.json       # Cached relevance labels
│   └── evaluation_results.json # Evaluation report artifact
├── embeddings/
│   ├── doc_embeddings.npy     # Cached semantic embeddings
│   └── doc_pmids.json         # Embedding cache index
└── notebooks/
    ├── 01_data_pipeline.ipynb
    ├── 02_retrieval_analysis.ipynb
    └── 03_rag_demo.ipynb
```
