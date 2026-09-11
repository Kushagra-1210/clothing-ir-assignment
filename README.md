# Clothing Information Retrieval System

## Group Members
- Kushagra Bansal (2410110184)
- Bharat Parashar (2310110079)

An information retrieval (IR) system built for **Assignment 1 (CSD358)**. It
indexes a corpus of 100 clothing product descriptions and supports ranked
free-text search (Vector Space Model, lnc.ltc weighting), exact phrase
search, and ordered proximity search over a positional index.

## Project Files

| File | Description |
|------|-------------|
| `corpus_100.txt` | Input corpus: 100 clothing documents in XML-like format |
| `clothing_ir_model.py` | The IR system: indexer, positional index, search engine, interactive CLI |
| `run_tests.py` | Part E test runner — free-text, phrase, proximity, and OOV queries |
| `test_results.txt` | Output of `run_tests.py` (generated; see "How to Run") |
| `dictionary_output.txt` | Full dictionary/inverted index dump (generated via the `export` command) |
| `positional_index_output.txt` | Full positional index dump (generated via the `export` command) |

Each document in the corpus has four fields:

```
<DOC>
  <DOCID>     unique identifier (D001 ... D100)
  <CATEGORY>  product type (T-Shirt, Shirt, Jeans, Kurta, Saree, Dress, Hoodie, Jacket, Leggings, Sweatshirt)
  <TITLE>     product name
  <TEXT>      product description
</DOC>
```

There are 10 documents per category = 100 documents total.

## How It Works

### 1. Preprocessing (Part A)
Each document's `TITLE` and `TEXT` are combined, then:
- Lowercased
- Punctuation/special characters stripped (`[^a-z0-9\s'-]` removed)
- Split into tokens
- **Stop-words removed** — a hand-curated list of ~90 common English function
  words (articles, prepositions, pronouns, auxiliary verbs, conjunctions;
  see `tokenize()`). We use a custom list rather than a large standard one
  (e.g. NLTK's) because the corpus vocabulary is small and domain-specific;
  a long generic list risks stripping words that matter for clothing search
  (e.g. some standard lists include words like "will" or "may" which we do
  want removed, but also risk removing short domain terms). The list is
  applied identically to documents and queries.
- Single-character tokens dropped
- **Porter stemming** applied (self-contained implementation, no external
  library) — e.g. "checked" → "check", "breathable" → "breathabl"

After this pipeline, the corpus yields **122 unique terms** across 100
documents.

### 2. Inverted Index
`term -> set of docIDs`, built during `build_index()`. Powers Boolean search
and candidate-set lookup for ranked retrieval.

### 3. Positional Index (Part C)
`term -> {docID: [positions]}`, built alongside the inverted index in the
same pass. Term frequency for a (term, doc) pair is simply the length of its
position list. This enables:
- **Exact phrase search** (`phrase [query]`) — terms must be adjacent and in
  order (implemented as proximity search with k=1).
- **Ordered proximity search** (`near [k] [query]`) — terms must occur in
  order, with a gap of at most k token positions between consecutive query
  terms. k defaults to 4 if omitted, e.g. `near 3 cotton shirt` uses k=3.

Both commands print the actual matched positions for the top result as
evidence that retrieval is driven by the positional index and not just
"do both terms occur somewhere in this document".

### 4. Vector Space Model — lnc.ltc (Part B, the assignment's required scheme)
This is the **default free-text search** (just type a query with no command
prefix). Implemented in `lnc_ltc_search()`:
- **Document weight ("lnc"):** `1 + log10(tf)` for tf > 0, **no idf**, then
  the document vector is cosine-normalized.
- **Query weight ("ltc"):** `(1 + log10(tf)) * log10(N/df)`, then the query
  vector is cosine-normalized (N = 100).
- Because both vectors are pre-normalized, cosine similarity is just their
  dot product.
- Returns up to 10 results, sorted by decreasing similarity; **ties are
  broken by increasing document ID**, as the assignment specifies.

### 5. Extra ranking methods (novelty)
These are not required by the assignment but are included as additional,
clearly-separated retrieval methods for comparison:
- **`augtfidf`** — an augmented-tf / BM25-style-idf cosine search (this was
  the original default before the assignment's exact lnc.ltc scheme was
  added).
- **`bm25`** — Okapi BM25 (k1=1.5, b=0.75).
- **`jaccard`** — Jaccard similarity between query and document term sets.
- **`and` / `or` / `not`** — Boolean retrieval over the inverted index.
- **`expand`** — query expansion via term co-occurrence, then lnc.ltc-style
  ranking.
- **`eval`** — precision/recall/F1/AP/NDCG evaluation against 5 hand-labeled
  test queries, compared across TF-IDF, BM25, and Jaccard.

### 6. Index export (deliverable)
The `export` command writes:
- `dictionary_output.txt` — every term, its document frequency, and its full
  postings list `(docID, tf)`.
- `positional_index_output.txt` — every term, its document frequency, and
  its full postings list `(docID, tf, [positions])`.

## How to Run

```bash
cd clothing-ir-assignment
python clothing_ir_model.py
```

(On Windows, `python` may need to be `py` depending on your install.)

This will:
1. Load and parse all 100 documents.
2. Build the inverted index, positional index, and lnc.ltc vectors.
3. Print corpus statistics (document/term counts, category distribution,
   top-20 terms by IDF).
4. Start the interactive prompt.

### Interactive Commands

| Command | Description |
|---------|-------------|
| `your query here` | **lnc.ltc cosine similarity search (default, Part B)** |
| `augtfidf your query` | Augmented-TF / BM25-idf cosine search (extra) |
| `bm25 your query` | BM25 ranked search (extra) |
| `jaccard your query` | Jaccard similarity search (extra) |
| `phrase your query` | Exact phrase search (terms adjacent, in order) |
| `near your query` | Proximity search, default k=4 |
| `near k your query` | Proximity search with a specific k, e.g. `near 3 cotton shirt` |
| `and` / `or` / `not term1 term2` | Boolean search (extra) |
| `expand your query` | Query expansion + ranked search (extra) |
| `cat category_name` | List documents in a category, e.g. `cat kurta` |
| `stats` | Show corpus statistics |
| `export` | Save the full dictionary and positional index to files |
| `history` | Show query history for this session |
| `eval` | Run P/R/F1/MAP/NDCG evaluation (extra) |
| `help` | Show available commands |
| `quit` | Exit |

### Example session

```
Query> black cotton t-shirt
[lnc.ltc results with docID, category, title, score]

Query> phrase cotton shirt
[exact phrase matches + matched positions for the top result]

Query> near 3 winter wear
[proximity matches within k=3 token positions]

Query> export
Dictionary exported to dictionary_output.txt
Positional index exported to positional_index_output.txt

Query> quit
```

## Part E — Running the Test Suite

```bash
python run_tests.py
```

This runs the assignment's required test mix — 10 free-text queries, 6 exact
phrase queries, 3 proximity queries at different k values, and 1
out-of-vocabulary query — and writes everything, plus a written comparison
of cases where positional retrieval changes the result set, to
`test_results.txt`.

## Summary of Components

```
clothing_ir_model.py
├── PorterStemmer                self-contained English stemmer
├── Document (dataclass)         holds one indexed document
├── ClothingIRModel               main engine
│   ├── load_corpus()             parse corpus_100.txt
│   ├── tokenize()                preprocessing + stemming pipeline
│   ├── build_index()             inverted index, positional index,
│   │                             lnc.ltc doc vectors, BM25-idf doc vectors
│   ├── lnc_ltc_search()          REQUIRED: exact lnc.ltc VSM ranking (Part B)
│   ├── phrase_search()           exact phrase / ordered proximity (Part C)
│   ├── print_term_positions()    positional-index evidence for a match (Part D)
│   ├── export_dictionary()       dictionary deliverable
│   ├── export_positional_index() positional index deliverable
│   ├── tfidf_search()            augmented-tf/BM25-idf ranking (extra)
│   ├── bm25_search()             BM25 ranking (extra)
│   ├── jaccard_search()          Jaccard ranking (extra)
│   ├── boolean_search()          AND/OR/NOT retrieval (extra)
│   ├── query_expansion()         co-occurrence expansion (extra)
│   ├── evaluate() / evaluate_all() P/R/F1/AP/NDCG (extra)
│   └── print_stats() / print_query_history()
└── interactive_mode()            command-line UI
run_tests.py                      Part E test runner
```

## Code Documentation

The module includes docstrings for every class and public method, and
inline comments wherever a formula (lnc.ltc weighting, BM25 idf, Porter
stemmer measure/CVC rules) is easy to misread without context.