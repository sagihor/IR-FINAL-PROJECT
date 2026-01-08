
# General Engine (server) Overview

The search engine is implemented as a Flask server in `search_frontend.py`.
The server follows a modular design in which the search endpoint does not perform retrieval by itself, but instead aggregates and fuses results from several specialized endpoints. Each endpoint is responsible for a different relevance signal, and the final ranking is produced by a weighted combination of these signals, followed by light boosting using global importance measures.

# Endpoints and Their Role in the Search Pipeline:

1. search_body : Body-based retrieval (TF-IDF + Cosine)

This endpoint performs classical information retrieval over the article body text.
It uses the inverted index of the body text of the corpus and computes TF-IDF vectors for the query and documents, followed by cosine similarity.

- Only posting lists of query terms are accessed (sparse computation).
- Term frequency is normalized as 1 + log(tf).
- Document norms are precomputed offline and retrieved on demand.
- The endpoint returns the top 100 documents by cosine score by (doc_id, title) list.

In addition to the endpoint itself, an internal helper function is used by /search to retrieve a larger candidate set (up to 1000 documents).
This helper returns only (doc_id, score) pairs and is used for fusion with other signals.

2. search_title : Title-based retrieval

This endpoint Returns ALL (not just top 100) search results that contain a query word in the title of articles,
ordered in descending order of the number of query words that appear in the title.

- Uses the title inverted index of the whole corpus.
- The endpoint returns all matching documents as (doc_id, title) list.
- Does not perform TF-IDF, it is purely coverage-based.


For the main search pipeline, an internal scoring function returns only (doc_id, score) pairs.
Titles are fetched only at the wrapper function or search function the final stage, after the top results are selected.

3. search_anchor : Anchor-text retrieval

This endpoint operates on anchor texts of links.
Returns all (not just top 100) search results that contain a query word in the anchor text of articles, ordered in descending order
of the number of query words that appear in anchor text linking to the page.


- Uses the anchor-text inverted index.
- Ranking is based on the number of distinct query terms appearing in anchors.
- The endpoint returns all matching documents as (doc_id, title) list.

Scores are passed to search function as (doc_id, score) pairs without titles.
Titles are fetched only at the wrapper function or search function the final stage, after the top results are selected.

4. /get_pagerank and /get_pageview : Popularity signals

These endpoints return external importance measures:
- PageRank values.
- PageViews counts.

They are not used as standalone search mechanisms, but as light boosting signals inside search.
Boosting is applied only to the top 500 candidates after textual ranking.

5. BM25 Integration:

The system includes a dedicated BM25 implementation.

BM25 is not exposed as a standalone endpoint.
Instead, it is integrated directly into the main search function:

- search_body remains TF-IDF + cosine, as required.
- search uses an internal BM25 scorer for body relevance.

The BM25 implementation:
- Uses posting lists from the body index.
- Relies on document length mapping and average document length.
- Applies the standard BM25 formula with parameters k1 and b.

BM25 was selected for the final version because it showed more stable behavior on multi-term queries and partial textual overlap.

# Fusion Strategy in /search

The search endpoint performs:

1. Retrieve scores from:
   - Body (BM25)
   - Title (coverage)
   - Anchor (coverage)

2. Normalize each score source independently.

3. Apply weighted fusion:
   - Body: 40%
   - Title: 40%
   - Anchor: 20%

4. Select the top 500 candidates.

5. Apply light boosts using:
   - PageRank (log-scaled, normalized)
   - PageViews (log-scaled, normalized)

6. Sort and return the top 100 documents.

Titles are retrieved only at the final step.

# Data Files Used by the Server (Generated in GCP)

The server relies on precomputed files generated in GCP and downloaded locally at startup:

- Body, Title, Anchor inverted indexes.
- titles.tsv + titles.idx : doc_id to title mapping.
- pagerank.tsv + pagerank.idx : doc_id to PageRank mapping.
- pageviews.tsv + pageviews.idx : doc_id to PageViews mapping.
- dl_body.tsv : doc_id to document length mapping.
- avgdl_body.txt : average document length.
- docnorms_body.tsv + docnorms_body.idx : document norms for cosine similarity.

All retrieval and scoring is done locally, without accessing GCS at query time.

# Memory Management and TSV + IDX Pattern

At server startup, all required files are downloaded once from the public GCS bucket.
During query processing, no network access is performed.

Loading large mappings fully into RAM was not scalable.
Therefore, a TSV + IDX architecture is used:

- TSV stores the actual data (doc_id<TAB>value).
- IDX stores, for each doc_id, the byte offset and record length in the TSV.

For every pair, two core operations are used:
1. Binary search in the memory-mapped IDX.
2. Targeted read from the TSV using seek + read.

This allows:
- O(log n) lookup in the index.
- O(1) disk access to the exact record.
- No need to load millions of entries into memory.

# Repository Structure

The GitHub repository contains:
- search_frontend.py : the main Flask server.
- inverted_index_gcp.py : index reader provided by the course staff.
- Scripts and notebooks for building indexes in GCP.
- GCP_CREATION_NOTEBOOK: the notebook of generating the indexes and files in the GCP.

Large index files are not stored in the repository.
They are hosted in a public Google Storage Bucket, and links are provided in the report submitted via Moodle.

