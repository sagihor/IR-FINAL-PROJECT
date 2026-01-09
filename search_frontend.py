# =========================
# Imports
# =========================

import os
import re
import math
import pickle
import mmap
import struct
from collections import defaultdict, Counter
from flask import Flask, request, jsonify
from google.cloud import storage
from inverted_index_gcp import InvertedIndex
import threading
from array import array
import heapq



_titles_lock = threading.Lock()


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# =========================
# Config (GCS + project paths)
# =========================
GCS_BASE_DIR = "ir_project_indexes"
GCS_BUCKET_NAME = "ir-sagi-bucket"
# GCS folders for each index type
GCS_BODY_DIR   =  f"{GCS_BASE_DIR}/body_index"
GCS_TITLE_DIR  = f"{GCS_BASE_DIR}/title_index"
GCS_ANCHOR_DIR = f"{GCS_BASE_DIR}/anchor_index"

# =========================
# Local dirs to keep the posting lists and the tsv and idx of titles locally
# =========================

# We cache everything under ~/ir_local_cache so:
# - startup downloads once
# - query-time does not hit GCS (faster + more stable)
LOCAL_BASE_DIR = os.path.join(os.path.expanduser("~"), "ir_local_cache")
os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

LOCAL_INDEX_DIR  = os.path.join(LOCAL_BASE_DIR, "indexes")
LOCAL_TITLES_MAPPING_DIR = os.path.join(LOCAL_BASE_DIR, "titles")
# Ensure directories exist
os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
os.makedirs(LOCAL_TITLES_MAPPING_DIR, exist_ok=True)

# Local dirs
LOCAL_BODY_DIR   = os.path.join(LOCAL_INDEX_DIR, "body_index")
LOCAL_TITLE_DIR  = os.path.join(LOCAL_INDEX_DIR, "title_index")
LOCAL_ANCHOR_DIR = os.path.join(LOCAL_INDEX_DIR, "anchor_index")


# =========================
# GCS client (create once)
# =========================

gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)



# =========================
# Download indexes locally to VM once at startup
# =========================

# Download all blobs under a given GCS "folder" prefix into a local directory.
def download_prefix_to_local(gcs_prefix: str, local_dir: str):
    # - Query-time reads should be local disk reads (fast)
    # - Avoid repeated GCS network calls
    gcs_prefix = gcs_prefix.rstrip("/") + "/"
    os.makedirs(local_dir, exist_ok=True)
    for blob in gcs_bucket.list_blobs(prefix=gcs_prefix):
        if blob.name.endswith("/"):
            continue
        # Relative path within the prefix
        rel = blob.name[len(gcs_prefix):]
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)

# Download index folders (postings + metadata) once at startup
download_prefix_to_local(GCS_BODY_DIR, LOCAL_BODY_DIR)
download_prefix_to_local(GCS_TITLE_DIR, LOCAL_TITLE_DIR)
download_prefix_to_local(GCS_ANCHOR_DIR, LOCAL_ANCHOR_DIR)


# =========================
# Stopwords (download once at startup)
# =========================

STOPWORDS_BLOB = f"{GCS_BASE_DIR}/stopwords_en.txt"

with gcs_bucket.blob(STOPWORDS_BLOB).open("r") as f:
    english_stopwords = frozenset(line.strip() for line in f if line.strip())

# Token regex
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
# Additional corpus-specific stopwords
corpus_stopwords = frozenset([
    "category", "references", "also", "external", "links",
    "may", "first", "see", "history", "people", "one", "two",
    "part", "thumb", "including", "second", "following",
    "many", "however", "would", "became"
])
# Final stopwords set
all_stopwords = english_stopwords.union(corpus_stopwords)



# =========================
# Tokenizer: Lowercase + regex tokenization + stopword filtering
# =========================
def tokenize(text: str):
    tokens = [m.group() for m in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]



# ==========================================================
# Titles: TSV + IDX (download once at startup, local mmap)
# ==========================================================

# key idea:
# - Fast lookup by doc_id without loading all titles into RAM
# - Use memory-mapped IDX for O(log n) binary search
# - Use seek+read on TSV to fetch exact line segment

# Paths in GCS
TITLES_TSV_GCS = f"{GCS_BASE_DIR}/titles.tsv"
TITLES_IDX_GCS = f"{GCS_BASE_DIR}/titles.idx"

# Local paths on the server VM
LOCAL_TITLES_TSV = os.path.join(LOCAL_TITLES_MAPPING_DIR, "titles.tsv")
LOCAL_TITLES_IDX = os.path.join(LOCAL_TITLES_MAPPING_DIR, "titles.idx")
# Download once at server startup (not at query time)
if GCS_BUCKET_NAME is not None:
    if not os.path.exists(LOCAL_TITLES_TSV):
        gcs_bucket.blob(TITLES_TSV_GCS).download_to_filename(LOCAL_TITLES_TSV)
    if not os.path.exists(LOCAL_TITLES_IDX):
        gcs_bucket.blob(TITLES_IDX_GCS).download_to_filename(LOCAL_TITLES_IDX)

# IDX record: (doc_id:uint32, offset:uint64, length:uint32) = 16 bytes
REC = struct.Struct("<I Q I")
REC_SIZE = REC.size

# Open files and mmap the index
_titles_tsv_f = open(LOCAL_TITLES_TSV, "rb")
_titles_idx_f = open(LOCAL_TITLES_IDX, "rb")
_titles_idx_mm = mmap.mmap(_titles_idx_f.fileno(), 0, access=mmap.ACCESS_READ)

# Number of documents
_titles_count = _titles_idx_mm.size() // REC_SIZE


# Binary search in the sorted index -> (offset, length)
def _idx_get(doc_id: int):
    lo, hi = 0, _titles_count - 1
    doc_id = int(doc_id)

    while lo <= hi:
        mid = (lo + hi) // 2
        base = mid * REC_SIZE
        mid_id, off, ln = REC.unpack_from(_titles_idx_mm, base)
        if mid_id == doc_id:
            return off, ln
        if mid_id < doc_id:
            lo = mid + 1
        else:
            hi = mid - 1

    return None

# Fetch titles for multiple doc_ids and return {doc_id: title}
def get_titles(doc_ids):
    out = {}
    seen = set()
    for x in doc_ids:
        if x is None:
            continue
        doc_id = int(x)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        # Find TSV location for this doc_id
        hit = _idx_get(doc_id)
        if not hit:
            continue
        off, ln = hit
        if ln <= 0:
            continue
        # Thread-safe read: protect seek/read on shared file handle
        with _titles_lock:
            _titles_tsv_f.seek(off)
            line = _titles_tsv_f.read(ln)
        # Parse "id<TAB>title\n"
        tab = line.find(b"\t")
        if tab < 0:
            continue
        title = line[tab + 1:].rstrip(b"\n").decode("utf-8", errors="replace")
        out[doc_id] = title

    return out

# Total number of documents (used for IDF)
DOC_COUNT = _titles_count


# ==========================================================
# PageRank: TSV + IDX (download once at startup, local mmap)
# ==========================================================

# Same pattern as Titles:
# - binary search in IDX (mmap)
# - fetch value from TSV using offset/length
# - lock around seek/read

# Paths in GCS
PAGERANK_TSV_GCS = f"{GCS_BASE_DIR}/pagerank.tsv"
PAGERANK_IDX_GCS = f"{GCS_BASE_DIR}/pagerank.idx"

# Local paths on the server VM
LOCAL_PAGERANK_TSV = os.path.join(LOCAL_TITLES_MAPPING_DIR, "pagerank.tsv")
LOCAL_PAGERANK_IDX = os.path.join(LOCAL_TITLES_MAPPING_DIR, "pagerank.idx")

# Download once at server startup (not at query time)
if GCS_BUCKET_NAME is not None:
    if not os.path.exists(LOCAL_PAGERANK_TSV):
        gcs_bucket.blob(PAGERANK_TSV_GCS).download_to_filename(LOCAL_PAGERANK_TSV)
    if not os.path.exists(LOCAL_PAGERANK_IDX):
        gcs_bucket.blob(PAGERANK_IDX_GCS).download_to_filename(LOCAL_PAGERANK_IDX)

# Same record layout: doc_id:uint32 | offset:uint64 | length:uint32
PR_REC = struct.Struct("<I Q I")
PR_REC_SIZE = PR_REC.size

_pagerank_tsv_f = open(LOCAL_PAGERANK_TSV, "rb")
_pagerank_idx_f = open(LOCAL_PAGERANK_IDX, "rb")
_pagerank_idx_mm = mmap.mmap(_pagerank_idx_f.fileno(), 0, access=mmap.ACCESS_READ)
_pagerank_count = _pagerank_idx_mm.size() // PR_REC_SIZE

_pagerank_lock = threading.Lock()  # thread-safe


def _pagerank_idx_get(doc_id: int):
    #  Binary search in pagerank.idx to find (offset,length)
    lo, hi = 0, _pagerank_count - 1
    doc_id = int(doc_id)

    while lo <= hi:
        mid = (lo + hi) // 2
        base = mid * PR_REC_SIZE
        mid_id, off, ln = PR_REC.unpack_from(_pagerank_idx_mm, base)

        if mid_id == doc_id:
            return off, ln
        if mid_id < doc_id:
            lo = mid + 1
        else:
            hi = mid - 1

    return None

def pagerank_get(doc_id: int, default: float = 0.0) -> float:
    #  Fetch PageRank for a doc_id.
    hit = _pagerank_idx_get(doc_id)
    if not hit:
        return default
    off, ln = hit
    if ln <= 0:
        return default

    # Thread-safe seek/read
    with _pagerank_lock:
        _pagerank_tsv_f.seek(off)
        line = _pagerank_tsv_f.read(ln)

    tab = line.find(b"\t")
    if tab < 0:
        return default

    # Expected format: b"id\tpagerank\n"
    try:
        return float(line[tab + 1:].strip())
    except Exception:
        return default



# ==========================================================
# PageViews: TSV + IDX (download once at startup, local mmap)
# ==========================================================

# Paths in GCS
PAGEVIEWS_TSV_GCS = f"{GCS_BASE_DIR}/pageviews.tsv"
PAGEVIEWS_IDX_GCS = f"{GCS_BASE_DIR}/pageviews.idx"

# Local paths on the server VM
LOCAL_PAGEVIEWS_TSV = os.path.join(LOCAL_TITLES_MAPPING_DIR, "pageviews.tsv")
LOCAL_PAGEVIEWS_IDX = os.path.join(LOCAL_TITLES_MAPPING_DIR, "pageviews.idx")

# Download once at server startup (not at query time)
if GCS_BUCKET_NAME is not None:
    if not os.path.exists(LOCAL_PAGEVIEWS_TSV):
        gcs_bucket.blob(PAGEVIEWS_TSV_GCS).download_to_filename(LOCAL_PAGEVIEWS_TSV)
    if not os.path.exists(LOCAL_PAGEVIEWS_IDX):
        gcs_bucket.blob(PAGEVIEWS_IDX_GCS).download_to_filename(LOCAL_PAGEVIEWS_IDX)

# IDX record: (doc_id:uint32, offset:uint64, length:uint32) = 16 bytes (כמו titles/pagerank)
PV_REC = struct.Struct("<I Q I")
PV_REC_SIZE = PV_REC.size

_pageviews_tsv_f = open(LOCAL_PAGEVIEWS_TSV, "rb")
_pageviews_idx_f = open(LOCAL_PAGEVIEWS_IDX, "rb")
_pageviews_idx_mm = mmap.mmap(_pageviews_idx_f.fileno(), 0, access=mmap.ACCESS_READ)
_pageviews_count = _pageviews_idx_mm.size() // PV_REC_SIZE

_pageviews_lock = threading.Lock()  # thread-safe

def _pageviews_idx_get(doc_id: int):
    # Binary search in pageviews.idx to find (offset,length)
    lo, hi = 0, _pageviews_count - 1
    doc_id = int(doc_id)

    while lo <= hi:
        mid = (lo + hi) // 2
        base = mid * PV_REC_SIZE
        mid_id, off, ln = PV_REC.unpack_from(_pageviews_idx_mm, base)

        if mid_id == doc_id:
            return off, ln
        if mid_id < doc_id:
            lo = mid + 1
        else:
            hi = mid - 1

    return None

def pageviews_get(doc_id: int, default: int = 0) -> int:
    # etch PageViews for a doc_id
    hit = _pageviews_idx_get(doc_id)
    if not hit:
        return default
    off, ln = hit
    if ln <= 0:
        return default

    with _pageviews_lock:
        _pageviews_tsv_f.seek(off)
        line = _pageviews_tsv_f.read(ln)

    tab = line.find(b"\t")
    if tab < 0:
        return default

    # line: b"id\tviews\n"
    try:
        return int(line[tab + 1:].strip())
    except Exception:
        return default



# ==========================================================
# Body DL: TSV + IDX (download once at startup, local mmap)
# ==========================================================

DL_TSV_GCS = f"{GCS_BASE_DIR}/dl_body.tsv"
#DL_IDX_GCS = f"{GCS_BASE_DIR}/dl_body.idx"

LOCAL_DL_TSV = os.path.join(LOCAL_TITLES_MAPPING_DIR, "dl_body.tsv")
#LOCAL_DL_IDX = os.path.join(LOCAL_TITLES_MAPPING_DIR, "dl_body.idx")

# Download once at server startup
if GCS_BUCKET_NAME is not None:
    if not os.path.exists(LOCAL_DL_TSV):
        gcs_bucket.blob(DL_TSV_GCS).download_to_filename(LOCAL_DL_TSV)
    #if not os.path.exists(LOCAL_DL_IDX):
     #   gcs_bucket.blob(DL_IDX_GCS).download_to_filename(LOCAL_DL_IDX)


# ==========================================================
# AVGDL
# ==========================================================
AVGDL_BLOB = f"{GCS_BASE_DIR}/avgdl_body.txt"
with gcs_bucket.blob(AVGDL_BLOB).open("r") as f:
    AVGDL_BODY = float(f.read().strip())


# ==========================================================
# Body DocNorms: TSV + IDX (download once at startup, local mmap)
# ==========================================================

DOCNORMS_TSV_GCS = f"{GCS_BASE_DIR}/docnorms_body.tsv"
DOCNORMS_IDX_GCS = f"{GCS_BASE_DIR}/docnorms_body.idx"

LOCAL_DOCNORMS_TSV = os.path.join(LOCAL_TITLES_MAPPING_DIR, "docnorms_body.tsv")
LOCAL_DOCNORMS_IDX = os.path.join(LOCAL_TITLES_MAPPING_DIR, "docnorms_body.idx")

# Download once at server startup (not at query time)
if GCS_BUCKET_NAME is not None:
    if not os.path.exists(LOCAL_DOCNORMS_TSV):
        gcs_bucket.blob(DOCNORMS_TSV_GCS).download_to_filename(LOCAL_DOCNORMS_TSV)
    if not os.path.exists(LOCAL_DOCNORMS_IDX):
        gcs_bucket.blob(DOCNORMS_IDX_GCS).download_to_filename(LOCAL_DOCNORMS_IDX)

# same IDX record format: (doc_id:uint32, offset:uint64, length:uint32)
NORM_REC = struct.Struct("<I Q I")
NORM_REC_SIZE = NORM_REC.size

_docnorms_tsv_f = open(LOCAL_DOCNORMS_TSV, "rb")
_docnorms_idx_f = open(LOCAL_DOCNORMS_IDX, "rb")
_docnorms_idx_mm = mmap.mmap(_docnorms_idx_f.fileno(), 0, access=mmap.ACCESS_READ)
_docnorms_count = _docnorms_idx_mm.size() // NORM_REC_SIZE

_docnorms_lock = threading.Lock()


def _docnorms_idx_get(doc_id: int):
    lo, hi = 0, _docnorms_count - 1
    doc_id = int(doc_id)
    while lo <= hi:
        mid = (lo + hi) // 2
        base = mid * NORM_REC_SIZE
        mid_id, off, ln = NORM_REC.unpack_from(_docnorms_idx_mm, base)
        if mid_id == doc_id:
            return off, ln
        if mid_id < doc_id:
            lo = mid + 1
        else:
            hi = mid - 1
    return None

def docnorm_get(doc_id: int, default: float = 1.0) -> float:
    hit = _docnorms_idx_get(doc_id)
    if not hit:
        return default
    off, ln = hit
    if ln <= 0:
        return default

    with _docnorms_lock:
        _docnorms_tsv_f.seek(off)
        line = _docnorms_tsv_f.read(ln)

    tab = line.find(b"\t")
    if tab < 0:
        return default
    try:
        return float(line[tab + 1:].strip())
    except Exception:
        return default



# =========================
# Load indexes (once, global)
# =========================

# Read index metadata into memory; posting lists will be read from local files on demand.
# bucket_name=None because we are reading from LOCAL_* directories (not GCS) after download.

body_index   = InvertedIndex.read_index(LOCAL_BODY_DIR, "body",   bucket_name=None)
title_index  = InvertedIndex.read_index(LOCAL_TITLE_DIR,"title", bucket_name=None)
anchor_index = InvertedIndex.read_index(LOCAL_ANCHOR_DIR,"anchor",bucket_name=None)



# =========================
# FIX: posting_locs may contain GCS prefixes; when reading locally we need only the basename
# =========================
def _normalize_posting_locs(idx):
    idx.posting_locs = {
        term: [(os.path.basename(fname), offset) for fname, offset in locs]
        for term, locs in idx.posting_locs.items()
    }


_normalize_posting_locs(body_index)
_normalize_posting_locs(title_index)
_normalize_posting_locs(anchor_index)


# ==========================================================
# Body DL in RAM (FAST O(1) lookup by doc_id)
# ==========================================================

def build_dl_array(local_tsv_path: str) -> array:
    # Build a dense array DL where DL[doc_id] = document length.
    #Uses uint32 array to keep RAM small (~4 bytes per doc_id).

    # find max doc_id (first pass)
    max_id = 0
    with open(local_tsv_path, "rb") as f:
        for line in f:
            tab = line.find(b"\t")
            if tab < 0:
                continue
            try:
                doc_id = int(line[:tab])
                if doc_id > max_id:
                    max_id = doc_id
            except:
                continue


    # allocate dense array (uint32) initialized to 0
    dl = array("I", [0]) * (max_id + 1)

    # fill values (second pass)
    with open(local_tsv_path, "rb") as f:
        for line in f:
            tab = line.find(b"\t")
            if tab < 0:
                continue
            try:
                doc_id = int(line[:tab])
                val = int(line[tab + 1:].strip())
                if 0 <= doc_id < len(dl):
                    dl[doc_id] = val
            except:
                continue

    return dl



# Build once at startup
DL_ARR = build_dl_array(LOCAL_DL_TSV)

def dl_get_fast(doc_id: int, default: int) -> int:
    # O(1) doc length lookup from RAM array (fallback to default).
    doc_id = int(doc_id)
    if 0 <= doc_id < len(DL_ARR):
        v = DL_ARR[doc_id]
        if v > 0:
            return int(v)
    return int(default)




@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    #use BM25 for body scores, not the regular body_search
    body_scores = dict(search_body_bm25(query))
    title_scores = dict(search_title_scores(query))
    anchor_scores = dict(search_anchor_scores(query))

    # Scores normalization
    # Goal: prevent any one source (body/title/anchor) from dominating due to scale.
    # For title/anchor:
    # - divide by query coverage (q_len) to reward matching more distinct query tokens
    # - then max-normalize within the query

    q_len = len(set(tokenize(query))) or 1

    for d in (title_scores, anchor_scores):
        # Coverage normalization: scale down scores as query has more unique terms
        for doc_id in list(d.keys()):
            d[doc_id] /= q_len
        # Max-normalize to [0,1] within that query (if non-empty)
        if d:
            mx = max(d.values()) or 1.0
            for doc_id in list(d.keys()):
                d[doc_id] /= mx

    if body_scores:
        mx = max(body_scores.values()) or 1.0
        for doc_id in body_scores:
            body_scores[doc_id] /= mx

    # Weighted fusion for textual results
    final = defaultdict(float)
    # Current weights:
    # - body: 0.40
    # - title: 0.40
    # - anchor: 0.20
    for doc_id, score in body_scores.items():
        final[doc_id] += 0.40 * score
    for doc_id, score in title_scores.items():
        final[doc_id] += 0.40 * score
    for doc_id, score in anchor_scores.items():
        final[doc_id] += 0.20 * score

    if not final:
        return jsonify([])

    # Candidate pruning before PR/PV
    # To avoid calling pagerank/pageviews for too many docs, take top 500 by current score
    top_candidates = heapq.nlargest(500, final.items(), key=lambda x: x[1])
    candidate_ids = [doc_id for doc_id, _ in top_candidates]


    # PageRank + PageViews boosts:

    # Use log1p to compress heavy-tailed distributions.
    pagerank_dict = {doc_id: math.log1p(pagerank_get(doc_id, 0.0)) for doc_id in candidate_ids}
    pageviews_dict = {doc_id: math.log1p(pageviews_get(doc_id, 0)) for doc_id in candidate_ids}

    # Normalize each signal so its max becomes 1.0 (avoid one dominating)
    max_pagerank = max(pagerank_dict.values())
    max_pageview = max(pageviews_dict.values())
    max_pagerank = max_pagerank or 1.0
    max_pageview = max_pageview or 1.0

    # Add small boosts (0.05 each) to preserve primary relevance ordering
    for doc_id in candidate_ids:
        final[doc_id] += (pagerank_dict[doc_id] / max_pagerank) * 0.05
        final[doc_id] += (pageviews_dict[doc_id] / max_pageview) * 0.05

    # Rank final results and take top 100
    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    top_100 = ranked[:100]

    ids = [doc_id for doc_id, _ in top_100]
    id2title = get_titles(ids)
    res = [(str(doc_id), id2title.get(doc_id)) for doc_id, _ in top_100]

    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    top = search_body_scores(query)
    top100 = top[:100]
    ids = [doc_id for doc_id, _ in top100]
    id2title = get_titles(ids)
    res = [(str(doc_id), id2title.get(doc_id)) for doc_id, _ in top100]
    # END SOLUTION
    return jsonify(res)

def search_body_scores(query):
    #  Compute body TF-IDF cosine scores for a query
    # Key optimization:
    #     - Only iterate postings for query terms (sparse computation)
    #     - Numerator accumulates dot-product between query vector and doc vector
    #     - Doc norm is computed before for each doc

    # Returns: list[(doc_id:int, score:float)] sorted descending

    query_tokens = (tokenize(query))
    if not query_tokens:
        return []
    # Term frequency in the query
    query_tf = Counter(query_tokens)
    # Total doc count for IDF
    N = DOC_COUNT
    # Build query TF-IDF vector: tf = 1 + log(freq), idf = log(N/df)
    query_tfidf_vector = {}
    query_length = len(query_tokens)
    for term, frequency in query_tf.items():
        tf = 1 + math.log(frequency)
        df = body_index.df.get(term, 0)
        if df == 0:
            continue
        idf = math.log(N / df)
        query_tfidf_vector[term] = tf * idf
    if not query_tfidf_vector:
        return []
    # Query vector norm for cosine denominator
    query_size_for_cosine = math.sqrt(sum(w * w for w in query_tfidf_vector.values()))

    # Accumulate dot-products (numerator)
    numerators_per_doc = defaultdict(float)

    # Iterate over query terms only
    for term, term_tfidf_in_query in query_tfidf_vector.items():
        df = body_index.df.get(term, 0)
        if df == 0:
            continue
        idf = math.log(N / df)
        # Read posting list from local postings file(s)
        posting_list = body_index.read_a_posting_list(LOCAL_BODY_DIR, term,  bucket_name=None)
        for doc_id, frequency in posting_list:
            tf = 1.0 + math.log(frequency) # alternative calculate tfidf - known
            term_tfidf_in_doc = tf * idf
            numerators_per_doc[doc_id] += term_tfidf_in_doc * term_tfidf_in_query
    cosine_scores = []
    for doc_id, numerator in numerators_per_doc.items():
        # dn from norms
        dn = docnorm_get(doc_id, None)
        if dn is None or dn <= 0:
            continue
        score = numerator / (dn * query_size_for_cosine)
        cosine_scores.append((doc_id, score))
    ranked = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    top = ranked[:1000]
    return top




def search_body_bm25(query, k1=1.2, b=0.75):
    # BM25 for the search function
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25_scores = defaultdict(float)
    N = DOC_COUNT

    for term in set(query_tokens):
        df = body_index.df.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))

        posting_list = body_index.read_a_posting_list(LOCAL_BODY_DIR, term, bucket_name=None)
        for doc_id, freq in posting_list:
            #from loaded file
            doc_len = dl_get_fast(doc_id, int(AVGDL_BODY))
            #TSV + IDX : doc_len = dl_get(doc_id, int(AVGDL_BODY))
            #FULL DICT IN RAM: doc_len = DOC_LENGTHS.get(doc_id, AVGDL_BODY)

            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / AVGDL_BODY))
            bm25_scores[doc_id] += idf * (numerator / denominator)

    return heapq.nlargest(1000, bm25_scores.items(), key=lambda x: x[1])





@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    top = search_title_scores(query)
    ids = [doc_id for doc_id, _ in top]
    id2title = get_titles(ids)
    res = [(str(doc_id), id2title.get(doc_id)) for doc_id, _ in top]
    # END SOLUTION
    return jsonify(res)

def search_title_scores(query):
    # Compute title coverage scores:
    # For each distinct query term, add +1 to each document that contains it in title.
    query_tokens = set(tokenize(query))
    docs_and_counts = defaultdict(int)

    for term in query_tokens:
        if term not in title_index.df:
            continue
        for doc_id, tf in title_index.read_a_posting_list(LOCAL_TITLE_DIR, term, bucket_name=None):
            docs_and_counts[doc_id] += 1

    ranked_docs = sorted(docs_and_counts.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    top = search_anchor_scores(query)
    ids = [doc_id for doc_id, _ in top]
    id2title = get_titles(ids)
    res = [(str(doc_id), id2title.get(doc_id)) for doc_id, _ in top]
    # END SOLUTION
    return jsonify(res)

def search_anchor_scores(query):
    #  Compute anchor coverage scores:
    #  For each distinct query term, add +1 to each document that is linked by that term in anchor text.
    query_tokens = set(tokenize(query))
    docs_and_counts = defaultdict(int)
    for term in query_tokens:
        if term not in anchor_index.df:
            continue
        for doc_id, tf in anchor_index.read_a_posting_list(LOCAL_ANCHOR_DIR, term, bucket_name=None):
            docs_and_counts[doc_id] += 1

    ranked_docs = sorted(docs_and_counts.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    #res = [pagerank.get(w_id,0) for w_id in wiki_ids]
    res = [pagerank_get(int(w_id), 0.0) for w_id in wiki_ids]

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [pageviews_get(int(w_id), 0) for w_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)