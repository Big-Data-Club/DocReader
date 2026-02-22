"""
ChromaDB vector store — multilingual edition.

Key changes vs. v0.4:
1. Embedding model changed to `paraphrase-multilingual-MiniLM-L12-v2`
   • Supports 50+ languages including Vietnamese + English in ONE shared space
   • Same 384-dim output as the old all-MiniLM-L6-v2 → no schema change in Chroma
   • Vietnamese and English texts now map to compatible vector positions
   NOTE: If you have an existing ChromaDB index built with all-MiniLM-L6-v2 you
         MUST re-index all documents (delete data/chroma/* and re-upload).

2. `multilingual_search()` — searches with separate VI and EN query lists then
   merges with Reciprocal Rank Fusion via multilingual.rrf_merge().

3. Hypothetical questions generated in both languages are all stored in the
   same `chunk_questions` collection (model is now language-agnostic).
"""
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import settings

_chroma_client   = None
_embedding_model = None

# ── Embedding model ───────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _chroma_client


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().encode(texts, show_progress_bar=False).tolist()


# ── Collections ───────────────────────────────────────────────────────────────

def get_collection(name: str = "documents"):
    return get_chroma_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"},
    )


def get_questions_collection():
    """
    Separate collection for hypothetical questions per chunk (both languages).
    Queries matched here benefit from question-to-question embedding alignment.
    """
    return get_chroma_client().get_or_create_collection(
        name="chunk_questions", metadata={"hnsw:space": "cosine"},
    )


# ── Indexing ──────────────────────────────────────────────────────────────────

def add_chunks(doc_id: str, filename: str, chunks: list[dict]):
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    get_collection().upsert(
        ids=[f"{doc_id}_{c['chunk_index']}" for c in chunks],
        embeddings=embed_texts(texts),
        documents=texts,
        metadatas=[{
            "doc_id":        doc_id,
            "filename":      filename,
            "chunk_index":   c["chunk_index"],
            "start_char":    c["start_char"],
            "end_char":      c["end_char"],
            "page_estimate": c["page_estimate"],
            "section":       c.get("section", ""),
        } for c in chunks],
    )


def add_hypothetical_questions(
    doc_id: str, filename: str, chunk_index: int, questions: list[str]
):
    """
    Index hypothetical questions (may be in both VI and EN).
    All questions for one chunk share the same chunk_index pointer.
    """
    if not questions:
        return
    base = f"{doc_id}_{chunk_index}"
    get_questions_collection().upsert(
        ids=[f"{base}_q{i}" for i in range(len(questions))],
        embeddings=embed_texts(questions),
        documents=questions,
        metadatas=[{
            "doc_id":       doc_id,
            "filename":     filename,
            "chunk_index":  chunk_index,
            "source":       "hypothetical_question",
        } for _ in questions],
    )


# ── Search helpers ────────────────────────────────────────────────────────────

def search(query: str, n_results: int = 5, doc_id: str = None) -> list[dict]:
    col   = get_collection()
    count = col.count()
    if count == 0:
        return []
    results = col.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=min(n_results, count),
        where={"doc_id": doc_id} if doc_id else None,
        include=["documents", "metadatas", "distances"],
    )
    return [{
        "text":     doc,
        "score":    round(1 - results["distances"][0][i], 4),
        "metadata": results["metadatas"][0][i],
    } for i, doc in enumerate(results["documents"][0])]


def search_questions(query: str, n_results: int = 5, doc_id: str = None) -> list[dict]:
    """
    Match query against hypothetical-question index → return source chunks.
    Works in any language because model is multilingual.
    """
    col   = get_questions_collection()
    count = col.count()
    if count == 0:
        return []
    results = col.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=min(n_results * 2, count),
        where={"doc_id": doc_id} if doc_id else None,
        include=["documents", "metadatas", "distances"],
    )
    seen, chunks = set(), []
    for i, q_text in enumerate(results["documents"][0]):
        meta  = results["metadatas"][0][i]
        score = round(1 - results["distances"][0][i], 4)
        cid   = f"{meta['doc_id']}_{meta['chunk_index']}"
        if cid in seen:
            continue
        seen.add(cid)
        src = get_chunk_by_id(meta["doc_id"], meta["chunk_index"])
        if src:
            src["score"]        = score
            src["via_question"] = q_text
            chunks.append(src)
    return chunks[:n_results]


def multi_query_search(
    queries: list[str], n_results: int = 5, doc_id: str = None
) -> list[dict]:
    """Legacy: single-pool multi-query without language separation."""
    seen: dict[str, dict] = {}
    for q in queries:
        for chunk in search(q, n_results=n_results, doc_id=doc_id):
            meta = chunk["metadata"]
            cid  = f"{meta['doc_id']}_{meta['chunk_index']}"
            if cid not in seen or chunk["score"] > seen[cid]["score"]:
                seen[cid] = chunk
    return sorted(seen.values(), key=lambda x: x["score"], reverse=True)


# ── Multilingual search (main entry point) ────────────────────────────────────

def multilingual_search(
    queries_vi: list[str],
    queries_en: list[str],
    n_results: int = 8,
    doc_id: str = None,
) -> list[dict]:
    """
    Search with separate Vietnamese and English query lists, then fuse with RRF.

    Why separate lists?
      Even with a multilingual model, cosine similarity is slightly higher
      when query and document are in the same language.  Running distinct
      VI-only and EN-only searches then merging via RRF gives better recall
      across both language-matched and cross-language pairs than a single
      mixed-language search.

    Flow:
      1. Embed all VI queries → search collection → ranked_vi
      2. Embed all EN queries → search collection → ranked_en
      3. rrf_merge(ranked_vi, ranked_en) → deduplicated, rank-fused list
    """
    from app.multilingual import rrf_merge

    ranked_vi = _ranked_from_queries(queries_vi, n_results + 3, doc_id)
    ranked_en = _ranked_from_queries(queries_en, n_results + 3, doc_id)

    # Tag source language for transparency
    for c in ranked_vi:
        c.setdefault("search_lang", "vi")
    for c in ranked_en:
        c.setdefault("search_lang", "en")

    return rrf_merge(ranked_vi, ranked_en, top_n=n_results)


def _ranked_from_queries(
    queries: list[str], n_results: int, doc_id: str | None
) -> list[dict]:
    """
    Run each query independently, merge by best score (pre-RRF within one language).
    This is the inner-language de-dup before cross-language RRF.
    """
    best: dict[str, dict] = {}
    for q in queries:
        if not q or not q.strip():
            continue
        for chunk in search(q, n_results=n_results, doc_id=doc_id):
            meta = chunk["metadata"]
            cid  = f"{meta['doc_id']}_{meta['chunk_index']}"
            if cid not in best or chunk["score"] > best[cid]["score"]:
                best[cid] = chunk
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


# ── Hypothetical question multilingual search ─────────────────────────────────

def multilingual_search_questions(
    queries_vi: list[str],
    queries_en: list[str],
    n_results: int = 5,
    doc_id: str = None,
) -> list[dict]:
    """Search question index with both language variants, RRF-merge."""
    from app.multilingual import rrf_merge

    q_vi = _ranked_questions(queries_vi, n_results + 2, doc_id)
    q_en = _ranked_questions(queries_en, n_results + 2, doc_id)
    return rrf_merge(q_vi, q_en, top_n=n_results)


def _ranked_questions(
    queries: list[str], n_results: int, doc_id: str | None
) -> list[dict]:
    best: dict[str, dict] = {}
    for q in queries:
        if not q or not q.strip():
            continue
        for chunk in search_questions(q, n_results=n_results, doc_id=doc_id):
            meta = chunk["metadata"]
            cid  = f"{meta['doc_id']}_{meta['chunk_index']}"
            if cid not in best or chunk["score"] > best[cid]["score"]:
                best[cid] = chunk
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


# ── Chunk retrieval ───────────────────────────────────────────────────────────

def fetch_chunks_by_refs(chunk_refs: list[dict]) -> list[dict]:
    chunks = []
    for ref in chunk_refs:
        c = get_chunk_by_id(ref["doc_id"], ref["chunk_index"])
        if c:
            c["from_graph"] = True
            c["score"]      = 0.0
            chunks.append(c)
    return chunks


def get_chunk_by_id(doc_id: str, chunk_index: int) -> dict | None:
    try:
        result = get_collection().get(
            ids=[f"{doc_id}_{chunk_index}"],
            include=["documents", "metadatas"],
        )
        if result["documents"]:
            return {"text": result["documents"][0], "metadata": result["metadatas"][0]}
    except Exception:
        pass
    return None


def get_surrounding_chunks(doc_id: str, chunk_index: int, window: int = 1) -> list[dict]:
    chunks = []
    for idx in range(max(0, chunk_index - window), chunk_index + window + 1):
        c = get_chunk_by_id(doc_id, idx)
        if c:
            chunks.append(c)
    return chunks


def delete_document(doc_id: str):
    get_collection().delete(where={"doc_id": doc_id})
    col = get_questions_collection()
    if col.count() > 0:
        col.delete(where={"doc_id": doc_id})