"""
Doc Reader — GraphRAG + Multilingual + Personalization edition.
v0.6 — community detection, entity aliases, history memory, user profiling.
"""
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Doc Reader", version="0.6.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:              str
    doc_id:             Optional[str]  = None
    conversation_id:    Optional[str]  = None
    use_reasoning:      bool           = False
    n_results:          int            = 5
    use_multi_query:    bool           = True
    use_hyde:           bool           = True
    use_stepback:       bool           = True
    use_graph_traversal: bool          = True
    use_question_index: bool           = True
    use_rerank:         bool           = True
    use_global_fallback: bool          = True   # fall back to community/global search if no entities found


class StructuredRequest(BaseModel):
    doc_id: str
    prompt: str
    schema: dict


class TTSRequest(BaseModel):
    text:  str
    voice: str = "leah"


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationRename(BaseModel):
    title: str


# ══════════════════════════════════════════════════════════════════════════════
# Background tasks
# ══════════════════════════════════════════════════════════════════════════════

def _run_entity_summarization(doc_id: str):
    from app import graph, groq_client
    from app.config import settings
    if not settings.groq_api_key:
        return
    try:
        entities = graph.get_entities_without_summary(limit=30)
        for ent in entities:
            contexts = graph.get_entity_contexts(ent["entity_id"], max_contexts=5)
            if not contexts:
                continue
            summary = groq_client.summarize_entity(ent["name"], ent["type"], contexts)
            if summary:
                graph.update_entity_summary(ent["entity_id"], summary)
    except Exception as e:
        print(f"Entity summarization error: {e}")


def _run_question_indexing(doc_id: str, filename: str, chunks: list[dict]):
    """Index bilingual hypothetical questions for every chunk."""
    from app import vectorstore, groq_client
    from app.config import settings
    if not settings.groq_api_key:
        return
    try:
        for chunk in chunks[:20]:
            questions = groq_client.generate_chunk_questions(chunk["text"], n=3)
            if questions:
                vectorstore.add_hypothetical_questions(
                    doc_id, filename, chunk["chunk_index"], questions
                )
    except Exception as e:
        print(f"Question indexing error: {e}")


def _run_community_detection(doc_id: str):
    """Build communities after new document is ingested."""
    from app import graph, groq_client
    from app.config import settings
    if not settings.groq_api_key:
        return
    try:
        graph.build_communities()
        # Summarize newly discovered communities
        for cid in graph.get_all_communities():
            ents = graph.get_community_entities(cid)
            if len(ents) >= 2:
                summary = groq_client.build_community_summary(ents)
                if summary:
                    graph.update_community_summary(cid, summary)
    except Exception as e:
        print(f"Community detection error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Upload
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    from app import extractor, groq_client, graph, vectorstore, storage
    from app.config import settings

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    content_type = file.content_type or extractor.guess_mime_type(file.filename)
    meta    = storage.upload_document(file_bytes, file.filename, content_type)
    doc_id  = meta["id"]

    extraction  = extractor.extract_text(file_bytes, content_type)
    text        = extraction["content"]
    total_pages = int(extraction["metadata"].get("page_count", 1) or 1)

    if not text.strip():
        return JSONResponse({**meta, "warning": "No text extracted.", "extraction": extraction})

    chunks    = extractor.chunk_text(text, max_chars=800, overlap=100, total_pages=total_pages)
    vectorstore.add_chunks(doc_id, file.filename, chunks)

    chunk_map  = {c["chunk_index"]: c["text"] for c in chunks}
    graph_data = {"entities": [], "relations": []}
    if settings.groq_api_key:
        try:
            graph_data = groq_client.extract_knowledge_graph(text, chunk_map=chunk_map)
        except Exception as e:
            print(f"KG extraction error: {e}")

    graph.add_document_node(doc_id, file.filename)
    graph.add_entities_and_relations(doc_id, graph_data)

    if settings.groq_api_key:
        background_tasks.add_task(_run_entity_summarization, doc_id)
        background_tasks.add_task(_run_question_indexing, doc_id, file.filename, chunks)
        background_tasks.add_task(_run_community_detection, doc_id)

    return JSONResponse({
        **meta,
        "extraction": {
            "char_count":  extraction["char_count"],
            "word_count":  extraction["word_count"],
            "chunk_count": len(chunks),
            "tables":      len(extraction["tables"]),
            "total_pages": total_pages,
        },
        "knowledge_graph": {
            "entities":     len(graph_data["entities"]),
            "relations":    len(graph_data["relations"]),
            "chunk_linked": sum(1 for e in graph_data["entities"] if e.get("chunk_index") is not None),
            "aliases":      sum(len(e.get("aliases", [])) for e in graph_data["entities"]),
        },
        "background_tasks": [
            "entity_summarization",
            "bilingual_question_indexing",
            "community_detection",
        ],
        "status": "processed",
    })


# ══════════════════════════════════════════════════════════════════════════════
# Documents
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/documents")
async def list_documents():
    from app import storage
    return JSONResponse(storage.list_documents())


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    from app import vectorstore, graph
    vectorstore.delete_document(doc_id)
    graph.delete_document_graph(doc_id)
    return JSONResponse({"deleted": doc_id})


# ══════════════════════════════════════════════════════════════════════════════
# Conversations
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/conversations")
async def list_conversations():
    from app.history import list_conversations
    return JSONResponse(list_conversations())


@app.post("/api/conversations")
async def create_conversation(body: ConversationCreate):
    from app.history import create_conversation
    return JSONResponse(create_conversation(body.title or "New conversation"))


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    from app.history import get_conversation, get_messages
    conv = get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return JSONResponse({**conv, "messages": get_messages(conv_id)})


@app.patch("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, body: ConversationRename):
    from app.history import rename_conversation
    rename_conversation(conv_id, body.title)
    return JSONResponse({"ok": True})


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    from app.history import delete_conversation
    delete_conversation(conv_id)
    return JSONResponse({"deleted": conv_id})


# ══════════════════════════════════════════════════════════════════════════════
# RAG Query — full pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/query")
async def query_documents(req: QueryRequest):
    from app import vectorstore, graph, groq_client
    from app import history as hist
    from app.multilingual import detect_language, rrf_merge

    # ── Conversation bookkeeping ──────────────────────────────────────────────
    detected_lang = detect_language(req.query)
    conv_id       = req.conversation_id
    if not conv_id:
        conv    = hist.create_conversation(
            hist.auto_title_from_query(req.query), lang=detected_lang
        )
        conv_id = conv["id"]
    hist.add_message(conv_id, "user", req.query, doc_id=req.doc_id)

    # ── Step 1: Multilingual query bundle ─────────────────────────────────────
    query_bundle = {
        "original": req.query, "detected_lang": detected_lang,
        "all_search_queries_vi": [req.query],
        "all_search_queries_en": [req.query],
    }
    try:
        query_bundle = groq_client.build_query_bundle(
            query=req.query,
            use_multi_query=req.use_multi_query,
            use_hyde=req.use_hyde,
            use_stepback=req.use_stepback,
        )
    except Exception as e:
        print(f"Query bundle error: {e}")

    queries_vi = query_bundle.get("all_search_queries_vi", [req.query])
    queries_en = query_bundle.get("all_search_queries_en", [req.query])

    # ── Step 2: Multilingual vector search ───────────────────────────────────
    vector_chunks = vectorstore.multilingual_search(
        queries_vi=queries_vi,
        queries_en=queries_en,
        n_results=req.n_results + 3,
        doc_id=req.doc_id,
    )

    # ── Step 3: Bilingual question index ─────────────────────────────────────
    q_chunks: list[dict] = []
    if req.use_question_index:
        try:
            q_chunks = vectorstore.multilingual_search_questions(
                queries_vi=[query_bundle.get("query_vi", req.query)],
                queries_en=[query_bundle.get("query_en", req.query)],
                n_results=3,
                doc_id=req.doc_id,
            )
        except Exception as e:
            print(f"Question search error: {e}")

    # ── Step 4: Graph traversal (alias-aware, 2-hop) ──────────────────────────
    graph_context: dict = {"entities": [], "chunk_refs": [], "relations": [], "communities": []}
    graph_chunks:  list[dict] = []
    translated_q = query_bundle.get(
        "query_en" if detected_lang == "vi" else "query_vi", ""
    )

    if req.use_graph_traversal:
        try:
            seed_names = graph.extract_query_entities(
                req.query,
                doc_id=req.doc_id,
                translated_query=translated_q,
            )
            if seed_names:
                graph_context = graph.graph_local_search(seed_names, depth=2)
                if graph_context["chunk_refs"]:
                    graph_chunks = vectorstore.fetch_chunks_by_refs(
                        graph_context["chunk_refs"][:10]
                    )
        except Exception as e:
            print(f"Graph traversal error: {e}")

    # ── Step 4b: Global fallback — community search ───────────────────────────
    global_chunks: list[dict] = []
    if req.use_global_fallback and not seed_names if req.use_graph_traversal else True:
        try:
            # Extract keywords from query for global entity search
            import re as _re
            kws = _re.findall(r"\b\w{4,}\b", req.query.lower())[:5]
            global_ents = graph.global_entity_search(kws, top_k=5)
            if global_ents:
                # Get chunks from global entity matches
                global_refs = []
                for ent in global_ents:
                    eid    = ent["id"]
                    result = graph.get_db().execute(
                        "MATCH (e:Entity {entity_id: $eid})-[:APPEARS_IN]->(c:Chunk) "
                        "RETURN c.doc_id, c.chunk_index LIMIT 3",
                        {"eid": eid},
                    )
                    while result.has_next():
                        r = result.get_next()
                        global_refs.append({"doc_id": r[0], "chunk_index": r[1]})
                if global_refs:
                    global_chunks = vectorstore.fetch_chunks_by_refs(global_refs[:6])
                    for c in global_chunks:
                        c["from_global"] = True
        except Exception as e:
            print(f"Global search error: {e}")

    # ── Step 5: Personalization context ──────────────────────────────────────
    entity_ids_in_graph = [e["id"] for e in graph_context.get("entities", [])]
    personalization_ctx = {}
    try:
        personalization_ctx = hist.build_personalization_context(
            query=req.query,
            doc_id=req.doc_id,
            entity_ids=entity_ids_in_graph,
            lang=detected_lang,
            n_history=3,
        )
    except Exception as e:
        print(f"Personalization error: {e}")

    past_exchanges = personalization_ctx.get("past_exchanges", [])
    recent_turns   = hist.get_recent_messages(conv_id, n=6)

    # ── Step 6: RRF merge across ALL sources ─────────────────────────────────
    merged = rrf_merge(
        vector_chunks,
        q_chunks,
        graph_chunks,
        global_chunks,
        top_n=(req.n_results + 5),
    )

    # Preserve from_graph + from_global flags
    graph_ids  = {
        f"{c.get('metadata',{}).get('doc_id','')}_{c.get('metadata',{}).get('chunk_index','')}"
        for c in graph_chunks
    }
    global_ids = {
        f"{c.get('metadata',{}).get('doc_id','')}_{c.get('metadata',{}).get('chunk_index','')}"
        for c in global_chunks
    }
    for c in merged:
        m   = c.get("metadata", {})
        cid = f"{m.get('doc_id','')}_{m.get('chunk_index','')}"
        if cid in graph_ids:
            c["from_graph"]  = True
        if cid in global_ids:
            c["from_global"] = True

    if not merged:
        answer = (
            "Không tìm thấy nội dung liên quan trong tài liệu."
            if detected_lang == "vi"
            else "No relevant content found in documents."
        )
        hist.add_message(conv_id, "assistant", answer, doc_id=req.doc_id)
        return JSONResponse({
            "answer": answer, "sources": [], "conversation_id": conv_id,
            "new_conversation": not req.conversation_id,
            "query_bundle": _safe_bundle(query_bundle),
        })

    # ── Step 7: Language-aware reranking + profile boost ─────────────────────
    if req.use_rerank and len(merged) > req.n_results:
        try:
            ranked = groq_client.rerank_chunks(
                req.query, merged, top_k=req.n_results,
                query_translated=translated_q,
                use_profile_boost=True,
            )
        except Exception as e:
            print(f"Rerank error: {e}")
            ranked = merged[:req.n_results]
    else:
        ranked = merged[:req.n_results]

    # ── Step 8: Generate answer ───────────────────────────────────────────────
    analysis = None
    if req.use_reasoning:
        result   = groq_client.reasoning_query(
            req.query,
            "\n\n".join(c["text"] for c in ranked),
            detected_lang=detected_lang,
        )
        answer   = result["answer"]
        analysis = result["analysis"]
    else:
        answer = groq_client.rag_answer(
            query=req.query,
            context_chunks=ranked,
            graph_context=graph_context,
            past_exchanges=past_exchanges,
            recent_turns=recent_turns,
            detected_lang=detected_lang,
            personalization_context=personalization_ctx,
        )

    # ── Step 9: Persist answer with entity IDs ───────────────────────────────
    sources_meta = [{
        "filename":      c["metadata"].get("filename"),
        "score":         round(c.get("rerank_score", c.get("rrf_score", c["score"])), 3),
        "page_estimate": c["metadata"].get("page_estimate"),
        "section":       c["metadata"].get("section", ""),
    } for c in ranked]

    hist.add_message(
        conv_id, "assistant", answer,
        doc_id=req.doc_id,
        entity_ids=entity_ids_in_graph[:10],
        metadata={
            "sources":       sources_meta,
            "entities":      graph_context.get("entities", [])[:3],
            "detected_lang": detected_lang,
            "query_bundle":  {
                "original":  query_bundle["original"],
                "query_vi":  query_bundle.get("query_vi", ""),
                "query_en":  query_bundle.get("query_en", ""),
            },
        },
    )

    # Periodically refresh profile vector index (every ~5 turns)
    try:
        from app.history import index_user_interests_into_vector
        with hist.get_conn() as c:
            cnt = c.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conv_id,)
            ).fetchone()[0]
        if cnt % 5 == 0:
            index_user_interests_into_vector()
    except Exception:
        pass

    # ── Step 10: Build response ───────────────────────────────────────────────
    sources = []
    for i, c in enumerate(ranked):
        meta = c["metadata"]
        sources.append({
            "citation_number": i + 1,
            "text":            c["text"][:300] + ("…" if len(c["text"]) > 300 else ""),
            "filename":        meta.get("filename", ""),
            "doc_id":          meta.get("doc_id", ""),
            "chunk_index":     meta.get("chunk_index"),
            "start_char":      meta.get("start_char"),
            "end_char":        meta.get("end_char"),
            "page_estimate":   meta.get("page_estimate"),
            "section":         meta.get("section", ""),
            "score":           round(c.get("score", 0), 3),
            "rrf_score":       round(c.get("rrf_score", 0), 4),
            "rerank_score":    round(c.get("rerank_score", c.get("rrf_score", c["score"])), 3),
            "llm_relevance":   c.get("llm_relevance"),
            "profile_affinity": c.get("profile_affinity"),
            "from_graph":      c.get("from_graph", False),
            "from_global":     c.get("from_global", False),
            "via_question":    c.get("via_question"),
            "search_lang":     c.get("search_lang", ""),
        })

    return JSONResponse({
        "answer":           answer,
        "analysis":         analysis,
        "conversation_id":  conv_id,
        "new_conversation": not req.conversation_id,
        "sources":          sources,
        "graph_context": {
            "entities":          graph_context.get("entities", [])[:6],
            "relations":         graph_context.get("relations", [])[:5],
            "communities":       graph_context.get("communities", [])[:3],
            "chunks_from_graph": len(graph_chunks),
            "chunks_from_global": len(global_chunks),
        },
        "personalization": {
            "interests":        personalization_ctx.get("user_interests", [])[:5],
            "language_hint":    personalization_ctx.get("language_hint", "en"),
            "past_context_used": len(past_exchanges) > 0,
        },
        "retrieval_stats": {
            "detected_lang":       detected_lang,
            "vector_chunks":       len(vector_chunks),
            "question_chunks":     len(q_chunks),
            "graph_chunks":        len(graph_chunks),
            "global_chunks":       len(global_chunks),
            "total_before_rerank": len(merged),
            "final_chunks":        len(ranked),
            "queries_vi":          len(queries_vi),
            "queries_en":          len(queries_en),
        },
        "query_bundle": _safe_bundle(query_bundle),
    })


def _safe_bundle(b: dict) -> dict:
    return {
        "original":      b.get("original", ""),
        "detected_lang": b.get("detected_lang", "en"),
        "query_vi":      b.get("query_vi", ""),
        "query_en":      b.get("query_en", ""),
        "multi_vi":      b.get("multi_vi", []),
        "multi_en":      b.get("multi_en", []),
        "hyde_vi":       b.get("hyde_vi", ""),
        "hyde_en":       b.get("hyde_en", ""),
        "stepback_vi":   b.get("stepback_vi", ""),
        "stepback_en":   b.get("stepback_en", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Passage viewer
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/passage/{doc_id}/{chunk_index}")
async def get_passage(doc_id: str, chunk_index: int, window: int = 1):
    from app import vectorstore
    chunks = vectorstore.get_surrounding_chunks(doc_id, chunk_index, window=window)
    if not chunks:
        raise HTTPException(404, "Chunk not found")
    return JSONResponse({
        "doc_id": doc_id, "chunk_index": chunk_index, "window": window,
        "passage": [{
            "text":          c["text"],
            "chunk_index":   c["metadata"].get("chunk_index"),
            "start_char":    c["metadata"].get("start_char"),
            "end_char":      c["metadata"].get("end_char"),
            "page_estimate": c["metadata"].get("page_estimate"),
            "section":       c["metadata"].get("section", ""),
            "is_target":     c["metadata"].get("chunk_index") == chunk_index,
        } for c in chunks],
    })


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge graph endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/graph/{doc_id}")
async def get_knowledge_graph(doc_id: str):
    from app import graph
    return JSONResponse(graph.get_graph_for_doc(doc_id))


@app.get("/api/entity/{entity_id}")
async def get_entity_detail(entity_id: str):
    from app import graph
    from app.history import get_entity_conversation_history
    neighborhood = graph.get_entity_neighborhood(entity_id)
    if not neighborhood:
        raise HTTPException(404, "Entity not found")
    # Attach conversation history for this entity
    neighborhood["conversation_history"] = get_entity_conversation_history(entity_id)
    return JSONResponse(neighborhood)


@app.get("/api/user/interests")
async def get_user_interests():
    from app.history import get_user_interests
    return JSONResponse(get_user_interests(top_k=20))


@app.get("/api/user/profile")
async def get_user_profile():
    """Return aggregated user profile for debugging/display."""
    from app.history import get_user_interests, _infer_lang_preference
    return JSONResponse({
        "interests":       get_user_interests(top_k=15),
        "language_pref":   _infer_lang_preference(),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Structured output, TTS, Health, Frontend
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/extract-structured")
async def extract_structured(req: StructuredRequest):
    from app import vectorstore, groq_client
    chunks  = vectorstore.search(req.prompt, n_results=8, doc_id=req.doc_id)
    context = "\n\n".join(c["text"] for c in chunks)
    return JSONResponse(groq_client.structured_output(req.prompt, req.schema, context))


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    from app import groq_client
    try:
        audio = groq_client.text_to_speech(req.text, voice=req.voice)
        return Response(content=audio, media_type="audio/wav",
                        headers={"Content-Disposition": "inline; filename=speech.wav"})
    except Exception as e:
        raise HTTPException(500, f"TTS error: {str(e)}")


@app.get("/api/tts/voices")
async def get_voices():
    from app.groq_client import TTS_VOICES
    return JSONResponse(TTS_VOICES)


@app.get("/api/health")
async def health():
    from app.config import settings
    return JSONResponse({
        "status":          "ok",
        "version":         "0.6.0",
        "groq_configured": bool(settings.groq_api_key),
        "minio_endpoint":  settings.minio_endpoint,
        "embedding_model": settings.embedding_model,
    })


@app.get("/", response_class=HTMLResponse)
async def frontend():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")