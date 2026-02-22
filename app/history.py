"""
Conversation history — GraphRAG-aware, multilingual, personalization engine.

Architecture:
─────────────────────────────────────────────────────────────────────────
SQLite (persistent structured store)
  • conversations     — conversation metadata + language preference
  • messages          — full turn history with doc/entity linkage
  • entity_mentions   — which entities were discussed in which conv turn
  • user_interests    — LLM-extracted topic preferences aggregated over time
  • conversation_summaries — compressed summaries of older conversations

ChromaDB (semantic vector index, 3 separate collections)
  • history_answers   — assistant answers (for future relevant-context lookup)
  • history_questions — user questions (for "have I asked this before?" detection)
  • user_profile_vec  — user interest text embeddings for personalization

Design principles:
  1. Dual-index: questions AND answers indexed separately — matching question
     to question gives much higher precision than matching query to answer.
  2. Entity threading: each message logs which KG entities were involved.
     Future queries can retrieve history by entity overlap, not just semantics.
  3. Conversation compression: when a conversation exceeds COMPRESS_THRESHOLD
     turns, the old turns are summarized into a compact text and removed from
     the LLM's window — preserving memory without burning context tokens.
  4. User interest profile: across all conversations, we track topic frequencies
     and build a persistent "who is this user?" model that biases the RAG prompt.
  5. Language preference: tracked per conversation and overall — used to bias
     default response language.
─────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings

DB_PATH          = str(Path(settings.data_dir) / "history.db")
COMPRESS_AFTER   = 12   # compress older turns when conversation exceeds this
SUMMARY_KEEP     = 6    # keep this many recent turns uncompressed


# ══════════════════════════════════════════════════════════════════════════════
# SQLite layer
# ══════════════════════════════════════════════════════════════════════════════

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id            TEXT PRIMARY KEY,
                title         TEXT,
                lang_pref     TEXT DEFAULT 'en',   -- detected primary language
                doc_ids       TEXT DEFAULT '[]',   -- JSON list of docs referenced
                created_at    TEXT,
                updated_at    TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                doc_id          TEXT,
                entity_ids      TEXT DEFAULT '[]',  -- JSON list of KG entity IDs
                metadata        TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Compressed summaries for old turns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                summary         TEXT NOT NULL,       -- LLM-generated digest
                turn_range      TEXT,                -- e.g. "0-10"
                entity_ids      TEXT DEFAULT '[]',   -- entities in summarized turns
                created_at      TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Per-message entity tracking for entity-based history search
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id              TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                message_id      TEXT NOT NULL,
                entity_id       TEXT NOT NULL,
                entity_name     TEXT,
                created_at      TEXT NOT NULL
            )
        """)

        # Aggregated user interest profile
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_interests (
                topic       TEXT PRIMARY KEY,
                score       REAL DEFAULT 1.0,        -- frequency-weighted score
                last_seen   TEXT,
                doc_ids     TEXT DEFAULT '[]',        -- which docs relate to this topic
                lang        TEXT DEFAULT 'en'
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_conv ON entity_mentions(conversation_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON entity_mentions(entity_id)")
        conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Conversation CRUD
# ══════════════════════════════════════════════════════════════════════════════

def create_conversation(title: str = "New conversation", lang: str = "en") -> dict:
    cid = str(uuid.uuid4())
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?)",
            (cid, title, lang, "[]", now, now),
        )
        conn.commit()
    return {"id": cid, "title": title, "lang_pref": lang, "created_at": now, "updated_at": now}


def list_conversations(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conv_id: str, title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conv_id),
        )
        conn.commit()


def delete_conversation(conv_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM entity_mentions WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversation_summaries WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
    _delete_conv_from_index(conv_id)


# ══════════════════════════════════════════════════════════════════════════════
# Message management
# ══════════════════════════════════════════════════════════════════════════════

def add_message(
    conv_id:    str,
    role:       str,
    content:    str,
    doc_id:     str = None,
    metadata:   dict = None,
    entity_ids: list[str] = None,
) -> dict:
    """
    Persist a message and:
    • update conversation lang_pref from metadata
    • log entity_mentions
    • index in ChromaDB (dual-index: questions + answers)
    • trigger interest extraction on assistant turns
    • trigger compression if conversation is getting long
    """
    msg_id    = str(uuid.uuid4())
    now       = _now()
    meta      = metadata or {}
    ents      = entity_ids or meta.get("entity_ids", []) or []
    meta_json = json.dumps(meta)
    ents_json = json.dumps(ents)

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, doc_id or "", ents_json, meta_json, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )
        # Update doc_ids on conversation
        if doc_id:
            row = conn.execute(
                "SELECT doc_ids FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if row:
                current = json.loads(row["doc_ids"] or "[]")
                if doc_id not in current:
                    current.append(doc_id)
                    conn.execute(
                        "UPDATE conversations SET doc_ids = ? WHERE id = ?",
                        (json.dumps(current), conv_id),
                    )
        conn.commit()

    # Log entity mentions
    for eid in ents:
        _log_entity_mention(conv_id, msg_id, eid, now)

    msg = {
        "id":              msg_id,
        "conversation_id": conv_id,
        "role":            role,
        "content":         content,
        "doc_id":          doc_id,
        "entity_ids":      ents,
        "metadata":        meta,
        "created_at":      now,
    }

    # Dual ChromaDB indexing
    _index_message(msg)

    if role == "assistant":
        # Background: extract user interests from this exchange
        try:
            _update_user_interests_from_meta(meta, doc_id)
        except Exception as e:
            print(f"Interest update error: {e}")

        # Check if we should compress old turns
        _maybe_compress(conv_id)

    return msg


def get_messages(conv_id: str, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at LIMIT ?",
            (conv_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d          = dict(r)
        d["metadata"]   = json.loads(d["metadata"] or "{}")
        d["entity_ids"] = json.loads(d["entity_ids"] or "[]")
        result.append(d)
    return result


def get_recent_messages(conv_id: str, n: int = 6) -> list[dict]:
    """Last n turns for LLM context — prepends compressed summary if available."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (conv_id, n),
        ).fetchall()
    turns = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # Prepend most recent summary (if any) as a [system-context] message
    summary = _get_latest_summary(conv_id)
    if summary:
        turns.insert(0, {
            "role":    "user",
            "content": f"[Conversation summary so far]: {summary}",
        })
    return turns


# ══════════════════════════════════════════════════════════════════════════════
# Conversation compression
# ══════════════════════════════════════════════════════════════════════════════

def _maybe_compress(conv_id: str):
    """Compress the oldest turns if the conversation is long enough."""
    with get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()[0]

    if total < COMPRESS_AFTER:
        return

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, entity_ids FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (conv_id, total - SUMMARY_KEEP),
        ).fetchall()

    if not rows:
        return

    # Check we haven't already summarized these
    existing = _get_latest_summary(conv_id)
    if existing and len(rows) <= COMPRESS_AFTER - SUMMARY_KEEP:
        return

    try:
        _compress_turns(conv_id, [dict(r) for r in rows])
    except Exception as e:
        print(f"Compression error: {e}")


def _compress_turns(conv_id: str, turns: list[dict]):
    """LLM-summarize a batch of turns, store summary, delete turns."""
    from app.config import settings
    from app.groq_client import get_client

    if not settings.groq_api_key:
        return

    transcript = "\n".join(
        f"{t['role'].upper()}: {t['content'][:400]}" for t in turns
    )
    prompt = (
        "Summarize the following conversation turns into a concise bullet-point digest. "
        "Preserve key facts, entities, decisions, and questions asked. "
        "Output ONLY the summary, no preamble.\n\n" + transcript
    )
    try:
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
        )
        summary_text = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary LLM error: {e}")
        return

    # Collect all entity_ids from the turns being compressed
    all_eids: list[str] = []
    for t in turns:
        all_eids.extend(json.loads(t.get("entity_ids") or "[]"))

    now = _now()
    sid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
            (
                sid, conv_id, summary_text,
                f"0-{len(turns)}",
                json.dumps(list(set(all_eids))),
                now,
            ),
        )
        # Remove compressed turns
        ids_to_delete = [t["id"] for t in turns]
        placeholders  = ",".join("?" * len(ids_to_delete))
        conn.execute(
            f"DELETE FROM entity_mentions WHERE message_id IN ({placeholders})",
            ids_to_delete,
        )
        conn.execute(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            ids_to_delete,
        )
        conn.commit()


def _get_latest_summary(conv_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT summary FROM conversation_summaries
            WHERE conversation_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (conv_id,),
        ).fetchone()
    return row["summary"] if row else ""


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB dual-index
# ══════════════════════════════════════════════════════════════════════════════

def _get_answers_collection():
    from app.vectorstore import get_chroma_client
    return get_chroma_client().get_or_create_collection(
        "history_answers", metadata={"hnsw:space": "cosine"}
    )


def _get_questions_collection():
    from app.vectorstore import get_chroma_client
    return get_chroma_client().get_or_create_collection(
        "history_questions", metadata={"hnsw:space": "cosine"}
    )


def _get_profile_collection():
    from app.vectorstore import get_chroma_client
    return get_chroma_client().get_or_create_collection(
        "user_profile_vec", metadata={"hnsw:space": "cosine"}
    )


def _index_message(msg: dict):
    """
    Dual-index: user questions → history_questions, assistant answers → history_answers.

    Indexing both sides means:
    • New user query retrieves similar past QUESTIONS → "did I ask this before?"
    • New user query also retrieves relevant past ANSWERS → inject as context
    """
    try:
        from app.vectorstore import embed_texts
        meta_base = {
            "conversation_id": msg["conversation_id"],
            "message_id":      msg["id"],
            "doc_id":          msg.get("doc_id") or "",
            "role":            msg["role"],
        }
        text = msg["content"][:1200]

        if msg["role"] == "user":
            col = _get_questions_collection()
        else:
            col = _get_answers_collection()

        col.upsert(
            ids=[msg["id"]],
            embeddings=[embed_texts([text])[0]],
            documents=[text],
            metadatas=[meta_base],
        )
    except Exception as e:
        print(f"Message index error: {e}")


def _delete_conv_from_index(conv_id: str):
    where = {"conversation_id": conv_id}
    for col_fn in [_get_answers_collection, _get_questions_collection]:
        try:
            col = col_fn()
            if col.count() > 0:
                col.delete(where=where)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# History search — entity-aware + dual-index
# ══════════════════════════════════════════════════════════════════════════════

def search_history(
    query:     str,
    n_results: int  = 3,
    doc_id:    str  = None,
    entity_ids: list[str] = None,
) -> list[dict]:
    """
    Retrieves relevant past exchanges via three complementary strategies:

    1. Semantic answer search — "has the assistant said something relevant?"
    2. Semantic question search — "has the user asked something similar?"
    3. Entity-based search — "has this entity been discussed?" (SQLite lookup)

    Results are RRF-merged, deduplicated, and scored.
    """
    from app.vectorstore import embed_texts
    from app.multilingual import rrf_merge

    ranked_answers   = _semantic_history_search(_get_answers_collection(),   query, n_results, doc_id)
    ranked_questions = _semantic_history_search(_get_questions_collection(), query, n_results, doc_id)
    entity_results   = _entity_history_search(entity_ids, n_results) if entity_ids else []

    def _key(item):
        return item.get("message_id", item.get("text", ""))[:50]

    # Convert to uniform format for RRF
    def _norm(items, label):
        out = []
        for it in items:
            out.append({
                "text":            it.get("text", ""),
                "message_id":      it.get("message_id", ""),
                "conversation_id": it.get("conversation_id", ""),
                "score":           it.get("score", 0.5),
                "source":          label,
                "metadata": {"doc_id": it.get("doc_id", ""),
                              "conversation_id": it.get("conversation_id", "")},
            })
        return out

    a_norm = _norm(ranked_answers,   "answer")
    q_norm = _norm(ranked_questions, "question")
    e_norm = _norm(entity_results,   "entity")

    # RRF with a custom key that works on message_id
    merged = rrf_merge(
        a_norm, q_norm, e_norm,
        key_fn=lambda x: x.get("message_id") or x.get("text", "")[:40],
        top_n=n_results,
    )
    return merged


def _semantic_history_search(
    col, query: str, n_results: int, doc_id: str = None
) -> list[dict]:
    try:
        from app.vectorstore import embed_texts
        if col.count() == 0:
            return []
        where  = {"doc_id": doc_id} if doc_id else None
        emb    = embed_texts([query])[0]
        result = col.query(
            query_embeddings=[emb],
            n_results=min(n_results, col.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i]
            items.append({
                "text":            doc,
                "message_id":      meta.get("message_id", ""),
                "conversation_id": meta.get("conversation_id", ""),
                "doc_id":          meta.get("doc_id", ""),
                "score":           round(1 - result["distances"][0][i], 3),
            })
        return items
    except Exception as e:
        print(f"History semantic search error: {e}")
        return []


def _entity_history_search(entity_ids: list[str], limit: int = 5) -> list[dict]:
    """Find past messages where specific KG entities were mentioned."""
    if not entity_ids:
        return []
    try:
        placeholders = ",".join("?" * len(entity_ids[:10]))
        with get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT em.message_id, em.conversation_id, m.content
                FROM entity_mentions em
                JOIN messages m ON m.id = em.message_id
                WHERE em.entity_id IN ({placeholders}) AND m.role = 'assistant'
                ORDER BY em.created_at DESC
                LIMIT ?
                """,
                (*entity_ids[:10], limit),
            ).fetchall()
        return [
            {
                "text":            r["content"][:600],
                "message_id":      r["message_id"],
                "conversation_id": r["conversation_id"],
                "score":           0.7,
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Entity history search error: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# User interest / personalization
# ══════════════════════════════════════════════════════════════════════════════

def _log_entity_mention(conv_id: str, msg_id: str, entity_id: str, now: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO entity_mentions VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()), conv_id, msg_id, entity_id,
                entity_id.split("_", 1)[-1].replace("_", " ")[:60],
                now,
            ),
        )
        conn.commit()


def _update_user_interests_from_meta(meta: dict, doc_id: str = None):
    """
    Extract topics from assistant answer metadata and upsert into user_interests.
    Runs on every assistant turn — lightweight SQLite upsert.
    """
    topics: list[str] = []

    # From KG entities attached to the answer
    for ent in meta.get("entities", []):
        if isinstance(ent, dict):
            topics.append(ent.get("name", ""))
        elif isinstance(ent, str):
            topics.append(ent)

    # From the query_bundle — the original query IS a topic signal
    qb = meta.get("query_bundle", {})
    original = qb.get("original", "")
    if original:
        topics.extend(re.findall(r"\b\w{4,}\b", original.lower())[:5])

    if not topics:
        return

    now  = _now()
    lang = meta.get("detected_lang", "en")
    with get_conn() as conn:
        for t in topics:
            t = t.strip().lower()
            if not t or len(t) < 3:
                continue
            conn.execute(
                """
                INSERT INTO user_interests(topic, score, last_seen, doc_ids, lang)
                VALUES (?, 1.0, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    score    = score * 0.9 + 1.0,
                    last_seen = excluded.last_seen,
                    lang      = excluded.lang
                """,
                (t, now, json.dumps([doc_id] if doc_id else []), lang),
            )
        conn.commit()


def get_user_interests(top_k: int = 10, lang: str = None) -> list[dict]:
    """Return top-k interest topics, optionally filtered by language."""
    with get_conn() as conn:
        if lang:
            rows = conn.execute(
                "SELECT topic, score FROM user_interests WHERE lang = ? ORDER BY score DESC LIMIT ?",
                (lang, top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT topic, score FROM user_interests ORDER BY score DESC LIMIT ?",
                (top_k,),
            ).fetchall()
    return [{"topic": r["topic"], "score": round(r["score"], 3)} for r in rows]


def build_personalization_context(
    query:      str,
    doc_id:     str  = None,
    entity_ids: list[str] = None,
    lang:       str  = "en",
    n_history:  int  = 3,
) -> dict:
    """
    Build a rich personalization context dict for injection into the RAG prompt.

    Returns:
    {
        "user_interests":     [str],   # top topics this user cares about
        "past_exchanges":     [dict],  # relevant past Q&A (RRF-merged)
        "entity_history":     [dict],  # past messages mentioning same entities
        "language_hint":      str,     # "vi" | "en" based on user history
        "personalization_str": str,    # formatted string ready for LLM injection
    }
    """
    interests    = get_user_interests(top_k=8, lang=lang)
    past         = search_history(query, n_results=n_history, doc_id=doc_id, entity_ids=entity_ids)
    lang_hint    = _infer_lang_preference()

    # Format for LLM
    parts = []
    if interests:
        topic_list = ", ".join(i["topic"] for i in interests[:6])
        parts.append(f"User's frequent topics: {topic_list}.")
    if past:
        excerpts = "; ".join(p["text"][:150] for p in past[:2] if p.get("text"))
        if excerpts:
            parts.append(f"Relevant past context: {excerpts}")

    return {
        "user_interests":      [i["topic"] for i in interests],
        "past_exchanges":      past,
        "language_hint":       lang_hint,
        "personalization_str": " ".join(parts),
    }


def _infer_lang_preference() -> str:
    """Return 'vi' or 'en' based on majority of stored interests."""
    with get_conn() as conn:
        vi_count = conn.execute(
            "SELECT COUNT(*) FROM user_interests WHERE lang = 'vi'"
        ).fetchone()[0]
        en_count = conn.execute(
            "SELECT COUNT(*) FROM user_interests WHERE lang = 'en'"
        ).fetchone()[0]
    return "vi" if vi_count > en_count else "en"


# ══════════════════════════════════════════════════════════════════════════════
# User profile vector index
# ══════════════════════════════════════════════════════════════════════════════

def index_user_interests_into_vector():
    """
    Embed user interest topics and upsert into user_profile_vec collection.
    Call this periodically (e.g., after every 5 assistant turns) rather than
    on every message to control embedding cost.
    """
    try:
        from app.vectorstore import embed_texts
        interests = get_user_interests(top_k=20)
        if not interests:
            return
        col    = _get_profile_collection()
        texts  = [i["topic"] for i in interests]
        embs   = embed_texts(texts)
        col.upsert(
            ids   =[f"interest_{t.replace(' ','_')[:40]}" for t in texts],
            embeddings=embs,
            documents =texts,
            metadatas =[{"score": i["score"]} for i in interests],
        )
    except Exception as e:
        print(f"Profile index error: {e}")


def score_chunk_by_user_profile(chunk_text: str) -> float:
    """
    Return a personalization affinity score [0, 1] for a chunk.
    Higher = more aligned with user's historical interests.
    Used as a soft re-ranking signal (small weight).
    """
    try:
        from app.vectorstore import embed_texts
        col = _get_profile_collection()
        if col.count() == 0:
            return 0.5
        emb    = embed_texts([chunk_text[:500]])[0]
        result = col.query(
            query_embeddings=[emb],
            n_results=1,
            include=["distances"],
        )
        if result["distances"] and result["distances"][0]:
            return round(1 - result["distances"][0][0], 3)
    except Exception:
        pass
    return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def auto_title_from_query(query: str) -> str:
    title = query.strip()
    return title[:47] + "..." if len(title) > 50 else title


def get_entity_conversation_history(entity_id: str, limit: int = 5) -> list[dict]:
    """All past conversations where this entity was mentioned — for entity detail page."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT em.conversation_id, c.title, em.created_at
            FROM entity_mentions em
            JOIN conversations c ON c.id = em.conversation_id
            WHERE em.entity_id = ?
            ORDER BY em.created_at DESC
            LIMIT ?
            """,
            (entity_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _now() -> str:
    return datetime.utcnow().isoformat()


# Init DB on import
init_db()