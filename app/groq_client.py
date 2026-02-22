"""
Groq API client — multilingual + personalization edition.

Changes vs. v0.5:
• KG extraction now requests cross-lingual aliases for each entity
  (e.g., "machine learning" → alias "học máy") — fed into graph.add_entity_alias()
• rag_answer() accepts personalization_context and weaves it into the system prompt
• rerank_chunks() adds a small user-profile affinity boost
• build_community_summary() — LLM summarizes a community of entities for global search
"""
import json
from groq import Groq
from app.config import settings

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def chat(messages, model=None, temperature=0.1, max_tokens=2048) -> str:
    model = model or settings.groq_chat_model
    resp  = get_client().chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ── Query bundle ──────────────────────────────────────────────────────────────

def build_query_bundle(
    query: str,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    use_stepback: bool = True,
    context_hint: str = "",
) -> dict:
    try:
        from app.multilingual import build_multilingual_query_bundle
        return build_multilingual_query_bundle(
            query=query,
            use_multi_query=use_multi_query,
            use_hyde=use_hyde,
            use_stepback=use_stepback,
        )
    except Exception as e:
        print(f"Query bundle error: {e}")
        return {
            "original": query, "detected_lang": "en",
            "query_vi": query, "query_en": query,
            "multi_vi": [query], "multi_en": [query],
            "hyde_vi": "", "hyde_en": "",
            "stepback_vi": "", "stepback_en": "",
            "all_search_queries_vi": [query],
            "all_search_queries_en": [query],
            "all_search_queries": [query],
            "multi": [query], "hyde": "", "stepback": "",
        }


# ── Reranking with personalization boost ─────────────────────────────────────

def rerank_chunks(
    query:             str,
    chunks:            list[dict],
    top_k:             int  = 5,
    query_translated:  str  = "",
    use_profile_boost: bool = True,
) -> list[dict]:
    """
    LLM reranker + optional user-profile affinity boost.
    Score = 0.35 * rrf_score + 0.55 * llm_relevance + 0.10 * profile_affinity
    """
    if not chunks or len(chunks) <= top_k:
        return chunks

    query_header = f"Question: {query}"
    if query_translated and query_translated.strip() and query_translated != query:
        query_header += f"\nAlso expressed as: {query_translated}"

    numbered = "\n\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(chunks))
    prompt   = (
        f"{query_header}\n\n"
        f"Rate each passage's relevance (0-10). "
        f"Passages may be in a different language — judge meaning, not language.\n"
        f"Return ONLY JSON: {{\"scores\": [n, n, ...]}}\n\n"
        f"Passages:\n{numbered}"
    )
    try:
        resp       = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0, max_tokens=128,
        )
        raw        = json.loads(resp.choices[0].message.content)
        scores     = raw.get("scores", raw) if isinstance(raw, dict) else raw
        llm_scores = [float(s) / 10.0 for s in scores]
    except Exception as e:
        print(f"Rerank LLM error: {e}")
        llm_scores = [0.5] * len(chunks)

    # Optional profile affinity
    profile_scores = [0.5] * len(chunks)
    if use_profile_boost:
        try:
            from app.history import score_chunk_by_user_profile
            profile_scores = [score_chunk_by_user_profile(c["text"]) for c in chunks]
        except Exception:
            pass

    if len(llm_scores) == len(chunks):
        for i, c in enumerate(chunks):
            base   = c.get("rrf_score", c.get("score", 0.5))
            llm_s  = llm_scores[i]
            prof_s = profile_scores[i]
            c["llm_relevance"]    = round(llm_s, 3)
            c["profile_affinity"] = round(prof_s, 3)
            if use_profile_boost:
                c["rerank_score"] = round(0.35 * base + 0.55 * llm_s + 0.10 * prof_s, 4)
            else:
                c["rerank_score"] = round(0.35 * base + 0.65 * llm_s, 4)
    else:
        for c in chunks:
            c["rerank_score"] = c.get("rrf_score", c.get("score", 0))

    return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ── Entity summarization ──────────────────────────────────────────────────────

def summarize_entity(entity_name: str, entity_type: str, contexts: list[str]) -> str:
    if not contexts:
        return ""
    ctx_text = "\n---\n".join(contexts[:6])
    prompt   = (
        f"Based on these excerpts, write a concise 2-3 sentence factual summary "
        f"of the {entity_type} entity \"{entity_name}\". "
        f"Focus on key attributes, role, or significance. Return ONLY the summary.\n\n"
        f"Excerpts:\n{ctx_text}"
    )
    try:
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Entity summary error: {e}"); return ""


# ── Community summary ─────────────────────────────────────────────────────────

def build_community_summary(
    community_entities: list[dict],
    sample_text: str = "",
) -> str:
    """
    Generate a thematic summary for a community of related entities.
    Enables global search: broad queries match community summaries even when
    no specific entity is found in the query text.
    """
    if not community_entities:
        return ""
    entity_lines = "\n".join(
        f"• {e['name']} ({e['type']})"
        + (f": {e['summary'][:100]}" if e.get("summary") else "")
        for e in community_entities[:15]
    )
    sample_part = f"\n\nSample text:\n{sample_text[:500]}" if sample_text else ""
    prompt = (
        f"These entities form a connected knowledge cluster:\n{entity_lines}"
        f"{sample_part}\n\n"
        f"Write a 2-3 sentence thematic summary of what this cluster is about. "
        f"Return ONLY the summary."
    )
    try:
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Community summary error: {e}"); return ""


# ── Bilingual hypothetical questions ─────────────────────────────────────────

def generate_chunk_questions(chunk_text: str, n: int = 3) -> list[str]:
    """Generate hypothetical questions in BOTH Vietnamese and English."""
    prompt = (
        f"Generate {n} concise questions in ENGLISH and {n} in VIETNAMESE "
        f"that the following passage directly answers. "
        f"Questions should sound like real user queries. "
        f"Return ONLY JSON: {{\"questions_en\": [...], \"questions_vi\": [...]}}\n\n"
        f"Text:\n{chunk_text[:1200]}"
    )
    try:
        resp      = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5, max_tokens=512,
        )
        data      = json.loads(resp.choices[0].message.content)
        questions = []
        for key in ("questions_en", "questions_vi"):
            for q in data.get(key, []):
                if isinstance(q, str) and q.strip():
                    questions.append(q.strip())
        return questions[: n * 2]
    except Exception as e:
        print(f"Question generation error: {e}"); return []


# ── KG extraction (alias-aware) ───────────────────────────────────────────────

def extract_knowledge_graph(text: str, chunk_map: dict = None) -> dict:
    client = get_client()
    if chunk_map:
        return _extract_per_chunk(client, chunk_map)
    return _extract_full_text(client, text)


def _extract_per_chunk(client, chunk_map: dict) -> dict:
    all_entities, all_relations = [], []
    seen_entity_chunk: set[tuple] = set()

    for cidx, chunk_text in list(chunk_map.items())[:15]:
        prompt = (
            f"Extract key entities and relationships from this text chunk.\n"
            f"Text may be Vietnamese, English, or both.\n"
            f"For each entity, include 'aliases': cross-lingual alternate names "
            f"(e.g. if name is 'machine learning', add alias 'học máy'; "
            f"if name is 'trí tuệ nhân tạo', add alias 'artificial intelligence').\n"
            f"Return JSON:\n"
            f"- \"entities\": [{{\"name\": str, "
            f"\"type\": \"PERSON|ORGANIZATION|CONCEPT|PLACE|DATE|TECHNOLOGY|TOPIC\", "
            f"\"context\": str, \"aliases\": [str]}}]\n"
            f"- \"relations\": [{{\"source\": str, \"target\": str, \"relation\": str}}]\n"
            f"Max 8 entities, 6 relations. ONLY valid JSON.\n\n"
            f"Text: {chunk_text[:1500]}"
        )
        try:
            resp = client.chat.completions.create(
                model=settings.groq_fast_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=1200,
            )
            data = json.loads(resp.choices[0].message.content)
            for ent in data.get("entities", []):
                name = ent.get("name", "").strip()
                if not name:
                    continue
                key = (name.lower(), cidx)
                if key not in seen_entity_chunk:
                    seen_entity_chunk.add(key)
                    ent["chunk_index"] = cidx
                    all_entities.append(ent)
            all_relations.extend(data.get("relations", []))
        except Exception as e:
            print(f"KG chunk {cidx} error: {e}")

    return {"entities": all_entities[:40], "relations": all_relations[:50]}


def _extract_full_text(client, text: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract knowledge graph. Text may be Vietnamese or English. "
                "For each entity, include 'aliases' list with cross-lingual names. "
                "Max 20 entities, 30 relations. ONLY JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract from:\n\n{text[:6000]}\n\n"
                f"JSON: {{\"entities\": [{{\"name\": str, \"type\": str, "
                f"\"context\": str, \"aliases\": [str]}}], "
                f"\"relations\": [{{\"source\": str, \"target\": str, \"relation\": str}}]}}"
            ),
        },
    ]
    resp = client.chat.completions.create(
        model=settings.groq_fast_model, messages=messages,
        response_format={"type": "json_object"}, temperature=0.0, max_tokens=2048,
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return {"entities": data.get("entities", []), "relations": data.get("relations", [])}
    except Exception:
        return {"entities": [], "relations": []}


# ── RAG answer (personalization-aware) ───────────────────────────────────────

def rag_answer(
    query:                   str,
    context_chunks:          list[dict],
    graph_context:           dict       = None,
    past_exchanges:          list[dict] = None,
    recent_turns:            list[dict] = None,
    model:                   str        = None,
    detected_lang:           str        = "en",
    personalization_context: dict       = None,
) -> str:
    model = model or settings.groq_chat_model

    context_parts = []
    for i, c in enumerate(context_chunks[:6]):
        meta     = c.get("metadata", {})
        loc      = _fmt_location(meta)
        origin   = " (graph)" if c.get("from_graph") else ""
        lang_tag = f" [{c.get('search_lang', '?').upper()}]" if c.get("search_lang") else ""
        context_parts.append(f"[{i+1}] {loc}{origin}{lang_tag}\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    graph_info = ""
    if graph_context:
        ents = graph_context.get("entities", [])
        if ents:
            lines = [
                f"• {e['name']} ({e['type']})"
                + (f": {e['summary']}" if e.get("summary") else "")
                for e in ents[:8]
            ]
            graph_info = "\n\nKnowledge graph entities:\n" + "\n".join(lines)
        rels = graph_context.get("relations", [])[:5]
        if rels:
            graph_info += "\n\nRelationships:\n" + "\n".join(
                f"  {r.get('from_id','')} → {r.get('relation','')} → {r.get('to_name','')}"
                for r in rels
            )

    history_hint = ""
    if past_exchanges:
        excerpts = [p["text"][:200] for p in past_exchanges[:2] if p.get("text")]
        if excerpts:
            history_hint = "\n\nRelevant past context:\n" + "\n".join(f"- {e}" for e in excerpts)

    persona_hint = ""
    if personalization_context:
        ps = personalization_context.get("personalization_str", "")
        if ps:
            persona_hint = f"\n\nUser context: {ps}"

    if detected_lang == "vi":
        lang_instruction = (
            "The user's question is in Vietnamese. "
            "ALWAYS respond in Vietnamese regardless of source language. "
        )
    elif detected_lang == "mixed":
        lang_instruction = "Question is mixed VI/EN. Respond primarily in Vietnamese. "
    else:
        lang_instruction = "Respond in English. "

    system_prompt = (
        f"You are a helpful personal document assistant with memory and personalization. "
        f"Answer using the document passages and knowledge graph context provided. "
        f"{lang_instruction}"
        f"Cite passages as [1], [2], etc. "
        f"Entity summaries provide background knowledge — use them to enrich answers. "
        f"If a passage is in a different language, translate its content naturally. "
        f"Be concise and accurate. Say so if context is insufficient."
    )

    user_prompt = (
        f"Document passages:\n{context}"
        f"{graph_info}{history_hint}{persona_hint}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite with [1], [2], etc.):"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if recent_turns:
        for turn in recent_turns[:-1][-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})
    return chat(messages, model=model, temperature=0.1)


def _fmt_location(meta: dict) -> str:
    parts = []
    if meta.get("filename"):      parts.append(meta["filename"])
    if meta.get("section"):       parts.append(f'§ {meta["section"][:60]}')
    if meta.get("page_estimate"): parts.append(f'p.{meta["page_estimate"]}')
    if meta.get("start_char") is not None:
        parts.append(f'chars {meta["start_char"]}–{meta["end_char"]}')
    return " · ".join(parts) or "unknown"


# ── Reasoning query ───────────────────────────────────────────────────────────

def reasoning_query(query: str, context: str, detected_lang: str = "en") -> dict:
    lang_note = "Respond in Vietnamese." if detected_lang == "vi" else "Respond in English."
    messages  = [
        {
            "role":    "system",
            "content": f"Expert document analyst. Think step by step. {lang_note}\nFormat:\nANALYSIS: [reasoning]\nANSWER: [answer]",
        },
        {"role": "user", "content": f"Context:\n{context[:5000]}\n\nQuestion: {query}"},
    ]
    resp    = get_client().chat.completions.create(
        model=settings.groq_chat_model, messages=messages,
        temperature=0.2, max_tokens=4096,
    )
    content  = resp.choices[0].message.content
    analysis, answer = "", content
    if "ANALYSIS:" in content and "ANSWER:" in content:
        parts    = content.split("ANSWER:", 1)
        analysis = parts[0].replace("ANALYSIS:", "").strip()
        answer   = parts[1].strip()
    return {"analysis": analysis, "answer": answer, "full": content}


# ── Structured output ─────────────────────────────────────────────────────────

def structured_output(prompt: str, schema: dict, context: str = "", model: str = None) -> dict:
    model = model or settings.groq_chat_model
    sys_p = f"Precise data extraction. Return ONLY JSON matching:\n{json.dumps(schema, indent=2)}\nUse null for missing."
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"Context:\n{context}\n\nInstruction: {prompt}"},
    ]
    resp = get_client().chat.completions.create(
        model=model, messages=messages,
        response_format={"type": "json_object"}, temperature=0.0, max_tokens=4096,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"error": "JSON parse failed", "raw": resp.choices[0].message.content}


# ── TTS ───────────────────────────────────────────────────────────────────────

def text_to_speech(text: str, voice: str = None) -> bytes:
    voice = voice or settings.groq_tts_voice
    if len(text) > 3500:
        text = text[:3500] + "..."
    resp = get_client().audio.speech.create(
        model=settings.groq_tts_model, voice=voice, input=text, response_format="wav",
    )
    return resp.read()


TTS_VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]