"""
Multilingual query pipeline — Vietnamese ↔ English.

Strategy:
1. Detect query language (VI / EN / mixed) using Unicode heuristic — no extra dep.
2. Translate query to the OTHER language via Groq fast model.
3. Build a cross-lingual query bundle: multi-query variants + HyDE + step-back,
   all generated in BOTH languages so embeddings are always language-matched.
4. RRF (Reciprocal Rank Fusion) merges ranked lists from each language pass.

Why this works:
  • Monolingual embeddings score highest within their own language space.
    A Vietnamese query against an English chunk gets poor cosine sim even when
    the meaning is identical — and vice-versa.
  • By searching with BOTH language variants we guarantee language-matched
    recall from every language direction.
  • RRF is rank-based and therefore immune to score-scale differences between
    language-specific embedding distributions.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ── Language detection ────────────────────────────────────────────────────────

_VI_CHARS = re.compile(
    r"[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
    r"ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ]",
    re.UNICODE,
)


def detect_language(text: str) -> str:
    """
    Return 'vi', 'en', or 'mixed'.
    Heuristic: ratio of Vietnamese diacritical characters to total alpha chars.
    Threshold at 3 % → vi; below → en; texts with both → mixed.
    """
    if not text:
        return "en"
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return "en"
    vi_chars = _VI_CHARS.findall(text)
    ratio = len(vi_chars) / len(alpha)
    if ratio > 0.03:
        return "vi"
    # Check for some common Vietnamese words even without diacritics
    plain_vi = re.search(
        r"\b(cua|cai|mot|hai|ba|bon|nam|bay|tam|chin|muoi|va|hoac|khong|co|la|len|xuong|nha|nuoc)\b",
        text.lower(),
    )
    if plain_vi and ratio > 0.01:
        return "vi"
    return "en"


def is_mixed(text: str) -> bool:
    """True when text contains both English and Vietnamese content."""
    vi = _VI_CHARS.search(text) is not None
    en = bool(re.search(r"[a-zA-Z]{3,}", text))
    return vi and en


# ── Translation via Groq ──────────────────────────────────────────────────────

def translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """
    Translate text to target_lang ('vi' | 'en') using Groq fast model.
    Returns original text on failure.
    """
    from app.config import settings
    if not settings.groq_api_key or not text.strip():
        return text

    lang_name = {"vi": "Vietnamese", "en": "English"}.get(target_lang, "English")
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Return ONLY the translation, no explanation.\n\n{text}"
    )
    try:
        from app.groq_client import get_client
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        translated = resp.choices[0].message.content.strip()
        # Sanity: ignore if model returned a very different length (likely hallucination)
        if 0.2 < len(translated) / max(len(text), 1) < 5.0:
            return translated
    except Exception as e:
        print(f"Translation error: {e}")
    return text


# ── Cross-lingual query expansion ────────────────────────────────────────────

def build_multilingual_query_bundle(
    query: str,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    use_stepback: bool = True,
) -> dict:
    """
    Produce a language-aware query bundle.

    Returns:
    {
        "original":         str,
        "detected_lang":    "vi" | "en" | "mixed",
        "query_vi":         str,   # query in Vietnamese
        "query_en":         str,   # query in English
        "multi_vi":         [str], # multi-query variants in Vietnamese
        "multi_en":         [str], # multi-query variants in English
        "hyde_vi":          str,   # HyDE passage in Vietnamese
        "hyde_en":          str,   # HyDE passage in English
        "stepback_vi":      str,
        "stepback_en":      str,
        "all_search_queries_vi": [str],  # all VI queries for vector search
        "all_search_queries_en": [str],  # all EN queries for vector search
    }
    """
    from app import groq_client

    lang = detect_language(query)

    # Build base translations
    if lang == "vi":
        query_vi = query
        query_en = translate(query, "en")
    elif lang == "en":
        query_en = query
        query_vi = translate(query, "vi")
    else:  # mixed — treat original as both
        query_vi = query
        query_en = query

    bundle: dict = {
        "original":      query,
        "detected_lang": lang,
        "query_vi":      query_vi,
        "query_en":      query_en,
    }

    # ── Multi-query variants ──────────────────────────────────────────────────
    if use_multi_query:
        multi_vi = _rewrite_multi_query_lang(query_vi, lang="vi", n=3)
        multi_en = _rewrite_multi_query_lang(query_en, lang="en", n=3)
    else:
        multi_vi = [query_vi]
        multi_en = [query_en]
    bundle["multi_vi"] = multi_vi
    bundle["multi_en"] = multi_en

    # ── HyDE (hypothetical document passage) ─────────────────────────────────
    hyde_vi = hyde_en = ""
    if use_hyde:
        hyde_vi = _rewrite_hyde_lang(query_vi, lang="vi")
        hyde_en = _rewrite_hyde_lang(query_en, lang="en")
    bundle["hyde_vi"] = hyde_vi
    bundle["hyde_en"] = hyde_en

    # ── Step-back ─────────────────────────────────────────────────────────────
    stepback_vi = stepback_en = ""
    if use_stepback:
        # Generate step-back in original lang then translate
        if lang == "vi":
            stepback_vi = _rewrite_stepback_lang(query_vi, lang="vi")
            stepback_en = translate(stepback_vi, "en") if stepback_vi else ""
        else:
            stepback_en = _rewrite_stepback_lang(query_en, lang="en")
            stepback_vi = translate(stepback_en, "vi") if stepback_en else ""
    bundle["stepback_vi"] = stepback_vi
    bundle["stepback_en"] = stepback_en

    # ── Aggregate per language ────────────────────────────────────────────────
    bundle["all_search_queries_vi"] = _dedup_queries(
        multi_vi, [hyde_vi] if hyde_vi else [], [stepback_vi] if stepback_vi else []
    )
    bundle["all_search_queries_en"] = _dedup_queries(
        multi_en, [hyde_en] if hyde_en else [], [stepback_en] if stepback_en else []
    )

    # Backward-compat key used by old pipeline
    bundle["all_search_queries"] = (
        bundle["all_search_queries_vi"] + bundle["all_search_queries_en"]
    )
    bundle["multi"]    = multi_vi + multi_en
    bundle["hyde"]     = hyde_vi or hyde_en
    bundle["stepback"] = stepback_vi or stepback_en

    return bundle


# ── Language-specific rewrite helpers ────────────────────────────────────────

def _rewrite_multi_query_lang(query: str, lang: str = "en", n: int = 3) -> list[str]:
    """Generate n alternative queries in the given language."""
    import json
    from app.config import settings
    if not settings.groq_api_key:
        return [query]

    lang_name = "Vietnamese" if lang == "vi" else "English"
    prompt = (
        f"Generate {n} different search queries in {lang_name} for the same information need. "
        f"Use different vocabulary or angles. "
        f"Return ONLY JSON: {{\"queries\": [\"...\", \"...\"]}}\n\n"
        f"Original ({lang_name}): {query}"
    )
    try:
        from app.groq_client import get_client
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=512,
        )
        variants = json.loads(resp.choices[0].message.content).get("queries", [])
        seen, result = {query.lower()}, [query]
        for v in variants:
            if isinstance(v, str) and v.strip() and v.lower() not in seen:
                seen.add(v.lower())
                result.append(v.strip())
        return result[: n + 1]
    except Exception as e:
        print(f"Multi-query ({lang}) error: {e}")
        return [query]


def _rewrite_hyde_lang(query: str, lang: str = "en") -> str:
    """Generate a hypothetical document passage in the given language."""
    from app.config import settings
    if not settings.groq_api_key:
        return query

    lang_name = "Vietnamese" if lang == "vi" else "English"
    prompt = (
        f"Write a short factual passage in {lang_name} (2-4 sentences) that directly answers the question. "
        f"Write like an excerpt from a document — not conversational.\n\n"
        f"Question: {query}\nPassage:"
    )
    try:
        from app.groq_client import get_client
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"HyDE ({lang}) error: {e}")
        return query


def _rewrite_stepback_lang(query: str, lang: str = "en") -> str:
    """Produce a broader step-back question in the given language."""
    from app.config import settings
    if not settings.groq_api_key:
        return query

    lang_name = "Vietnamese" if lang == "vi" else "English"
    prompt = (
        f"Given the specific question in {lang_name}, produce ONE broader general question "
        f"for background context. Return ONLY the broader question in {lang_name}.\n\n"
        f"Specific: {query}"
    )
    try:
        from app.groq_client import get_client
        resp = get_client().chat.completions.create(
            model=settings.groq_fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=128,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Step-back ({lang}) error: {e}")
        return query


# ── RRF merge ─────────────────────────────────────────────────────────────────

def rrf_merge(
    *ranked_lists: list[dict],
    key_fn=None,
    k: int = 60,
    top_n: int | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over any number of ranked chunk lists.

    key_fn: callable that returns a unique string key per chunk.
            Defaults to "{doc_id}_{chunk_index}".

    Each list contributes  1 / (k + rank)  to each document's total RRF score.
    The final list is sorted by descending RRF score.

    Benefits vs. simple score averaging:
      • Rank-based → immune to score scale differences between
        language-specific embedding distributions.
      • A chunk that appears near the top in multiple lists (e.g. once from
        VI query, once from EN query) is heavily promoted.
      • Handles deduplication naturally.
    """
    if key_fn is None:
        def key_fn(c):
            m = c.get("metadata", {})
            return f"{m.get('doc_id', '')}_{m.get('chunk_index', '')}"

    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, dict]  = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = key_fn(chunk)
            rrf_scores[key]  = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunk_store:
                chunk_store[key] = chunk

    merged = sorted(chunk_store.values(), key=lambda c: rrf_scores[key_fn(c)], reverse=True)
    for chunk in merged:
        chunk["rrf_score"] = round(rrf_scores[key_fn(chunk)], 6)
        # Surface the best single-list score too (used downstream for display)
        chunk.setdefault("score", chunk["rrf_score"])

    return merged[:top_n] if top_n else merged


# ── Utility ───────────────────────────────────────────────────────────────────

def _dedup_queries(*query_lists: list[str]) -> list[str]:
    seen, result = set(), []
    for lst in query_lists:
        for q in lst:
            if q and q.strip() and q.lower() not in seen:
                seen.add(q.lower())
                result.append(q.strip())
    return result