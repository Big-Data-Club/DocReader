"""
Kuzu knowledge graph — GraphRAG v2.
"""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import kuzu

from app.config import settings

_db   = None
_conn = None


# ── DB bootstrap ──────────────────────────────────────────────────────────────

def get_db() -> kuzu.Connection:
    global _db, _conn
    if _db is None:
        _db   = kuzu.Database(settings.kuzu_dir + "/db")
        _conn = kuzu.Connection(_db)
        _init_schema()
    return _conn


def _init_schema():
    c = _conn

    c.execute("""
        CREATE NODE TABLE IF NOT EXISTS Document(
            doc_id   STRING,
            filename STRING,
            PRIMARY KEY(doc_id)
        )
    """)

    c.execute("""
        CREATE NODE TABLE IF NOT EXISTS Entity(
            entity_id    STRING,
            name         STRING,
            entity_type  STRING,
            summary      STRING,
            community_id STRING,
            PRIMARY KEY(entity_id)
        )
    """)

    # Alias node — one per alternate name/language variant
    c.execute("""
        CREATE NODE TABLE IF NOT EXISTS EntityAlias(
            alias_id   STRING,
            alias_text STRING,
            lang       STRING,
            PRIMARY KEY(alias_id)
        )
    """)

    c.execute("""
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            chunk_id    STRING,
            doc_id      STRING,
            chunk_index INT64,
            PRIMARY KEY(chunk_id)
        )
    """)

    # ── Relationships ─────────────────────────────────────────────────────────

    c.execute("""
        CREATE REL TABLE IF NOT EXISTS MENTIONED_IN(
            FROM Entity TO Document,
            context STRING
        )
    """)

    c.execute("""
        CREATE REL TABLE IF NOT EXISTS APPEARS_IN(
            FROM Entity TO Chunk,
            context STRING
        )
    """)

    c.execute("""
        CREATE REL TABLE IF NOT EXISTS RELATED_TO(
            FROM Entity TO Entity,
            relation STRING,
            doc_id   STRING,
            weight   INT64
        )
    """)

    # Alias → canonical entity
    c.execute("""
        CREATE REL TABLE IF NOT EXISTS ALIAS_OF(
            FROM EntityAlias TO Entity
        )
    """)

    # Implicit co-occurrence: entities sharing a chunk
    c.execute("""
        CREATE REL TABLE IF NOT EXISTS COOCCURS_WITH(
            FROM Entity TO Entity,
            chunk_id STRING,
            doc_id   STRING
        )
    """)


# ── Canonical entity ID ───────────────────────────────────────────────────────

def _canonical_id(name: str, etype: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return f"{etype.lower()}_{norm}"


def _alias_id(alias_text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "_", alias_text.strip().lower()).strip("_")
    return f"alias_{norm}"


# ── Document node ─────────────────────────────────────────────────────────────

def add_document_node(doc_id: str, filename: str):
    conn = get_db()
    conn.execute(
        "MERGE (d:Document {doc_id: $doc_id}) SET d.filename = $filename",
        {"doc_id": doc_id, "filename": filename},
    )


# ── Entity alias management ───────────────────────────────────────────────────

def add_entity_alias(entity_id: str, alias_text: str, lang: str = "en"):
    """
    Register an alternate name for an entity.
    Called during KG extraction when the LLM provides a translated name,
    or when the multilingual pipeline detects a cross-lingual pair.
    """
    if not alias_text or not alias_text.strip():
        return
    conn = get_db()
    aid  = _alias_id(alias_text)
    conn.execute(
        "MERGE (a:EntityAlias {alias_id: $aid}) SET a.alias_text = $txt, a.lang = $lang",
        {"aid": aid, "txt": alias_text.strip(), "lang": lang},
    )
    # Link alias → entity (ignore if entity doesn't exist yet)
    try:
        conn.execute(
            """
            MATCH (a:EntityAlias {alias_id: $aid}), (e:Entity {entity_id: $eid})
            MERGE (a)-[:ALIAS_OF]->(e)
            """,
            {"aid": aid, "eid": entity_id},
        )
    except Exception:
        pass


def get_entity_aliases(entity_id: str) -> list[str]:
    conn = get_db()
    result = conn.execute(
        """
        MATCH (a:EntityAlias)-[:ALIAS_OF]->(e:Entity {entity_id: $eid})
        RETURN a.alias_text
        """,
        {"eid": entity_id},
    )
    aliases = []
    while result.has_next():
        v = result.get_next()[0]
        if v:
            aliases.append(v)
    return aliases


def resolve_alias(alias_text: str) -> str | None:
    """Return entity_id if alias_text matches a registered alias, else None."""
    conn = get_db()
    aid  = _alias_id(alias_text)
    result = conn.execute(
        """
        MATCH (a:EntityAlias {alias_id: $aid})-[:ALIAS_OF]->(e:Entity)
        RETURN e.entity_id
        """,
        {"aid": aid},
    )
    return result.get_next()[0] if result.has_next() else None


# ── Add entities + relations ──────────────────────────────────────────────────

def add_entities_and_relations(doc_id: str, graph_data: dict):
    """
    graph_data:
        entities:  [{name, type, context, chunk_index?, aliases?: [str]}]
        relations: [{source, target, relation}]

    Additions vs v1:
    • Registers `aliases` list on each entity (cross-lingual names from LLM)
    • Builds COOCCURS_WITH edges between entities sharing the same chunk
    • Increments RELATED_TO weight on repeated relations
    """
    conn        = get_db()
    name_to_id: dict[str, str] = {}
    chunk_to_entities: dict[int, list[str]] = {}  # cidx → [entity_id]

    for ent in graph_data.get("entities", []):
        name    = ent.get("name", "").strip()
        etype   = ent.get("type", "CONCEPT").strip().upper()
        context = ent.get("context", "")
        cidx    = ent.get("chunk_index")
        aliases = ent.get("aliases", [])
        if not name:
            continue

        eid = _canonical_id(name, etype)
        name_to_id[name.lower()] = eid

        # Upsert entity
        conn.execute(
            """
            MERGE (e:Entity {entity_id: $eid})
            SET e.name = $name, e.entity_type = $etype
            """,
            {"eid": eid, "name": name, "etype": etype},
        )

        # Entity → Document
        conn.execute(
            """
            MATCH (e:Entity {entity_id: $eid}), (d:Document {doc_id: $doc_id})
            MERGE (e)-[r:MENTIONED_IN]->(d)
            SET r.context = $context
            """,
            {"eid": eid, "doc_id": doc_id, "context": context},
        )

        # Entity → Chunk
        if cidx is not None:
            chunk_id = f"{doc_id}_{cidx}"
            conn.execute(
                "MERGE (c:Chunk {chunk_id: $cid}) SET c.doc_id=$doc_id, c.chunk_index=$cidx",
                {"cid": chunk_id, "doc_id": doc_id, "cidx": int(cidx)},
            )
            conn.execute(
                """
                MATCH (e:Entity {entity_id: $eid}), (c:Chunk {chunk_id: $cid})
                MERGE (e)-[r:APPEARS_IN]->(c)
                SET r.context = $context
                """,
                {"eid": eid, "cid": chunk_id, "context": context},
            )
            chunk_to_entities.setdefault(cidx, []).append(eid)

        # Register aliases (cross-lingual names)
        for alias in aliases:
            if isinstance(alias, str) and alias.strip() and alias.lower() != name.lower():
                lang = _guess_lang(alias)
                add_entity_alias(eid, alias, lang=lang)

    # Relations with weight
    for rel in graph_data.get("relations", []):
        src      = rel.get("source", "").strip()
        tgt      = rel.get("target", "").strip()
        relation = rel.get("relation", "RELATED_TO")
        if not src or not tgt:
            continue
        src_id = name_to_id.get(src.lower()) or _canonical_id(src, "CONCEPT")
        tgt_id = name_to_id.get(tgt.lower()) or _canonical_id(tgt, "CONCEPT")
        # Upsert with weight increment
        try:
            conn.execute(
                """
                MATCH (a:Entity {entity_id: $src}), (b:Entity {entity_id: $tgt})
                MERGE (a)-[r:RELATED_TO {doc_id: $doc_id}]->(b)
                SET r.relation = $rel, r.weight = coalesce(r.weight, 0) + 1
                """,
                {"src": src_id, "tgt": tgt_id, "doc_id": doc_id, "rel": relation},
            )
        except Exception as e:
            print(f"Relation upsert error: {e}")

    # COOCCURS_WITH: entities sharing the same chunk
    for cidx, eids in chunk_to_entities.items():
        if len(eids) < 2:
            continue
        chunk_id = f"{doc_id}_{cidx}"
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                a, b = eids[i], eids[j]
                try:
                    conn.execute(
                        """
                        MATCH (a:Entity {entity_id: $a}), (b:Entity {entity_id: $b})
                        MERGE (a)-[r:COOCCURS_WITH {chunk_id: $cid}]->(b)
                        SET r.doc_id = $doc_id
                        """,
                        {"a": a, "b": b, "cid": chunk_id, "doc_id": doc_id},
                    )
                except Exception:
                    pass


# ── Entity summaries ──────────────────────────────────────────────────────────

def update_entity_summary(entity_id: str, summary: str):
    get_db().execute(
        "MATCH (e:Entity {entity_id: $eid}) SET e.summary = $summary",
        {"eid": entity_id, "summary": summary},
    )


def get_entities_without_summary(limit: int = 50) -> list[dict]:
    conn   = get_db()
    result = conn.execute(
        """
        MATCH (e:Entity)
        WHERE e.summary IS NULL OR e.summary = ''
        RETURN e.entity_id, e.name, e.entity_type
        LIMIT $limit
        """,
        {"limit": limit},
    )
    rows = []
    while result.has_next():
        r = result.get_next()
        rows.append({"entity_id": r[0], "name": r[1], "type": r[2]})
    return rows


def get_entity_contexts(entity_id: str, max_contexts: int = 5) -> list[str]:
    conn   = get_db()
    result = conn.execute(
        "MATCH (e:Entity {entity_id: $eid})-[r:MENTIONED_IN]->() RETURN r.context LIMIT $lim",
        {"eid": entity_id, "lim": max_contexts},
    )
    ctxs = []
    while result.has_next():
        v = result.get_next()[0]
        if v:
            ctxs.append(v)
    return ctxs


# ── Community detection ───────────────────────────────────────────────────────

def build_communities(min_size: int = 2) -> dict[str, str]:
    """
    Assign community_id to entities via BFS over RELATED_TO + COOCCURS_WITH edges.

    Returns {entity_id: community_id}.
    Entities in the same strongly-connected component get the same community_id.
    Singletons (no edges) get their own community.

    Call this after ingesting a batch of documents.
    """
    conn   = get_db()
    result = conn.execute(
        "MATCH (a:Entity)-[:RELATED_TO|COOCCURS_WITH]->(b:Entity) RETURN a.entity_id, b.entity_id"
    )
    adjacency: dict[str, set[str]] = {}
    while result.has_next():
        a, b = result.get_next()
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # Also grab isolated entities
    result = conn.execute("MATCH (e:Entity) RETURN e.entity_id")
    all_ids = set()
    while result.has_next():
        all_ids.add(result.get_next()[0])
    for eid in all_ids:
        adjacency.setdefault(eid, set())

    # BFS
    visited:   dict[str, str] = {}
    community_leader: dict[str, list[str]] = {}

    for start in sorted(adjacency.keys()):
        if start in visited:
            continue
        queue     = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited[node] = start
            component.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        community_leader[start] = component

    # Persist community_id to Kuzu
    for leader, members in community_leader.items():
        cid = f"c_{leader[:20]}"
        for eid in members:
            try:
                conn.execute(
                    "MATCH (e:Entity {entity_id: $eid}) SET e.community_id = $cid",
                    {"eid": eid, "cid": cid},
                )
            except Exception:
                pass

    return visited


def get_community_entities(community_id: str) -> list[dict]:
    conn   = get_db()
    result = conn.execute(
        """
        MATCH (e:Entity {community_id: $cid})
        RETURN e.entity_id, e.name, e.entity_type, e.summary
        """,
        {"cid": community_id},
    )
    ents = []
    while result.has_next():
        r = result.get_next()
        ents.append({"id": r[0], "name": r[1], "type": r[2], "summary": r[3] or ""})
    return ents


def get_all_communities() -> list[str]:
    conn   = get_db()
    result = conn.execute(
        "MATCH (e:Entity) WHERE e.community_id IS NOT NULL RETURN DISTINCT e.community_id"
    )
    ids = []
    while result.has_next():
        ids.append(result.get_next()[0])
    return ids


def update_community_summary(community_id: str, summary: str):
    """
    Store community-level summary on the community's representative entity node.
    Used for global search — when query doesn't match specific entities but matches a topic.
    """
    conn = get_db()
    conn.execute(
        """
        MATCH (e:Entity {community_id: $cid})
        SET e.summary = CASE WHEN e.summary IS NULL OR e.summary = ''
                        THEN $summary
                        ELSE e.summary END
        """,
        {"cid": community_id, "summary": summary},
    )


# ── Global search (community-level) ──────────────────────────────────────────

def global_entity_search(query_keywords: list[str], top_k: int = 5) -> list[dict]:
    """
    Global search: find entity summaries that mention any keyword.
    Returns top-k entities with summaries for broad/thematic queries.
    Used when graph local search finds no seed entities.
    """
    conn    = get_db()
    results = []
    seen    = set()

    for kw in query_keywords:
        kw_lower = kw.lower()
        result = conn.execute(
            """
            MATCH (e:Entity)
            WHERE e.summary IS NOT NULL AND e.summary <> ''
            RETURN e.entity_id, e.name, e.entity_type, e.summary, e.community_id
            LIMIT 100
            """
        )
        while result.has_next():
            r = result.get_next()
            eid, name, etype, summary, cid = r
            if eid in seen:
                continue
            if kw_lower in (summary or "").lower() or kw_lower in name.lower():
                seen.add(eid)
                results.append({
                    "id":           eid,
                    "name":         name,
                    "type":         etype,
                    "summary":      summary or "",
                    "community_id": cid or "",
                    "is_seed":      True,
                })
        if len(results) >= top_k:
            break

    return results[:top_k]


# ── Smart query entity extraction ─────────────────────────────────────────────

def extract_query_entities(
    query: str,
    doc_id: str = None,
    translated_query: str = "",
) -> list[str]:
    """
    Multi-strategy entity extraction from a query string.

    Strategy (in order, stops once we have ≥3 matches):
    1. Exact literal match in original query
    2. Exact literal match in translated_query (cross-lingual)
    3. Alias lookup: does any registered alias appear in the query?
    4. Prefix/partial match for entity names > 4 chars

    Returns list of entity names (not IDs) for graph_local_search.
    """
    conn        = get_db()
    query_lower = query.lower()
    alt_lower   = (translated_query or "").lower()

    # Fetch all candidates
    if doc_id:
        result = conn.execute(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {doc_id: $doc_id})
            RETURN e.name, e.entity_id
            """,
            {"doc_id": doc_id},
        )
    else:
        result = conn.execute("MATCH (e:Entity) RETURN e.name, e.entity_id")

    matches: dict[str, str] = {}
    while result.has_next():
        name, eid = result.get_next()
        if not name:
            continue
        name_l = name.lower()
        if name_l in query_lower or (alt_lower and name_l in alt_lower):
            matches[name] = eid
        elif len(name) >= 4:
            for word in re.findall(r"\w+", query_lower + " " + alt_lower):
                if name_l in word or word in name_l:
                    matches[name] = eid
                    break
        if len(matches) >= 10:
            break

    if len(matches) < 3:
        for token in re.findall(r"\w{3,}", query_lower + " " + alt_lower):
            eid = resolve_alias(token)
            if eid:
                name_result = conn.execute(
                    "MATCH (e:Entity {entity_id: $eid}) RETURN e.name",
                    {"eid": eid},
                )
                if name_result.has_next():
                    matches[name_result.get_next()[0]] = eid
            if len(matches) >= 10:
                break

    return list(matches.keys())


# ── GraphRAG Local Search ─────────────────────────────────────────────────────

def graph_local_search(
    query_entity_names: list[str],
    depth: int = 2,
    limit_per_hop: int = 8,
    use_cooccurrence: bool = True,
) -> dict:
    """
    GraphRAG local search with extended traversal.

    Traverses RELATED_TO edges (weighted, primary) and optionally
    COOCCURS_WITH edges (implicit co-occurrence, secondary).

    Returns entities, chunk_refs, relations, and now also community_ids
    so the caller can optionally fetch community-level context.
    """
    conn = get_db()

    # Resolve seed IDs
    seed_ids: set[str] = set()
    for name in query_entity_names:
        result = conn.execute(
            "MATCH (e:Entity) WHERE lower(e.name) = $n RETURN e.entity_id",
            {"n": name.strip().lower()},
        )
        while result.has_next():
            seed_ids.add(result.get_next()[0])
        # Also try alias resolution
        eid = resolve_alias(name)
        if eid:
            seed_ids.add(eid)

    if not seed_ids:
        return {"entities": [], "chunk_refs": [], "relations": [], "communities": []}

    visited_ids: set[str] = set(seed_ids)
    frontier:    set[str] = set(seed_ids)
    all_relations: list[dict] = []

    for hop in range(depth):
        next_frontier: set[str] = set()
        for eid in list(frontier):
            result = conn.execute(
                """
                MATCH (a:Entity {entity_id: $eid})-[r:RELATED_TO]->(b:Entity)
                RETURN b.entity_id, r.relation, b.name, coalesce(r.weight, 1)
                ORDER BY coalesce(r.weight, 1) DESC
                LIMIT $lim
                """,
                {"eid": eid, "lim": limit_per_hop},
            )
            while result.has_next():
                row  = result.get_next()
                bid, rel, bname, weight = row[0], row[1], row[2], row[3]
                if bid not in visited_ids:
                    next_frontier.add(bid)
                all_relations.append({
                    "from_id": eid, "to_id": bid,
                    "to_name": bname, "relation": rel, "weight": weight,
                    "edge_type": "related",
                })

            if use_cooccurrence and hop == 0:
                result = conn.execute(
                    """
                    MATCH (a:Entity {entity_id: $eid})-[r:COOCCURS_WITH]->(b:Entity)
                    RETURN b.entity_id, b.name, r.chunk_id
                    LIMIT $lim
                    """,
                    {"eid": eid, "lim": limit_per_hop // 2},
                )
                while result.has_next():
                    row  = result.get_next()
                    bid, bname, cid = row[0], row[1], row[2]
                    if bid not in visited_ids:
                        next_frontier.add(bid)
                    all_relations.append({
                        "from_id": eid, "to_id": bid,
                        "to_name": bname, "relation": "co-occurs",
                        "edge_type": "cooccurrence", "chunk_id": cid,
                    })

        visited_ids |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    # Collect entity details
    entities: list[dict] = []
    community_ids: set[str] = set()
    for eid in visited_ids:
        result = conn.execute(
            """
            MATCH (e:Entity {entity_id: $eid})
            RETURN e.name, e.entity_type, e.summary, e.community_id
            """,
            {"eid": eid},
        )
        if result.has_next():
            row = result.get_next()
            cid = row[3] or ""
            entities.append({
                "id":           eid,
                "name":         row[0],
                "type":         row[1],
                "summary":      row[2] or "",
                "community_id": cid,
                "is_seed":      eid in seed_ids,
            })
            if cid:
                community_ids.add(cid)

    # Collect reachable chunks
    chunk_refs: list[dict] = []
    seen_chunks: set[str] = set()
    for eid in visited_ids:
        result = conn.execute(
            """
            MATCH (e:Entity {entity_id: $eid})-[:APPEARS_IN]->(c:Chunk)
            RETURN c.chunk_id, c.doc_id, c.chunk_index
            """,
            {"eid": eid},
        )
        while result.has_next():
            row = result.get_next()
            cid, doc_id, cidx = row[0], row[1], row[2]
            if cid not in seen_chunks:
                seen_chunks.add(cid)
                chunk_refs.append({"chunk_id": cid, "doc_id": doc_id, "chunk_index": cidx})

    return {
        "entities":   entities,
        "chunk_refs": chunk_refs,
        "relations":  all_relations,
        "communities": list(community_ids),
    }


# ── Entity neighborhood ─────────────────────────────────────────────

def get_entity_neighborhood(entity_id: str) -> dict:
    """
    Full 2-hop neighborhood for an entity — used in entity detail page
    and for enriching RAG context when a single key entity is identified.
    Returns: entity, aliases, neighbors, chunks, co-occurrences.
    """
    conn = get_db()

    # Core entity
    result = conn.execute(
        "MATCH (e:Entity {entity_id: $eid}) RETURN e.name, e.entity_type, e.summary, e.community_id",
        {"eid": entity_id},
    )
    if not result.has_next():
        return {}
    row    = result.get_next()
    entity = {"id": entity_id, "name": row[0], "type": row[1],
               "summary": row[2] or "", "community_id": row[3] or ""}

    # Aliases
    entity["aliases"] = get_entity_aliases(entity_id)

    # Neighbors (RELATED_TO)
    result = conn.execute(
        """
        MATCH (e:Entity {entity_id: $eid})-[r:RELATED_TO]->(n:Entity)
        RETURN n.entity_id, n.name, n.entity_type, r.relation, coalesce(r.weight, 1)
        ORDER BY coalesce(r.weight, 1) DESC LIMIT 15
        """,
        {"eid": entity_id},
    )
    neighbors = []
    while result.has_next():
        r = result.get_next()
        neighbors.append({"id": r[0], "name": r[1], "type": r[2], "relation": r[3], "weight": r[4]})
    entity["neighbors"] = neighbors

    # Chunks
    result = conn.execute(
        """
        MATCH (e:Entity {entity_id: $eid})-[:APPEARS_IN]->(c:Chunk)
        RETURN c.chunk_id, c.doc_id, c.chunk_index
        """,
        {"eid": entity_id},
    )
    chunk_refs = []
    while result.has_next():
        r = result.get_next()
        chunk_refs.append({"chunk_id": r[0], "doc_id": r[1], "chunk_index": r[2]})
    entity["chunk_refs"] = chunk_refs

    return entity


# ── Graph UI helpers ──────────────────────────────────────────────────────────

def search_entities(query: str, doc_id: str = None, limit: int = 10) -> list[dict]:
    conn        = get_db()
    query_lower = query.lower()
    if doc_id:
        result = conn.execute(
            """
            MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {doc_id: $doc_id})
            RETURN e.name, e.entity_type, e.entity_id, e.summary
            """,
            {"doc_id": doc_id},
        )
    else:
        result = conn.execute(
            "MATCH (e:Entity) RETURN e.name, e.entity_type, e.entity_id, e.summary"
        )
    matches = []
    while result.has_next():
        row = result.get_next()
        name, etype, eid, summary = row[0], row[1], row[2], row[3] or ""
        if query_lower in name.lower():
            matches.append({"name": name, "type": etype, "id": eid, "summary": summary})
        if len(matches) >= limit:
            break
    return matches


def get_graph_for_doc(doc_id: str) -> dict:
    conn = get_db()
    result = conn.execute(
        """
        MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {doc_id: $doc_id})
        RETURN e.name, e.entity_type, e.entity_id
        """,
        {"doc_id": doc_id},
    )
    entities = []
    while result.has_next():
        row = result.get_next()
        entities.append({"name": row[0], "type": row[1], "id": row[2]})

    result = conn.execute(
        """
        MATCH (a:Entity)-[:MENTIONED_IN]->(d:Document {doc_id: $doc_id})
        MATCH (a)-[r:RELATED_TO]->(b:Entity)
        RETURN a.name, r.relation, b.name, coalesce(r.weight, 1)
        """,
        {"doc_id": doc_id},
    )
    relations = []
    while result.has_next():
        row = result.get_next()
        relations.append({
            "source": row[0], "relation": row[1],
            "target": row[2], "weight": row[3],
        })
    return {"entities": entities, "relations": relations, "doc_id": doc_id}


def delete_document_graph(doc_id: str):
    conn = get_db()
    conn.execute("MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c", {"doc_id": doc_id})
    conn.execute(
        "MATCH (e:Entity)-[r:MENTIONED_IN]->(d:Document {doc_id: $doc_id}) DELETE r",
        {"doc_id": doc_id},
    )
    conn.execute("MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d", {"doc_id": doc_id})
    conn.execute(
        "MATCH (e:Entity) WHERE NOT (e)-[:MENTIONED_IN]->(:Document) DETACH DELETE e"
    )


# ── Language heuristic ────────────────────────────────────────────────────────

def _guess_lang(text: str) -> str:
    """Quick heuristic to tag an alias as 'vi' or 'en'."""
    vi_re = re.compile(
        r"[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ]",
        re.UNICODE,
    )
    return "vi" if vi_re.search(text) else "en"