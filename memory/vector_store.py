"""
Vector memory for MARA.

Stores completed research sessions as embeddings in Qdrant (local).
On query, searches for semantically similar past sessions and returns
a formatted context string that the Planner can use.

Embedding model : all-MiniLM-L6-v2 (80 MB, CPU-only, 384 dimensions)
Vector DB       : Qdrant local in-process (.qdrant_storage/)
Fallback        : ChromaDB in-process (no file persistence)
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

COLLECTION_NAME = "research_sessions"
VECTOR_DIM = 384
SIMILARITY_THRESHOLD = 0.70

# ── LAZY SINGLETONS ───────────────────────────────────────────────────────────

_embedding_model = None
_qdrant_client = None
_chroma_client = None
_using_chroma = False


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("[Memory] Loading all-MiniLM-L6-v2 embedding model…")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("[Memory] Embedding model ready")
    return _embedding_model


def _embed(text: str) -> List[float]:
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def _get_qdrant():
    global _qdrant_client, _using_chroma
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(path=".qdrant_storage")

        existing = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info("[Memory] Created Qdrant collection '%s'", COLLECTION_NAME)

        _qdrant_client = client
        logger.info("[Memory] Qdrant initialised at .qdrant_storage/")
        return _qdrant_client

    except Exception as e:
        logger.warning("[Memory] Qdrant init failed (%s) — falling back to ChromaDB", e)
        _using_chroma = True
        return None


def _get_chroma():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client
    import chromadb
    _chroma_client = chromadb.Client()
    _chroma_client.get_or_create_collection(COLLECTION_NAME)
    logger.info("[Memory] ChromaDB fallback initialised (in-process, no persistence)")
    return _chroma_client


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def store_research_session(state) -> Optional[str]:
    try:
        query = state.get("query", "")
        report = state.get("final_report", "")
        sub_qs = state.get("sub_questions", [])
        sources = [r.get("url", "") for r in state.get("search_results", [])]

        if not query or not report:
            logger.warning("[Memory] Skipping store — query or report is empty")
            return None

        text_to_embed = f"{query} {report[:500]}"
        vector = _embed(text_to_embed)

        payload = {
            "query": query,
            "sub_questions": sub_qs,
            "final_report": report[:3000],
            "sources": list(dict.fromkeys(sources)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        point_id = str(uuid.uuid4())

        if not _using_chroma:
            qdrant = _get_qdrant()
            if qdrant:
                from qdrant_client.models import PointStruct
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
                logger.info("[Memory] Session stored to Qdrant (id=%s…)", point_id[:8])
                return point_id

        chroma = _get_chroma()
        col = chroma.get_collection(COLLECTION_NAME)
        col.add(
            ids=[point_id],
            embeddings=[vector],
            documents=[payload["final_report"]],
            metadatas=[{k: str(v) for k, v in payload.items() if k != "final_report"}],
        )
        logger.info("[Memory] Session stored to ChromaDB (id=%s…)", point_id[:8])
        return point_id

    except Exception as e:
        logger.warning("[Memory] store_research_session failed (non-fatal): %s", e)
        return None


def search_past_research(query: str, limit: int = 3) -> List[dict]:
    try:
        vector = _embed(query)

        if not _using_chroma:
            qdrant = _get_qdrant()
            if qdrant:
                response = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    limit=limit,
                    score_threshold=SIMILARITY_THRESHOLD,
                )
                return [
                    {**pt.payload, "score": pt.score}
                    for pt in response.points
                ]

        chroma = _get_chroma()
        col = chroma.get_collection(COLLECTION_NAME)
        results = col.query(query_embeddings=[vector], n_results=limit)
        output = []
        for i, doc in enumerate(results.get("documents", [[]])[0]):
            meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
            dist = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
            score = 1.0 - dist
            if score >= SIMILARITY_THRESHOLD:
                output.append({"final_report": doc, **meta, "score": score})
        return output

    except Exception as e:
        logger.warning("[Memory] search_past_research failed (non-fatal): %s", e)
        return []


def get_research_context(query: str) -> Optional[str]:
    results = search_past_research(query, limit=1)
    if not results:
        return None

    best = results[0]
    score = best.get("score", 0)
    past_q = best.get("query", "")
    report = best.get("final_report", "")

    logger.info(
        "[Memory] Found prior session (score=%.2f) for query: %r", score, past_q
    )

    context = (
        f"PRIOR RESEARCH CONTEXT (similarity={score:.2f}):\n"
        f"A previous research session answered a similar question: \"{past_q}\"\n\n"
        f"Key findings from that session:\n{report[:600]}…\n\n"
        f"Use this context to avoid duplicating known findings and to focus "
        f"sub-questions on gaps not already covered."
    )
    return context
