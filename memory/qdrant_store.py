import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tools.embeddings import embed

COLLECTION = "mara_sessions"
VECTOR_DIM = 384

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path="./qdrant_storage")
        collections = [c.name for c in _client.get_collections().collections]
        if COLLECTION not in collections:
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _client

def save_session(query: str, report: str) -> None:
    client = get_client()
    vector = embed([query])[0]
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"query": query, "report": report},
        )],
    )

def retrieve_similar(query: str, top_k: int = 3) -> list[dict]:
    client = get_client()
    vector = embed([query])[0]
    hits = client.search(collection_name=COLLECTION, query_vector=vector, limit=top_k)
    return [{"query": h.payload["query"], "report": h.payload["report"], "score": h.score} for h in hits]
