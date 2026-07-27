"""
Qdrant vector store — cloud-only via Qdrant Cloud.
Uses the modern query_points() API (qdrant-client >= 1.7).

Usage: QdrantVectorStore(url="https://xxxx.cloud.qdrant.io:6333", api_key="...", ...)
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import uuid


class QdrantVectorStore:
    """Persistent vector store backed by Qdrant Cloud."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str = "davinci_resolve_guide",
        vector_size: int = 1536,
    ):
        if not url or not api_key:
            raise ValueError(
                "Qdrant Cloud requires both 'url' and 'api_key'. "
                "Set QDRANT_URL and QDRANT_API_KEY in your .env file."
            )
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            check_compatibility=False,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        """Create the collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict] | None = None,
    ):
        """Upload documents with embeddings into Qdrant."""
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {"text": text}
            if metadata:
                payload.update(metadata[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return top_k most similar documents using Qdrant cosine similarity."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": hit.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in hit.payload.items() if k != "text"
                },
                "score": hit.score,
            }
            for hit in results.points
        ]

    def count(self) -> int:
        """Return the number of documents in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def collection_exists_with_data(self) -> bool:
        """Check if the collection already has data (skip re-ingestion)."""
        return self.count() > 0

    def delete_collection(self):
        """Delete the entire collection (for reset/re-ingest)."""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
