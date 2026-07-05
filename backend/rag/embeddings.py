"""
Embeddings module using OpenAI text-embedding-3-small.
Drop-in replacement for the Gemini embedder.
"""
from openai import OpenAI


class OpenAIEmbedder:
    """Generate embeddings using OpenAI text-embedding-3-small (1536-dim)."""

    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts for document storage (batch)."""
        response = self.client.embeddings.create(
            model=self.MODEL,
            input=texts,
        )
        # Sort by index to preserve order
        sorted_data = sorted(response.data, key=lambda e: e.index)
        return [item.embedding for item in sorted_data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query for retrieval."""
        response = self.client.embeddings.create(
            model=self.MODEL,
            input=[query],
        )
        return response.data[0].embedding
