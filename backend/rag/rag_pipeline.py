"""
RAG Pipeline — orchestrates loading, embedding, storing, and querying.
Uses Qdrant as persistent vector store instead of the in-memory store.
"""
from .embeddings import OpenAIEmbedder
from .vector_store import QdrantVectorStore
from .document_loader import DocumentLoader
from .generator import OpenAIGenerator

# Minimum cosine similarity score for a retrieved chunk to be considered relevant.
# Queries whose best match falls below this threshold are rejected as off-topic.
MIN_RELEVANCE_SCORE = 0.35
OFF_TOPIC_MESSAGE = (
    "I can only answer questions about DaVinci Resolve based on the Beginner's Guide PDF. "
    "Your question doesn't appear to be related to this guide. "
    "Please ask something about DaVinci Resolve — for example editing, color grading, "
    "the timeline, exporting, or the Fusion/Fairlight pages."
)


class RAGPipeline:
    """End-to-end RAG pipeline: load docs → embed (OpenAI) → store in Qdrant → query → generate (OpenAI)."""

    def __init__(
        self,
        api_key: str,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str = "davinci_resolve_guide",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.embedder = OpenAIEmbedder(api_key=api_key)

        if qdrant_url and qdrant_api_key:
            # Cloud connection
            self.store = QdrantVectorStore(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=collection_name,
                vector_size=OpenAIEmbedder.DIMENSION,  # 1536 for text-embedding-3-small
            )
        else:
            # Local connection (default: localhost:6333)
            self.store = QdrantVectorStore(
                host=qdrant_host or "localhost",
                port=qdrant_port or 6333,
                collection_name=collection_name,
                vector_size=OpenAIEmbedder.DIMENSION,
            )
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.generator = OpenAIGenerator(api_key=api_key)

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    def ingest_pdf(self, file_path: str, force: bool = False) -> int:
        """
        Load, chunk, embed, and store a PDF file.

        Args:
            file_path: Path to the PDF.
            force: If True, delete existing data and re-ingest.

        Returns:
            Number of chunks ingested.
        """
        if not force and self.store.collection_exists_with_data():
            return self.store.count()

        if force:
            self.store.delete_collection()

        docs = self.loader.load_pdf(file_path)
        self._embed_and_store(docs)
        return len(docs)

    def ingest_text(self, text: str, source: str = "inline"):
        """Chunk, embed, and store raw text."""
        docs = self.loader.load_text(text, source=source)
        self._embed_and_store(docs)

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(self, question: str, top_k: int = 5) -> dict:
        """Retrieve relevant chunks and generate an answer.

        Returns an off-topic refusal if the best matching chunk scores below
        MIN_RELEVANCE_SCORE — meaning the question is unrelated to the PDF.
        """
        query_embedding = self.embedder.embed_query(question)
        results = self.store.search(query_embedding, top_k=top_k)

        # Guard: reject off-topic questions based on similarity score
        if not results or results[0]["score"] < MIN_RELEVANCE_SCORE:
            return {
                "question": question,
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
            }

        answer = self.generator.generate(question, results)
        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant chunks without generating an answer."""
        query_embedding = self.embedder.embed_query(question)
        return self.store.search(query_embedding, top_k=top_k)

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def status(self) -> dict:
        """Return pipeline status info."""
        count = self.store.count()
        return {
            "ready": count > 0,
            "chunks_indexed": count,
            "collection": self.store.collection_name,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _embed_and_store(self, docs: list[dict]):
        """Embed documents in batches and upload to Qdrant."""
        if not docs:
            return

        # Batch in groups of 100 to avoid API limits
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [doc["text"] for doc in batch]
            metadata = [doc["metadata"] for doc in batch]
            embeddings = self.embedder.embed_texts(texts)
            self.store.add_documents(texts, embeddings, metadata)
            print(f"  Uploaded batch {i // batch_size + 1} ({len(batch)} chunks)")
