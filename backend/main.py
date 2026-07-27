"""
FastAPI backend for the DaVinci Resolve RAG system.

Endpoints:
  GET  /status          — Check if the index is ready
  POST /ingest          — Trigger PDF ingestion (idempotent, or forced)
  POST /query           — Ask a question, get an answer + sources
  GET  /health          — Health check
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag import RAGPipeline

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")           # cloud: "https://xxxx.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")    # cloud: Qdrant cloud API key
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # local fallback
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))  # local fallback
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "davinci_resolve_guide")

# Path to the DaVinci Resolve Beginner's Guide PDF
PDF_PATH = os.getenv(
    "PDF_PATH",
    "/Users/jaypatel/Downloads/DaVinci-Resolve-16_Beginners-Guide.pdf",
)

pipeline: RAGPipeline | None = None


# ---------------------------------------------------------------------------
# Lifespan — initialize pipeline on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    pipeline = RAGPipeline(
        api_key=OPENAI_API_KEY,
        qdrant_url=QDRANT_URL or None,
        qdrant_api_key=QDRANT_API_KEY or None,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
    )
    print("✅ RAG pipeline initialized")
    yield
    print("👋 Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DaVinci Resolve RAG API",
    description="Ask questions about DaVinci Resolve powered by Qdrant + OpenAI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    pdf_path: str = Field(
        default=PDF_PATH,
        description="Absolute path to the PDF file to ingest.",
    )
    force: bool = Field(
        default=False,
        description="If True, delete existing index and re-ingest.",
    )


class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")


class Source(BaseModel):
    text: str
    score: float
    page: int | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class StatusResponse(BaseModel):
    ready: bool
    chunks_indexed: int
    collection: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def status():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.status()


@app.post("/ingest", tags=["Ingestion"])
async def ingest(req: IngestRequest):
    """
    Ingest the PDF into Qdrant. Idempotent by default — skips if data already exists.
    Set force=true to wipe and re-ingest.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if not os.path.exists(req.pdf_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found at: {req.pdf_path}",
        )

    print(f"📄 Ingesting PDF: {req.pdf_path} (force={req.force})")
    count = pipeline.ingest_pdf(req.pdf_path, force=req.force)
    return {
        "success": True,
        "chunks_indexed": count,
        "message": f"Indexed {count} chunks from the PDF.",
    }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(req: QueryRequest):
    """Ask a question about DaVinci Resolve and get an AI-generated answer."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    status = pipeline.status()
    if not status["ready"]:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Please call /ingest first.",
        )

    result = pipeline.query(req.question, top_k=req.top_k)

    sources = [
        Source(
            text=s["text"],
            score=round(s["score"], 4),
            page=s.get("metadata", {}).get("page"),
            source=s.get("metadata", {}).get("source"),
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources,
    )
