# DaVinci Resolve RAG System

> Ask questions about DaVinci Resolve — powered by **Qdrant** vector DB, **Google Gemini** embeddings & generation, and a **React** frontend.

## Architecture

```
frontend/       React + Vite UI
backend/
  main.py       FastAPI server (GET /status, POST /ingest, POST /query)
  rag/
    document_loader.py   PDF → text chunks (PyMuPDF)
    embeddings.py        Gemini embedding model (3072-dim)
    vector_store.py      Qdrant persistent vector store
    generator.py         Gemini answer generation
    rag_pipeline.py      Orchestrates everything
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for Qdrant)
- A Google Gemini API key

## Setup

### 1. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy env and add your Gemini API key
cp .env.example .env
# Edit .env — set GEMINI_API_KEY

uvicorn main:app --reload --port 8000
```

### 3. Ingest the PDF

Once the server is running, trigger ingestion via the UI **"Start Ingestion"** button, or via curl:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/Users/jaypatel/Downloads/DaVinci-Resolve-16_Beginners-Guide.pdf"}'
```

### 4. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## API Endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| GET    | /health   | Health check                       |
| GET    | /status   | Index readiness & chunk count      |
| POST   | /ingest   | Ingest PDF into Qdrant             |
| POST   | /query    | Ask a question, get answer+sources |

## Environment Variables

| Variable           | Default                | Description                |
|--------------------|------------------------|----------------------------|
| `GEMINI_API_KEY`   | —                      | **Required** Gemini API key |
| `QDRANT_HOST`      | `localhost`            | Qdrant host                |
| `QDRANT_PORT`      | `6333`                 | Qdrant port                |
| `QDRANT_COLLECTION`| `davinci_resolve_guide`| Collection name            |
| `PDF_PATH`         | *(hardcoded path)*     | Path to the PDF            |
