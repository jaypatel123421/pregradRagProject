"""
Document loader with PDF support using PyMuPDF.
Adapted from gemini-rag example, extended with PDF parsing and smarter chunking.
"""
import os
import fitz  # PyMuPDF


class DocumentLoader:
    """Load and chunk text/PDF documents."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_pdf(self, file_path: str) -> list[dict]:
        """Load a PDF file, extract text page by page, and split into chunks."""
        doc = fitz.open(file_path)
        full_text = ""
        page_map: list[tuple[int, int]] = []  # (char_start, page_number)

        for page_num, page in enumerate(doc, start=1):
            start = len(full_text)
            text = page.get_text()
            full_text += text
            page_map.append((start, page_num))

        doc.close()
        return self._chunk_text(full_text, source=file_path, page_map=page_map)

    def load_text(self, text: str, source: str = "inline") -> list[dict]:
        """Chunk raw text string into documents."""
        return self._chunk_text(text, source=source)

    def _chunk_text(
        self,
        text: str,
        source: str,
        page_map: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        """Split text into overlapping chunks, tracking page numbers if available."""
        if not text.strip():
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                page_num = 1
                if page_map:
                    for char_start, pnum in page_map:
                        if char_start <= start:
                            page_num = pnum
                        else:
                            break

                chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": os.path.basename(source),
                        "chunk_index": len(chunks),
                        "page": page_num,
                    },
                })
            start += self.chunk_size - self.chunk_overlap

        return chunks
