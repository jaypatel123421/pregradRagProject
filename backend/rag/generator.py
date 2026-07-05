"""
Answer generator using OpenAI chat completions.
Strictly restricted to DaVinci Resolve content from the PDF.
"""
from openai import OpenAI


class OpenAIGenerator:
    """Generate answers using an OpenAI chat model with retrieved context."""

    SYSTEM_PROMPT = (
        "You are a strict assistant for the DaVinci Resolve Beginner's Guide PDF. "
        "Your ONLY job is to answer questions about DaVinci Resolve based on the "
        "provided context excerpts from that PDF.\n\n"
        "Rules you MUST follow:\n"
        "1. ONLY answer using information found in the provided context excerpts. "
        "Do NOT use any external knowledge, even if you know the answer.\n"
        "2. If the question is NOT about DaVinci Resolve, respond with exactly: "
        "'I can only answer questions about DaVinci Resolve based on the Beginner's Guide PDF.'\n"
        "3. If the question IS about DaVinci Resolve but the context does not contain "
        "enough information to answer it, respond with: "
        "'The provided excerpts from the guide don't cover this topic. "
        "Try rephrasing or asking about a different aspect of DaVinci Resolve.'\n"
        "4. Never make up steps, features, shortcuts, or settings that are not "
        "explicitly mentioned in the context.\n"
        "5. Keep answers concise, clear, and reference page numbers when available."
    )

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Generate an answer strictly grounded in the retrieved context chunks."""
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            page = chunk.get("metadata", {}).get("page", "?")
            context_parts.append(
                f"[Excerpt {i} — Page {page}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_message = (
            f"Context excerpts from the DaVinci Resolve Beginner's Guide PDF:\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer strictly using only the context above:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,  # deterministic — no creative deviation
        )
        return response.choices[0].message.content
