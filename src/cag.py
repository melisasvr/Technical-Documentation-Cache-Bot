"""
CAG (Cache-Augmented Generation), simulated with prompt caching.

Idea: put ALL documents into the system prompt, once, and keep it byte-for-byte
identical on every request. Providers that support prompt caching (Groq does this
automatically on supported models) can then reuse the already-processed prefix,
so follow-up questions are faster and cheaper. No retrieval step is needed.

IMPORTANT for caching: the big context must come FIRST and never change between
calls. Only the chat history + new question come after it.
"""
import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import BotAnswer, history_to_messages, read_usage

SYSTEM_TEMPLATE = """You are a technical documentation assistant.
Answer ONLY using the documentation below. If the answer is not in it, say you
don't know. Mention the source file name when helpful. Be concise.

=== DOCUMENTATION START ===
{context}
=== DOCUMENTATION END ==="""


def build_context(docs: list[Document], max_chars: int) -> tuple[str, bool]:
    """Join all docs into one string. Returns (context, was_truncated)."""
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = d.metadata["source"] + (f" (page {page + 1})" if page is not None else "")
        parts.append(f"\n--- {label} ---\n{d.page_content.strip()}")
    context = "\n".join(parts)
    if len(context) > max_chars:
        return context[:max_chars], True
    return context, False


class CAGBot:
    def __init__(self, docs: list[Document], llm, max_chars: int):
        self.llm = llm
        self.context, self.truncated = build_context(docs, max_chars)
        # Built ONCE. Reusing the exact same string enables prefix caching.
        self.system_prompt = SYSTEM_TEMPLATE.format(context=self.context)
        self.sources = sorted({d.metadata["source"] for d in docs})

    def ask(self, question: str, history: list[dict] | None = None) -> BotAnswer:
        messages = [
            SystemMessage(content=self.system_prompt),  # big, static, cacheable
            *history_to_messages(history),
            HumanMessage(content=question),
        ]
        start = time.perf_counter()
        response = self.llm.invoke(messages)
        latency = time.perf_counter() - start

        in_tok, cached = read_usage(response)
        return BotAnswer(
            text=response.content,
            latency_s=latency,
            sources=self.sources,
            input_tokens=in_tok,
            cached_tokens=cached,
        )
