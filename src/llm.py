"""Shared LLM helpers used by both the CAG and RAG bots."""
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq


@dataclass
class BotAnswer:
    """What every bot returns, so the UI can treat CAG and RAG the same way."""
    text: str
    latency_s: float                      # total time for the full answer
    sources: list[str] = field(default_factory=list)
    input_tokens: int | None = None
    cached_tokens: int | None = None      # only filled if Groq reports it
    retrieval_s: float | None = None      # RAG only


def get_llm(api_key: str, model: str) -> ChatGroq:
    # temperature=0 -> stable, factual answers (also nicer for fair comparisons)
    return ChatGroq(api_key=api_key, model=model, temperature=0)


def history_to_messages(history: list[dict] | None, max_turns: int = 6) -> list[BaseMessage]:
    """Convert [{'role': 'user'|'assistant', 'content': str}] to LangChain messages."""
    msgs: list[BaseMessage] = []
    for m in (history or [])[-max_turns:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    return msgs


def read_usage(response) -> tuple[int | None, int | None]:
    """Pull token counts from the response, if the API provided them."""
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return usage.get("input_tokens"), details.get("cache_read")
