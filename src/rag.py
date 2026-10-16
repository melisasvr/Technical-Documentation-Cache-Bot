"""
RAG baseline: split -> embed -> store in Chroma -> retrieve top-k -> answer.
Only the few relevant chunks are sent to the LLM on each question.
"""
import time
import uuid

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .llm import BotAnswer, history_to_messages, read_usage

SYSTEM_PROMPT = """You are a technical documentation assistant.
Answer ONLY using the context excerpts provided with the question. If the answer
is not in them, say you don't know. Be concise."""


class LocalEmbeddings(Embeddings):
    """
    Wraps Chroma's built-in local embedding model (all-MiniLM-L6-v2 via ONNX).
    Groq has no embeddings API, so we embed locally. The model downloads
    automatically the first time it's used (~80 MB).
    """
    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _label(meta: dict) -> str:
    """'file.pdf p.3' for PDFs, 'file.md' for text files."""
    return meta["source"] + (f" p.{meta['page'] + 1}" if "page" in meta else "")


class RAGBot:
    def __init__(self, docs: list[Document], llm, chunk_size: int, chunk_overlap: int, top_k: int):
        self.llm = llm
        self.top_k = top_k

        # 1) Split documents into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = splitter.split_documents(docs)

        # 2) Embed + store in an in-memory Chroma collection.
        #    A unique name avoids stale data when Streamlit reruns the script.
        self.store = Chroma.from_documents(
            self.chunks,
            embedding=LocalEmbeddings(),
            collection_name=f"docs_{uuid.uuid4().hex[:8]}",
        )

    def ask(self, question: str, history: list[dict] | None = None) -> BotAnswer:
        start = time.perf_counter()

        # 3) Retrieve the most similar chunks
        hits = self.store.similarity_search(question, k=self.top_k)
        retrieval_s = time.perf_counter() - start

        context = "\n\n".join(f"[{_label(h.metadata)}]\n{h.page_content}" for h in hits)

        # 4) Ask the LLM, using only the retrieved chunks as context
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history_to_messages(history),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        response = self.llm.invoke(messages)
        latency = time.perf_counter() - start

        in_tok, cached = read_usage(response)
        return BotAnswer(
            text=response.content,
            latency_s=latency,
            sources=sorted({_label(h.metadata) for h in hits}),
            input_tokens=in_tok,
            cached_tokens=cached,
            retrieval_s=retrieval_s,
        )
