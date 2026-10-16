"""Streamlit UI: chat with your docs using CAG or RAG and compare latency."""
import streamlit as st

from src import config
from src.cag import CAGBot
from src.llm import get_llm
from src.loaders import SUPPORTED, load_folder, load_uploaded
from src.rag import RAGBot

st.set_page_config(page_title="Doc Cache Bot", page_icon="📚", layout="wide")
st.title("📚 Technical Documentation Bot: CAG vs RAG")

# ---------- session state (survives Streamlit reruns) ----------
defaults = {"docs": [], "cag": None, "rag": None, "messages": [], "stats": {"CAG": [], "RAG": []}}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Settings")
    api_key = st.text_input("Groq API key", value=config.GROQ_API_KEY, type="password")
    model = st.selectbox("Model", config.AVAILABLE_MODELS)
    mode = st.radio("Answer mode", ["CAG", "RAG"], help="CAG = full docs in prompt. RAG = retrieve chunks.")
    max_chars = st.number_input("CAG max context (chars)", 5_000, 400_000, config.CAG_MAX_CHARS, step=5_000)

    st.header("2. Documents")
    uploads = st.file_uploader("Upload PDF / Markdown", type=[s.strip(".") for s in SUPPORTED],
                               accept_multiple_files=True)
    use_samples = st.checkbox(f"Also load files from {config.SAMPLE_DOCS_DIR}/", value=not uploads)

    if st.button("Build CAG + RAG", type="primary"):
        if not api_key:
            st.error("Add your Groq API key first.")
        else:
            docs = load_folder(config.SAMPLE_DOCS_DIR) if use_samples else []
            for f in uploads or []:
                docs.extend(load_uploaded(f))
            if not docs:
                st.error("No documents found.")
            else:
                with st.spinner("Preparing context and vector index..."):
                    llm = get_llm(api_key, model)
                    st.session_state.docs = docs
                    st.session_state.cag = CAGBot(docs, llm, max_chars)
                    st.session_state.rag = RAGBot(docs, llm, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.TOP_K)
                    st.session_state.messages = []
                    st.session_state.stats = {"CAG": [], "RAG": []}
                st.success(f"Loaded {len(docs)} page(s)/section(s).")

    if st.session_state.cag:
        cag, rag = st.session_state.cag, st.session_state.rag
        st.caption(f"CAG context: ~{len(cag.context) // 4:,} tokens | RAG chunks: {len(rag.chunks)}")
        if cag.truncated:
            st.warning("Docs exceeded the CAG limit and were truncated.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.stats = {"CAG": [], "RAG": []}
        st.rerun()

# ---------- latency comparison ----------
st.subheader("⏱️ Latency comparison")
cols = st.columns(2)
for col, name in zip(cols, ["CAG", "RAG"]):
    times = [s.latency_s for s in st.session_state.stats[name]]
    col.metric(f"{name} avg response", f"{sum(times) / len(times):.2f}s" if times else "–",
               help=f"{len(times)} question(s) answered")
    if times:
        col.caption(f"last: {times[-1]:.2f}s | questions: {len(times)}")

st.divider()

# ---------- chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

question = st.chat_input("Ask something about your docs...")
if question:
    bot = st.session_state.cag if mode == "CAG" else st.session_state.rag
    if bot is None:
        st.warning("Load documents and click 'Build CAG + RAG' first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({mode})..."):
            try:
                # history = everything BEFORE this question
                answer = bot.ask(question, history=st.session_state.messages)
            except Exception as e:  # e.g. rate limit / bad key
                st.error(f"Groq error: {e}")
                st.stop()

        meta = f"{mode} · {answer.latency_s:.2f}s"
        if answer.retrieval_s is not None:
            meta += f" (retrieval {answer.retrieval_s:.2f}s)"
        if answer.input_tokens:
            meta += f" · {answer.input_tokens:,} input tokens"
        if answer.cached_tokens:
            meta += f" ({answer.cached_tokens:,} cached)"
        if answer.sources:
            meta += " · sources: " + ", ".join(answer.sources)
        st.markdown(answer.text)
        st.caption(meta)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text, "meta": meta},
    ]
    st.session_state.stats[mode].append(answer)
    st.rerun()  # refresh the latency metrics at the top
