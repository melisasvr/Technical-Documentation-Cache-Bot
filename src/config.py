"""Central settings. Values can be overridden in the .env file."""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq retires models from time to time. Check https://console.groq.com/docs/models
AVAILABLE_MODELS = list(dict.fromkeys([DEFAULT_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]))

# CAG: max characters of document text placed in the system prompt (~4 chars = 1 token).
# Free-tier Groq accounts have tokens-per-minute limits, so keep this modest.
CAG_MAX_CHARS = int(os.getenv("CAG_MAX_CHARS", "12000"))

# RAG: chunking and retrieval settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

SAMPLE_DOCS_DIR = "data/sample_docs"
