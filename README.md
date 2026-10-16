# 📚 Technical Documentation Cache Bot (CAG vs RAG)

A small Streamlit app that answers questions about technical documents (PDF or Markdown) using two different strategies, so you can compare them side by side:

| | **CAG** (Cache-Augmented Generation) | **RAG** (Retrieval-Augmented Generation) |
|---|---|---|
| **Idea** | Put the *whole* document set into one static system prompt | Split docs into chunks, retrieve only the relevant ones |
| **Retrieval step** | None | Yes (local embeddings + Chroma) |
| **Tokens sent per question** | All the docs (cached by Groq when possible) | Only the top-k chunks |
| **Sees the whole document?** | Yes, up to the context limit | No, only the retrieved chunks |

**Stack:** Python · [Groq](https://groq.com) (LLM) · LangChain · pypdf · Chroma · Streamlit

---

## Features

- Upload PDFs / Markdown files, or use the bundled sample doc
- **CAG mode:** full documents in a static system prompt (simulates CAG using Groq's automatic prompt caching)
- **RAG mode:** chunking → local embeddings → Chroma → top-k retrieval → answer
- Chat interface with a mode switch
- Latency shown per answer and averaged per mode (RAG also shows retrieval time)
- Token usage shown per answer, including **cached tokens** when Groq reports them
- Automatic PDF clean-up (removes repeated headers/footers and table-of-contents dot leaders to save tokens)

---

## Project structure

```
doc-cache-bot/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example            # copy to .env and add your key
├── README.md
├── data/
│   └── sample_docs/        # small sample doc loaded by default
└── src/
    ├── config.py           # settings (model, chunk size, limits)
    ├── cleaning.py         # removes repeated PDF headers/footers
    ├── loaders.py          # PDF / Markdown loading (pypdf)
    ├── llm.py              # Groq client + shared helpers
    ├── cag.py              # CAG bot
    └── rag.py              # RAG bot (Chroma)
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd doc-cache-bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your Groq API key

1. Create a free key at <https://console.groq.com/keys>.
2. Copy the example env file and edit it:

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides
# GROQ_MODEL=openai/gpt-oss-120b
# CAG_MAX_CHARS=18000
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

You can also paste the key into the app's sidebar instead.

### 3. Run

```bash
streamlit run app.py
```

---

## How to use

1. In the sidebar, upload PDF/Markdown files (or leave the sample doc checked).
2. Click **Build CAG + RAG**. The first RAG build downloads a small local embedding model (~80 MB).
3. Choose **CAG** or **RAG** and ask questions in the chat.
4. Ask the same questions in both modes and compare the latency boxes at the top and the captions under each answer.

**Tip:** ask the same CAG question twice. If Groq's prompt caching applies, the second answer is usually faster and the caption shows a cached-token count.

---

## Important: free-tier limits and the CAG context size

CAG sends the document text with **every** question, so it is limited by your Groq plan.

- On Groq's free tier, `openai/gpt-oss-120b` and `openai/gpt-oss-20b` allow about **8,000 tokens per minute** (input + output combined). Check the [current limits](https://console.groq.com/docs/rate-limits).
- The sidebar setting **"CAG max context (chars)"** caps how much document text goes into the prompt (roughly 3.3–3.5 characters per token).
- If the documents are longer than the cap, the text is **truncated**: anything after the cut-off is never seen by the model.

| Setting | Approx. input tokens | Notes |
|---|---|---|
| 12,000 chars | ~3,000–3,500 | Safe, but covers only the first part of a ~33-page PDF |
| 18,000–20,000 chars | ~5,500–6,000 | Near the free-tier limit once the 1,200-token output cap is added |
| Developer plan | much higher | Lets you fit larger documents |

RAG is not affected as much because it only sends a few chunks per question.

---

## Example results

These come from small manual tests on a free Groq account, with one run per question. They show how the two modes behaved in this setup. They are **not** a rigorous benchmark.

### Sample doc (5 questions, tiny document)

- CAG averaged about 0.6 s per answer and RAG about 1.05 s.
- Roughly half a second of RAG's time was the local retrieval step.
- Answer quality was the same, and both modes correctly said "I don't know" for a question the doc doesn't cover.

### RFC 8259 (JSON spec, 33-page PDF), CAG limited to 12,000 characters

The free tier allows about 8,000 tokens per minute, so CAG could only load roughly the first third of the PDF. RAG searches the whole document, so it wasn't limited the same way.

| Question | CAG (12,000 chars) | RAG |
|---|---|---|
| What encoding must JSON use between systems? | Partial (said "Unicode", missed the UTF-8 MUST) | Correct (UTF-8) |
| What does the RFC say about duplicate names? | Correct | Correct |
| Can I add a byte order mark? | "I don't know" | Correct (MUST NOT) |
| Summarize the changes from RFC 7159 | "I don't know" | Correct (listed from Appendix A) |
| Parser vs generator requirements | "I don't know" | Correct |
| Are leading zeros allowed? | Not answered (cut off) | Correct (one quoted sentence was cut mid-way) |
| Why is `eval()` risky? | "I don't know" | Correct |

- **Input size:** CAG sent about 3,100-3,500 input tokens per question. RAG sent about 1,000-1,200.
- **Latency:** RAG answered in about 0.9-1.2 s, including about 0.35-0.45 s of retrieval. CAG answered in about 0.8-1.5 s in most cases, with one slower outlier (about 3.3 s). With one run each, there is no clear speed winner.
- **Prompt caching:** on repeated CAG questions Groq reported about 3,072 cached tokens, and some answers came back faster (for example about 0.76 s versus 1.5 s on the first question).

**What this does and doesn't show**

- With a document longer than the free-tier limit, RAG answered more questions because CAG had to truncate. That is a limit of the setup, not proof that RAG is better in general.
- CAG's "I don't know" answers were questions whose answers sat after the 12,000-character cut-off. The model never saw that text.
- A fairer comparison would raise CAG to about 20,000 characters (near the free-tier limit), use a shorter document that fits entirely, or use a paid plan with a higher limit. That rerun has not been included here yet.

## Known limitations

- **Truncation can mislead.** If the cut-off removes the exact sentence you need, CAG may answer from nearby text (such as the table of contents) and sound confident but incomplete. Check the "CAG context" size in the sidebar and the truncation warning.
- **RAG chunk boundaries.** Chunks can split a sentence, so a quoted passage may end mid-sentence (this happened once in the leading-zeros answer). RAG can also miss questions that need information spread across many sections.
- **Naive truncation.** CAG simply cuts the text at the character limit. It does not pick the most relevant parts.
- **Local embeddings.** Groq has no embeddings API, so RAG uses Chroma's built-in local model. Retrieval costs about half a second per question on a typical CPU.
- **In-memory index.** The Chroma collection is rebuilt on each "Build" click and is not saved to disk.
- **Chat history.** Only the last two exchanges are sent to the model.
- **Citation markers.** The `gpt-oss` models sometimes add markers like `【file.pdf (page 7)】` to answers. This is cosmetic.
- **Text-only PDFs.** Scanned PDFs (images) need OCR first. pypdf cannot read them.
- Caching behavior depends on Groq's current model support and prompt size, so check [Groq's prompt caching docs](https://console.groq.com/docs/prompt-caching).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `404 model_not_found` | Groq retires models from time to time. Check <https://console.groq.com/docs/models> and set `GROQ_MODEL` in `.env`. |
| `413 Request too large ... tokens per minute` | Lower "CAG max context", use fewer/shorter docs, or upgrade your Groq plan. |
| `429` rate-limit error | Wait about a minute between CAG questions on the free tier. |
| Code changes don't seem to apply | Files inside `src/` are not hot-reloaded. Stop Streamlit (Ctrl+C), run it again, then click **Build CAG + RAG** again. |
| "I don't know" for something that is in the PDF | Check whether it falls after the CAG cut-off. Raise the limit (within your token budget) or try RAG mode. |

---

## Ideas for next steps

- A "Compare both" button that sends one question to CAG and RAG side by side
- Smarter CAG context selection instead of plain truncation
- Persistent Chroma storage
- A small benchmark script with a fixed question set and accuracy scoring
- OCR support for scanned PDFs

## Contributing
- Contributions are welcome.
1. Fork the repository and create a branch: `git checkout -b feature/your-feature`
2. Make your changes and keep them focused on one topic.
3. `streamlit run dashboard.py` still run.
4. Commit with a clear message and open a pull request describing what changed and why.
5. Please open an issue first for larger changes, and include steps to reproduce when reporting a bug.

---

## Acknowledgements

Built with [LangChain](https://www.langchain.com/), [Chroma](https://www.trychroma.com/), [Streamlit](https://streamlit.io/), [pypdf](https://pypdf.readthedocs.io/) and [Groq](https://groq.com). The CAG idea follows the "preload knowledge, skip retrieval" approach to document Q&A.

---

## License
- This project is licensed under the MIT License.
- MIT License
```
Copyright (c) 2026 Melisa Sever

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```