"""Upload a PDF, ask questions about it, get answers grounded in its content."""

import hashlib
import io
import os
import threading
from collections import OrderedDict
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from landingai_ade import LandingAIADE, LandingAiadeError
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 75
TOP_K = 8
FETCH_K = 25
MMR_LAMBDA = 0.5
MAX_PAGES = 7
MAX_FILE_MB = 10
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_SESSION_QUESTIONS = int(os.environ.get("MAX_SESSION_QUESTIONS", "10"))
MAX_DAILY_REQUESTS = int(os.environ.get("MAX_DAILY_REQUESTS", "200"))
PDF_CACHE_MAX_ENTRIES = 50

client = OpenAI()
ade_client = LandingAIADE()  # reads VISION_AGENT_API_KEY
encoding = tiktoken.get_encoding("cl100k_base")

# Process-wide (shared across all sessions) usage tracking. A plain dict guarded by a
# lock is enough here since this runs as a single process on a single instance; it
# would need a real store (Redis/DB) behind a load balancer with multiple workers.
_usage_lock = threading.Lock()
_usage_state = {"date": date.today(), "count": 0}

_pdf_cache_lock = threading.Lock()
_pdf_cache: "OrderedDict[str, Tuple[List[dict], np.ndarray]]" = OrderedDict()


def consume_global_quota() -> bool:
    """Returns False once the shared daily request cap is hit, so a spike in traffic
    (or abuse) can't run up an unbounded OpenAI/LandingAI bill. Every call that spends
    money (a PDF parse+embed, or a question) must go through this first."""
    with _usage_lock:
        today = date.today()
        if _usage_state["date"] != today:
            _usage_state["date"] = today
            _usage_state["count"] = 0
        if _usage_state["count"] >= MAX_DAILY_REQUESTS:
            return False
        _usage_state["count"] += 1
        return True


def get_cached_pdf(file_hash: str) -> Optional[Tuple[List[dict], np.ndarray]]:
    """Shared across sessions/users: if someone else already uploaded this exact file
    (common for popular T&Cs/contracts), skip paying for ADE parsing + embeddings again."""
    with _pdf_cache_lock:
        cached = _pdf_cache.get(file_hash)
        if cached is not None:
            _pdf_cache.move_to_end(file_hash)
        return cached


def set_cached_pdf(file_hash: str, value: Tuple[List[dict], np.ndarray]) -> None:
    with _pdf_cache_lock:
        _pdf_cache[file_hash] = value
        _pdf_cache.move_to_end(file_hash)
        while len(_pdf_cache) > PDF_CACHE_MAX_ENTRIES:
            _pdf_cache.popitem(last=False)


def get_page_count(file_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(file_bytes)).pages)


def extract_pages(filename: str, file_bytes: bytes) -> List[str]:
    """Parse the PDF with LandingAI's Agentic Document Extraction, which reconstructs
    real table structure (rows, columns, merged cells) instead of flattening tables
    into disconnected text like plain PDF text extraction does. Returns one markdown
    string per page."""
    try:
        parsed = ade_client.v2.parse(
            document=(filename, file_bytes, "application/pdf"),
            options={"inline_markdown": True},
        )
    except LandingAiadeError as e:
        st.error(f"Couldn't parse this PDF: {e}")
        return []

    if parsed.structure and parsed.structure.children:
        return [page.markdown or "" for page in parsed.structure.children]
    return [parsed.markdown or ""]


def chunk_pages(pages: List[str]) -> List[dict]:
    """Split each page's text into overlapping token chunks, tagged with page number."""
    chunks = []
    for page_no, text in enumerate(pages, start=1):
        text = text.strip()
        if not text:
            continue
        tokens = encoding.encode(text)
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_TOKENS, len(tokens))
            chunk_text = encoding.decode(tokens[start:end])
            chunks.append({"text": chunk_text, "page": page_no})
            if end == len(tokens):
                break
            start = end - CHUNK_OVERLAP
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts, normalized for cosine similarity via dot product."""
    if not texts:
        return np.zeros((0, 1536))
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def search(
    query: str,
    chunks: List[dict],
    embeddings: np.ndarray,
    k: int = TOP_K,
    fetch_k: int = FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> List[dict]:
    """Retrieve chunks via Maximal Marginal Relevance: relevant to the query, but
    diverse from each other, so a broad question (e.g. "what are the important
    things to consider?") pulls a spread of the document instead of several
    near-duplicate passages from the single most similar section."""
    query_vec = embed_texts([query])[0]
    sims_to_query = embeddings @ query_vec

    fetch_k = min(fetch_k, len(chunks))
    k = min(k, len(chunks))
    candidates = list(np.argsort(sims_to_query)[::-1][:fetch_k])

    selected: List[int] = []
    while candidates and len(selected) < k:
        if not selected:
            best = candidates[0]
        else:
            selected_vecs = embeddings[selected]
            best, best_score = None, -np.inf
            for idx in candidates:
                redundancy = np.max(embeddings[idx] @ selected_vecs.T)
                score = lambda_mult * sims_to_query[idx] - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best, best_score = idx, score
        selected.append(best)
        candidates.remove(best)

    return [chunks[i] for i in selected]


def build_context(results: List[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"[Page {r['page']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def get_chat_response(messages: List[dict], context: str):
    system_prompt = (
        "You are a helpful assistant that answers questions using only the provided "
        "excerpts from a PDF document. Cite the page number(s) you used, like (p. 3).\n\n"
        "For broad or open-ended questions (e.g. 'what are the important things to "
        "consider', 'summarize this', 'what should I watch out for'), don't require an "
        "excerpt to literally contain that phrasing — instead synthesize across ALL the "
        "excerpts below and surface anything a reader would want to know: obligations, "
        "costs or fees, deadlines, liability, termination or renewal terms, restrictions, "
        "and similar noteworthy points. Present it as a short list, each item citing its "
        "page.\n\n"
        "Only say you couldn't find an answer if the excerpts truly have nothing relevant "
        "to draw on."
        f"\n\nExcerpts:\n{context}"
    )
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=0.3,
        stream=True,
    )
    return st.write_stream(stream)


def process_pdf(filename: str, file_bytes: bytes, file_hash: str):
    cached = get_cached_pdf(file_hash)
    if cached is not None:
        return cached

    if not file_bytes.startswith(b"%PDF-"):
        st.error("This doesn't look like a valid PDF file.")
        return None, None

    if len(file_bytes) > MAX_FILE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        st.error(f"This file is {size_mb:.1f} MB, over the {MAX_FILE_MB} MB limit.")
        return None, None

    try:
        page_count = get_page_count(file_bytes)
    except Exception:
        st.error("Couldn't read this file. Is it a valid PDF?")
        return None, None

    if page_count > MAX_PAGES:
        st.error(f"This PDF has {page_count} pages; the limit is {MAX_PAGES} pages for now.")
        return None, None

    if not consume_global_quota():
        st.error(f"We've hit today's usage limit ({MAX_DAILY_REQUESTS} requests). Please try again tomorrow.")
        return None, None

    with st.spinner("Reading and indexing PDF..."):
        pages = extract_pages(filename, file_bytes)
        chunks = chunk_pages(pages)
        if not chunks:
            st.error("Couldn't extract any text from this PDF. It may be scanned images without a text layer.")
            return None, None
        embeddings = embed_texts([c["text"] for c in chunks])

    set_cached_pdf(file_hash, (chunks, embeddings))
    return chunks, embeddings


def main():
    st.set_page_config(page_title="PDF Q&A", page_icon="📄")
    st.title("📄 PDF Upload & Ask")
    st.caption("Upload a PDF, then ask questions about its contents.")
    st.caption(
        "Your document's content is sent to OpenAI and LandingAI for processing — "
        "avoid uploading anything highly sensitive."
    )

    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
        st.session_state.chunks = None
        st.session_state.embeddings = None
        st.session_state.messages = []
        st.session_state.question_count = 0

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help=f"Max {MAX_PAGES} pages, {MAX_FILE_MB} MB.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if file_hash != st.session_state.file_hash:
            chunks, embeddings = process_pdf(uploaded_file.name, file_bytes, file_hash)
            st.session_state.file_hash = file_hash
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.messages = []
            st.session_state.question_count = 0
            if chunks:
                st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}.")

    if not st.session_state.chunks:
        st.info("Upload a PDF to get started.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the document"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.question_count >= MAX_SESSION_QUESTIONS:
            with st.chat_message("assistant"):
                st.warning(
                    f"You've reached the {MAX_SESSION_QUESTIONS}-question limit for this "
                    "session. Refresh the page to start a new one."
                )
            return

        if not consume_global_quota():
            with st.chat_message("assistant"):
                st.error(f"We've hit today's usage limit ({MAX_DAILY_REQUESTS} requests). Please try again tomorrow.")
            return

        st.session_state.question_count += 1

        with st.status("Searching document...", expanded=False):
            results = search(prompt, st.session_state.chunks, st.session_state.embeddings)
            context = build_context(results)
            for r in results:
                st.markdown(f"**Page {r['page']}**: {r['text'][:200]}...")

        with st.chat_message("assistant"):
            response = get_chat_response(st.session_state.messages, context)
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
