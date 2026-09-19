# Pdfuploadandask

Upload a PDF, ask questions about it, and get answers grounded in the document's content (with page citations). Limited to 7 pages / 10 MB per document for now.

## How it works

1. You upload a PDF through the Streamlit UI (rejected if it's over 7 pages or 10 MB).
2. [LandingAI's Agentic Document Extraction](https://landing.ai/agentic-document-extraction) parses each page into markdown, correctly reconstructing tables (rows, columns, merged cells) instead of flattening them into disconnected text.
3. Each page's markdown is split into overlapping chunks and embedded with OpenAI's `text-embedding-3-small`, kept in memory for the session.
4. When you ask a question, it's embedded too and matched against the chunks using MMR (Maximal Marginal Relevance) — relevant to the question, but diverse from each other, so a broad question ("what should I watch out for?") pulls a spread of the document instead of near-duplicate passages from one section.
5. `gpt-4o-mini` answers using only that context, synthesizing across excerpts for open-ended questions, and cites the page numbers it drew from.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys:

   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY and VISION_AGENT_API_KEY
   ```

   Get a LandingAI API key at [landing.ai](https://landing.ai) (used for `VISION_AGENT_API_KEY`).

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually http://localhost:8501), upload a PDF, and start asking questions.

## Limits and cost protection

- 7 pages / 10 MB per PDF, and uploads are checked for a real PDF file signature, not just the `.pdf` extension.
- 10 questions per session (`MAX_SESSION_QUESTIONS`) — refresh the page to start a new session.
- A shared daily cap of 200 paid requests (`MAX_DAILY_REQUESTS`) across all users, as a hard ceiling on OpenAI/LandingAI spend regardless of traffic. Both are configurable via env vars — see `.env.example`.
- Identical PDFs (by content hash) are parsed and embedded only once and reused across all users/sessions, so re-uploading a popular document (e.g. a well-known app's ToS) doesn't re-pay for it.
- This is process-wide, in-memory tracking — fine for the single-instance deployment this is built for, but wouldn't hold up behind multiple app instances/workers without a shared store (Redis, a database) backing it instead.

## Notes

- Everything else is kept in-memory for the current session — nothing is persisted to disk. Uploading a new PDF re-indexes and replaces the previous one.
- Scanned PDFs without a text layer won't extract any text; LandingAI's extraction does OCR, but a page that's entirely a photo/scan with no legible text still won't produce useful content.
- Each upload costs a small number of LandingAI credits (roughly 1/page) — negligible at the 7-page cap, and only paid once per unique document thanks to the cache above.
