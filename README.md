# Pdfuploadandask

Upload a PDF, ask questions about it, and get answers grounded in the document's content (with page citations).

## How it works

1. You upload a PDF through the Streamlit UI.
2. The text is extracted per page and split into overlapping chunks.
3. Each chunk is embedded with OpenAI's `text-embedding-3-small` and kept in memory for the session.
4. When you ask a question, it's embedded too, compared against the chunks with cosine similarity, and the top matches are used as context.
5. `gpt-4o-mini` answers using only that context and cites the page numbers it drew from.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key:

   ```bash
   cp .env.example .env
   # then edit .env and set OPENAI_API_KEY
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL Streamlit prints (usually http://localhost:8501), upload a PDF, and start asking questions.

## Notes

- Everything is kept in-memory for the current session — nothing is persisted to disk. Uploading a new PDF re-indexes and replaces the previous one.
- Scanned PDFs without a text layer won't extract any text (no OCR step); use a text-based PDF or add OCR pre-processing if you need that.
- The `knowledge/` and `patterns/` folders are reference examples (Docling-based RAG pipeline and LLM workflow patterns) that informed this app's design — they aren't part of the running app.
