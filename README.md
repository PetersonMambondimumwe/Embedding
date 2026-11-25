# Embedding

A hands-on Python CLI project that demonstrates how to build a vector search pipeline using **Google Gemini embeddings** (`gemini-embedding-001`) and **Neon Serverless PostgreSQL** with the **`pgvector`** extension.

It chunks a Markdown or text document by section, converts each chunk into a 768-dimensional embedding vector, stores the chunks and metadata in Neon with an HNSW index, and runs fast cosine-similarity queries from the command line.

---

## How It Works

1. **Section-Aware Chunking (`chunk_markdown`)**
   - Parses Markdown headings (`#`, `##`, etc.) so each chunk keeps track of the section it belongs to.
   - Packs consecutive paragraphs into chunks of up to `700` characters, falling back to word-boundary splits if a single paragraph is longer.
   - Records `source`, `section` (stored as JSONB metadata), and `chunk_index`.

2. **Embedding Generation (`embed_texts`)**
   - Uses the official `google-genai` SDK with `gemini-embedding-001` (`output_dimensionality=768`).
   - Applies retrieval-specific task types so document chunks and search questions align properly in vector space:
     - `RETRIEVAL_DOCUMENT` (with document `title`) during ingestion.
     - `RETRIEVAL_QUERY` when embedding user search queries.
   - Processes chunks in batches of `50`.

3. **Neon Postgres + `pgvector` Storage (`connect_neon`, `upsert_chunks`)**
   - Enables the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector;`) before registering `pgvector` adapters on the `psycopg` connection.
   - Creates the `documents` table and an HNSW index (`vector_cosine_ops`) for fast Approximate Nearest Neighbor (ANN) search.
   - Uses idempotent upserts on `(source, chunk_index)` and removes stale trailing chunks when a file is re-ingested after editing.

4. **Semantic Search (`search_documents`)**
   - Embeds your natural-language question and ranks stored chunks using `pgvector`'s cosine distance operator (`<=>`), reporting similarity as `1 - (embedding <=> query_vector)`.

---

## Project Structure

| File | Purpose |
| --- | --- |
| `embed_and_search.py` | Main CLI tool (`demo`, `ingest`, `query`) handling chunking, Gemini embedding calls, Neon schema setup, upserts, and cosine search. |
| `sample_document.md` | Sample multi-section Markdown article (*Deep-Sea Hydrothermal Vents and Bioluminescent Ecosystems*) used by the `demo` command. |
| `requirements.txt` | Pinned Python dependencies (`google-genai`, `psycopg[binary]`, `pgvector`, `python-dotenv`). |
| `.env.example` | Environment variable template for `GEMINI_API_KEY` and `DATABASE_URL`. |
| `.gitignore` | Excludes `.env`, `.venv/`, and Python cache files from version control. |

---

## Prerequisites

- **Python 3.10+**
- **Gemini API Key:** Get a free key from Google AI Studio at https://aistudio.google.com/
- **Neon Postgres Database:** Create a free project at https://neon.tech/ and copy its PostgreSQL connection string (ensure `?sslmode=require` is included).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PetersonMambondimumwe/Embedding.git
cd Embedding
```

### 2. Create a virtual environment and install dependencies

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file from the template (never commit `.env`):

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
DATABASE_URL=postgresql://user:password@ep-your-endpoint.region.aws.neon.tech/neondb?sslmode=require
```

---

## Usage

Run all commands using the virtual environment's Python executable (`.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on macOS/Linux).

### Run the end-to-end demo

Ingests `sample_document.md` into Neon and runs the default query (`"How do giant tube worms survive without sunlight?"`):

```powershell
.\.venv\Scripts\python.exe embed_and_search.py demo
```

You can override the demo file, query, or number of results:

```powershell
.\.venv\Scripts\python.exe embed_and_search.py demo --query "How do deep-sea animals produce light?" --top-k 2
```

### Ingest a custom document

Chunk, embed, and upsert any Markdown or plain-text file:

```powershell
.\.venv\Scripts\python.exe embed_and_search.py ingest --file path\to\your_document.md
```

### Query stored embeddings

Search vectors already stored in Neon without re-ingesting:

```powershell
.\.venv\Scripts\python.exe embed_and_search.py query "Why are mining companies interested in hydrothermal vents?" --top-k 3
```

---

## Database Schema

When `embed_and_search.py` connects to Neon, it automatically provisions:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(768) NOT NULL,
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
ON documents USING hnsw (embedding vector_cosine_ops);
```
