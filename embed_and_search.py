"""Chunk, embed, store, and query documents in Neon Postgres using Gemini embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector import Vector
from pgvector.psycopg import register_vector
import psycopg
from psycopg.types.json import Jsonb

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
DEFAULT_SAMPLE_FILE = Path(__file__).with_name("sample_document.md")
DEFAULT_DEMO_QUERY = "How do giant tube worms survive without sunlight?"


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a single chunk extracted from a source document."""

    source: str
    section: str
    chunk_index: int
    content: str


def require_env(name: str, placeholder_prefix: str) -> str:
    """Return a configured environment variable or exit with a clear hint."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith(placeholder_prefix):
        print(
            f"Missing {name}. Copy .env.example to .env and set {name} before running.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def chunk_markdown(text: str, source: str, max_chars: int = 700) -> list[DocumentChunk]:
    """Split markdown text into section-aware chunks bounded by max_chars."""
    sections = _split_by_headings(text)
    chunks: list[DocumentChunk] = []

    for section, body in sections:
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        for part in _pack_paragraphs(paragraphs, max_chars):
            chunks.append(
                DocumentChunk(
                    source=source,
                    section=section,
                    chunk_index=len(chunks),
                    content=part,
                )
            )
    return chunks


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Group markdown lines under their nearest heading."""
    sections: list[tuple[str, str]] = []
    current_heading = "Introduction"
    buffer: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = stripped.lstrip("#").strip() or current_heading
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))
    return [(heading, body) for heading, body in sections if body]


def _pack_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Pack consecutive paragraphs into chunks up to max_chars."""
    packed: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                packed.append("\n\n".join(current))
                current, current_len = [], 0
            packed.extend(_split_long_paragraph(paragraph, max_chars))
            continue

        added_len = len(paragraph) + (2 if current else 0)
        if current and current_len + added_len > max_chars:
            packed.append("\n\n".join(current))
            current, current_len = [paragraph], len(paragraph)
        else:
            current.append(paragraph)
            current_len += added_len

    if current:
        packed.append("\n\n".join(current))
    return packed


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Fallback split on word boundaries when a single paragraph exceeds max_chars."""
    words = paragraph.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars:
            parts.append(" ".join(current))
            current, current_len = [word], len(word)
        else:
            current.append(word)
            current_len += added_len

    if current:
        parts.append(" ".join(current))
    return parts


def embed_texts(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    title: str | None = None,
    batch_size: int = 50,
) -> list[list[float]]:
    """Generate Gemini embeddings for a list of strings."""
    if not texts:
        return []

    config_kwargs: dict[str, object] = {
        "task_type": task_type,
        "output_dimensionality": EMBEDDING_DIM,
    }
    if title and task_type == "RETRIEVAL_DOCUMENT":
        config_kwargs["title"] = title
    config = types.EmbedContentConfig(**config_kwargs)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=config,
        )
        if not response.embeddings:
            raise RuntimeError("Gemini API returned no embeddings.")
        vectors.extend(list(item.values) for item in response.embeddings)
    return vectors


def connect_neon(database_url: str) -> psycopg.Connection:
    """Open a Neon connection, enable pgvector, and initialize schema."""
    conn = psycopg.connect(database_url)
    with conn.cursor() as cur:
        # Must exist before register_vector() looks up the vector type OID.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: psycopg.Connection) -> None:
    """Create the documents table and HNSW cosine index if missing."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                UNIQUE (source, chunk_index)
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
            ON documents USING hnsw (embedding vector_cosine_ops);
            """
        )
    conn.commit()


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
) -> int:
    """Insert or update document chunks and remove stale trailing chunks."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding count must match.")
    if not chunks:
        return 0

    source = chunks[0].source
    rows = [
        (
            chunk.source,
            chunk.chunk_index,
            chunk.content,
            Jsonb({"section": chunk.section}),
            Vector(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (source, chunk_index, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source, chunk_index)
            DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding;
            """,
            rows,
        )
        cur.execute(
            "DELETE FROM documents WHERE source = %s AND chunk_index >= %s;",
            (source, len(chunks)),
        )
    conn.commit()
    return len(rows)


def search_documents(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[tuple[str, int, str, dict[str, object], float]]:
    """Run cosine similarity search against Neon pgvector."""
    query_vec = Vector(query_embedding)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                source,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM documents
            ORDER BY embedding <=> %s ASC
            LIMIT %s;
            """,
            (query_vec, query_vec, top_k),
        )
        return cur.fetchall()


def run_ingest(file_path: Path, client: genai.Client, conn: psycopg.Connection) -> None:
    """Read, chunk, embed, and store a document in Neon."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(text, source=file_path.name)
    print(f"Chunked '{file_path.name}' into {len(chunks)} chunks.")

    embeddings = embed_texts(
        client,
        [chunk.content for chunk in chunks],
        task_type="RETRIEVAL_DOCUMENT",
        title=file_path.stem.replace("_", " ").title(),
    )
    stored = upsert_chunks(conn, chunks, embeddings)
    print(f"Stored {stored} vectors (dim={EMBEDDING_DIM}) in Neon table 'documents'.")


def run_query(
    question: str,
    top_k: int,
    client: genai.Client,
    conn: psycopg.Connection,
) -> None:
    """Embed a query and display the closest matching chunks from Neon."""
    print(f"\nQuery: {question}")
    query_embedding = embed_texts(client, [question], task_type="RETRIEVAL_QUERY")[0]
    results = search_documents(conn, query_embedding, top_k=top_k)

    if not results:
        print("No documents found in 'documents'. Run 'ingest' first.")
        return

    for rank, (source, chunk_index, content, metadata, similarity) in enumerate(results, start=1):
        section = metadata.get("section", "Unknown")
        print(f"\n--- Result #{rank} | similarity={similarity:.4f} | {source} [{section}] (chunk {chunk_index}) ---")
        print(content)


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI subcommands."""
    parser = argparse.ArgumentParser(
        description="Embed documents with Gemini and search them in Neon pgvector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = subparsers.add_parser("ingest", help="Chunk, embed, and store a document.")
    ingest_cmd.add_argument("--file", type=Path, default=DEFAULT_SAMPLE_FILE)

    query_cmd = subparsers.add_parser("query", help="Search stored embeddings.")
    query_cmd.add_argument("question", help="Natural-language search query.")
    query_cmd.add_argument("--top-k", type=int, default=3)

    demo_cmd = subparsers.add_parser(
        "demo", help="Ingest the sample document and run a similarity query."
    )
    demo_cmd.add_argument("--file", type=Path, default=DEFAULT_SAMPLE_FILE)
    demo_cmd.add_argument("--query", default=DEFAULT_DEMO_QUERY)
    demo_cmd.add_argument("--top-k", type=int, default=3)

    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    gemini_api_key = require_env("GEMINI_API_KEY", "your_gemini_api_key")
    database_url = require_env("DATABASE_URL", "postgresql://user:password@host")

    client = genai.Client(api_key=gemini_api_key)
    with connect_neon(database_url) as conn:
        if args.command == "ingest":
            run_ingest(args.file, client, conn)
        elif args.command == "query":
            run_query(args.question, args.top_k, client, conn)
        elif args.command == "demo":
            run_ingest(args.file, client, conn)
            run_query(args.query, args.top_k, client, conn)


if __name__ == "__main__":
    main()
