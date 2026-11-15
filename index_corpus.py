"""Index the PDFs in the corpus directory into Typesense for semantic search.

Pipeline: PDF -> text (per page) -> overlapping word chunks -> JSON documents
(with a reference to the source PDF and pages) -> Typesense import.

Embeddings are computed inside Typesense by its built-in local model, so no
external embedding service is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterator

import typesense
from pypdf import PdfReader

DEFAULT_CORPUS = Path(__file__).parent / "corpus"
DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_MODEL = "ts/all-MiniLM-L12-v2"


def collection_schema(name: str, model: str) -> dict:
    return {
        "name": name,
        "fields": [
            {"name": "doc_id", "type": "string", "facet": True},
            {"name": "file_name", "type": "string", "facet": True},
            {"name": "file_path", "type": "string", "index": False, "optional": True},
            {"name": "title", "type": "string", "optional": True},
            {"name": "chunk_index", "type": "int32"},
            {"name": "page_start", "type": "int32"},
            {"name": "page_end", "type": "int32"},
            {"name": "text", "type": "string"},
            {
                "name": "embedding",
                "type": "float[]",
                "embed": {"from": ["text"], "model_config": {"model_name": model}},
            },
        ],
        "default_sorting_field": "chunk_index",
    }


def extract_pages(pdf_path: Path) -> tuple[list[str], str | None]:
    """Return the normalized text of each page and the PDF title (if any)."""
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # Re-join words hyphenated across line breaks, then collapse whitespace.
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        pages.append(re.sub(r"\s+", " ", text).strip())
    title = None
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip() or None
    return pages, title


def chunk_pages(
    pages: list[str], chunk_size: int, overlap: int
) -> Iterator[tuple[str, int, int]]:
    """Yield (text, page_start, page_end) word-based chunks with overlap.

    Page numbers are 1-based and track which pages each chunk spans.
    """
    words: list[str] = []
    word_pages: list[int] = []
    for page_no, page_text in enumerate(pages, start=1):
        for word in page_text.split():
            words.append(word)
            word_pages.append(page_no)

    if not words:
        return

    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end]), word_pages[start], word_pages[end - 1]
        if end == len(words):
            break


def build_documents(
    pdf_path: Path, corpus_dir: Path, chunk_size: int, overlap: int
) -> list[dict]:
    pages, title = extract_pages(pdf_path)
    rel_path = pdf_path.relative_to(corpus_dir).as_posix()
    doc_id = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]

    docs = []
    for i, (text, page_start, page_end) in enumerate(
        chunk_pages(pages, chunk_size, overlap)
    ):
        doc = {
            "id": f"{doc_id}-{i:05d}",
            "doc_id": doc_id,
            "file_name": pdf_path.name,
            "file_path": rel_path,
            "chunk_index": i,
            "page_start": page_start,
            "page_end": page_end,
            "text": text,
        }
        if title:
            doc["title"] = title
        docs.append(doc)
    return docs


def make_client(host: str, port: int, protocol: str, api_key: str) -> typesense.Client:
    return typesense.Client(
        {
            "nodes": [{"host": host, "port": port, "protocol": protocol}],
            "api_key": api_key,
            # Embedding generation during import can be slow on CPU.
            "connection_timeout_seconds": 300,
        }
    )


def ensure_collection(client: typesense.Client, name: str, model: str, recreate: bool) -> None:
    exists = True
    try:
        client.collections[name].retrieve()
    except typesense.exceptions.ObjectNotFound:
        exists = False

    if exists and recreate:
        print(f"Dropping existing collection '{name}'")
        client.collections[name].delete()
        exists = False

    if not exists:
        print(f"Creating collection '{name}' (embedding model: {model})")
        client.collections.create(collection_schema(name, model))


def delete_existing_chunks(client: typesense.Client, collection: str, doc_id: str) -> None:
    """Remove previous chunks of a PDF so re-indexing never leaves stale chunks."""
    client.collections[collection].documents.delete({"filter_by": f"doc_id:={doc_id}"})


def index_documents(
    client: typesense.Client, collection: str, docs: list[dict], batch_size: int
) -> int:
    failures = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        results = client.collections[collection].documents.import_(
            batch, {"action": "upsert"}
        )
        for doc, res in zip(batch, results):
            if not res.get("success"):
                failures += 1
                print(f"  ! failed to index {doc['id']}: {res.get('error')}", file=sys.stderr)
    return failures


def index_pdf(
    client: typesense.Client,
    collection: str,
    pdf_path: Path,
    corpus_dir: Path,
    chunk_size: int = 200,
    overlap: int = 50,
    batch_size: int = 100,
) -> tuple[int, int]:
    """Index (or re-index) a single PDF. Returns (indexed_chunks, total_chunks).

    Raises ValueError if the PDF has no extractable text; pypdf errors propagate.
    """
    docs = build_documents(pdf_path, corpus_dir, chunk_size, overlap)
    if not docs:
        raise ValueError("no extractable text (scanned PDF?)")
    delete_existing_chunks(client, collection, docs[0]["doc_id"])
    failures = index_documents(client, collection, docs, batch_size)
    return len(docs) - failures, len(docs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Directory containing PDFs (searched recursively)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Typesense built-in embedding model")
    p.add_argument("--chunk-size", type=int, default=200, help="Words per chunk")
    p.add_argument("--overlap", type=int, default=50, help="Words shared by consecutive chunks")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--recreate", action="store_true", help="Drop and recreate the collection first")
    p.add_argument("--host", default=os.getenv("TYPESENSE_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("TYPESENSE_PORT", "8108")))
    p.add_argument("--protocol", default=os.getenv("TYPESENSE_PROTOCOL", "http"))
    p.add_argument("--api-key", default=os.getenv("TYPESENSE_API_KEY", "dev-secret-key"))
    args = p.parse_args()
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        p.error("require chunk-size > 0 and 0 <= overlap < chunk-size")
    return args


def main() -> int:
    args = parse_args()
    corpus_dir = args.corpus.resolve()
    pdfs = sorted(corpus_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {corpus_dir}", file=sys.stderr)
        return 1

    client = make_client(args.host, args.port, args.protocol, args.api_key)
    ensure_collection(client, args.collection, args.model, args.recreate)

    total_chunks = total_failures = 0
    for pdf in pdfs:
        try:
            indexed, total = index_pdf(
                client, args.collection, pdf, corpus_dir,
                args.chunk_size, args.overlap, args.batch_size,
            )
        except typesense.exceptions.TypesenseClientError:
            raise
        except Exception as exc:
            print(f"Skipping {pdf.name}: {exc}", file=sys.stderr)
            continue

        print(f"{pdf.name}: {indexed}/{total} chunks indexed")
        total_chunks += total
        total_failures += total - indexed

    print(f"Done: {total_chunks - total_failures}/{total_chunks} chunks from {len(pdfs)} PDFs "
          f"in collection '{args.collection}'")
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
