import os
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── Constants ────────────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "neuro_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


# ── Singletons ────────────────────────────────────────────────────────────────
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(client: Optional[chromadb.PersistentClient] = None):
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _file_hash(path: str) -> str:
    """SHA-256 of file bytes — used to detect duplicates."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _extract_text_by_page(path: str) -> list[dict]:
    """Return list of {page: int, text: str} dicts."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def _chunk_pages(pages: list[dict], filename: str) -> list[dict]:
    """
    Split each page's text into overlapping chunks.
    Returns list of chunk dicts with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for j, text in enumerate(splits):
            chunks.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page["page"],
                    "chunk_index": j,
                }
            )
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────
def ingest_pdf(path: str) -> dict:
    """
    Parse, chunk, embed and store a PDF in ChromaDB.
    Returns a summary dict: {filename, pages, chunks, skipped}.
    """
    path = str(Path(path).resolve())
    filename = Path(path).name
    file_hash = _file_hash(path)

    collection = get_collection()
    embedder = get_embedder()

    # Duplicate check via stored metadata
    existing = collection.get(where={"file_hash": file_hash}, limit=1)
    if existing["ids"]:
        return {"filename": filename, "pages": 0, "chunks": 0, "skipped": True}

    pages = _extract_text_by_page(path)
    chunks = _chunk_pages(pages, filename)

    if not chunks:
        return {"filename": filename, "pages": 0, "chunks": 0, "skipped": False}

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": c["filename"],
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "file_hash": file_hash,
        }
        for c in chunks
    ]

    # ChromaDB add in batches of 500 to avoid memory spikes
    batch = 500
    for start in range(0, len(ids), batch):
        collection.add(
            ids=ids[start : start + batch],
            embeddings=embeddings[start : start + batch],
            documents=texts[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "skipped": False,
    }


def list_ingested_files() -> list[str]:
    """Return sorted list of unique filenames already in the collection."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    seen = set()
    for meta in results["metadatas"]:
        seen.add(meta["filename"])
    return sorted(seen)


def delete_pdf(filename: str) -> int:
    """Remove all chunks belonging to a given filename. Returns deleted count."""
    collection = get_collection()
    results = collection.get(where={"filename": filename})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])
