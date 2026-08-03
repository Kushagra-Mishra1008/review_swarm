"""
Indexer: embeds code chunks and stores them in ChromaDB, with an
incremental reindex path based on a file-hash manifest so re-runs only
touch files that actually changed.

Uses sentence-transformers/all-MiniLM-L6-v2 locally on CPU — free, fast,
and touches zero rate limit or token budget.
"""

import hashlib
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

from retrieval.chunker import chunk_source_file

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = ".cache/chroma_db"
MANIFEST_PATH = ".cache/chroma_db/file_manifest.json"
COLLECTION_NAME = "code_chunks"


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(chunk: dict) -> str:
    """Stable, unique ID per chunk: file + symbol + line range."""
    raw = f"{chunk['file_path']}:{chunk['symbol_name']}:{chunk['start_line']}:{chunk['end_line']}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CodeIndexer:
    def __init__(self, persist_dir: str = CHROMA_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, str]:
        if not os.path.exists(MANIFEST_PATH):
            return {}
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self) -> None:
        os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
        with open(MANIFEST_PATH, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def index_repo(self, repo_files: dict[str, str]) -> dict:
        """
        repo_files: {file_path: source_code} for every file to consider.
        Only files whose hash changed since last run get re-embedded.
        Files removed since last run get their chunks deleted.

        Returns a summary dict: {"reindexed": [...], "unchanged": [...], "removed": [...]}.
        """
        reindexed = []
        unchanged = []

        current_paths = set(repo_files.keys())
        previous_paths = set(self._manifest.keys())
        removed_paths = previous_paths - current_paths

        for file_path, source in repo_files.items():
            new_hash = _file_hash(source)
            old_hash = self._manifest.get(file_path)

            if old_hash == new_hash:
                unchanged.append(file_path)
                continue

            # Changed or new — remove any old chunks for this file first,
            # then chunk + embed + store fresh.
            self._delete_file_chunks(file_path)

            chunks = chunk_source_file(file_path, source)
            if chunks:
                self._embed_and_store(chunks)

            self._manifest[file_path] = new_hash
            reindexed.append(file_path)

        for file_path in removed_paths:
            self._delete_file_chunks(file_path)
            del self._manifest[file_path]

        self._save_manifest()

        return {
            "reindexed": reindexed,
            "unchanged": unchanged,
            "removed": list(removed_paths),
        }

    def _embed_and_store(self, chunks: list[dict], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [self._format_for_embedding(c) for c in batch]
            embeddings = self._embedder.encode(texts, convert_to_numpy=True).tolist()

            self._collection.add(
                ids=[_chunk_id(c) for c in batch],
                embeddings=embeddings,
                documents=[c["source"] for c in batch],
                metadatas=[
                    {
                        "file_path": c["file_path"],
                        "symbol_name": c["symbol_name"],
                        "kind": c["kind"],
                        "language": c["language"],
                        "start_line": c["start_line"],
                        "end_line": c["end_line"],
                    }
                    for c in batch
                ],
            )

    def _delete_file_chunks(self, file_path: str) -> None:
        """Delete all chunks belonging to a given file, if any exist."""
        self._collection.delete(where={"file_path": file_path})

    @staticmethod
    def _format_for_embedding(chunk: dict) -> str:
        """
        Prepend symbol name + kind to the source before embedding — this
        gives the embedding model a bit of extra signal beyond raw code,
        which noticeably helps for short/generic-looking functions.
        """
        return f"{chunk['kind']} {chunk['symbol_name']}\n{chunk['source']}"

    @property
    def collection(self):
        return self._collection