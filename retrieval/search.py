"""
Hybrid search: combines vector similarity (ChromaDB) with exact string
matching (ripgrep) and merges the two via reciprocal rank fusion.

Pure vector search is genuinely bad at exact symbol lookup — searching
for "get_definition" might rank a semantically-similar-but-wrong function
above the actual one. Grep fixes that. This file also exposes the three
query functions that become MCP tools in Phase 4:
  - search_code(query, k)
  - find_callers(symbol)
  - get_definition(symbol)
"""

import os
import subprocess
from dataclasses import dataclass

from retrieval.indexer import CodeIndexer

RRF_K = 60  # standard reciprocal rank fusion constant


@dataclass
class SearchResult:
    file_path: str
    symbol_name: str
    kind: str
    start_line: int
    end_line: int
    source: str
    score: float


class HybridSearch:
    def __init__(self, indexer: CodeIndexer, repo_root: str):
        self._indexer = indexer
        self._repo_root = repo_root

    # ---- Public query functions (become MCP tools in Phase 4) ----------

    def search_code(self, query: str, k: int = 5) -> list[SearchResult]:
        """Semantic + exact-match hybrid search, merged via RRF."""
        vector_hits = self._vector_search(query, k=k * 3)
        grep_hits = self._grep_search(query, k=k * 3)
        return self._reciprocal_rank_fusion(vector_hits, grep_hits, k=k)

    def find_callers(self, symbol: str) -> list[SearchResult]:
        """
        Exact-match only — 'who calls this' is a grep problem, not a
        semantic one. Excludes the definition line itself where possible
        by filtering out chunks whose symbol_name equals the target
        (best-effort; a call site inside another function with the same
        name would still slip through, which is an acceptable edge case).
        """
        raw_hits = self._grep_raw(symbol)
        results = []
        for file_path, line_num, line_text in raw_hits:
            chunk = self._chunk_containing_line(file_path, line_num)
            if chunk and chunk["symbol_name"] != symbol:
                results.append(self._chunk_to_result(chunk, score=1.0))
        return self._dedupe(results)

    def get_definition(self, symbol: str) -> SearchResult | None:
        """
        Exact match on symbol_name in the chunk metadata — the most
        precise possible lookup, no ranking needed.
        """
        collection = self._indexer.collection
        hits = collection.get(
            where={"symbol_name": symbol},
            include=["documents", "metadatas"],
        )
        if not hits["ids"]:
            return None

        # Prefer a function/class definition over a method if multiple
        # symbols share the same name across files.
        for meta, doc in zip(hits["metadatas"], hits["documents"]):
            if meta["kind"] in ("function", "class"):
                return SearchResult(
                    file_path=meta["file_path"],
                    symbol_name=meta["symbol_name"],
                    kind=meta["kind"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    source=doc,
                    score=1.0,
                )

        meta, doc = hits["metadatas"][0], hits["documents"][0]
        return SearchResult(
            file_path=meta["file_path"],
            symbol_name=meta["symbol_name"],
            kind=meta["kind"],
            start_line=meta["start_line"],
            end_line=meta["end_line"],
            source=doc,
            score=1.0,
        )

    # ---- Internals -------------------------------------------------------

    def _vector_search(self, query: str, k: int) -> list[SearchResult]:
        embedder = self._indexer._embedder
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()

        hits = self._indexer.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )
        results = []
        if not hits["ids"] or not hits["ids"][0]:
            return results

        for meta, doc, distance in zip(
            hits["metadatas"][0], hits["documents"][0], hits["distances"][0]
        ):
            results.append(
                SearchResult(
                    file_path=meta["file_path"],
                    symbol_name=meta["symbol_name"],
                    kind=meta["kind"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    source=doc,
                    score=1.0 / (1.0 + distance),  # lower distance = higher score
                )
            )
        return results

    def _grep_search(self, query: str, k: int) -> list[SearchResult]:
        raw_hits = self._grep_raw(query)
        results = []
        for file_path, line_num, _ in raw_hits[:k]:
            chunk = self._chunk_containing_line(file_path, line_num)
            if chunk:
                results.append(self._chunk_to_result(chunk, score=1.0))
        return self._dedupe(results)

    def _grep_raw(self, pattern: str) -> list[tuple[str, int, str]]:
        """
        Runs ripgrep over the repo root. Returns (file_path, line_number,
        line_text) tuples, with file_path normalized to be relative to
        repo_root — matching the format indexer.py stores in metadata.
        Falls back to an empty list if ripgrep isn't installed or the
        search errors — hybrid search degrades to vector-only rather
        than crashing.

        encoding/errors are forced to utf-8/replace explicitly: on
        Windows, subprocess defaults to the system codepage (cp1252),
        which crashes on any non-cp1252 byte ripgrep's output contains.
        """
        try:
            proc = subprocess.run(
                ["rg", "--line-number", "--no-heading", "--fixed-strings", pattern, self._repo_root],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        if proc.returncode not in (0, 1):  # 1 = no matches, still fine
            return []
        if proc.stdout is None:
            return []

        results = []
        for line in proc.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            raw_file_path, line_num_str, line_text = parts
            try:
                line_num = int(line_num_str)
            except ValueError:
                continue

            # Normalize to match the relative-path format used in the
            # index (relative to repo_root), since ripgrep returns paths
            # relative to whatever cwd/argument it was invoked with.
            try:
                rel_path = os.path.relpath(raw_file_path, self._repo_root)
            except ValueError:
                rel_path = raw_file_path

            results.append((rel_path, line_num, line_text))
        return results
    def _chunk_containing_line(self, file_path: str, line_num: int) -> dict | None:
        """Find the chunk metadata whose line range contains line_num."""
        hits = self._indexer.collection.get(
            where={"file_path": file_path},
            include=["documents", "metadatas"],
        )
        for meta, doc in zip(hits["metadatas"], hits["documents"]):
            if meta["start_line"] <= line_num <= meta["end_line"]:
                return {**meta, "source": doc}
        return None

    @staticmethod
    def _chunk_to_result(chunk: dict, score: float) -> SearchResult:
        return SearchResult(
            file_path=chunk["file_path"],
            symbol_name=chunk["symbol_name"],
            kind=chunk["kind"],
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            source=chunk["source"],
            score=score,
        )

    @staticmethod
    def _dedupe(results: list[SearchResult]) -> list[SearchResult]:
        seen = set()
        deduped = []
        for r in results:
            key = (r.file_path, r.symbol_name, r.start_line)
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_hits: list[SearchResult],
        grep_hits: list[SearchResult],
        k: int,
    ) -> list[SearchResult]:
        """
        RRF score for a chunk = sum over each ranked list of 1/(RRF_K + rank).
        A chunk appearing near the top of both lists outranks one that's
        only near the top of one list.
        """
        scores: dict[tuple, float] = {}
        chunk_lookup: dict[tuple, SearchResult] = {}

        for ranked_list in (vector_hits, grep_hits):
            for rank, result in enumerate(ranked_list):
                key = (result.file_path, result.symbol_name, result.start_line)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
                chunk_lookup[key] = result

        ranked_keys = sorted(scores.keys(), key=lambda k_: scores[k_], reverse=True)
        top_results = []
        for key in ranked_keys[:k]:
            result = chunk_lookup[key]
            result.score = scores[key]
            top_results.append(result)
        return top_results