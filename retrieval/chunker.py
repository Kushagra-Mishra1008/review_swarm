"""
Chunker: splits a Python source file into function/class-level chunks
using tree-sitter, instead of naive fixed-line-count slicing.

Each chunk is a complete, self-contained unit of code (a whole function,
a whole class) with metadata attached — never a fragment cut mid-body.
This is what makes the embeddings in indexer.py meaningful.
"""

from dataclasses import dataclass, asdict

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

# Node types we care about extracting as standalone chunks.
CHUNKABLE_NODE_TYPES = {
    "function_definition": "function",
    "class_definition": "class",
}


@dataclass
class CodeChunk:
    file_path: str
    symbol_name: str
    kind: str          # "function" | "class" | "method"
    language: str
    start_line: int      # 1-indexed, inclusive
    end_line: int          # 1-indexed, inclusive
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


class PythonChunker:
    """
    Parses a single Python file and returns one CodeChunk per top-level
    and nested function/class definition.

    A method inside a class is still emitted as its own chunk (kind=
    "method"), separately from the class chunk itself — both are useful:
    the class chunk gives structure/context, the method chunk gives a
    precise, small, embeddable unit for search.
    """

    def __init__(self):
        self._parser = Parser(PY_LANGUAGE)

    def chunk_file(self, file_path: str, source_code: str) -> list[CodeChunk]:
        source_bytes = source_code.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        chunks: list[CodeChunk] = []
        self._walk(root, source_bytes, file_path, chunks, inside_class=False)
        return chunks

    def _walk(
        self,
        node,
        source_bytes: bytes,
        file_path: str,
        chunks: list[CodeChunk],
        inside_class: bool,
    ) -> None:
        for child in node.children:
            if child.type in CHUNKABLE_NODE_TYPES:
                kind = CHUNKABLE_NODE_TYPES[child.type]
                if kind == "function" and inside_class:
                    kind = "method"

                symbol_name = self._extract_name(child, source_bytes)
                chunk_source = source_bytes[child.start_byte:child.end_byte].decode("utf-8")

                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        symbol_name=symbol_name,
                        kind=kind,
                        language="python",
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        source=chunk_source,
                    )
                )

                # Recurse into function/class bodies to find nested defs
                # (methods inside classes, nested functions).
                is_class = child.type == "class_definition"
                self._walk(child, source_bytes, file_path, chunks, inside_class=is_class)
            else:
                # Not a chunkable node itself — keep walking down in case
                # a def/class is nested inside (e.g. inside an if-block).
                self._walk(child, source_bytes, file_path, chunks, inside_class=inside_class)

    @staticmethod
    def _extract_name(node, source_bytes: bytes) -> str:
        """
        Function and class definition nodes have a direct 'identifier'
        child holding the name. Fall back to '<anonymous>' if the grammar
        shape ever changes underneath us.
        """
        for child in node.children:
            if child.type == "identifier":
                return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
        return "<anonymous>"


def chunk_source_file(file_path: str, source_code: str) -> list[dict]:
    """
    Convenience wrapper: parse one file, return chunks as plain dicts
    (ready for indexer.py to embed and store).
    """
    chunker = PythonChunker()
    return [c.to_dict() for c in chunker.chunk_file(file_path, source_code)]