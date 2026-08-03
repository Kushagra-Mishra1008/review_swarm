"""
Verification gate for the Phase 1 retrieval layer.

Clones/pulls a real repo (FastAPI), indexes it, and runs three manual
spot-checks per the plan:
  - get_definition("APIRouter") returns the actual class
  - find_callers on a known utility returns real call sites
  - search_code(...) returns plausible chunks for a natural-language query

Zero LLM calls, zero token cost — this whole phase is local.
"""

import sys
import time

from retrieval.indexer import CodeIndexer
from retrieval.repo import Repo, RepoError
from retrieval.search import HybridSearch

FASTAPI_CLONE_URL = "https://github.com/fastapi/fastapi.git"
FASTAPI_LOCAL_PATH = "./data/fastapi"


def check(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main() -> int:
    all_passed = True

    # --- Step 1: clone/pull ---
    print("Cloning/pulling FastAPI...")
    repo = Repo(clone_url=FASTAPI_CLONE_URL, local_path=FASTAPI_LOCAL_PATH)
    try:
        repo.clone_or_pull()
    except RepoError as e:
        print(f"[FAIL] Repo clone/pull failed: {e}")
        return 1

    files = repo.load_python_files()
    all_passed &= check(f"Loaded Python files ({len(files)} found)", len(files) > 50)

    # --- Step 2: index ---
    print("\nIndexing (this can take a couple minutes on first run)...")
    start = time.time()
    indexer = CodeIndexer()
    summary = indexer.index_repo(files)
    elapsed = time.time() - start

    print(f"       Reindexed: {len(summary['reindexed'])} files")
    print(f"       Unchanged: {len(summary['unchanged'])} files")
    print(f"       Removed:   {len(summary['removed'])} files")
    print(f"       Time:      {elapsed:.1f}s")

    all_passed &= check(
        "Index has chunks",
        indexer.collection.count() > 0,
    )
    print(f"       Total chunks in collection: {indexer.collection.count()}")

    # --- Step 3: manual spot-checks ---
    search = HybridSearch(indexer, repo_root=FASTAPI_LOCAL_PATH)

    print("\n--- get_definition('APIRouter') ---")
    definition = search.get_definition("APIRouter")
    found_def = definition is not None
    all_passed &= check("get_definition('APIRouter') found something", found_def)
    if definition:
        print(f"       {definition.file_path}:{definition.start_line}-{definition.end_line} ({definition.kind})")
        all_passed &= check(
            "Definition kind is 'class'",
            definition.kind == "class",
        )

    print("\n--- find_callers('jsonable_encoder') ---")
    callers = search.find_callers("jsonable_encoder")
    all_passed &= check("find_callers('jsonable_encoder') found call sites", len(callers) > 0)
    for c in callers[:5]:
        print(f"       {c.file_path}:{c.start_line} ({c.symbol_name})")

    print("\n--- search_code('how are exceptions handled in middleware') ---")
    results = search.search_code("how are exceptions handled in middleware", k=5)
    all_passed &= check("search_code returned results", len(results) > 0)
    for r in results:
        print(f"       [{r.score:.3f}] {r.file_path}:{r.start_line} — {r.kind} {r.symbol_name}")

    print()
    if all_passed:
        print("Phase 1 retrieval verification: ALL CHECKS PASSED")
        return 0
    else:
        print("Phase 1 retrieval verification: SOME CHECKS FAILED (see above — inspect manually)")
        return 1


if __name__ == "__main__":
    sys.exit(main())