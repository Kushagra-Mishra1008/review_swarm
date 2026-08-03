"""
Repo: clone or pull a target repository, then walk it to produce the
{file_path: source_code} dict that indexer.py needs.

Kept deliberately dumb — this file's only job is getting source code
onto disk and into memory. No chunking, no embedding, no search logic
lives here.
"""

import os
import subprocess

# Directories we never want to index — dependencies, build artifacts,
# version control internals, virtual envs. Extend as needed per repo.
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv", "env",
    "build", "dist", ".pytest_cache", ".mypy_cache", "site-packages",
    ".tox", "egg-info",
}


class RepoError(Exception):
    pass


class Repo:
    """
    Usage:
        repo = Repo(clone_url="https://github.com/fastapi/fastapi.git",
                     local_path="./data/fastapi")
        repo.clone_or_pull()
        files = repo.load_python_files()
    """

    def __init__(self, clone_url: str, local_path: str):
        self.clone_url = clone_url
        self.local_path = local_path

    def clone_or_pull(self) -> None:
        """
        Clones the repo if it doesn't exist locally yet, otherwise pulls
        the latest changes. Shallow clone (--depth 1) since we only need
        current source, not full history.
        """
        if os.path.exists(os.path.join(self.local_path, ".git")):
            self._pull()
        else:
            self._clone()

    def _clone(self) -> None:
        os.makedirs(os.path.dirname(self.local_path) or ".", exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", self.clone_url, self.local_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RepoError(f"git clone failed: {result.stderr.strip()}")

    def _pull(self) -> None:
        result = subprocess.run(
            ["git", "-C", self.local_path, "pull", "--ff-only"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RepoError(f"git pull failed: {result.stderr.strip()}")

    def load_python_files(self) -> dict[str, str]:
        """
        Walks the local repo path and reads every .py file, skipping
        SKIP_DIRS. Returns {relative_file_path: source_code}.

        Files that fail to decode as UTF-8 are silently skipped rather
        than crashing the whole index run — a handful of odd files
        shouldn't block indexing the other 500.
        """
        if not os.path.exists(self.local_path):
            raise RepoError(f"Repo not cloned yet: {self.local_path}")

        files: dict[str, str] = {}

        for root, dirs, filenames in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue

                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, self.local_path)

                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        files[rel_path] = f.read()
                except (UnicodeDecodeError, OSError):
                    continue

        return files