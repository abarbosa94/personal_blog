"""Select source notebooks for the post-merge translation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def frontmatter_value(notebook_path: Path, key: str) -> str | None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not notebook["cells"]:
        return None
    source = "".join(notebook["cells"][0].get("source", []))
    prefix = f"{key}:"
    for line in source.splitlines():
        if line.strip().startswith(prefix):
            return line.split(":", 1)[1].strip().strip("'\"")
    return None


def select_sources(
    changed_paths: list[str],
    *,
    repository: Path,
    manual_source: str | None = None,
) -> list[str]:
    """Return English sources, excluding the reciprocal translation-merge loop."""
    normalized_changes = {Path(path).as_posix() for path in changed_paths}
    candidates = [manual_source] if manual_source else sorted(normalized_changes)
    selected: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        relative = Path(candidate).as_posix()
        path = repository / relative
        if (
            not relative.startswith("posts/")
            or path.suffix != ".ipynb"
            or not path.exists()
            or frontmatter_value(path, "language-version") == "translation"
        ):
            continue
        translation = frontmatter_value(path, "translation")
        if (
            not manual_source
            and translation
            and (path.parent / translation).relative_to(repository).as_posix()
            in normalized_changes
        ):
            continue
        selected.append(relative)
    return selected


def changed_notebooks(before: str, after: str, repository: Path) -> list[str]:
    if not before or set(before) == {"0"}:
        before = f"{after}^"
    result = subprocess.run(
        ["git", "diff", "--name-only", before, after, "--", "posts/*.ipynb"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--manual-source")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    changes = (
        []
        if args.manual_source
        else changed_notebooks(args.before, args.after, args.repository)
    )
    sources = select_sources(
        changes,
        repository=args.repository,
        manual_source=args.manual_source,
    )
    print(json.dumps({"source": sources}, separators=(",", ":")))
