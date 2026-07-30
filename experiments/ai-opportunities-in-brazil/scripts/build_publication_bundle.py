"""Build the compact, Git-tracked evidence bundle used by the blog post."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "analysis"
PUBLICATION = ROOT / "publication"

PUBLIC_FILES = [
    (ROOT / "analysis-specs" / "country-comparison-panel.csv", "country-comparison-panel.csv"),
    (ROOT / "data" / "sources" / "registry.csv", "source-registry.csv"),
    (ANALYSIS / "baseline-seven-indicator-evidence-matrix.csv", "baseline-seven-indicator-evidence-matrix.csv"),
    (ANALYSIS / "baseline-seven-indicator-evidence-matrix.md", "baseline-seven-indicator-evidence-matrix.md"),
    (ANALYSIS / "conference-presence-2025-seven-venues-brief.md", "conference-presence-2025-seven-venues-brief.md"),
    (ANALYSIS / "conference-presence-2025-seven-venues-pooled.csv", "conference-presence-2025-seven-venues-pooled.csv"),
    (ANALYSIS / "conference-presence-2025-seven-venues-pooled.metadata.json", "conference-presence-2025-seven-venues-pooled.metadata.json"),
    (ANALYSIS / "conference-presence-2025-seven-venues-sensitivity.csv", "conference-presence-2025-seven-venues-sensitivity.csv"),
    (ANALYSIS / "conference-presence-2025-seven-venues-sensitivity.metadata.json", "conference-presence-2025-seven-venues-sensitivity.metadata.json"),
    (ANALYSIS / "ai-index-economy-2025.csv", "ai-index-economy-2025.csv"),
    (ANALYSIS / "ai-index-economy-2025.metadata.json", "ai-index-economy-2025.metadata.json"),
    (ANALYSIS / "epoch-notable-models-2025.csv", "epoch-notable-models-2025.csv"),
    (ANALYSIS / "epoch-notable-models-2025.metadata.json", "epoch-notable-models-2025.metadata.json"),
    (ANALYSIS / "epoch-notable-models-2026-ytd.csv", "epoch-notable-models-2026-ytd.csv"),
    (ANALYSIS / "epoch-notable-models-2026-ytd.metadata.json", "epoch-notable-models-2026-ytd.metadata.json"),
    (ANALYSIS / "top500-2025-11.csv", "top500-2025-11.csv"),
    (ANALYSIS / "top500-2025-11.metadata.json", "top500-2025-11.metadata.json"),
    (ANALYSIS / "top500-2026-06.csv", "top500-2026-06.csv"),
    (ANALYSIS / "top500-2026-06.metadata.json", "top500-2026-06.metadata.json"),
    (ANALYSIS / "world-bank-factor-context.csv", "world-bank-factor-context.csv"),
    (ANALYSIS / "world-bank-factor-context-manifest.json", "world-bank-factor-context-manifest.json"),
    (ANALYSIS / "hypothesis-assessments.csv", "hypothesis-assessments.csv"),
    (ANALYSIS / "hypothesis-assessments.md", "hypothesis-assessments.md"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    missing = [relative_to_root(source) for source, _ in PUBLIC_FILES if not source.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Regenerate the frozen analysis outputs before building the "
            f"publication bundle. Missing files:\n{formatted}"
        )

    PUBLICATION.mkdir(parents=True, exist_ok=True)
    manifest_files = []

    for source, published_name in PUBLIC_FILES:
        destination = PUBLICATION / published_name
        shutil.copyfile(source, destination)
        source_hash = sha256(source)
        published_hash = sha256(destination)
        if source_hash != published_hash:
            raise RuntimeError(f"Copied file hash mismatch: {published_name}")
        manifest_files.append(
            {
                "published_file": published_name,
                "source_artifact": relative_to_root(source),
                "bytes": destination.stat().st_size,
                "sha256": published_hash,
            }
        )

    manifest = {
        "bundle_version": "1.0",
        "frozen_on": "2026-07-30",
        "purpose": "Compact evidence snapshot supporting the published blog analysis.",
        "files": manifest_files,
    }
    (PUBLICATION / "source-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (PUBLICATION / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['published_file']}\n" for item in manifest_files),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
