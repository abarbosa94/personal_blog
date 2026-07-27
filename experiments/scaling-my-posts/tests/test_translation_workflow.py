from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "scaling-my-posts" / "src"
sys.path.insert(0, str(SRC))

from translation_workflow import select_sources  # noqa: E402


def write_notebook(path: Path, frontmatter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "raw",
                        "metadata": {},
                        "source": frontmatter.splitlines(keepends=True),
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_an_english_only_change_starts_translation(tmp_path: Path) -> None:
    source = "posts/example.ipynb"
    write_notebook(
        tmp_path / source,
        "---\nlang: en\ntranslation: example-pt-BR.ipynb\n---\n",
    )
    assert select_sources([source], repository=tmp_path) == [source]


def test_a_reciprocal_pair_change_does_not_loop(tmp_path: Path) -> None:
    source = "posts/example.ipynb"
    translation = "posts/example-pt-BR.ipynb"
    write_notebook(
        tmp_path / source,
        "---\nlang: en\ntranslation: example-pt-BR.ipynb\n---\n",
    )
    write_notebook(
        tmp_path / translation,
        "---\nlang: pt-BR\nlanguage-version: translation\n"
        "translation: example.ipynb\n---\n",
    )
    assert select_sources(
        [source, translation], repository=tmp_path
    ) == []


def test_manual_dispatch_can_regenerate_a_paired_source(tmp_path: Path) -> None:
    source = "posts/example.ipynb"
    write_notebook(tmp_path / source, "---\nlang: en\n---\n")
    assert select_sources(
        [], repository=tmp_path, manual_source=source
    ) == [source]
