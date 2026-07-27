from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "scaling-my-posts" / "src"
sys.path.insert(0, str(SRC))

from translate_notebook import (  # noqa: E402
    is_structural_block,
    materialize_translation,
    protect,
    source_frontmatter,
    split_markdown,
    translate_around_protected,
    validate_heading_parity,
    validate_notebook_structure,
)


def test_split_markdown_retains_blank_lines() -> None:
    source = "# Heading\n\nA paragraph.\n\n- A list\n"
    assert "".join(split_markdown(source)) == source


def test_structural_blocks_are_not_translated() -> None:
    assert is_structural_block("$$\nx + y\n$$")
    assert is_structural_block(":::")
    assert is_structural_block("![](image.png)")
    assert not is_structural_block("## A heading")


def test_materialize_accepts_an_already_reconstructed_fallback() -> None:
    protected = {"ZXQPROTECTED0000QXZ": "[the article](https://example.com)"}
    translated = "Leia [o artigo](https://example.com)."
    assert materialize_translation(translated, protected) == translated


def test_source_frontmatter_adds_an_idempotent_reciprocal_pair() -> None:
    source = "---\ntitle: Example\nlang: en\n---"
    paired = source_frontmatter(source, "example-pt-BR.ipynb")
    assert paired.count("translation: example-pt-BR.ipynb") == 1
    assert "language-version:" not in paired
    assert source_frontmatter(paired, "example-pt-BR.ipynb") == paired


def test_published_notebook_image_paths_are_portable() -> None:
    posts = ROOT / "posts"
    for notebook_path in posts.glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown = "".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        assert "](" not in markdown or not any(
            "\\" in destination.split(")", 1)[0]
            for destination in markdown.split("](")[1:]
        ), f"{notebook_path.name} contains a Windows-style Markdown link"


def test_folded_notebook_code_matches_the_experiment_sources() -> None:
    notebook_path = ROOT / "posts" / "2026-07-23-Scaling-MyPost-WithAIAgents.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    expected_sources = {
        "alignment-code": (
            SRC / "translation_eval.py",
            {
                "Sentence",
                "AlignmentStep",
                "joined_text",
                "_transition_score",
                "monotonic_align",
            },
        ),
        "screening-code": (
            SRC / "screening.py",
            {"TOKEN_PATTERN", "RepetitionResult", "repetition_result"},
        ),
        "judge-prompts": (
            SRC / "prompts.py",
            {"MQM_SYSTEM_PROMPT", "PAIRWISE_SYSTEM_PROMPT"},
        ),
    }

    def named_nodes(code: str) -> dict[str, ast.AST]:
        result: dict[str, ast.AST] = {}
        for node in ast.parse(code).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                result[node.name] = node
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                result[node.targets[0].id] = node
        return result

    for cell_id, (source_path, symbol_names) in expected_sources.items():
        lines = [
            line
            for line in cells[cell_id]["source"]
            if not line.startswith("#|")
        ]
        embedded_code = "".join(lines).lstrip()
        source_code = source_path.read_text(encoding="utf-8")
        embedded_nodes = named_nodes(embedded_code)
        source_nodes = named_nodes(source_code)
        for symbol_name in symbol_names:
            assert symbol_name in embedded_nodes
            assert ast.dump(
                embedded_nodes[symbol_name],
                include_attributes=False,
            ) == ast.dump(
                source_nodes[symbol_name],
                include_attributes=False,
            ), (
                f"The folded {symbol_name!r} excerpt no longer matches "
                f"{source_path.name}"
            )


def test_published_notebooks_have_balanced_markdown_structure() -> None:
    source_path = ROOT / "posts" / "2026-07-23-Scaling-MyPost-WithAIAgents.ipynb"
    translation_path = (
        ROOT / "posts" / "2026-07-23-Scaling-MyPost-WithAIAgents-pt-BR.ipynb"
    )
    paths = [source_path]
    if translation_path.exists():
        paths.append(translation_path)

    notebooks = []
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        validate_notebook_structure(notebook)
        notebooks.append(notebook)
    if len(notebooks) == 2:
        validate_heading_parity(*notebooks)


def test_heading_parity_rejects_model_invented_sections() -> None:
    source = {
        "cells": [{"cell_type": "markdown", "source": ["# Source\n\nText."]}]
    }
    translated = {
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# Fonte\n\n## Seção inventada\n"],
            }
        ]
    }
    with pytest.raises(ValueError, match="changed heading structure"):
        validate_heading_parity(source, translated)


def test_multiline_hallucinated_link_label_is_rejected() -> None:
    class HallucinatingTranslator:
        def translate_batch(
            self,
            texts: list[str],
            *,
            preserve_placeholders: bool = True,
            short_link_label: bool = False,
            max_new_tokens: int | None = None,
        ) -> list[str]:
            if short_link_label:
                assert max_new_tokens == 32
                return ["# Segurança de dados\n\n## Componentes"]
            return texts

    source = "Read [English](https://example.com)."
    masked, protected = protect(source)
    result = translate_around_protected(
        HallucinatingTranslator(),
        masked,
        protected,
    )
    assert result == source
