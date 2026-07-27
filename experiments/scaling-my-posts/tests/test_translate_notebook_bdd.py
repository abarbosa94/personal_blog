from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "scaling-my-posts" / "src"
sys.path.insert(0, str(SRC))

from translate_notebook import (  # noqa: E402
    portuguese_frontmatter,
    protect,
    restore,
    translate_around_protected,
)


scenarios(
    str(Path(__file__).resolve().parent / "features" / "translate_notebook.feature")
)


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@given(
    "a source paragraph contains a Markdown link, inline code, math, and a citation"
)
def english_markdown(context: dict[str, Any]) -> None:
    context["source"] = (
        "Read [`code`](https://example.com), evaluate $x + y$, "
        "and compare the result [@paper]."
    )


@when("the generator replaces every protected construct with an internal token")
def protect_markdown(context: dict[str, Any]) -> None:
    masked, protected = protect(context["source"])
    context["masked"] = masked
    context["protected"] = protected


@when("the generator restores the original constructs without calling Tower+")
def restore_markdown(context: dict[str, Any]) -> None:
    context["result"] = restore(context["masked"], context["protected"])


@then("the complete restored paragraph exactly matches the source paragraph")
def restored_paragraph_matches_source(context: dict[str, Any]) -> None:
    assert context["result"] == context["source"]


@then("the link, code, math, and citation are preserved byte for byte")
def protected_constructs_are_unchanged(context: dict[str, Any]) -> None:
    assert set(context["protected"].values()) == {
        "[`code`](https://example.com)",
        "$x + y$",
        "[@paper]",
    }


@given(
    parsers.parse(
        'an English paragraph says "{english_word}" immediately before '
        "a protected Markdown link"
    )
)
def protected_paragraph(
    context: dict[str, Any],
    english_word: str,
) -> None:
    context["english_word"] = english_word
    context["source"] = f"{english_word} [the article](https://example.com)."
    context["masked"], context["protected"] = protect(context["source"])


@given("Tower+ cannot be trusted to copy an internal placeholder")
def model_may_drop_placeholder(context: dict[str, Any]) -> None:
    context["placeholder_loss_expected"] = True


@when("the fallback separates translatable prose from protected Markdown")
def prepare_fallback_translator(context: dict[str, Any]) -> None:
    class RecordingTranslator:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], bool]] = []

        def translate_batch(
            self,
            texts: list[str],
            *,
            preserve_placeholders: bool = True,
            short_link_label: bool = False,
            max_new_tokens: int | None = None,
        ) -> list[str]:
            self.calls.append((texts, preserve_placeholders))
            return [
                text.replace(
                    context["english_word"],
                    context["portuguese_word"],
                ).replace("the article", "o artigo")
                for text in texts
            ]

    context["translator"] = RecordingTranslator()


@when(parsers.parse('the fallback translates "{english_word}" to "{portuguese_word}"'))
def translate_prose_around_markdown(
    context: dict[str, Any],
    english_word: str,
    portuguese_word: str,
) -> None:
    assert context["placeholder_loss_expected"] is True
    assert context["english_word"] == english_word
    context["portuguese_word"] = portuguese_word
    context["result"] = translate_around_protected(
        context["translator"],
        context["masked"],
        context["protected"],
    )
    context["calls"] = context["translator"].calls


@then("no internal placeholder is sent to Tower+ during the fallback")
def fallback_model_does_not_see_tokens(context: dict[str, Any]) -> None:
    assert context["calls"]
    for texts, preserve_placeholders in context["calls"]:
        assert preserve_placeholders is False
        assert all("ZXQPROTECTED" not in text for text in texts)


@then(parsers.parse('the result is "{expected}"'))
def translated_paragraph_retains_link(
    context: dict[str, Any],
    expected: str,
) -> None:
    assert context["result"] == expected


@given(parsers.parse('an English paragraph contains the short link label "{label}"'))
def paragraph_with_short_link_label(
    context: dict[str, Any],
    label: str,
) -> None:
    context["label"] = label
    context["source"] = f"Read [{label}](https://example.com)."
    context["masked"], context["protected"] = protect(context["source"])


@given("Tower+ expands that label into a multiline article with Markdown headings")
def hallucinating_link_translator(context: dict[str, Any]) -> None:
    context["hallucination"] = (
        "# Data Security\n\n## Core Components\n\nUnrelated generated content."
    )

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
                return [context["hallucination"] for _ in texts]
            return texts

    context["translator"] = HallucinatingTranslator()


@when("the fallback validates the translated link label")
def validate_hallucinated_link_label(context: dict[str, Any]) -> None:
    context["result"] = translate_around_protected(
        context["translator"],
        context["masked"],
        context["protected"],
    )


@then("the unsafe label translation is discarded")
def unsafe_link_label_is_discarded(context: dict[str, Any]) -> None:
    assert context["hallucination"] not in context["result"]


@then(parsers.parse('the original link "{expected_link}" is retained'))
def original_link_is_retained(
    context: dict[str, Any],
    expected_link: str,
) -> None:
    assert expected_link in context["result"]


@then("no new Markdown heading is introduced")
def no_heading_is_introduced(context: dict[str, Any]) -> None:
    assert "\n#" not in context["result"]


@given("source metadata describes a published English notebook")
def english_frontmatter(context: dict[str, Any]) -> None:
    context["source"] = "---\nlang: en\ntitle: 'English'\ndraft: false\n---"


@when(
    parsers.parse(
        'Portuguese metadata is generated for the source file "{source_name}"'
    )
)
def generate_portuguese_frontmatter(
    context: dict[str, Any],
    source_name: str,
) -> None:
    context["source_name"] = source_name
    context["result"] = portuguese_frontmatter(context["source"], source_name)


@then("the generated language is Brazilian Portuguese")
def language_is_portuguese(context: dict[str, Any]) -> None:
    assert "lang: pt-BR" in context["result"]


@then(parsers.parse('its translation link points back to "{source_name}"'))
def document_links_to_source(
    context: dict[str, Any],
    source_name: str,
) -> None:
    assert context["source_name"] == source_name
    assert f"translation: {source_name}" in context["result"]


@then("it is marked as the secondary translation")
def document_is_secondary_translation(context: dict[str, Any]) -> None:
    assert "language-version: translation" in context["result"]


@then("it remains a draft until an author reviews it")
def generated_document_is_draft(context: dict[str, Any]) -> None:
    assert "draft: true" in context["result"]
