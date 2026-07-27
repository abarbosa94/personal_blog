"""Generate a reviewable translated Quarto notebook with Tower+.

The command preserves notebook structure and protects Markdown constructs that a
translation model must not rewrite. The generated document remains a draft and is
marked as a machine translation until a human reviews it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MODEL = "Unbabel/Tower-Plus-2B"
REVISION = "4d779ca939174189c0677d4a75642d36d6a33b66"
PROTECTED_CACHE_REVISION = "translate-link-labels-v2"
PLACEHOLDER_PREFIX = "ZXQPROTECTED"
PROTECTED = re.compile(
    r"`[^`\n]+`"
    r"|\$\$[\s\S]*?\$\$"
    r"|\$[^$\n]+\$"
    r"|\[@[^\]]+\]"
    r"|!\[[^\]]*\]\([^)]+\)"
    r"|(?<!!)\[[^\]]+\]\([^)]+\)"
    r"|\]\([^)]+\)"
    r"|https?://[^\s)>]+"
)
CALLOUT_TITLE = re.compile(r'(title=")([^"]+)(")')
FENCE_LINE = re.compile(r"^\s*(`{3,}|~{3,})(?:markdown|md)?\s*$", re.IGNORECASE)


def split_markdown(text: str) -> list[str]:
    """Split at blank lines while retaining the original separators."""
    return [part for part in re.split(r"(\n[ \t]*\n)", text) if part]


def is_separator(part: str) -> bool:
    return bool(re.fullmatch(r"\n[ \t]*\n", part))


def is_structural_block(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return True
    if stripped == ":::":
        return True
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return True
    if stripped.startswith("$$") and stripped.endswith("$$"):
        return True
    if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", stripped):
        return True
    if all(re.fullmatch(r"[|:\- ]+", line) for line in stripped.splitlines()):
        return True
    return False


def validate_notebook_structure(notebook: dict[str, object]) -> None:
    """Reject translated Markdown with unbalanced code fences or Quarto divs."""
    div_depth = 0
    for cell_index, cell in enumerate(notebook["cells"]):  # type: ignore[index]
        if cell["cell_type"] != "markdown":  # type: ignore[index]
            continue
        active_fence: str | None = None
        for line in "".join(cell.get("source", [])).splitlines():  # type: ignore[union-attr]
            stripped = line.strip()
            fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
            if fence_match:
                marker = fence_match.group(1)[0]
                if active_fence is None:
                    active_fence = marker
                elif active_fence == marker:
                    active_fence = None
                continue
            if active_fence is not None:
                continue
            if re.match(r"^:{3,}\s+\{", stripped):
                div_depth += 1
            elif re.fullmatch(r":{3,}", stripped):
                div_depth -= 1
                if div_depth < 0:
                    raise ValueError(
                        f"Markdown cell {cell_index} closes an unopened Quarto div"
                    )
        if active_fence is not None:
            raise ValueError(
                f"Markdown cell {cell_index} contains an unclosed code fence"
            )
    if div_depth:
        raise ValueError(f"Notebook contains {div_depth} unclosed Quarto div(s)")


def validate_heading_parity(
    source_notebook: dict[str, object],
    translated_notebook: dict[str, object],
) -> None:
    """Require translated Markdown to retain the source heading hierarchy."""
    source_cells = source_notebook["cells"]  # type: ignore[index]
    translated_cells = translated_notebook["cells"]  # type: ignore[index]
    for cell_index, (source_cell, translated_cell) in enumerate(
        zip(source_cells, translated_cells, strict=True)
    ):
        if source_cell["cell_type"] != "markdown":  # type: ignore[index]
            continue
        source_levels = [
            len(match.group(1))
            for match in re.finditer(
                r"^(#{1,6})\s+",
                "".join(source_cell.get("source", [])),  # type: ignore[union-attr]
                flags=re.MULTILINE,
            )
        ]
        translated_levels = [
            len(match.group(1))
            for match in re.finditer(
                r"^(#{1,6})\s+",
                "".join(translated_cell.get("source", [])),  # type: ignore[union-attr]
                flags=re.MULTILINE,
            )
        ]
        if translated_levels != source_levels:
            raise ValueError(
                f"Markdown cell {cell_index} changed heading structure: "
                f"{source_levels} -> {translated_levels}"
            )


def protect(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        key = f"{PLACEHOLDER_PREFIX}{len(protected):04d}QXZ"
        protected[key] = match.group(0)
        return key

    return PROTECTED.sub(replace, text), protected


def restore(text: str, protected: dict[str, str]) -> str:
    for key, value in protected.items():
        if key not in text:
            raise ValueError(f"Tower+ dropped protected Markdown placeholder {key}")
        text = text.replace(key, value)
    if PLACEHOLDER_PREFIX in text:
        raise ValueError("Tower+ returned an unknown protected Markdown placeholder")
    return text


def materialize_translation(text: str, protected: dict[str, str]) -> str:
    """Restore a primary result or accept an already reconstructed fallback."""
    if not protected or PLACEHOLDER_PREFIX not in text:
        return text
    return restore(text, protected)


def remove_model_code_fence_wrapper(source: str, translation: str) -> str:
    """Remove only an outer Markdown fence invented by the translation model."""
    if re.search(r"^(`{3,}|~{3,})", source, flags=re.MULTILINE):
        return translation
    lines = translation.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if nonempty:
        first = nonempty[0]
        opening = FENCE_LINE.fullmatch(lines[first])
        if opening:
            del lines[first]
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            if nonempty:
                last = nonempty[-1]
                closing = FENCE_LINE.fullmatch(lines[last])
                if closing and closing.group(1)[0] == opening.group(1)[0]:
                    del lines[last]
    result = "\n".join(lines).strip()
    if re.search(r"^(`{3,}|~{3,})", result, flags=re.MULTILINE):
        raise ValueError("Tower+ invented a code fence inside translated prose")
    return result


def translation_cache_key(text: str, protected: dict[str, str]) -> str:
    identity = f"{MODEL}@{REVISION}"
    if protected:
        identity += f"@{PROTECTED_CACHE_REVISION}"
    return hashlib.sha256(f"{identity}\0{text}".encode("utf-8")).hexdigest()


def translate_around_protected(
    translator: "TowerMarkdownTranslator",
    masked: str,
    protected: dict[str, str],
) -> str:
    """Translate prose fragments without exposing protected tokens to the model."""
    if not protected:
        return translator.translate_batch([masked])[0]
    marker = re.compile(
        "(" + "|".join(re.escape(key) for key in protected) + ")"
    )
    parts = marker.split(masked)
    prose_indexes = [
        index
        for index, part in enumerate(parts)
        if part and part not in protected
    ]
    link_labels: list[tuple[str, str, str]] = []
    for key, value in protected.items():
        match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", value)
        if match and "`" not in match.group(1):
            link_labels.append((key, match.group(1), match.group(2)))

    prose_inputs = [parts[index] for index in prose_indexes]
    prose_translations = (
        translator.translate_batch(
            prose_inputs,
            preserve_placeholders=False,
        )
        if prose_inputs
        else []
    )
    label_translations = (
        translator.translate_batch(
            [label for _, label, _ in link_labels],
            preserve_placeholders=False,
            short_link_label=True,
            max_new_tokens=32,
        )
        if link_labels
        else []
    )
    for index, value in zip(prose_indexes, prose_translations, strict=True):
        parts[index] = value
    restored_values = dict(protected)
    for (key, source_label, destination), label in zip(
        link_labels,
        label_translations,
        strict=True,
    ):
        if (
            "\n" in label
            or "[" in label
            or "]" in label
            or re.search(r"(^|\s)#{1,6}\s", label)
            or len(label) > max(80, len(source_label) * 4)
        ):
            label = source_label
        restored_values[key] = f"[{label}]({destination})"
    return restore("".join(parts), restored_values)


def portuguese_frontmatter(source: str, english_name: str) -> str:
    lines = source.splitlines()
    replacements = {
        "description": (
            "Como usei agentes de IA, LLMs e GitHub Actions para traduzir "
            "automaticamente meus posts em notebooks Jupyter"
        ),
        "draft": "true",
        "lang": "pt-BR",
        "title": "Tornando Meus Pensamentos Bilíngues com IA Agêntica",
    }
    result: list[str] = []
    inserted_pairing = False
    for line in lines:
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key in {"translation", "language-version"}:
            continue
        if key in replacements:
            value = replacements[key]
            quote = "'" if key == "title" else ""
            result.append(f"{key}: {quote}{value}{quote}")
            if key == "lang":
                result.append(f"translation: {english_name}")
                result.append("language-version: translation")
                inserted_pairing = True
        else:
            result.append(line)
    if not inserted_pairing:
        raise ValueError("Notebook front matter must contain a lang field")
    return "\n".join(result)


def source_frontmatter(source: str, translation_name: str) -> str:
    """Add or update the reciprocal translation link in source front matter."""
    lines = source.splitlines()
    result: list[str] = []
    inserted_pairing = False
    found_lang = False
    for line in lines:
        key = line.split(":", 1)[0].strip() if ":" in line else ""
        if key == "language-version":
            continue
        if key == "translation":
            if not inserted_pairing:
                result.append(f"translation: {translation_name}")
                inserted_pairing = True
            continue
        result.append(line)
        if key == "lang":
            found_lang = True
            if not inserted_pairing:
                result.append(f"translation: {translation_name}")
                inserted_pairing = True
    if not found_lang:
        raise ValueError("Notebook front matter must contain a lang field")
    return "\n".join(result)


def pair_source_notebook(source: Path, translation_name: str) -> None:
    """Persist the English side of a reciprocal notebook translation pair."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    first_cell = notebook["cells"][0]
    raw = "".join(first_cell["source"])
    first_cell["source"] = source_frontmatter(
        raw, translation_name
    ).splitlines(keepends=True)
    source.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


class TowerMarkdownTranslator:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            torch_dtype=torch.bfloat16,
        ).to("cpu")
        self.model.eval()

    def translate_batch(
        self,
        texts: list[str],
        *,
        preserve_placeholders: bool = True,
        short_link_label: bool = False,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if short_link_label:
            instruction = (
                "Translate this short Markdown link label from English to "
                "Brazilian Portuguese. Do not expand, explain, or add information. "
                "Return exactly one short line containing only the translated label."
            )
        else:
            instruction = (
                "Translate the following technical blog Markdown from English to "
                "Brazilian Portuguese. Preserve every Markdown marker, line break, "
                "table column, and Quarto directive exactly. Do not wrap the "
                "translation in a Markdown code fence. "
            )
            if preserve_placeholders:
                instruction += (
                    f"Copy every token beginning with {PLACEHOLDER_PREFIX} exactly. "
                )
            instruction += "Return only the translated Markdown."
        prompts = [
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": f"{instruction}\n\nEnglish Markdown:\n{text}",
                        }
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for text in texts
            ]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to("cpu")
        input_length = encoded["input_ids"].shape[1]
        source_lengths = self.tokenizer(
            texts, add_special_tokens=False, truncation=True, max_length=1024
        )["input_ids"]
        max_source = max(len(tokens) for tokens in source_lengths)
        # A translation should be close to the source length. The cap prevents a
        # missing EOS token from turning one Markdown block into an hour-long run.
        generation_limit = max_new_tokens or min(
            512,
            max(64, int(max_source * 1.5) + 48),
        )
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=generation_limit,
                no_repeat_ngram_size=8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return [
            item.strip()
            for item in self.tokenizer.batch_decode(
                generated[:, input_length:], skip_special_tokens=True
            )
        ]


def translate_notebook(
    source: Path,
    output: Path,
    *,
    pair_source: bool = False,
) -> None:
    source_text = source.read_text(encoding="utf-8")
    source_notebook = json.loads(source_text)
    notebook = json.loads(source_text)
    tasks: list[tuple[int, int, dict[str, str]]] = []
    cell_parts: dict[int, list[str]] = {}

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        parts = split_markdown("".join(cell.get("source", [])))
        cell_parts[cell_index] = parts
        for part_index, part in enumerate(parts):
            if is_separator(part) or is_structural_block(part):
                continue
            masked, protected = protect(part)
            tasks.append((cell_index, part_index, protected))
            parts[part_index] = masked

    masked_texts = [cell_parts[cell][part] for cell, part, _ in tasks]
    progress_path = output.with_suffix(output.suffix + ".tower-plus-progress.json")
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    translations: list[str | None] = [None] * len(tasks)
    pending: list[int] = []
    for index, text in enumerate(masked_texts):
        key = translation_cache_key(text, tasks[index][2])
        cached = progress.get(key)
        if isinstance(cached, str):
            translations[index] = cached
        else:
            pending.append(index)

    if pending:
        translator = TowerMarkdownTranslator()
        batch_size = 4
        for start in range(0, len(pending), batch_size):
            indexes = pending[start : start + batch_size]
            batch = translator.translate_batch([masked_texts[index] for index in indexes])
            for index, translation in zip(indexes, batch, strict=True):
                # Batched decoding occasionally drops a placeholder. Retry only
                # that block in isolation before accepting or rejecting it.
                try:
                    restore(translation, tasks[index][2])
                except ValueError:
                    print(
                        f"Reconstructing block {index + 1} after placeholder loss",
                        flush=True,
                    )
                    translation = translate_around_protected(
                        translator,
                        masked_texts[index],
                        tasks[index][2],
                    )
                translations[index] = translation
                key = translation_cache_key(
                    masked_texts[index],
                    tasks[index][2],
                )
                progress[key] = translation
                progress_path.write_text(
                    json.dumps(progress, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8",
                )
            completed = len(tasks) - len(pending) + min(
                start + batch_size, len(pending)
            )
            print(f"Translated {completed}/{len(tasks)} blocks", flush=True)

    if any(translation is None for translation in translations):
        raise RuntimeError("Translation checkpoint is incomplete")
    for (cell_index, part_index, protected), translation in zip(
        tasks, translations, strict=True
    ):
        assert translation is not None
        source_part = cell_parts[cell_index][part_index]
        cell_parts[cell_index][part_index] = remove_model_code_fence_wrapper(
            source_part,
            materialize_translation(
                translation,
                protected,
            ),
        )

    for cell_index, parts in cell_parts.items():
        notebook["cells"][cell_index]["source"] = "".join(parts).splitlines(
            keepends=True
        )

    raw = "".join(notebook["cells"][0]["source"])
    notebook["cells"][0]["source"] = portuguese_frontmatter(
        raw, source.name
    ).splitlines(keepends=True)

    notice = (
        '::: {.callout-warning title="Rascunho de tradução gerado por máquina"}\n'
        "Esta versão foi gerada localmente com o Tower+ 2B e ainda precisa de "
        "revisão humana antes da publicação.\n"
        ":::\n\n"
    )
    first_markdown = next(
        cell for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    first_markdown["source"] = (
        notice + "".join(first_markdown["source"])
    ).splitlines(keepends=True)

    validate_notebook_structure(notebook)
    validate_heading_parity(source_notebook, notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    if pair_source:
        pair_source_notebook(source, output.name)
    progress_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pair-source",
        action="store_true",
        help="Add the reciprocal translation metadata to the source notebook.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    translate_notebook(args.source, args.output, pair_source=args.pair_source)
