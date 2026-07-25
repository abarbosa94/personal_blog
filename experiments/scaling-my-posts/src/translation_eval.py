"""Sentence-level EN <-> PT-BR translation evaluation for the blog.

The CLI deliberately separates the reproducible, offline stages from the two
credentialed evaluators:

    align -> review CSV -> freeze -> predict -> judge/xcomet -> aggregate

Run ``python experiments/scaling-my-posts/src/translation_eval.py --help`` from
the repository root for the complete command line interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from judge_factory import (
    DEFAULT_JUDGE_PROVIDER,
    available_judge_providers,
    create_judge_adapter,
    resolve_judge_configuration,
)
from judge_interface import JudgeConfiguration
from prompts import MQM_SYSTEM_PROMPT, PAIRWISE_SYSTEM_PROMPT, PROMPT_VERSION

HISTORICAL_COMMIT = "e78386012f37512a5ebd316a1389fabf9bf3b707"
ENGLISH_NOTEBOOK = "_notebooks/2020-09-19-Distilling-BERT.ipynb"
PORTUGUESE_NOTEBOOK = "_notebooks/2020-09-19-Distilling-BERT-pt.ipynb"
ALIGNED_CELL_PAIRS = [
    (3, 3),
    (8, 8),
    (9, 9),
    (21, 21),
    (24, 24),
    (25, 25),
    (27, 27),
    (32, 32),
    (34, 34),
    (37, 37),
    (45, 46),
    (48, 49),
    (50, 51),
]
MODEL_KEYS = ("marian", "nllb", "tower")
MODEL_DISPLAY_NAMES = {
    "marian": "Marian OPUS-MT",
    "nllb": "NLLB-200 distilled 600M",
    "tower": "Tower+ 2B",
}
DEFAULT_JUDGE_MODEL_KEYS = ("marian", "tower")
LABSE_MODEL = "sentence-transformers/LaBSE"
LABSE_REVISION = "fa02c71f7e1d1f5a02b0d1a31cada51f564c7198"
XCOMET_MODEL = "Unbabel/XCOMET-XL"
RANDOM_SEED = 42

MQM_CATEGORIES = (
    "accuracy",
    "omission",
    "addition",
    "fluency",
    "terminology",
    "locale",
    "style",
    "formatting",
)
SEVERITY_WEIGHTS = {"minor": 1, "major": 5, "critical": 10}
TERMINAL_REVIEW_STATUSES = {"accept", "exclude", "localized"}


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    text: str


@dataclass(frozen=True)
class AlignmentStep:
    english_start: int
    english_count: int
    portuguese_start: int
    portuguese_count: int
    similarity: float | None
    score: float


class MQMError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span: str = Field(
        description="An exact substring of the candidate, or an empty string for omissions."
    )
    category: Literal[
        "accuracy",
        "omission",
        "addition",
        "fluency",
        "terminology",
        "locale",
        "style",
        "formatting",
    ]
    severity: Literal["minor", "major", "critical"]
    explanation: str


class MQMJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    errors: list[MQMError]
    summary: str


class PairwiseJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["A", "B", "tie"]
    decisive_categories: list[
        Literal[
            "accuracy",
            "omission",
            "addition",
            "fluency",
            "terminology",
            "locale",
            "style",
            "formatting",
        ]
    ]
    explanation: str


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_json(repo: Path, commit: str, path: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        text=True,
        encoding="utf-8",
    )
    return json.loads(raw)


def _overlap_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    import sacrebleu

    return {
        "bleu": sacrebleu.corpus_bleu(
            predictions, [references], tokenize="flores200"
        ).score,
        "chrf": sacrebleu.corpus_chrf(
            predictions, [references], word_order=2
        ).score,
        "ter": sacrebleu.corpus_ter(predictions, [references]).score,
    }


def _strip_legacy_markdown(value: str) -> str:
    value = re.sub(r"{%\s*fn\s+\d+\s*%}", "", value)
    value = re.sub(
        r"{{\s*['\"]?(.*?)['\"]?\s*\|\s*fndetail:\s*\d+\s*}}",
        r"\1",
        value,
        flags=re.DOTALL,
    )
    value = re.sub(r"!\[[^]]*]\([^\n]*\)", "", value)
    value = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def markdown_prose_blocks(value: str) -> list[str]:
    """Return prose blocks while keeping headings separate and dropping code/images."""
    blocks: list[str] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        if not paragraph:
            return
        cleaned = _strip_legacy_markdown(" ".join(paragraph))
        if cleaned:
            blocks.append(cleaned)
        paragraph.clear()

    for raw_line in value.splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line:
            flush()
            continue
        if re.fullmatch(r"!\[[^]]*]\([^)]*\)", line):
            flush()
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            flush()
            cleaned = _strip_legacy_markdown(heading.group(1))
            if cleaned:
                blocks.append(cleaned)
            continue
        line = re.sub(r"^>\s?(?:\w+:\s*)?", "", line)
        line = re.sub(r"^(?:[-*+] |\d+[.)]\s+)", "", line)
        paragraph.append(line)
    flush()
    return blocks


_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Prof.",
    "Fig.",
    "Eq.",
    "ex.",
    "p.ex.",
)


def split_sentences(block: str) -> list[str]:
    """A deterministic bilingual splitter suited to this small technical corpus."""
    protected = block
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<DOT>"))
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", protected)
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'“”‘’(]*[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ])", protected)
    return [piece.replace("<DOT>", ".").strip() for piece in pieces if piece.strip()]


def segment_markdown(value: str, language_prefix: str) -> list[Sentence]:
    sentences: list[str] = []
    for block in markdown_prose_blocks(value):
        sentences.extend(split_sentences(block))
    return [
        Sentence(f"{language_prefix}{index:02d}", sentence)
        for index, sentence in enumerate(sentences, start=1)
    ]


def joined_text(sentences: list[Sentence], start: int, count: int) -> str:
    return " ".join(sentence.text for sentence in sentences[start : start + count])


def _transition_score(
    english_text: str,
    portuguese_text: str,
    english_count: int,
    portuguese_count: int,
    similarity: float,
) -> float:
    merge_penalty = 0.08 * ((english_count - 1) + (portuguese_count - 1))
    length_ratio = (len(english_text) + 1) / (len(portuguese_text) + 1)
    length_penalty = 0.08 * abs(math.log(length_ratio))
    return similarity - merge_penalty - length_penalty


def monotonic_align(
    english: list[Sentence],
    portuguese: list[Sentence],
    similarity_fn: Callable[[str, str], float],
    max_group: int = 3,
    gap_penalty: float = -0.45,
) -> list[AlignmentStep]:
    """Find the maximum-scoring ordered, non-overlapping many-to-many alignment."""
    n, m = len(english), len(portuguese)
    scores = np.full((n + 1, m + 1), -np.inf, dtype=float)
    previous: dict[tuple[int, int], tuple[int, int, AlignmentStep]] = {}
    scores[0, 0] = 0.0
    transitions = [
        (a, b)
        for a in range(1, max_group + 1)
        for b in range(1, max_group + 1)
        if a == 1 or b == 1 or (a, b) == (2, 2)
    ] + [(1, 0), (0, 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(scores[i, j]):
                continue
            for english_count, portuguese_count in transitions:
                ni, nj = i + english_count, j + portuguese_count
                if ni > n or nj > m:
                    continue
                similarity: float | None
                if english_count == 0 or portuguese_count == 0:
                    similarity = None
                    step_score = gap_penalty
                else:
                    english_text = joined_text(english, i, english_count)
                    portuguese_text = joined_text(portuguese, j, portuguese_count)
                    similarity = float(similarity_fn(english_text, portuguese_text))
                    step_score = _transition_score(
                        english_text,
                        portuguese_text,
                        english_count,
                        portuguese_count,
                        similarity,
                    )
                candidate = scores[i, j] + step_score
                if candidate > scores[ni, nj]:
                    scores[ni, nj] = candidate
                    step = AlignmentStep(
                        i,
                        english_count,
                        j,
                        portuguese_count,
                        similarity,
                        step_score,
                    )
                    previous[(ni, nj)] = (i, j, step)

    if (n, m) not in previous and (n, m) != (0, 0):
        raise RuntimeError("No complete alignment path was found")
    steps: list[AlignmentStep] = []
    position = (n, m)
    while position != (0, 0):
        prior_i, prior_j, step = previous[position]
        steps.append(step)
        position = (prior_i, prior_j)
    return list(reversed(steps))


class LabseSimilarity:
    """Lazy, revision-pinned LaBSE cosine similarity with per-run caching."""

    def __init__(self, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SystemExit(
                "The align command requires sentence-transformers; install the benchmark requirements."
            ) from error
        self.model = SentenceTransformer(
            LABSE_MODEL, revision=LABSE_REVISION, device=device
        )
        self.cache: dict[str, np.ndarray] = {}

    def prepare(self, texts: Iterable[str]) -> None:
        missing = sorted(set(texts) - self.cache.keys())
        if not missing:
            return
        vectors = self.model.encode(
            missing,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.cache.update(zip(missing, vectors, strict=True))

    def __call__(self, left: str, right: str) -> float:
        self.prepare([left, right])
        return float(np.dot(self.cache[left], self.cache[right]))


def _all_windows(sentences: list[Sentence], max_group: int = 3) -> list[str]:
    return [
        joined_text(sentences, start, count)
        for start in range(len(sentences))
        for count in range(1, max_group + 1)
        if start + count <= len(sentences)
    ]


def generate_alignment_candidates(
    repo: Path, similarity: LabseSimilarity
) -> pd.DataFrame:
    english_notebook = _git_json(repo, HISTORICAL_COMMIT, ENGLISH_NOTEBOOK)
    portuguese_notebook = _git_json(repo, HISTORICAL_COMMIT, PORTUGUESE_NOTEBOOK)
    rows: list[dict[str, Any]] = []

    for pair_id, (english_cell, portuguese_cell) in enumerate(
        ALIGNED_CELL_PAIRS, start=1
    ):
        english_source = "".join(english_notebook["cells"][english_cell]["source"])
        portuguese_source = "".join(
            portuguese_notebook["cells"][portuguese_cell]["source"]
        )
        english = segment_markdown(english_source, f"p{pair_id:02d}-en")
        portuguese = segment_markdown(portuguese_source, f"p{pair_id:02d}-pt")
        similarity.prepare(_all_windows(english) + _all_windows(portuguese))
        steps = monotonic_align(english, portuguese, similarity)

        for step_index, step in enumerate(steps, start=1):
            english_slice = english[
                step.english_start : step.english_start + step.english_count
            ]
            portuguese_slice = portuguese[
                step.portuguese_start : step.portuguese_start
                + step.portuguese_count
            ]
            warnings: list[str] = []
            if step.english_count == 0 or step.portuguese_count == 0:
                warnings.append("unmatched sentence")
            if step.english_count != 1 or step.portuguese_count != 1:
                warnings.append("many-to-many proposal")
            if step.similarity is not None and step.similarity < 0.65:
                warnings.append("low semantic similarity")
            rows.append(
                {
                    "alignment_id": f"p{pair_id:02d}-a{step_index:02d}",
                    "pair_id": pair_id,
                    "english_cell": english_cell,
                    "portuguese_cell": portuguese_cell,
                    "english_sentence_ids": "|".join(
                        sentence.sentence_id for sentence in english_slice
                    ),
                    "portuguese_sentence_ids": "|".join(
                        sentence.sentence_id for sentence in portuguese_slice
                    ),
                    "english": " ".join(sentence.text for sentence in english_slice),
                    "portuguese": " ".join(
                        sentence.text for sentence in portuguese_slice
                    ),
                    "alignment_type": f"{step.english_count}:{step.portuguese_count}",
                    "labse_similarity": step.similarity,
                    "transition_score": step.score,
                    "review_priority": "high" if warnings else "normal",
                    "automatic_warning": "; ".join(warnings),
                    "review_status": "needs_review",
                    "reviewed_english": "",
                    "reviewed_portuguese": "",
                    "review_note": "",
                }
            )
    return pd.DataFrame(rows)


def freeze_reviewed_alignments(review: pd.DataFrame) -> pd.DataFrame:
    required = {
        "alignment_id",
        "pair_id",
        "english_sentence_ids",
        "portuguese_sentence_ids",
        "english",
        "portuguese",
        "review_status",
        "reviewed_english",
        "reviewed_portuguese",
        "review_note",
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Alignment review is missing columns: {sorted(missing)}")
    statuses = set(review["review_status"].fillna(""))
    invalid = statuses - TERMINAL_REVIEW_STATUSES
    if invalid:
        raise ValueError(
            "Every alignment needs a terminal review_status; unresolved values: "
            + ", ".join(sorted(invalid))
        )
    accepted = review.loc[review["review_status"] == "accept"].copy()
    if accepted.empty:
        raise ValueError("The review contains no accepted alignments")

    for column in ("english_sentence_ids", "portuguese_sentence_ids"):
        used: set[str] = set()
        for identifiers in accepted[column].fillna(""):
            for identifier in filter(None, str(identifiers).split("|")):
                if identifier in used:
                    raise ValueError(f"Accepted sentence is reused: {identifier}")
                used.add(identifier)

    def reviewed_or_original(row: pd.Series, language: str) -> str:
        reviewed = row[f"reviewed_{language}"]
        if pd.notna(reviewed) and str(reviewed).strip():
            return str(reviewed).strip()
        return str(row[language]).strip()

    return pd.DataFrame(
        {
            "segment_id": accepted["alignment_id"],
            "pair_id": accepted["pair_id"].astype(int),
            "english_sentence_ids": accepted["english_sentence_ids"],
            "portuguese_sentence_ids": accepted["portuguese_sentence_ids"],
            "english": accepted.apply(
                lambda row: reviewed_or_original(row, "english"), axis=1
            ),
            "portuguese": accepted.apply(
                lambda row: reviewed_or_original(row, "portuguese"), axis=1
            ),
            "reference_was_edited": accepted.apply(
                lambda row: bool(
                    (pd.notna(row["reviewed_english"]) and str(row["reviewed_english"]).strip())
                    or (
                        pd.notna(row["reviewed_portuguese"])
                        and str(row["reviewed_portuguese"]).strip()
                    )
                ),
                axis=1,
            ),
            "review_note": accepted["review_note"].fillna(""),
        }
    ).sort_values(["pair_id", "segment_id"])


def predict_segments(
    segments: pd.DataFrame,
    model_keys: list[str],
    device: str,
    predictions_checkpoint: Path | None = None,
    runtime_checkpoint: Path | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from translation_benchmark import MODEL_SPECS

    if (predictions_checkpoint is None) != (runtime_checkpoint is None):
        raise ValueError("Both prediction and runtime checkpoints must be supplied")

    prediction_columns = [
        "segment_id",
        "pair_id",
        "model",
        "direction",
        "source",
        "reference",
        "prediction",
    ]
    runtime_columns = [
        "model",
        "direction",
        "segments",
        "parameters_millions",
        "seconds",
        "segments_per_second",
        "bleu",
        "chrf",
        "ter",
    ]

    def load_checkpoint(path: Path | None, columns: list[str]) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame(columns=columns)
        frame = pd.read_csv(path, keep_default_na=False)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Checkpoint {path} is missing columns: {sorted(missing)}")
        return frame

    prediction_frame = load_checkpoint(predictions_checkpoint, prediction_columns)
    runtime_frame = load_checkpoint(runtime_checkpoint, runtime_columns)
    directions = [
        ("en", "pt-BR", "english", "portuguese"),
        ("pt-BR", "en", "portuguese", "english"),
    ]

    def reusable_task(
        model_name: str,
        direction: str,
        sources: list[str],
        references: list[str],
    ) -> bool:
        predictions = prediction_frame.loc[
            (prediction_frame["model"] == model_name)
            & (prediction_frame["direction"] == direction)
        ]
        runtime = runtime_frame.loc[
            (runtime_frame["model"] == model_name)
            & (runtime_frame["direction"] == direction)
        ]
        if len(predictions) != len(segments) or len(runtime) != 1:
            return False
        if predictions["segment_id"].duplicated().any():
            return False
        actual = predictions.set_index("segment_id")
        expected_ids = [str(value) for value in segments["segment_id"]]
        if set(actual.index.astype(str)) != set(expected_ids):
            return False
        for segment_id, source, reference in zip(
            expected_ids, sources, references, strict=True
        ):
            row = actual.loc[segment_id]
            if str(row["source"]) != str(source) or str(row["reference"]) != str(reference):
                return False
            if not str(row["prediction"]).strip():
                return False
        return True

    def drop_task(model_name: str, direction: str) -> None:
        nonlocal prediction_frame, runtime_frame
        prediction_frame = prediction_frame.loc[
            ~(
                (prediction_frame["model"] == model_name)
                & (prediction_frame["direction"] == direction)
            )
        ].copy()
        runtime_frame = runtime_frame.loc[
            ~(
                (runtime_frame["model"] == model_name)
                & (runtime_frame["direction"] == direction)
            )
        ].copy()

    for model_key in model_keys:
        specification = MODEL_SPECS[model_key]
        pending = []
        for source_lang, target_lang, source_column, reference_column in directions:
            direction = f"{source_lang} -> {target_lang}"
            sources = segments[source_column].astype(str).tolist()
            references = segments[reference_column].astype(str).tolist()
            if reusable_task(specification.name, direction, sources, references):
                if show_progress:
                    print(f"resume: {specification.name} | {direction} already complete")
                continue
            drop_task(specification.name, direction)
            pending.append(
                (source_lang, target_lang, source_column, reference_column)
            )
        if not pending:
            continue

        translator = specification.loader(device)
        try:
            for source_lang, target_lang, source_column, reference_column in pending:
                direction = f"{source_lang} -> {target_lang}"
                sources = segments[source_column].astype(str).tolist()
                references = segments[reference_column].astype(str).tolist()
                progress_bar = None
                if show_progress:
                    from tqdm.auto import tqdm

                    progress_bar = tqdm(
                        total=len(sources),
                        desc=f"{specification.name} | {direction}",
                        unit="segment",
                        dynamic_ncols=True,
                    )
                started = time.perf_counter()
                try:
                    predictions = translator.translate(
                        sources,
                        source_lang,
                        target_lang,
                        progress_bar.update if progress_bar is not None else None,
                    )
                finally:
                    elapsed = time.perf_counter() - started
                    if progress_bar is not None:
                        progress_bar.close()
                if len(predictions) != len(sources):
                    raise ValueError(
                        f"{specification.name} returned {len(predictions)} translations "
                        f"for {len(sources)} sources in {direction}"
                    )
                if any(not str(prediction).strip() for prediction in predictions):
                    raise ValueError(
                        f"{specification.name} returned an empty translation in {direction}"
                    )
                scores = _overlap_scores(predictions, references)
                summary = {
                    "model": specification.name,
                    "direction": direction,
                    "segments": len(sources),
                    "parameters_millions": translator.parameter_count / 1_000_000,
                    "seconds": elapsed,
                    "segments_per_second": len(sources) / elapsed,
                    **scores,
                }
                task_rows = []
                for segment, source, reference, prediction in zip(
                    segments.to_dict("records"),
                    sources,
                    references,
                    predictions,
                    strict=True,
                ):
                    task_rows.append(
                        {
                            "segment_id": segment["segment_id"],
                            "pair_id": segment["pair_id"],
                            "model": specification.name,
                            "direction": direction,
                            "source": source,
                            "reference": reference,
                            "prediction": prediction,
                        }
                    )
                task_frame = pd.DataFrame(task_rows)
                summary_frame = pd.DataFrame([summary])
                prediction_frame = (
                    task_frame
                    if prediction_frame.empty
                    else pd.concat([prediction_frame, task_frame], ignore_index=True)
                )
                runtime_frame = (
                    summary_frame
                    if runtime_frame.empty
                    else pd.concat([runtime_frame, summary_frame], ignore_index=True)
                )
                prediction_frame = prediction_frame.sort_values(
                    ["model", "direction", "segment_id"]
                ).reset_index(drop=True)
                runtime_frame = runtime_frame.sort_values(
                    ["model", "direction"]
                ).reset_index(drop=True)
                if predictions_checkpoint is not None and runtime_checkpoint is not None:
                    _atomic_write_csv(prediction_frame, predictions_checkpoint)
                    _atomic_write_csv(runtime_frame, runtime_checkpoint)
        finally:
            translator.close()
            del translator
    return runtime_frame, prediction_frame


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe without exposing a partially written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _judge_payload(
    direction: str, source: str, reference: str, **candidates: str
) -> dict[str, str]:
    target_locale = "Brazilian Portuguese" if direction.endswith("pt-BR") else "English"
    return {
        "translation_direction": direction,
        "target_locale": target_locale,
        "source": source,
        "human_reference": reference,
        **candidates,
    }


def filter_prediction_models(
    predictions: pd.DataFrame, model_keys: Iterable[str]
) -> pd.DataFrame:
    """Select benchmark model display names from stable command-line keys."""
    if "model" not in predictions.columns:
        raise ValueError("Predictions are missing the model column")
    keys = list(dict.fromkeys(model_keys))
    unknown = set(keys) - set(MODEL_DISPLAY_NAMES)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    selected_names = [MODEL_DISPLAY_NAMES[key] for key in keys]
    available = set(predictions["model"].astype(str))
    missing = set(selected_names) - available
    if missing:
        raise ValueError(f"Predictions are missing requested models: {sorted(missing)}")
    return predictions.loc[predictions["model"].isin(selected_names)].copy()


def default_judgments_path(provider: str) -> Path:
    if provider not in available_judge_providers():
        raise ValueError(f"Unknown judge provider: {provider!r}")
    return Path(f"posts/data/translation-eval-{provider}-judgments.jsonl")


def build_judge_jobs(
    predictions: pd.DataFrame,
    mode: Literal["mqm", "pairwise", "all"] = "all",
    judge_configuration: JudgeConfiguration | None = None,
) -> list[dict[str, Any]]:
    required = {
        "segment_id",
        "model",
        "direction",
        "source",
        "reference",
        "prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    configuration = judge_configuration or resolve_judge_configuration()
    request_settings = configuration.request_settings()
    jobs: list[dict[str, Any]] = []
    ordered = predictions.sort_values(["direction", "segment_id", "model"])

    if mode in {"mqm", "all"}:
        for row in ordered.to_dict("records"):
            payload = _judge_payload(
                row["direction"],
                row["source"],
                row["reference"],
                candidate_translation=row["prediction"],
            )
            identity = {
                "kind": "mqm",
                "segment_id": row["segment_id"],
                "direction": row["direction"],
                "candidate_model": row["model"],
                "judge_provider": configuration.provider,
                "judge_provider_name": configuration.provider_name,
                "judge_model": configuration.model,
                "judge_configuration": request_settings,
                "prompt_version": PROMPT_VERSION,
                "payload": payload,
            }
            jobs.append({**identity, "request_id": sha256_json(identity)})

    if mode in {"pairwise", "all"}:
        for (direction, segment_id), group in ordered.groupby(
            ["direction", "segment_id"], sort=True
        ):
            records = {row["model"]: row for row in group.to_dict("records")}
            for left_model, right_model in itertools.combinations(sorted(records), 2):
                for order, (model_a, model_b) in enumerate(
                    ((left_model, right_model), (right_model, left_model)), start=1
                ):
                    row_a, row_b = records[model_a], records[model_b]
                    payload = _judge_payload(
                        direction,
                        row_a["source"],
                        row_a["reference"],
                        candidate_A=row_a["prediction"],
                        candidate_B=row_b["prediction"],
                    )
                    identity = {
                        "kind": "pairwise",
                        "segment_id": segment_id,
                        "direction": direction,
                        "model_a": model_a,
                        "model_b": model_b,
                        "model_pair": " || ".join((left_model, right_model)),
                        "order": order,
                        "judge_provider": configuration.provider,
                        "judge_provider_name": configuration.provider_name,
                        "judge_model": configuration.model,
                        "judge_configuration": request_settings,
                        "prompt_version": PROMPT_VERSION,
                        "payload": payload,
                    }
                    jobs.append({**identity, "request_id": sha256_json(identity)})
    return jobs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_mqm_spans(judgment: MQMJudgment, candidate: str) -> None:
    for error in judgment.errors:
        if error.category == "omission":
            if error.span:
                raise ValueError("Omission errors must use an empty span")
        elif not error.span or error.span not in candidate:
            raise ValueError(f"MQM span is not an exact candidate substring: {error.span!r}")


def _judge_run_metadata(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "judge_provider",
        "judge_provider_name",
        "judge_model",
        "judge_configuration",
        "prompt_version",
    )
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"Judge row is missing run metadata: {missing}")
    return {field: row[field] for field in fields}


def _single_judge_run_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No judge rows were supplied")
    runs = {sha256_json(_judge_run_metadata(row)): _judge_run_metadata(row) for row in rows}
    if len(runs) != 1:
        raise ValueError(
            "Judge rows contain multiple providers, models, prompts, or inference settings"
        )
    return next(iter(runs.values()))


def run_judge(
    jobs: list[dict[str, Any]],
    output_path: Path,
    limit: int | None = None,
    show_progress: bool = False,
) -> tuple[int, int]:
    completed_rows = read_jsonl(output_path)
    if completed_rows and jobs:
        existing_run = _single_judge_run_metadata(completed_rows)
        requested_run = _single_judge_run_metadata(jobs)
        if existing_run != requested_run:
            raise ValueError(
                "The output file belongs to a different judge configuration; "
                "choose a new --output path"
            )
    completed = {row["request_id"] for row in completed_rows}
    pending = [job for job in jobs if job["request_id"] not in completed]
    if limit is not None:
        pending = pending[:limit]
    run_metadata = _single_judge_run_metadata(jobs) if jobs else None
    adapter = None
    configuration = None
    if run_metadata is not None:
        adapter = create_judge_adapter(
            run_metadata["judge_provider"], run_metadata["judge_model"]
        )
        configuration = adapter.configuration(**run_metadata["judge_configuration"])
        if adapter.provider_name != run_metadata["judge_provider_name"]:
            raise ValueError("Judge provider metadata does not match the registered adapter")
    written = 0
    progress_bar = None
    if show_progress:
        from tqdm.auto import tqdm

        description = (
            "judge"
            if run_metadata is None
            else f"{run_metadata['judge_provider']}/{run_metadata['judge_model']}"
        )
        progress_bar = tqdm(
            total=len(pending),
            desc=description,
            unit="request",
            dynamic_ncols=True,
        )
    try:
        for job in pending:
            if adapter is None or configuration is None:
                raise ValueError("No judge adapter was configured")
            schema = MQMJudgment if job["kind"] == "mqm" else PairwiseJudgment
            system_prompt = (
                MQM_SYSTEM_PROMPT if job["kind"] == "mqm" else PAIRWISE_SYSTEM_PROMPT
            )
            for attempt in range(4):
                try:
                    completion = adapter.judge(
                        system_prompt,
                        job["payload"],
                        schema,
                        configuration,
                    )
                    parsed = completion.result
                    if isinstance(parsed, MQMJudgment):
                        _validate_mqm_spans(
                            parsed, job["payload"]["candidate_translation"]
                        )
                    stored = {key: value for key, value in job.items() if key != "payload"}
                    stored.update(
                        {
                            "input_sha256": sha256_json(job["payload"]),
                            "prompt_sha256": hashlib.sha256(
                                system_prompt.encode("utf-8")
                            ).hexdigest(),
                            "created_at": utc_now(),
                            "response_model": completion.response_model,
                            "finish_reason": completion.finish_reason,
                            "result": parsed.model_dump(),
                            "usage": completion.usage,
                            "reasoning_content_omitted": (
                                completion.reasoning_content_omitted
                            ),
                            "api_response": completion.api_response,
                        }
                    )
                    append_jsonl(output_path, stored)
                    written += 1
                    if progress_bar is not None:
                        progress_bar.update(1)
                    break
                except Exception:  # API and validation errors share retry behavior.
                    if attempt == 3:
                        raise
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(f"retry {attempt + 1}/3")
                    time.sleep(2**attempt)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return written, len(pending)


def mqm_penalty(result: dict[str, Any]) -> int:
    return sum(SEVERITY_WEIGHTS[error["severity"]] for error in result["errors"])


def _bootstrap_mean_ci(
    values: list[float], iterations: int = 2_000, seed: int = RANDOM_SEED
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(seed)
    samples = generator.choice(array, size=(iterations, len(array)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pairwise_decisions(rows: list[dict[str, Any]]) -> pd.DataFrame:
    mapped: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "pairwise":
            continue
        winner = row["result"]["winner"]
        preferred = (
            "tie"
            if winner == "tie"
            else row["model_a"]
            if winner == "A"
            else row["model_b"]
        )
        mapped.append({**row, "preferred_model": preferred})
    if not mapped:
        return pd.DataFrame()
    frame = pd.DataFrame(mapped)
    decisions: list[dict[str, Any]] = []
    for key, group in frame.groupby(
        ["direction", "segment_id", "model_pair"], sort=True
    ):
        preferences = group["preferred_model"].tolist()
        stable = len(preferences) == 2 and len(set(preferences)) == 1
        decisions.append(
            {
                "direction": key[0],
                "segment_id": key[1],
                "model_pair": key[2],
                "preferred_model": preferences[0] if stable else "unstable",
                "stable": stable,
                "orders_present": len(preferences),
            }
        )
    return pd.DataFrame(decisions)


def aggregate_results(
    predictions_path: Path,
    judge_path: Path,
    output_dir: Path,
    xcomet_path: Path | None = None,
    model_keys: Iterable[str] | None = None,
) -> dict[str, Path]:
    predictions = pd.read_csv(predictions_path)
    if model_keys is not None:
        predictions = filter_prediction_models(predictions, model_keys)
    judge_rows = read_jsonl(judge_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    run_metadata = _single_judge_run_metadata(judge_rows)
    if run_metadata["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Judge output does not match the configured prompt version")
    adapter = create_judge_adapter(
        run_metadata["judge_provider"], run_metadata["judge_model"]
    )
    if adapter.provider_name != run_metadata["judge_provider_name"]:
        raise ValueError("Judge provider metadata does not match the registered adapter")
    judge_configuration = adapter.configuration(**run_metadata["judge_configuration"])
    expected_jobs = build_judge_jobs(
        predictions,
        "all",
        judge_configuration,
    )
    expected_ids = {job["request_id"] for job in expected_jobs}
    actual_ids = {row.get("request_id") for row in judge_rows}
    missing_judgments = expected_ids - actual_ids
    if missing_judgments:
        raise ValueError(
            f"Judge output is incomplete: {len(missing_judgments)} of "
            f"{len(expected_ids)} expected judgments are missing"
        )

    mqm_rows = []
    for row in judge_rows:
        if row.get("kind") != "mqm":
            continue
        errors = row["result"]["errors"]
        mqm_row = {
                "segment_id": row["segment_id"],
                "direction": row["direction"],
                "model": row["candidate_model"],
                "mqm_penalty": mqm_penalty(row["result"]),
                "minor_errors": sum(e["severity"] == "minor" for e in errors),
                "major_errors": sum(e["severity"] == "major" for e in errors),
                "critical_errors": sum(e["severity"] == "critical" for e in errors),
            }
        mqm_row.update(
            {
                f"{category}_errors": sum(e["category"] == category for e in errors)
                for category in MQM_CATEGORIES
            }
        )
        mqm_rows.append(mqm_row)
    mqm_frame = pd.DataFrame(mqm_rows)
    if not mqm_frame.empty:
        segments_path = output_dir / "translation-eval-mqm-segments.csv"
        mqm_frame.to_csv(segments_path, index=False)
        written["mqm_segments"] = segments_path
        summary_rows = []
        for (model, direction), group in mqm_frame.groupby(["model", "direction"]):
            values = group["mqm_penalty"].astype(float).tolist()
            low, high = _bootstrap_mean_ci(values)
            summary_row = {
                    "model": model,
                    "direction": direction,
                    "segments": len(group),
                    "mean_mqm_penalty": float(np.mean(values)),
                    "median_mqm_penalty": float(np.median(values)),
                    "mqm_ci_low": low,
                    "mqm_ci_high": high,
                    "minor_errors": int(group["minor_errors"].sum()),
                    "major_errors": int(group["major_errors"].sum()),
                    "critical_errors": int(group["critical_errors"].sum()),
                }
            summary_row.update(
                {
                    f"{category}_errors": int(group[f"{category}_errors"].sum())
                    for category in MQM_CATEGORIES
                }
            )
            summary_rows.append(summary_row)
        path = output_dir / "translation-eval-mqm-summary.csv"
        pd.DataFrame(summary_rows).to_csv(path, index=False)
        written["mqm"] = path

    decisions = pairwise_decisions(judge_rows)
    if not decisions.empty:
        pairwise_summary = []
        models = sorted(predictions["model"].unique())
        for direction in sorted(decisions["direction"].unique()):
            direction_rows = decisions.loc[decisions["direction"] == direction]
            for model in models:
                points: list[float] = []
                relevant = direction_rows.loc[
                    direction_rows["model_pair"].str.split(r" \|\| ", regex=True).apply(
                        lambda pair: model in pair
                    )
                ]
                for row in relevant.to_dict("records"):
                    if not row["stable"]:
                        continue
                    if row["preferred_model"] == "tie":
                        points.append(0.5)
                    elif row["preferred_model"] == model:
                        points.append(1.0)
                    else:
                        points.append(0.0)
                low, high = _bootstrap_mean_ci(points)
                pairwise_summary.append(
                    {
                        "model": model,
                        "direction": direction,
                        "stable_comparisons": len(points),
                        "preference_rate": float(np.mean(points)) if points else math.nan,
                        "preference_ci_low": low,
                        "preference_ci_high": high,
                        "unstable_comparisons": int((~relevant["stable"]).sum()),
                    }
                )
        decisions_path = output_dir / "translation-eval-pairwise-decisions.csv"
        summary_path = output_dir / "translation-eval-pairwise-summary.csv"
        decisions.to_csv(decisions_path, index=False)
        pd.DataFrame(pairwise_summary).to_csv(summary_path, index=False)
        written["pairwise_decisions"] = decisions_path
        written["pairwise"] = summary_path

    overlap_rows = []
    for (model, direction), group in predictions.groupby(["model", "direction"]):
        scores = _overlap_scores(
            group["prediction"].tolist(), group["reference"].tolist()
        )
        overlap_rows.append(
            {"model": model, "direction": direction, "segments": len(group), **scores}
        )
    overlap_path = output_dir / "translation-eval-overlap-appendix.csv"
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
    written["overlap"] = overlap_path

    if xcomet_path is not None and xcomet_path.exists():
        xcomet_rows = read_jsonl(xcomet_path)
        xcomet_frame = pd.DataFrame(xcomet_rows)
        if not xcomet_frame.empty:
            selected_models = set(predictions["model"].astype(str))
            xcomet_frame = xcomet_frame.loc[
                xcomet_frame["model"].isin(selected_models)
            ].copy()
            expected_xcomet = set(
                zip(
                    predictions["segment_id"],
                    predictions["model"],
                    predictions["direction"],
                    strict=True,
                )
            )
            actual_xcomet = set(
                zip(
                    xcomet_frame["segment_id"],
                    xcomet_frame["model"],
                    xcomet_frame["direction"],
                    strict=True,
                )
            )
            if expected_xcomet != actual_xcomet:
                raise ValueError(
                    "xCOMET output does not have exactly one result per prediction"
                )
            summary = (
                xcomet_frame.groupby(["model", "direction"], as_index=False)
                .agg(segments=("score", "size"), mean_xcomet=("score", "mean"))
            )
            path = output_dir / "translation-eval-xcomet-summary.csv"
            summary.to_csv(path, index=False)
            written["xcomet"] = path

    manifest = {
        "created_at": utc_now(),
        "judge_provider": run_metadata["judge_provider"],
        "judge_provider_name": run_metadata["judge_provider_name"],
        "judge_model": run_metadata["judge_model"],
        "judge_configuration": judge_configuration.request_settings(),
        "response_models": sorted(
            {
                row["response_model"]
                for row in judge_rows
                if row.get("response_model")
            }
        ),
        "prompt_version": PROMPT_VERSION,
        "severity_weights": SEVERITY_WEIGHTS,
        "files": {key: path.name for key, path in written.items()},
    }
    manifest_path = output_dir / "translation-eval-results-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written["manifest"] = manifest_path
    return written


def run_xcomet(
    predictions: pd.DataFrame,
    output_path: Path,
    metadata_path: Path,
    batch_size: int = 1,
    revision: str | None = None,
) -> int:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "Set HF_TOKEN after accepting the gated Unbabel/XCOMET-XL model license."
        )
    try:
        from comet import load_from_checkpoint
        from huggingface_hub import model_info, snapshot_download
    except ImportError as error:
        raise SystemExit(
            "Install unbabel-comet and huggingface-hub before running xCOMET."
        ) from error

    token = os.environ["HF_TOKEN"]
    if revision is None and metadata_path.exists():
        revision = json.loads(metadata_path.read_text(encoding="utf-8"))["revision"]
    if revision is None:
        revision = model_info(XCOMET_MODEL, token=token).sha
    snapshot = Path(
        snapshot_download(repo_id=XCOMET_MODEL, revision=revision, token=token)
    )
    checkpoint = snapshot / "checkpoints" / "model.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"xCOMET checkpoint not found at {checkpoint}")
    model = load_from_checkpoint(str(checkpoint))
    completed = {
        (row["segment_id"], row["model"], row["direction"])
        for row in read_jsonl(output_path)
    }
    pending = [
        row
        for row in predictions.to_dict("records")
        if (row["segment_id"], row["model"], row["direction"]) not in completed
    ]
    if not pending:
        return 0
    data = [
        {"src": row["source"], "mt": row["prediction"], "ref": row["reference"]}
        for row in pending
    ]
    result = model.predict(data, batch_size=batch_size, gpus=0, num_workers=0)
    error_spans = getattr(getattr(result, "metadata", None), "error_spans", None)
    for index, (row, score) in enumerate(zip(pending, result.scores, strict=True)):
        spans = error_spans[index] if error_spans is not None else []
        append_jsonl(
            output_path,
            {
                "segment_id": row["segment_id"],
                "model": row["model"],
                "direction": row["direction"],
                "score": float(score),
                "error_spans": json.loads(
                    json.dumps(spans, default=lambda value: asdict(value) if hasattr(value, "__dataclass_fields__") else str(value))
                ),
                "xcomet_model": XCOMET_MODEL,
                "xcomet_revision": revision,
                "created_at": utc_now(),
            },
        )
    metadata_path.write_text(
        json.dumps(
            {
                "model": XCOMET_MODEL,
                "revision": revision,
                "device": "cpu",
                "checkpoint": "checkpoints/model.ckpt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(pending)


def make_human_sample(
    predictions: pd.DataFrame,
    judge_rows: list[dict[str, Any]],
    review_path: Path,
    key_path: Path,
    xcomet_rows: list[dict[str, Any]] | None = None,
) -> int:
    decisions = pairwise_decisions(judge_rows)
    if decisions.empty:
        raise ValueError("Pairwise judgments are required before sampling human review")
    mqm = {
        (row["segment_id"], row["direction"], row["candidate_model"]): mqm_penalty(
            row["result"]
        )
        for row in judge_rows
        if row.get("kind") == "mqm"
    }
    prediction_index = {
        (row["segment_id"], row["direction"], row["model"]): row
        for row in predictions.to_dict("records")
    }
    xcomet = {
        (row["segment_id"], row["direction"], row["model"]): float(row["score"])
        for row in (xcomet_rows or [])
    }
    candidates: list[dict[str, Any]] = []
    for decision in decisions.to_dict("records"):
        left, right = decision["model_pair"].split(" || ")
        left_penalty = mqm.get((decision["segment_id"], decision["direction"], left))
        right_penalty = mqm.get((decision["segment_id"], decision["direction"], right))
        mqm_preferred = None
        if left_penalty is not None and right_penalty is not None:
            mqm_preferred = (
                "tie"
                if left_penalty == right_penalty
                else left
                if left_penalty < right_penalty
                else right
            )
        left_xcomet = xcomet.get((decision["segment_id"], decision["direction"], left))
        right_xcomet = xcomet.get((decision["segment_id"], decision["direction"], right))
        xcomet_preferred = None
        if left_xcomet is not None and right_xcomet is not None:
            xcomet_preferred = (
                "tie"
                if math.isclose(left_xcomet, right_xcomet, abs_tol=1e-9)
                else left
                if left_xcomet > right_xcomet
                else right
            )
        if not decision["stable"] or (
            mqm_preferred is not None
            and decision["preferred_model"] not in {mqm_preferred, "tie"}
        ) or (
            xcomet_preferred is not None
            and decision["preferred_model"] not in {xcomet_preferred, "tie"}
        ):
            stratum = "disagreement"
        elif left_penalty is not None and abs(left_penalty - right_penalty) <= 2:
            stratum = "borderline"
        else:
            stratum = "clear"
        candidates.append(
            {
                **decision,
                "left_model": left,
                "right_model": right,
                "stratum": stratum,
                "penalty_gap": abs((left_penalty or 0) - (right_penalty or 0)),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    selected: list[dict[str, Any]] = []
    strata = ("clear", "borderline", "disagreement")
    for direction in sorted(candidate_frame["direction"].unique()):
        for model_pair in sorted(candidate_frame["model_pair"].unique()):
            pool = candidate_frame.loc[
                (candidate_frame["direction"] == direction)
                & (candidate_frame["model_pair"] == model_pair)
            ]
            for stratum in strata:
                preferred_pool = pool.loc[pool["stratum"] == stratum]
                preferred_pool = preferred_pool.loc[
                    ~preferred_pool["segment_id"].isin(
                        row["segment_id"] for row in selected
                    )
                ]
                rows = preferred_pool.sort_values(
                    ["penalty_gap", "segment_id"],
                    ascending=[stratum != "clear", True],
                ).head(3).to_dict("records")
                selected.extend(rows)
            direction_selected = sum(
                row["direction"] == direction and row["model_pair"] == model_pair
                for row in selected
            )
            if direction_selected < 9:
                fallback = pool.loc[
                    ~pool["segment_id"].isin(
                        row["segment_id"] for row in selected
                    )
                ].sort_values(["penalty_gap", "segment_id"], ascending=[True, True])
                selected.extend(fallback.head(9 - direction_selected).to_dict("records"))

    review_rows, key_rows = [], []
    for index, row in enumerate(selected[:18], start=1):
        left = prediction_index[(row["segment_id"], row["direction"], row["left_model"])]
        right = prediction_index[(row["segment_id"], row["direction"], row["right_model"])]
        swap = int(hashlib.sha256(row["segment_id"].encode()).hexdigest(), 16) % 2 == 0
        model_a, model_b = (
            (row["right_model"], row["left_model"])
            if swap
            else (row["left_model"], row["right_model"])
        )
        candidate_a = right["prediction"] if swap else left["prediction"]
        candidate_b = left["prediction"] if swap else right["prediction"]
        sample_id = f"human-{index:02d}"
        review_rows.append(
            {
                "sample_id": sample_id,
                "direction": row["direction"],
                "source": left["source"],
                "human_reference": left["reference"],
                "candidate_A": candidate_a,
                "candidate_B": candidate_b,
                "choice_A_B_or_tie": "",
                "review_status": "needs_review",
                "confidence": "",
                "failure_tags": "",
                "note": "",
                "add_to_golden": "false",
            }
        )
        judge_preferred = row["preferred_model"]
        judge_choice = (
            "unstable"
            if judge_preferred == "unstable"
            else "tie"
            if judge_preferred == "tie"
            else "A"
            if judge_preferred == model_a
            else "B"
        )
        key_rows.append(
            {
                "sample_id": sample_id,
                "segment_id": row["segment_id"],
                "stratum": row["stratum"],
                "model_A": model_a,
                "model_B": model_b,
                "judge_choice": judge_choice,
                "judge_stable": row["stable"],
            }
        )
    pd.DataFrame(review_rows).to_csv(review_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    return len(review_rows)


def human_agreement(review: pd.DataFrame, key: pd.DataFrame) -> dict[str, float | int]:
    merged = review.merge(key, on="sample_id", validate="one_to_one")
    human = merged["choice_A_B_or_tie"].fillna("").str.strip()
    completed = merged.loc[human.isin({"A", "B", "tie"})].copy()
    valid = completed.loc[completed["judge_choice"].isin({"A", "B", "tie"})].copy()
    if valid.empty:
        raise ValueError("The human review has no completed item with a stable judge choice")
    observed = float(
        (valid["choice_A_B_or_tie"] == valid["judge_choice"]).mean()
    )
    labels = ("A", "B", "tie")
    expected = sum(
        (valid["choice_A_B_or_tie"] == label).mean()
        * (valid["judge_choice"] == label).mean()
        for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    low, high = _bootstrap_mean_ci(
        (valid["choice_A_B_or_tie"] == valid["judge_choice"])
        .astype(float)
        .tolist()
    )
    return {
        "completed_items": len(completed),
        "stable_judge_items": len(valid),
        "unstable_judge_items": int((completed["judge_choice"] == "unstable").sum()),
        "raw_agreement": observed,
        "agreement_ci_low": low,
        "agreement_ci_high": high,
        "cohen_kappa": float(kappa),
        "passes_70_percent_gate": observed >= 0.70,
    }


def _write_alignment_metadata(path: Path, candidates: pd.DataFrame) -> None:
    metadata = {
        "historical_commit": HISTORICAL_COMMIT,
        "english_notebook": ENGLISH_NOTEBOOK,
        "portuguese_notebook": PORTUGUESE_NOTEBOOK,
        "aligned_cell_pairs": ALIGNED_CELL_PAIRS,
        "alignment_candidates": len(candidates),
        "labse_model": LABSE_MODEL,
        "labse_revision": LABSE_REVISION,
        "algorithm": {
            "type": "monotonic dynamic programming",
            "max_group": 3,
            "gap_penalty": -0.45,
            "merge_penalty_per_extra_sentence": 0.08,
            "length_penalty_weight": 0.08,
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_parser = subparsers.add_parser("align", help="propose LaBSE/DP alignments")
    align_parser.add_argument("--repo", type=Path, default=Path.cwd())
    align_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    align_parser.add_argument(
        "--output", type=Path, default=Path("posts/data/translation-eval-alignment-review.csv")
    )
    align_parser.add_argument(
        "--metadata", type=Path, default=Path("posts/data/translation-eval-alignment.json")
    )

    freeze_parser = subparsers.add_parser("freeze", help="freeze reviewed alignments")
    freeze_parser.add_argument(
        "--review", type=Path, default=Path("posts/data/translation-eval-alignment-review.csv")
    )
    freeze_parser.add_argument(
        "--output", type=Path, default=Path("posts/data/translation-eval-segments.csv")
    )

    predict_parser = subparsers.add_parser("predict", help="translate accepted segments")
    predict_parser.add_argument(
        "--segments", type=Path, default=Path("posts/data/translation-eval-segments.csv")
    )
    predict_parser.add_argument(
        "--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS)
    )
    predict_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    predict_parser.add_argument("--threads", type=int, default=None)
    predict_parser.add_argument(
        "--predictions", type=Path, default=Path("posts/data/translation-eval-predictions.csv")
    )
    predict_parser.add_argument(
        "--runtime", type=Path, default=Path("posts/data/translation-eval-runtime.csv")
    )
    predict_parser.add_argument(
        "--restart",
        action="store_true",
        help="discard prediction checkpoints and recompute requested models",
    )

    judge_parser = subparsers.add_parser("judge", help="run resumable LLM judgments")
    judge_parser.add_argument(
        "--predictions", type=Path, default=Path("posts/data/translation-eval-predictions.csv")
    )
    judge_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(DEFAULT_JUDGE_MODEL_KEYS),
        help="candidate translation models to judge (default: marian tower)",
    )
    judge_parser.add_argument(
        "--judge-provider",
        choices=available_judge_providers(),
        default=DEFAULT_JUDGE_PROVIDER,
    )
    judge_parser.add_argument(
        "--judge-model",
        help="override the provider adapter's default model",
    )
    judge_parser.add_argument("--mode", choices=["mqm", "pairwise", "all"], default="all")
    judge_parser.add_argument("--limit", type=int)
    judge_parser.add_argument(
        "--dry-run", action="store_true", help="print job counts without calling the API"
    )
    judge_parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help="provider-specific reasoning level (defaults to the adapter setting)",
    )
    judge_parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="provider-specific completion cap (defaults to the adapter setting)",
    )
    judge_parser.add_argument(
        "--output",
        type=Path,
        help="defaults to posts/data/translation-eval-<provider>-judgments.jsonl",
    )

    xcomet_parser = subparsers.add_parser("xcomet", help="run gated xCOMET-XL on CPU")
    xcomet_parser.add_argument(
        "--predictions", type=Path, default=Path("posts/data/translation-eval-predictions.csv")
    )
    xcomet_parser.add_argument("--batch-size", type=int, default=1)
    xcomet_parser.add_argument("--revision")
    xcomet_parser.add_argument(
        "--output", type=Path, default=Path("posts/data/translation-eval-xcomet.jsonl")
    )
    xcomet_parser.add_argument(
        "--metadata", type=Path, default=Path("posts/data/translation-eval-xcomet-model.json")
    )

    aggregate_parser = subparsers.add_parser("aggregate", help="build report-ready summaries")
    aggregate_parser.add_argument(
        "--predictions", type=Path, default=Path("posts/data/translation-eval-predictions.csv")
    )
    aggregate_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(DEFAULT_JUDGE_MODEL_KEYS),
    )
    aggregate_parser.add_argument(
        "--judge-provider",
        choices=available_judge_providers(),
        default=DEFAULT_JUDGE_PROVIDER,
        help="selects the default judgment artifact path",
    )
    aggregate_parser.add_argument(
        "--judgments",
        type=Path,
        help="defaults to the selected provider's judgment artifact",
    )
    aggregate_parser.add_argument(
        "--xcomet", type=Path, default=Path("posts/data/translation-eval-xcomet.jsonl")
    )
    aggregate_parser.add_argument("--output-dir", type=Path, default=Path("posts/data"))

    human_parser = subparsers.add_parser("human-sample", help="create 18 blinded review items")
    human_parser.add_argument(
        "--predictions", type=Path, default=Path("posts/data/translation-eval-predictions.csv")
    )
    human_parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=list(DEFAULT_JUDGE_MODEL_KEYS),
    )
    human_parser.add_argument(
        "--judge-provider",
        choices=available_judge_providers(),
        default=DEFAULT_JUDGE_PROVIDER,
        help="selects the default judgment artifact path",
    )
    human_parser.add_argument(
        "--judgments",
        type=Path,
        help="defaults to the selected provider's judgment artifact",
    )
    human_parser.add_argument(
        "--xcomet", type=Path, default=Path("posts/data/translation-eval-xcomet.jsonl")
    )
    human_parser.add_argument(
        "--review", type=Path, default=Path("posts/data/translation-eval-human-review.csv")
    )
    human_parser.add_argument(
        "--key", type=Path, default=Path("posts/data/translation-eval-human-key.csv")
    )

    agreement_parser = subparsers.add_parser("human-agreement", help="score completed human review")
    agreement_parser.add_argument(
        "--review", type=Path, default=Path("posts/data/translation-eval-human-review.csv")
    )
    agreement_parser.add_argument(
        "--key", type=Path, default=Path("posts/data/translation-eval-human-key.csv")
    )
    agreement_parser.add_argument(
        "--output", type=Path, default=Path("posts/data/translation-eval-human-agreement.json")
    )

    args = parser.parse_args()
    if args.command == "align":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        candidates = generate_alignment_candidates(
            args.repo.resolve(), LabseSimilarity(args.device)
        )
        candidates.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
        _write_alignment_metadata(args.metadata, candidates)
        print(f"Wrote {len(candidates)} alignment candidates to {args.output}")
    elif args.command == "freeze":
        segments = freeze_reviewed_alignments(pd.read_csv(args.review, keep_default_na=False))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        segments.to_csv(args.output, index=False)
        print(f"Froze {len(segments)} accepted segments to {args.output}")
    elif args.command == "predict":
        if args.threads is not None:
            import torch

            torch.set_num_threads(args.threads)
        if args.restart:
            args.predictions.unlink(missing_ok=True)
            args.runtime.unlink(missing_ok=True)
        runtime, predictions = predict_segments(
            pd.read_csv(args.segments),
            args.models,
            args.device,
            args.predictions,
            args.runtime,
            show_progress=True,
        )
        _atomic_write_csv(runtime, args.runtime)
        _atomic_write_csv(predictions, args.predictions)
        print(runtime.to_string(index=False))
    elif args.command == "judge":
        predictions = filter_prediction_models(pd.read_csv(args.predictions), args.models)
        judge_configuration = resolve_judge_configuration(
            args.judge_provider,
            args.judge_model,
            args.reasoning_effort,
            args.max_completion_tokens,
        )
        jobs = build_judge_jobs(
            predictions,
            args.mode,
            judge_configuration,
        )
        if args.dry_run:
            counts = pd.Series([job["kind"] for job in jobs]).value_counts()
            print(f"candidates {', '.join(args.models)}")
            print(
                f"judge {judge_configuration.provider}/{judge_configuration.model} | "
                f"reasoning {judge_configuration.reasoning_effort} | max completion "
                f"tokens {judge_configuration.max_completion_tokens}"
            )
            print(counts.to_string())
            print(f"total {len(jobs)}")
            return
        output_path = args.output or default_judgments_path(
            judge_configuration.provider
        )
        written, pending = run_judge(
            jobs, output_path, args.limit, show_progress=True
        )
        print(f"Wrote {written} of {pending} pending judgments to {output_path}")
    elif args.command == "xcomet":
        count = run_xcomet(
            pd.read_csv(args.predictions),
            args.output,
            args.metadata,
            args.batch_size,
            args.revision,
        )
        print(f"Wrote {count} xCOMET evaluations to {args.output}")
    elif args.command == "aggregate":
        judgments_path = args.judgments or default_judgments_path(
            args.judge_provider
        )
        written = aggregate_results(
            args.predictions,
            judgments_path,
            args.output_dir,
            args.xcomet,
            args.models,
        )
        print("\n".join(f"{key}: {path}" for key, path in written.items()))
    elif args.command == "human-sample":
        judgments_path = args.judgments or default_judgments_path(
            args.judge_provider
        )
        count = make_human_sample(
            filter_prediction_models(pd.read_csv(args.predictions), args.models),
            read_jsonl(judgments_path),
            args.review,
            args.key,
            read_jsonl(args.xcomet) if args.xcomet.exists() else None,
        )
        print(f"Wrote {count} blinded human-review items to {args.review}")
    elif args.command == "human-agreement":
        result = human_agreement(pd.read_csv(args.review), pd.read_csv(args.key))
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
