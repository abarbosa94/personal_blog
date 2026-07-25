"""Detect repeated n-gram degeneration in saved translation predictions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class RepetitionResult:
    flagged: bool
    max_occurrences: int
    repeated_ngram: str


def repetition_result(
    text: str,
    *,
    ngram_size: int = 4,
    minimum_occurrences: int = 4,
) -> RepetitionResult:
    """Flag text when one normalized n-gram occurs at least the threshold."""
    tokens = TOKEN_PATTERN.findall(text.casefold())
    ngrams = Counter(
        tuple(tokens[index : index + ngram_size])
        for index in range(max(0, len(tokens) - ngram_size + 1))
    )
    if not ngrams:
        return RepetitionResult(False, 0, "")
    repeated, count = ngrams.most_common(1)[0]
    return RepetitionResult(
        flagged=count >= minimum_occurrences,
        max_occurrences=count,
        repeated_ngram=" ".join(repeated),
    )


def screen_csv(
    path: Path,
    *,
    model: str,
    ngram_size: int = 4,
    minimum_occurrences: int = 4,
) -> list[dict[str, str | int]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    flagged: list[dict[str, str | int]] = []
    for row in rows:
        if row["model"] != model:
            continue
        result = repetition_result(
            row["prediction"],
            ngram_size=ngram_size,
            minimum_occurrences=minimum_occurrences,
        )
        if result.flagged:
            flagged.append(
                {
                    "direction": row["direction"],
                    "segment_id": row["segment_id"],
                    "max_occurrences": result.max_occurrences,
                    "repeated_ngram": result.repeated_ngram,
                }
            )
    return flagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        type=Path,
        nargs="?",
        default=Path("posts/data/translation-eval-predictions.csv"),
    )
    parser.add_argument("--model", default="NLLB-200 distilled 600M")
    parser.add_argument("--ngram-size", type=int, default=4)
    parser.add_argument("--minimum-occurrences", type=int, default=4)
    args = parser.parse_args()

    flagged = screen_csv(
        args.predictions,
        model=args.model,
        ngram_size=args.ngram_size,
        minimum_occurrences=args.minimum_occurrences,
    )
    if not flagged:
        print("No predictions crossed the repetition threshold.")
        return
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=[
            "direction",
            "segment_id",
            "max_occurrences",
            "repeated_ngram",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(flagged)


if __name__ == "__main__":
    main()
