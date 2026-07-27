from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from screening import repetition_result, screen_csv  # noqa: E402


def test_repetition_result_detects_generation_loop() -> None:
    result = repetition_result(
        "in accordance with the rules " * 5,
        ngram_size=4,
        minimum_occurrences=4,
    )

    assert result.flagged is True
    assert result.max_occurrences >= 4


def test_saved_nllb_predictions_reproduce_four_failures() -> None:
    repo = EXPERIMENT_ROOT.parents[1]
    flagged = screen_csv(
        repo / "posts" / "data" / "translation-eval-predictions.csv",
        model="NLLB-200 distilled 600M",
    )

    assert [(row["direction"], row["segment_id"]) for row in flagged] == [
        ("pt-BR -> en", "p05-a04"),
        ("pt-BR -> en", "p06-a02"),
        ("pt-BR -> en", "p08-a01"),
        ("pt-BR -> en", "p08-a03"),
    ]
