from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from translation_eval import (  # noqa: E402
    MQMError,
    MQMJudgment,
    Sentence,
    _validate_mqm_spans,
    aggregate_results,
    append_jsonl,
    build_judge_jobs,
    filter_prediction_models,
    freeze_reviewed_alignments,
    human_agreement,
    monotonic_align,
    pairwise_decisions,
    predict_segments,
    segment_markdown,
)
from translation_judges import (  # noqa: E402
    KimiJudgeAdapter,
    create_judge_adapter,
    kimi_response_format,
)


def test_markdown_segmentation_keeps_headings_and_abbreviations() -> None:
    source = "# A heading\n\nThis is e.g. useful. Next sentence.\n\n```python\nskip()\n```"

    sentences = segment_markdown(source, "en")

    assert [sentence.text for sentence in sentences] == [
        "A heading",
        "This is e.g. useful.",
        "Next sentence.",
    ]
    assert [sentence.sentence_id for sentence in sentences] == ["en01", "en02", "en03"]


def test_dynamic_programming_can_choose_two_to_one_alignment() -> None:
    english = [Sentence("e1", "First idea."), Sentence("e2", "Second idea.")]
    portuguese = [Sentence("p1", "Primeira ideia. Segunda ideia.")]

    def similarity(left: str, right: str) -> float:
        del right
        return 0.95 if "First idea. Second idea." == left else 0.10

    steps = monotonic_align(english, portuguese, similarity)

    assert len(steps) == 1
    assert (steps[0].english_count, steps[0].portuguese_count) == (2, 1)


def test_prediction_checkpoints_are_reused_and_content_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import translation_benchmark

    calls: list[tuple[str, str, tuple[str, ...]]] = []
    progress_updates: list[int] = []

    class FakeTranslator:
        parameter_count = 10_000_000

        def translate(
            self,
            texts: list[str],
            source_lang: str,
            target_lang: str,
            progress=None,
        ) -> list[str]:
            calls.append((source_lang, target_lang, tuple(texts)))
            if progress is not None:
                progress(len(texts))
                progress_updates.append(len(texts))
            return [f"{target_lang}: {text}" for text in texts]

        def close(self) -> None:
            return None

    monkeypatch.setitem(
        translation_benchmark.MODEL_SPECS,
        "marian",
        translation_benchmark.ModelSpec("Fake model", lambda _device: FakeTranslator()),
    )
    segments = pd.DataFrame(
        [
            {
                "segment_id": "s01",
                "pair_id": 1,
                "english": "First source.",
                "portuguese": "Primeira fonte.",
            },
            {
                "segment_id": "s02",
                "pair_id": 2,
                "english": "Second source.",
                "portuguese": "Segunda fonte.",
            },
        ]
    )
    predictions_path = tmp_path / "predictions.csv"
    runtime_path = tmp_path / "runtime.csv"

    runtime, predictions = predict_segments(
        segments,
        ["marian"],
        "cpu",
        predictions_path,
        runtime_path,
        show_progress=True,
    )

    assert len(predictions) == 4
    assert len(runtime) == 2
    assert len(calls) == 2
    assert progress_updates == [2, 2]
    assert len(pd.read_csv(predictions_path)) == 4
    assert len(pd.read_csv(runtime_path)) == 2

    predict_segments(
        segments,
        ["marian"],
        "cpu",
        predictions_path,
        runtime_path,
    )
    assert len(calls) == 2

    changed = segments.copy()
    changed.loc[0, "english"] = "Changed source."
    predict_segments(
        changed,
        ["marian"],
        "cpu",
        predictions_path,
        runtime_path,
    )
    assert len(calls) == 4
    checkpoint = pd.read_csv(predictions_path)
    assert "Changed source." in set(checkpoint["source"])


def _review_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "alignment_id": "p01-a01",
        "pair_id": 1,
        "english_sentence_ids": "p01-en01",
        "portuguese_sentence_ids": "p01-pt01",
        "english": "English source.",
        "portuguese": "Fonte em português.",
        "review_status": "needs_review",
        "reviewed_english": "",
        "reviewed_portuguese": "",
        "review_note": "",
    }
    row.update(overrides)
    return row


def test_freeze_requires_every_alignment_to_be_reviewed() -> None:
    with pytest.raises(ValueError, match="terminal review_status"):
        freeze_reviewed_alignments(pd.DataFrame([_review_row()]))


def test_freeze_uses_explicit_reviewed_text_and_rejects_reuse() -> None:
    accepted = pd.DataFrame(
        [
            _review_row(
                review_status="accept",
                reviewed_portuguese="Fonte em português corrigida.",
            )
        ]
    )
    frozen = freeze_reviewed_alignments(accepted)
    assert frozen.iloc[0]["portuguese"] == "Fonte em português corrigida."
    assert bool(frozen.iloc[0]["reference_was_edited"])

    duplicate = pd.DataFrame(
        [
            _review_row(review_status="accept"),
            _review_row(
                alignment_id="p01-a02",
                review_status="accept",
                portuguese_sentence_ids="p01-pt02",
            ),
        ]
    )
    with pytest.raises(ValueError, match="reused"):
        freeze_reviewed_alignments(duplicate)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id": "s1",
                "model": model,
                "direction": "en -> pt-BR",
                "source": "Source text.",
                "reference": "Texto de referência.",
                "prediction": f"Candidate {index}.",
            }
            for index, model in enumerate(("Marian", "NLLB", "Tower"), start=1)
        ]
    )


def test_judge_jobs_are_complete_stable_and_blinded() -> None:
    jobs = build_judge_jobs(_predictions())
    repeated = build_judge_jobs(_predictions())

    assert len(jobs) == 9  # 3 MQM + 3 model pairs x 2 candidate orders
    assert [job["request_id"] for job in jobs] == [
        job["request_id"] for job in repeated
    ]
    for job in jobs:
        assert job["judge_provider"] == "kimi"
        assert job["judge_provider_name"] == "Moonshot AI"
        assert job["judge_model"] == "kimi-k3"
        assert job["judge_configuration"] == {
            "reasoning_effort": "max",
            "max_completion_tokens": 8192,
        }
        serialized_payload = str(job["payload"])
        assert "Marian" not in serialized_payload
        assert "NLLB" not in serialized_payload
        assert "Tower" not in serialized_payload


def test_kimi_schema_is_strict_and_uses_no_local_references() -> None:
    response_format = kimi_response_format(MQMJudgment)
    schema = response_format["json_schema"]["schema"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["properties"]["errors"]["items"]["additionalProperties"] is False
    assert "$defs" not in str(schema)
    assert "$ref" not in str(schema)


def test_judge_factory_switches_provider_defaults() -> None:
    kimi = create_judge_adapter("kimi")
    openai = create_judge_adapter("openai")

    assert kimi.configuration().model == "kimi-k3"
    assert kimi.configuration().reasoning_effort == "max"
    assert openai.configuration().provider_name == "OpenAI"
    assert openai.configuration().reasoning_effort == "medium"
    with pytest.raises(ValueError, match="Unknown judge provider"):
        create_judge_adapter("missing")


def test_kimi_adapter_uses_chat_schema_and_omits_reasoning_content() -> None:
    captured: dict = {}
    content = '{"errors": [], "summary": "No errors."}'

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=content),
                    )
                ],
                model_dump=lambda **_kwargs: {
                    "id": "test-response",
                    "model": "kimi-k3",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": content,
                                "reasoning_content": "private reasoning",
                            },
                        }
                    ],
                    "usage": {"completion_tokens": 10},
                },
            )

    adapter = KimiJudgeAdapter()
    adapter._client = SimpleNamespace(  # noqa: SLF001 - intentional fake client
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    completion = adapter.judge(
        "Judge carefully.",
        {"candidate_translation": "Candidate."},
        MQMJudgment,
        adapter.configuration(),
    )

    assert captured["model"] == "kimi-k3"
    assert captured["reasoning_effort"] == "max"
    assert captured["max_completion_tokens"] == 8192
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert completion.result.summary == "No errors."
    assert completion.reasoning_content_omitted is True
    assert "reasoning_content" not in completion.api_response["choices"][0]["message"]


def test_prediction_model_filter_uses_stable_cli_keys() -> None:
    selected = filter_prediction_models(
        pd.DataFrame(
            {
                "model": [
                    "Marian OPUS-MT",
                    "NLLB-200 distilled 600M",
                    "Tower+ 2B",
                ]
            }
        ),
        ["marian", "tower"],
    )

    assert selected["model"].tolist() == ["Marian OPUS-MT", "Tower+ 2B"]


def test_mqm_spans_must_be_exact_candidate_substrings() -> None:
    valid = MQMJudgment(
        errors=[
            MQMError(
                span="bad term",
                category="terminology",
                severity="minor",
                explanation="It is inconsistent.",
            ),
            MQMError(
                span="",
                category="omission",
                severity="major",
                explanation="A source clause is absent.",
            ),
        ],
        summary="Two errors.",
    )
    _validate_mqm_spans(valid, "A bad term appears.")

    invalid = MQMJudgment(
        errors=[
            MQMError(
                span="invented span",
                category="fluency",
                severity="minor",
                explanation="Not really present.",
            )
        ],
        summary="Invalid span.",
    )
    with pytest.raises(ValueError, match="exact candidate substring"):
        _validate_mqm_spans(invalid, "Candidate text.")


def test_reversed_pairwise_orders_must_agree() -> None:
    base = {
        "kind": "pairwise",
        "segment_id": "s1",
        "direction": "en -> pt-BR",
        "model_pair": "Marian || Tower",
    }
    stable = pairwise_decisions(
        [
            {
                **base,
                "model_a": "Marian",
                "model_b": "Tower",
                "result": {"winner": "B"},
            },
            {
                **base,
                "model_a": "Tower",
                "model_b": "Marian",
                "result": {"winner": "A"},
            },
        ]
    )
    assert bool(stable.iloc[0]["stable"])
    assert stable.iloc[0]["preferred_model"] == "Tower"

    unstable = pairwise_decisions(
        [
            {
                **base,
                "model_a": "Marian",
                "model_b": "Tower",
                "result": {"winner": "A"},
            },
            {
                **base,
                "model_a": "Tower",
                "model_b": "Marian",
                "result": {"winner": "A"},
            },
        ]
    )
    assert not bool(unstable.iloc[0]["stable"])
    assert unstable.iloc[0]["preferred_model"] == "unstable"


def test_human_agreement_reports_gate_and_kappa() -> None:
    review = pd.DataFrame(
        {
            "sample_id": ["h1", "h2", "h3", "h4"],
            "choice_A_B_or_tie": ["A", "B", "tie", "A"],
        }
    )
    key = pd.DataFrame(
        {
            "sample_id": ["h1", "h2", "h3", "h4"],
            "judge_choice": ["A", "B", "tie", "B"],
        }
    )

    result = human_agreement(review, key)

    assert result["completed_items"] == 4
    assert result["raw_agreement"] == 0.75
    assert result["passes_70_percent_gate"] is True


def test_aggregation_requires_complete_cached_judgments(tmp_path: Path) -> None:
    predictions = _predictions()
    predictions_path = tmp_path / "predictions.csv"
    judgments_path = tmp_path / "judgments.jsonl"
    predictions.to_csv(predictions_path, index=False)
    jobs = build_judge_jobs(predictions)
    for job in jobs:
        stored = {key: value for key, value in job.items() if key != "payload"}
        stored["result"] = (
            {"errors": [], "summary": "No errors."}
            if job["kind"] == "mqm"
            else {
                "winner": "A",
                "decisive_categories": ["accuracy"],
                "explanation": "A is more faithful.",
            }
        )
        append_jsonl(judgments_path, stored)

    written = aggregate_results(predictions_path, judgments_path, tmp_path)

    assert written["mqm"].exists()
    assert written["pairwise"].exists()
    assert written["overlap"].exists()
    mqm = pd.read_csv(written["mqm"])
    assert set(mqm["mean_mqm_penalty"]) == {0.0}

    incomplete_path = tmp_path / "incomplete.jsonl"
    for row in jobs[:-1]:
        stored = {key: value for key, value in row.items() if key != "payload"}
        stored["result"] = (
            {"errors": [], "summary": "No errors."}
            if row["kind"] == "mqm"
            else {"winner": "A", "decisive_categories": [], "explanation": "A."}
        )
        append_jsonl(incomplete_path, stored)
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results(predictions_path, incomplete_path, tmp_path / "incomplete")
