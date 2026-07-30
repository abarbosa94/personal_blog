from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from conference_pipeline.manual_review import ReviewStore, build_rows, write_queue
from conference_pipeline.review_analysis import expected_country_codes


def record(
    paper_id: str,
    *,
    openalex_id: str | None = None,
    match_method: str | None = None,
    affiliations: list[dict[str, str | None]] | None = None,
) -> dict:
    return {
        "paper": {
            "paper_id": paper_id,
            "venue_key": "icml",
            "year": 2025,
            "track": "main",
            "title": f"Paper {paper_id}",
            "authors": ["A. Author"],
            "doi": None,
            "official_url": f"https://example.test/{paper_id}",
            "source_kind": "fixture",
            "pdf_url": f"https://example.test/{paper_id}.pdf",
        },
        "openalex_id": openalex_id,
        "match_method": match_method,
        "affiliations": affiliations or [],
    }


def queue(tmp_path: Path, records: list[dict]) -> Path:
    source = tmp_path / "sample.jsonl"
    source.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    output = tmp_path / "review.csv"
    write_queue(build_rows([source]), output)
    return output


def test_build_prioritizes_failures_but_keeps_clean_controls(tmp_path: Path) -> None:
    path = queue(
        tmp_path,
        [
            record(
                "clean",
                openalex_id="https://openalex.org/W1",
                match_method="title",
                affiliations=[
                    {
                        "institution_id": "I1",
                        "institution_name": "University",
                        "country_code": "BR",
                        "institution_type": "education",
                    }
                ],
            ),
            record("missing"),
        ],
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["paper_id"] for row in rows] == ["missing", "clean"]
    assert {row["paper_id"] for row in rows} == {"missing", "clean"}
    assert rows[0]["system_signal"] == "unreconciled"
    assert rows[1]["system_signal"] == "automatic_pass"


def test_openalex_work_without_affiliations_is_not_an_automatic_pass(
    tmp_path: Path,
) -> None:
    path = queue(
        tmp_path,
        [record("empty", openalex_id="https://openalex.org/W1", match_method="title")],
    )
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["system_signal"] == "unreconciled"
    assert row["priority"] == "high"


def test_pdf_ner_result_is_not_an_automatic_pass(tmp_path: Path) -> None:
    path = queue(
        tmp_path,
        [
            record(
                "ner",
                openalex_id="https://openalex.org/W1",
                match_method="title+v2-pdf-ner",
                affiliations=[
                    {
                        "institution_id": "I1",
                        "institution_name": "University",
                        "country_code": "US",
                        "institution_type": "education",
                    }
                ],
            )
        ],
    )
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["system_signal"] == "mixed_sources"
    assert row["priority"] == "medium"


def test_unresolved_multinational_branch_is_high_priority(tmp_path: Path) -> None:
    path = queue(
        tmp_path,
        [
            record(
                "branch",
                openalex_id="https://openalex.org/W1",
                match_method="title+v2-pdf-ner",
                affiliations=[
                    {
                        "institution_id": "",
                        "institution_name": "Google DeepMind",
                        "country_code": None,
                        "institution_type": None,
                    },
                    {
                        "institution_id": "I1",
                        "institution_name": "IIT Madras",
                        "country_code": "IN",
                        "institution_type": "education",
                    },
                ],
            )
        ],
    )
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["system_signal"] == "ambiguous_multinational_branch"
    assert row["priority"] == "high"
    assert "Google DeepMind" in row["priority_reason"]


def test_resolved_branch_suppresses_annotated_duplicate_branch_signal(
    tmp_path: Path,
) -> None:
    path = queue(
        tmp_path,
        [
            record(
                "resolved-branch",
                openalex_id="https://openalex.org/W1",
                match_method="title+v2-pdf-ner+v9-search-evidence",
                affiliations=[
                    {
                        "institution_id": "",
                        "institution_name": "Equal contribution 1Google DeepMind.",
                        "country_code": None,
                        "institution_type": None,
                    },
                    {
                        "institution_id": "",
                        "institution_name": "Google DeepMind.",
                        "country_code": "US",
                        "institution_type": "company",
                    },
                ],
            )
        ],
    )

    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["system_signal"] == "mixed_sources"
    assert row["priority"] == "medium"


def test_failed_review_requires_code_confidence_and_note(tmp_path: Path) -> None:
    store = ReviewStore(queue(tmp_path, [record("missing")]))
    with pytest.raises(ValueError, match="confidence"):
        store.update("icml:missing", {"review_status": "fail"})
    with pytest.raises(ValueError, match="failure code"):
        store.update(
            "icml:missing", {"review_status": "fail", "confidence": "high"}
        )
    with pytest.raises(ValueError, match="note"):
        store.update(
            "icml:missing",
            {
                "review_status": "fail",
                "confidence": "high",
                "failure_codes": "missing_affiliation",
            },
        )


def test_review_is_saved_and_can_be_revised(tmp_path: Path) -> None:
    store = ReviewStore(queue(tmp_path, [record("missing")]))
    store.update(
        "icml:missing",
        {
            "review_status": "fail",
            "confidence": "high",
            "failure_codes": "missing_affiliation",
            "review_note": "The official PDF lists an institution.",
            "add_to_regression": "true",
        },
    )
    revised = store.update(
        "icml:missing",
        {
            "review_status": "defer",
            "confidence": "low",
            "failure_codes": "",
            "review_note": "Needs a second reviewer.",
            "add_to_regression": "false",
        },
    )
    assert revised["item"]["review_status"] == "defer"
    assert ReviewStore(store.path).snapshot()["items"][0]["review_status"] == "defer"


def test_expected_countries_accept_explicit_iso_codes_without_matching_prose() -> None:
    note = "UCLA (US), Saarland University (DE), and London (UK)."
    assert expected_country_codes(note) == ("DE", "GB", "US")
def test_review_store_accepts_utf8_bom(tmp_path: Path) -> None:
    review_queue = queue(tmp_path, [record("bom")])
    content = review_queue.read_text(encoding="utf-8")
    review_queue.write_text(content, encoding="utf-8-sig")

    assert ReviewStore(review_queue).snapshot()["meta"]["total"] == 1
