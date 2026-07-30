from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conference_pipeline.review_analysis import (  # noqa: E402
    expected_country_codes,
    read_queue,
)


def test_expected_countries_ignore_wrong_pipeline_countries_in_note() -> None:
    note = """- Expected countries: Austria and United Kingdom
- Pipeline countries: Switzerland and United Kingdom
Expected country set: AT, GB—not CH, GB.
"""

    assert expected_country_codes(note) == ("AT", "GB")


def test_expected_country_falls_back_to_full_note_without_structured_line() -> None:
    assert expected_country_codes("Google New York means a clear US institution") == (
        "US",
    )


def test_expected_country_stops_before_same_line_pipeline_result() -> None:
    note = "Expected CN and SG; pipeline CN SG and US."

    assert expected_country_codes(note) == ("CN", "SG")


def test_expected_country_parser_accepts_extended_iso_set() -> None:
    note = "Expected AE HK ID KE MA MK NG PT RO RU RW SE ZA."

    assert expected_country_codes(note) == (
        "AE",
        "HK",
        "ID",
        "KE",
        "MA",
        "MK",
        "NG",
        "PT",
        "RO",
        "RU",
        "RW",
        "SE",
        "ZA",
    )


def test_review_queue_reader_accepts_utf8_bom(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    queue.write_text(
        "review_id,review_status\npaper-1,pass\n",
        encoding="utf-8-sig",
    )

    assert read_queue(queue) == [
        {"review_id": "paper-1", "review_status": "pass"}
    ]
