from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conference_pipeline.enumerators import (  # noqa: E402
    CrossrefAaaaiEnumerator,
    IclrOpenReviewEnumerator,
    KddAdsEnumerator,
)


def test_crossref_filters_container_and_volume() -> None:
    items = [
        {
            "DOI": "10.1609/aaai.v39i1.123",
            "title": ["A Valid Paper"],
            "container-title": [
                "Proceedings of the AAAI Conference on Artificial Intelligence"
            ],
            "volume": "39",
            "author": [{"given": "Ana", "family": "Silva"}],
            "URL": "https://doi.org/10.1609/aaai.v39i1.123",
        },
        {
            "DOI": "10.1609/other.1",
            "title": ["Wrong Container"],
            "container-title": ["Another Conference"],
            "volume": "39",
        },
    ]

    papers = CrossrefAaaaiEnumerator.from_items(items, 2025, "39")

    assert len(papers) == 1
    assert papers[0].authors == ("Ana Silva",)


def test_crossref_same_cursor_can_advance_until_an_empty_page() -> None:
    valid_item = {
        "DOI": "10.1609/aaai.v39i1.123",
        "title": ["A Valid Paper"],
        "container-title": [
            "Proceedings of the AAAI Conference on Artificial Intelligence"
        ],
        "volume": "39",
    }

    class FakeHttp:
        def __init__(self) -> None:
            self.calls = 0

        def get_json(self, url: str) -> dict:
            self.calls += 1
            items = [valid_item] if self.calls <= 2 else []
            return {"message": {"items": items, "next-cursor": "same-token"}}

    http = FakeHttp()
    papers = CrossrefAaaaiEnumerator(http).enumerate(2025, "39")

    assert http.calls == 3
    assert len(papers) == 2


def test_iclr_openreview_notes_map_to_unique_official_papers() -> None:
    class Note:
        id = "PwxYoMvmvy"
        content = {
            "title": {"value": "  Beyond Random Masking  "},
            "authors": {"value": ["Ada Author", "Bob Builder"]},
            "venue": {"value": "ICLR 2025 Poster"},
        }

    papers = IclrOpenReviewEnumerator.from_notes([Note(), Note()], 2025)

    assert len(papers) == 1
    assert papers[0].paper_id == "PwxYoMvmvy"
    assert papers[0].venue_key == "iclr"
    assert papers[0].track == "ICLR 2025 Poster"
    assert papers[0].authors == ("Ada Author", "Bob Builder")
    assert papers[0].official_url == "https://openreview.net/forum?id=PwxYoMvmvy"
    assert papers[0].pdf_url == "https://openreview.net/pdf?id=PwxYoMvmvy"
    assert papers[0].source_kind == "official_openreview"


def test_kdd_ads_parser_preserves_cycles_and_doi_ids() -> None:
    html = """
    <table>
      <tr><td><strong>February Paper</strong><br>
      DOI: https://doi.org/10.1145/3711896.3737183</td></tr>
      <tr><td>Ada Author (Company); Bob Builder (University)</td></tr>
    </table>
    <table>
      <tr><td><strong>August Paper</strong><br>
      DOI: https://doi.org/10.1145/3690624.3709406</td></tr>
      <tr><td>Carol Coder (Laboratory)</td></tr>
    </table>
    """

    papers = KddAdsEnumerator.parse(html, 2025, "https://kdd.example/papers")

    assert [paper.track for paper in papers] == [
        "ads-february-cycle",
        "ads-august-cycle",
    ]
    assert papers[0].paper_id == "10.1145/3711896.3737183"
    assert papers[0].authors == ("Ada Author", "Bob Builder")
    assert papers[1].official_url == "https://doi.org/10.1145/3690624.3709406"

    parser = __import__(
        "conference_pipeline.enumerators", fromlist=["_KddAdsParser"]
    )._KddAdsParser(2025, "https://kdd.example/papers")
    parser.feed(html)
    assert parser.affiliations_by_doi["10.1145/3711896.3737183"] == (
        "Company",
        "University",
    )
