from __future__ import annotations

import json
import sys
from pathlib import Path

from pytest_bdd import given, parsers, scenarios, then, when

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conference_pipeline.enumerators import (  # noqa: E402
    AclAnthologyEnumerator,
    NeuripsEnumerator,
    PmlrEnumerator,
)
from conference_pipeline.cli import select_validation_sample  # noqa: E402
from conference_pipeline.manual_review import build_rows  # noqa: E402
from conference_pipeline.models import Affiliation, Paper, ReconciledPaper  # noqa: E402
from conference_pipeline.quality import build_quality_report  # noqa: E402
from conference_pipeline.reconcile import (  # noqa: E402
    CountryMentionExtractor,
    PdfAffiliationExtractor,
    fractional_country_weights,
    full_country_weights,
)

scenarios("features/conference_pipeline.feature")


@given("the ICML PMLR fixture", target_fixture="html_fixture")
def icml_fixture() -> str:
    return (ROOT / "tests/fixtures/pmlr_icml_sample.html").read_text(
        encoding="utf-8"
    )


@when("I enumerate the ICML main proceedings", target_fixture="papers")
def enumerate_icml(html_fixture: str) -> list[Paper]:
    return PmlrEnumerator.parse(
        html_fixture, "icml", 2025, "https://proceedings.mlr.press/v267/"
    )


@then(parsers.parse("{count:d} papers are enumerated"))
def paper_count(papers: list[Paper], count: int) -> None:
    assert len(papers) == count


@then(parsers.parse('the second ICML title is "{title}"'))
def second_icml_title(papers: list[Paper], title: str) -> None:
    assert papers[1].title == title


@then(parsers.parse("the second ICML paper has {count:d} authors"))
def second_icml_authors(papers: list[Paper], count: int) -> None:
    assert len(papers[1].authors) == count


@given("the ACL Anthology fixture with front matter", target_fixture="html_fixture")
def acl_fixture() -> str:
    return (ROOT / "tests/fixtures/acl_sample.html").read_text(encoding="utf-8")


@when("I enumerate ACL long papers", target_fixture="papers")
def enumerate_acl(html_fixture: str) -> list[Paper]:
    return AclAnthologyEnumerator.parse(
        html_fixture,
        "acl",
        2025,
        "2025.acl-long",
        "https://aclanthology.org/volumes/2025.acl-long/",
    )


@then("the ACL front matter is excluded")
def acl_front_matter_excluded(papers: list[Paper]) -> None:
    assert all(not paper.paper_id.endswith(".0") for paper in papers)


@then(parsers.parse("{count:d} ACL research papers are enumerated"))
def acl_count(papers: list[Paper], count: int) -> None:
    assert len(papers) == count


@then("every ACL paper has a derived DOI")
def acl_dois(papers: list[Paper]) -> None:
    assert all(paper.doi and paper.doi.startswith("10.18653/v1/") for paper in papers)


@given(
    "the NeurIPS proceedings fixture with two tracks", target_fixture="html_fixture"
)
def neurips_fixture() -> str:
    return (ROOT / "tests/fixtures/neurips_sample.html").read_text(encoding="utf-8")


@when("I enumerate NeurIPS papers", target_fixture="papers")
def enumerate_neurips(html_fixture: str) -> list[Paper]:
    return NeuripsEnumerator.parse(
        html_fixture, 2025, "https://proceedings.neurips.cc/paper_files/paper/2025"
    )


@then(parsers.parse("{count:d} NeurIPS papers are enumerated"))
def neurips_count(papers: list[Paper], count: int) -> None:
    assert len(papers) == count


@then(parsers.parse('the NeurIPS tracks are "{tracks}"'))
def neurips_tracks(papers: list[Paper], tracks: str) -> None:
    assert ",".join(paper.track for paper in papers) == tracks


@given(
    "a reconciled paper with affiliations in Brazil and the United States",
    target_fixture="reconciled_paper",
)
def international_paper() -> ReconciledPaper:
    paper = Paper(
        paper_id="international",
        venue_key="neurips",
        year=2025,
        track="conference",
        title="International Collaboration",
        authors=(),
        doi=None,
        official_url="https://example.org",
        source_kind="fixture",
    )
    return ReconciledPaper(
        paper=paper,
        openalex_id="W1",
        match_method="title",
        affiliations=(
            Affiliation("I1", "Brazil University", "BR", "education"),
            Affiliation("I2", "US Company", "US", "company"),
        ),
    )


@when("I calculate country weights", target_fixture="country_weights")
def country_weights(reconciled_paper: ReconciledPaper) -> dict[str, dict[str, float]]:
    return {
        "full": full_country_weights(reconciled_paper),
        "fractional": fractional_country_weights(reconciled_paper),
    }


@then("full counting assigns 1 to each country")
def full_weights(country_weights: dict[str, dict[str, float]]) -> None:
    assert country_weights["full"] == {"BR": 1.0, "US": 1.0}


@then("fractional counting assigns 0.5 to each country")
def fractional_weights(country_weights: dict[str, dict[str, float]]) -> None:
    assert country_weights["fractional"] == {"BR": 0.5, "US": 0.5}


@given(
    "10 enumerated papers with countries available for 2 papers",
    target_fixture="quality_input",
)
def low_coverage_sample() -> tuple[list[ReconciledPaper], int]:
    records: list[ReconciledPaper] = []
    for index in range(10):
        affiliations = (
            (Affiliation(f"I{index}", "Institution", "BR", "education"),)
            if index < 2
            else ()
        )
        records.append(
            ReconciledPaper(
                paper=Paper(
                    paper_id=str(index),
                    venue_key="acl",
                    year=2025,
                    track="main",
                    title=f"Paper {index}",
                    authors=(),
                    doi=None,
                    official_url="https://example.org",
                    source_kind="fixture",
                ),
                openalex_id=f"W{index}",
                match_method="title",
                affiliations=affiliations,
            )
        )
    return records, 10


@when("I evaluate the publication quality gates", target_fixture="quality_report")
def quality_report(quality_input: tuple[list[ReconciledPaper], int]):
    records, official_total = quality_input
    return build_quality_report(records, official_total)


@then("country coverage is 0.2")
def country_coverage(quality_report) -> None:
    assert quality_report.country_coverage == 0.2


@then("the sample does not pass")
def sample_fails(quality_report) -> None:
    assert not quality_report.passes()


@given(
    "PDF front matter with a university and a company",
    target_fixture="pdf_text",
)
def pdf_front_matter() -> str:
    return """Paper title
University of Example, Brazil
Example AI Company Ltd., United Kingdom
Abstract
Summary of the paper.
1. Introduction
Another University appears only in the body.
"""


@when("I extract affiliation candidates", target_fixture="affiliation_candidates")
def extract_affiliation_candidates(pdf_text: str) -> tuple[str, ...]:
    return PdfAffiliationExtractor.candidate_lines(pdf_text)


@then(parsers.parse("the university and company are preserved as {count:d} candidates"))
def candidate_count(affiliation_candidates: tuple[str, ...], count: int) -> None:
    assert len(affiliation_candidates) == count


@then("body text after the introduction is excluded")
def body_excluded(affiliation_candidates: tuple[str, ...]) -> None:
    assert all("Another University" not in value for value in affiliation_candidates)


@given("100 officially enumerated papers", target_fixture="formal_population")
def formal_population() -> list[Paper]:
    return [
        Paper(
            paper_id=str(index),
            venue_key="neurips",
            year=2025,
            track="conference",
            title=f"Paper {index}",
            authors=(),
            doi=None,
            official_url=f"https://example.org/{index}",
            source_kind="fixture",
        )
        for index in range(100)
    ]


@when(
    parsers.parse("I select 50 papers twice with seed {seed:d}"),
    target_fixture="formal_samples",
)
def select_formal_samples(
    formal_population: list[Paper], seed: int
) -> tuple[list[Paper], list[Paper]]:
    return (
        select_validation_sample(formal_population, 50, seed),
        select_validation_sample(formal_population, 50, seed),
    )


@then("both formal samples contain the same paper identifiers")
def reproducible_formal_sample(
    formal_samples: tuple[list[Paper], list[Paper]],
) -> None:
    first, second = formal_samples
    assert [paper.paper_id for paper in first] == [
        paper.paper_id for paper in second
    ]


@then("the formal sample is not simply the first 50 papers")
def formal_sample_is_random(
    formal_samples: tuple[list[Paper], list[Paper]],
) -> None:
    first, _ = formal_samples
    assert [paper.paper_id for paper in first] != [str(index) for index in range(50)]


@given(
    "a formal sample with an automatic failure and a clean control",
    target_fixture="manual_review_sources",
)
def manual_review_sources(tmp_path: Path) -> list[Path]:
    def paper(paper_id: str) -> dict:
        return {
            "paper_id": paper_id,
            "venue_key": "icml",
            "year": 2025,
            "track": "main",
            "title": paper_id.title(),
            "authors": ["A"],
            "doi": None,
            "official_url": f"https://example.test/{paper_id}",
            "source_kind": "fixture",
            "pdf_url": None,
        }

    records = [
        {
            "paper": paper("clean"),
            "openalex_id": "https://openalex.org/W1",
            "match_method": "title",
            "affiliations": [
                {
                    "institution_id": "I1",
                    "institution_name": "University",
                    "country_code": "BR",
                    "institution_type": "education",
                }
            ],
        },
        {
            "paper": paper("failure"),
            "openalex_id": None,
            "match_method": None,
            "affiliations": [],
        },
    ]
    source = tmp_path / "formal.jsonl"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return [source]


@when("I build the manual review queue", target_fixture="manual_review_queue")
def build_manual_review_queue(
    manual_review_sources: list[Path],
) -> list[dict[str, str]]:
    return build_rows(manual_review_sources)


@then("the automatic failure appears before the clean control")
def failure_is_prioritized(manual_review_queue: list[dict[str, str]]) -> None:
    assert [row["paper_id"] for row in manual_review_queue] == ["failure", "clean"]


@then("both formal papers remain in the review queue")
def formal_sample_is_preserved(manual_review_queue: list[dict[str, str]]) -> None:
    assert len(manual_review_queue) == 2


@given(
    "a wrapped multinational affiliation block",
    target_fixture="wrapped_affiliation_block",
)
def wrapped_affiliation_block() -> str:
    return """1 University College London, United King-
dom 2 Institut Polytechnique de Paris, France 3 Stanford University, USA.
Correspondence to: author@example.org
Proceedings of ICML, Vancouver, Canada.
"""


@when(
    "I extract explicit affiliation countries",
    target_fixture="explicit_affiliation_countries",
)
def extract_explicit_affiliation_countries(
    wrapped_affiliation_block: str,
) -> tuple[str, ...]:
    region = PdfAffiliationExtractor.affiliation_region(wrapped_affiliation_block)
    return CountryMentionExtractor.country_codes(region)


@then(parsers.parse('the countries are "{expected}"'))
def explicit_countries_match(
    explicit_affiliation_countries: tuple[str, ...], expected: str
) -> None:
    assert set(explicit_affiliation_countries) == set(expected.split(","))


@then("the conference location is not an affiliation country")
def conference_country_is_excluded(
    explicit_affiliation_countries: tuple[str, ...],
) -> None:
    assert "CA" not in explicit_affiliation_countries
