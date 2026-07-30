from conference_pipeline.conference_indicators import country_rows
from conference_pipeline.models import Affiliation, Paper, ReconciledPaper


def paper(paper_id: str, venue: str, countries: tuple[str, ...]) -> ReconciledPaper:
    return ReconciledPaper(
        Paper(
            paper_id,
            venue,
            2025,
            "main",
            paper_id,
            (),
            None,
            f"https://example.test/{paper_id}",
            "fixture",
        ),
        None,
        None,
        tuple(Affiliation("", code, code, None) for code in countries),
    )


def test_country_rows_produce_full_fractional_and_coverage_metrics() -> None:
    panel = {
        "BR": {"country_name": "Brazil", "comparison_group": "focus"},
        "US": {"country_name": "United States", "comparison_group": "frontier"},
    }

    rows, metadata = country_rows(
        [
            paper("one", "icml", ("BR", "US")),
            paper("two", "icml", ("BR",)),
            paper("three", "neurips", ()),
        ],
        panel,
    )

    all_br = next(
        row
        for row in rows
        if row["venue"] == "all" and row["country_code"] == "BR"
    )
    assert all_br["papers_full_count"] == 2
    assert all_br["papers_fractional_count"] == 1.5
    assert metadata["paper_universe"] == 3
    assert metadata["papers_without_country"] == 1
    assert metadata["country_coverage"] == 2 / 3
