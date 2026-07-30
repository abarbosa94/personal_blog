import pytest

from conference_pipeline.coverage_sensitivity import (
    competition_ranks,
    sensitivity_rows,
)
from conference_pipeline.models import Affiliation, Paper, ReconciledPaper


def record(paper_id: str, venue: str, countries: tuple[str, ...]):
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


def test_competition_ranks_preserve_ties() -> None:
    assert competition_ranks({"BR": 2, "US": 4, "CN": 2}) == {
        "US": 1,
        "BR": 2,
        "CN": 2,
    }


def test_sensitivity_exposes_observed_mar_and_bounds() -> None:
    panel = {
        "BR": {"country_name": "Brazil", "comparison_group": "focus"},
        "US": {"country_name": "United States", "comparison_group": "frontier"},
    }
    rows, metadata = sensitivity_rows(
        [
            record("one", "venue", ("BR", "US")),
            record("two", "venue", ("US",)),
            record("missing", "venue", ()),
        ],
        panel,
    )
    brazil = next(row for row in rows if row["country_code"] == "BR")

    assert brazil["country_coverage"] == pytest.approx(2 / 3)
    assert brazil["observed_full_share"] == 0.5
    assert brazil["observed_fractional_share"] == 0.25
    assert brazil["mar_estimated_full_count"] == 1.5
    assert brazil["population_share_lower_bound"] == pytest.approx(1 / 3)
    assert brazil["population_share_upper_bound"] == pytest.approx(2 / 3)
    assert metadata["venues"]["venue"]["coverage_warning"] is True
