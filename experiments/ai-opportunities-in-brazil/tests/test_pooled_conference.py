from conference_pipeline.models import Affiliation, Paper, ReconciledPaper
from conference_pipeline.pooled_conference import gate_metadata, pooled_rows


def record(venue: str, paper_id: str, countries: tuple[str, ...]) -> ReconciledPaper:
    return ReconciledPaper(
        Paper(paper_id, venue, 2025, "main", paper_id, (), None, "", "fixture"),
        None,
        None,
        tuple(Affiliation("", code, code, None) for code in countries),
    )


def test_equal_venue_pool_does_not_weight_large_venue_more() -> None:
    by_venue = {
        "large": [record("large", str(i), ("US",)) for i in range(9)],
        "small": [record("small", "x", ("BR",))],
    }
    panel = {
        "BR": {"country_name": "Brazil", "comparison_group": "panel"},
        "US": {"country_name": "United States", "comparison_group": "panel"},
    }

    rows = pooled_rows(by_venue, panel)
    equal = {
        row["country_code"]: row["estimate"]
        for row in rows
        if row["scenario"] == "all_venues"
        and row["estimator"] == "equal_venue_fractional_share"
    }
    weighted = {
        row["country_code"]: row["estimate"]
        for row in rows
        if row["scenario"] == "all_venues"
        and row["estimator"] == "paper_weighted_fractional_share"
    }

    assert equal == {"BR": 0.5, "US": 0.5}
    assert weighted == {"BR": 0.1, "US": 0.9}


def test_gate_metadata_uses_official_denominators() -> None:
    by_venue = {"iclr": [record("iclr", str(i), ("US",)) for i in range(9)]}

    metadata = gate_metadata(by_venue, {"iclr": 10})

    assert metadata["venues"]["iclr"]["country_coverage"] == 0.9
    assert metadata["all_venues_pass_coverage"]
