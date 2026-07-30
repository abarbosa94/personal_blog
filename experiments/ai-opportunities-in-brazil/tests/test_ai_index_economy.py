from pathlib import Path

from conference_pipeline.ai_index_economy import indicator_rows, read_investment


def test_read_investment(tmp_path: Path) -> None:
    source = tmp_path / "investment.csv"
    source.write_text(
        "Total Investment (in Billions of U.S. Dollars),Country\n1.25,Brazil\n",
        encoding="utf-8",
    )
    assert read_investment(source) == {"Brazil": 1.25}


def test_unreported_top15_is_censored_not_zero() -> None:
    panel = {
        "BR": {"country_name": "Brazil", "comparison_group": "focus"},
        "US": {"country_name": "United States", "comparison_group": "frontier"},
    }
    rows = indicator_rows(
        {"United States": 10.0, "Canada": 2.0},
        panel,
        "newly_funded_ai_companies",
        "companies",
    )
    brazil = rows[0]
    assert brazil["value"] is None
    assert brazil["observation_status"] == "not_reported_top15"
    assert brazil["upper_bound_if_unreported"] == 2.0
    assert rows[1]["rank_among_reported_panel"] == 1
