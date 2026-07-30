from datetime import date

from conference_pipeline.epoch_models import country_sectors, derive_rows


def test_country_sectors_aligns_parallel_organization_fields() -> None:
    row = {
        "Country (of organization)": "Brazil,Brazil,United States of America",
        "Organization categorization": "Academia,Industry,Industry",
    }
    assert country_sectors(row) == {"BR": "mixed", "US": "industry"}


def test_derive_rows_filters_period_and_notability() -> None:
    records = [
        {
            "Publication date": "2025-03-01",
            "Notability criteria": "SOTA improvement",
            "Country (of organization)": "Brazil",
            "Organization categorization": "Academia",
        },
        {
            "Publication date": "2025-04-01",
            "Notability criteria": "",
            "Country (of organization)": "Brazil",
            "Organization categorization": "Industry",
        },
        {
            "Publication date": "2026-01-01",
            "Notability criteria": "Significant use",
            "Country (of organization)": "Brazil",
            "Organization categorization": "Industry",
        },
    ]
    panel = {"BR": {"country_name": "Brazil", "comparison_group": "focus"}}
    rows, metadata = derive_rows(
        records, panel, date(2025, 1, 1), date(2025, 12, 31)
    )
    academia = next(row for row in rows if row["sector"] == "academia")
    assert academia["model_count"] == 1
    assert academia["country_total"] == 1
    assert metadata["eligible_notable_models"] == 1
