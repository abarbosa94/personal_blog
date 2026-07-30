from conference_pipeline.openreview_profile_affiliations import (
    active_institution_names,
)


def test_active_institution_names_filter_by_paper_year() -> None:
    history = [
        {"start": 2024, "institution": {"name": "Current University"}},
        {"start": 2020, "end": 2023, "institution": {"name": "Old University"}},
        {"start": 2026, "institution": {"name": "Future University"}},
    ]

    assert active_institution_names(history, 2025) == ("Current University",)
