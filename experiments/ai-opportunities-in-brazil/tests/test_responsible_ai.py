from conference_pipeline.responsible_ai import screen_title, validation_sample


def test_screen_title_maps_explicit_dimensions() -> None:
    dimensions, ambiguous = screen_title(
        "Differentially Private Learning with Equalized Odds"
    )
    assert dimensions == ("privacy_data_governance", "fairness")
    assert not ambiguous


def test_generic_bias_requires_context_review() -> None:
    dimensions, ambiguous = screen_title("Inductive Biases for Graph Learning")
    assert dimensions == ()
    assert ambiguous


def test_robustness_alone_is_not_automatically_responsible_ai() -> None:
    dimensions, ambiguous = screen_title("Robustness of Bayesian Estimators")
    assert dimensions == ()
    assert ambiguous


def test_validation_sample_includes_screen_negatives_from_each_venue() -> None:
    rows = [
        {
            "venue": venue, "paper_id": f"{venue}-{i}", "dimensions": "",
            "screen_status": "screen_negative",
        }
        for venue in ("aies", "facct")
        for i in range(6)
    ]
    sample = validation_sample(rows)
    assert sum(row["venue"] == "aies" for row in sample) == 5
    assert sum(row["venue"] == "facct" for row in sample) == 5
