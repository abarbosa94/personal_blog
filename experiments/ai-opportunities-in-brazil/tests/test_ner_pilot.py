from conference_pipeline.ner_pilot import numbered_candidates


def test_numbered_candidates_split_compact_affiliation_list() -> None:
    text = (
        "†Joint project lead, 1Tsinghua University, 2Stanford University, "
        "3CMU, 4University of Pennsylvania, 5Tencent Hunyuan X, 6Fitten."
    )

    assert numbered_candidates(text) == (
        "Tsinghua University",
        "Stanford University",
        "CMU",
        "University of Pennsylvania",
        "Tencent Hunyuan X",
        "Fitten",
    )


def test_numbered_candidates_repair_wrapped_organization_name() -> None:
    text = "1North Carolina State Uni-\nversity 2Snap Inc."

    assert numbered_candidates(text) == (
        "North Carolina State University",
        "Snap Inc",
    )
