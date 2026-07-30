from pathlib import Path

from conference_pipeline.models import (
    Affiliation,
    Paper,
    ReconciledPaper,
    ReconciliationDiagnostic,
)
from conference_pipeline.v2_pipeline import (
    _merge_affiliations,
    augment_pdf_country_record,
    augment_records,
    complement_records,
    merge_evidence_records,
    missing_country_records,
    overlay_records,
    sample_records,
)


def test_merge_drops_acl_publisher_affiliation() -> None:
    merged = _merge_affiliations(
        (
            Affiliation(
                "https://ror.org/019sw1443",
                "Association for Computational Linguistics",
                "US",
                "other",
            ),
            Affiliation("", "Peking University", "CN", "education"),
        )
    )

    assert [(item.institution_name, item.country_code) for item in merged] == [
        ("Peking University", "CN")
    ]


def test_merge_resolved_affiliation_supersedes_annotated_unresolved_duplicate() -> None:
    merged = _merge_affiliations(
        (
            Affiliation("", "Google DeepMind.", None, None),
            Affiliation("", "Equal contribution 1Google DeepMind.", None, None),
            Affiliation("", "Google DeepMind.", "US", "company"),
        )
    )

    assert [(item.institution_name, item.country_code) for item in merged] == [
        ("Google DeepMind.", "US")
    ]


def test_merge_keeps_cleanest_unresolved_label_for_search() -> None:
    merged = _merge_affiliations(
        (
            Affiliation("", "Google DeepMind.", None, None),
            Affiliation("", "Equal contribution 1Google DeepMind.", None, None),
        )
    )

    assert [(item.institution_name, item.country_code) for item in merged] == [
        ("Google DeepMind.", None)
    ]


def records(count: int) -> list[ReconciledPaper]:
    return [
        ReconciledPaper(
            Paper(
                paper_id=str(index),
                venue_key="icml",
                year=2025,
                track="main",
                title=f"Paper {index}",
                authors=(),
                doi=None,
                official_url=f"https://example.test/{index}",
                source_kind="fixture",
            ),
            None,
            None,
            (Affiliation("", "Institution", "US", None),),
        )
        for index in range(count)
    ]


def test_v2_half_sample_is_reproducible_and_not_a_prefix() -> None:
    population = records(70)

    first = sample_records(population, 35, 20250727)
    second = sample_records(population, 35, 20250727)

    assert [item.paper.paper_id for item in first] == [
        item.paper.paper_id for item in second
    ]
    assert [item.paper.paper_id for item in first] != [
        item.paper.paper_id for item in population[:35]
    ]


def test_sample_and_complement_partition_the_population() -> None:
    population = records(70)
    sample = sample_records(population, 35, 20250727)
    complement = complement_records(population, 35, 20250727)
    sample_ids = {item.paper.paper_id for item in sample}
    complement_ids = {item.paper.paper_id for item in complement}
    assert len(sample_ids) == len(complement_ids) == 35
    assert not sample_ids & complement_ids
    assert sample_ids | complement_ids == {
        item.paper.paper_id for item in population
    }


def test_missing_country_and_overlay_support_targeted_census_updates() -> None:
    population = records(3)
    population[1] = ReconciledPaper(
        population[1].paper,
        None,
        None,
        (Affiliation("", "Unknown", None, None),),
    )
    assert [item.paper.paper_id for item in missing_country_records(population)] == [
        "1"
    ]
    repaired = ReconciledPaper(
        population[1].paper,
        None,
        "v9",
        (Affiliation("", "Repaired", "BR", None),),
    )
    merged = overlay_records(population, [repaired])
    assert [item.countries for item in merged] == [("US",), ("BR",), ("US",)]


def test_merge_evidence_unions_country_sources_without_losing_metadata() -> None:
    baseline = records(1)[0]
    update = ReconciledPaper(
        baseline.paper,
        "W123",
        "doi",
        (Affiliation("ror-br", "Brazil University", "BR", None),),
        (ReconciliationDiagnostic("openalex", "success"),),
    )

    merged = merge_evidence_records([baseline], [update])

    assert merged[0].openalex_id == "W123"
    assert set(merged[0].countries) == {"BR", "US"}
    assert merged[0].match_method == "doi"
    assert merged[0].diagnostics == update.diagnostics


def test_pdf_country_augment_adds_only_explicit_country_evidence(monkeypatch) -> None:
    class FakeHttp:
        def get_bytes(self, url):
            return b"fixture"

    monkeypatch.setattr(
        "conference_pipeline.v2_pipeline.PdfAffiliationExtractor.country_codes",
        lambda value: ("BR", "US"),
    )
    baseline = ReconciledPaper(
        Paper(
            "paper",
            "acl",
            2025,
            "main",
            "Paper",
            (),
            None,
            "https://example.test",
            "fixture",
            "https://example.test/paper.pdf",
        ),
        "W1",
        "doi",
        (),
    )
    result = augment_pdf_country_record(baseline, FakeHttp())
    assert result.countries == ("BR", "US")
    assert result.diagnostics[-1].stage == "pdf_country_census"


def test_pdf_fallback_affiliations_are_recomputed_instead_of_carried_forward(
    monkeypatch,
) -> None:
    baseline = ReconciledPaper(
        records(1)[0].paper,
        "https://openalex.org/W1",
        "title+affiliation-fallback",
        (Affiliation("wrong", "Wrong Institution", "MX", None),),
    )

    class FakeNlp:
        class Defaults:
            pass

    class FakeSpacy:
        @staticmethod
        def load(model):
            return FakeNlp()

    class FakeReconciler:
        def __init__(self, http, use_openalex=False):
            pass

        def reconcile(self, paper):
            return ReconciledPaper(
                paper,
                None,
                "pdf+ror",
                (Affiliation("right", "Right Institution", "US", None),),
            )

    class FakeHttp:
        def get_bytes(self, url):
            raise RuntimeError("no PDF needed")

    monkeypatch.setitem(__import__("sys").modules, "spacy", FakeSpacy)
    monkeypatch.setattr(
        "conference_pipeline.v2_pipeline.OpenAlexReconciler", FakeReconciler
    )
    result = augment_records([baseline], FakeHttp())
    assert set(result[0].countries) == {"US"}
