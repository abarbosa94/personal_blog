from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conference_pipeline.models import Affiliation, Paper  # noqa: E402
from conference_pipeline.quality import build_quality_report  # noqa: E402
from conference_pipeline.reconcile import (  # noqa: E402
    CountryMentionExtractor,
    GrobidAffiliationExtractor,
    OpenAlexReconciler,
    PdfAffiliationExtractor,
    RorAffiliationResolver,
)


def sample_paper() -> Paper:
    return Paper(
        paper_id="sample",
        venue_key="acl",
        year=2025,
        track="main",
        title="A Sample Paper",
        authors=("Ana Silva",),
        doi="10.1/sample",
        official_url="https://example.org/sample",
        source_kind="fixture",
    )


def test_failed_title_lookup_becomes_an_unmatched_record() -> None:
    class FailingHttp:
        def get_json(self, url: str) -> dict:
            raise RuntimeError("temporary API failure")

    paper = Paper(
        paper_id="no-doi",
        venue_key="icml",
        year=2025,
        track="main",
        title="A Paper Without a DOI",
        authors=(),
        doi=None,
        official_url="https://example.org",
        source_kind="fixture",
    )

    reconciled = OpenAlexReconciler(FailingHttp()).reconcile(paper)

    assert reconciled.openalex_id is None
    assert reconciled.affiliations == ()


def test_title_lookup_uses_exact_search_for_question_marks() -> None:
    class FakeHttp:
        url = ""

        def get_json(self, url: str) -> dict:
            self.url = url
            return {"results": []}

    http = FakeHttp()
    diagnostics = []

    OpenAlexReconciler(http)._get_by_title(
        "Can Diffusion Models Disentangle? A Theoretical Perspective",
        diagnostics,
    )

    assert "search.exact=" in http.url
    assert "%3F" in http.url
    assert diagnostics[0].outcome == "not_found"


def test_title_lookup_tolerates_candidates_with_null_titles() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            return {
                "results": [
                    {"id": "W-null", "title": None},
                    {"id": "W-match", "title": "A Sample Paper"},
                ]
            }

    diagnostics = []

    result = OpenAlexReconciler(FakeHttp())._get_by_title(
        "A Sample Paper",
        diagnostics,
    )

    assert result["id"] == "W-match"
    assert diagnostics[0].outcome == "success"


def test_extracts_affiliations_from_grobid_tei() -> None:
    xml = (ROOT / "tests/fixtures/grobid_affiliations.xml").read_text(
        encoding="utf-8"
    )

    affiliations = GrobidAffiliationExtractor.parse(xml)

    assert len(affiliations) == 2
    assert {
        (affiliation.institution_name, affiliation.country_code)
        for affiliation in affiliations
    } == {("University One", "BR"), ("Company Two", "US")}


def test_quality_thresholds_are_explicit() -> None:
    work = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {
                "institutions": [
                    {
                        "id": "I1",
                        "display_name": "University",
                        "country_code": "BR",
                        "type": "education",
                    }
                ]
            }
        ],
    }
    reconciled = OpenAlexReconciler.from_work(sample_paper(), work, "doi")

    report = build_quality_report([reconciled], official_total=1)

    assert report.passes()
    assert report.country_coverage == 1.0


def test_pdf_front_matter_yields_conservative_affiliation_candidates() -> None:
    text = """A Paper Title
Ana Silva, Bob Smith
University of Example, São Paulo, Brazil
Google DeepMind, London, United Kingdom
Abstract
Some abstract text.
1. Introduction
This research studies a university dataset but is not an affiliation.
"""

    values = PdfAffiliationExtractor.candidate_lines(text)

    assert values == (
        "University of Example, São Paulo, Brazil",
        "Google DeepMind, London, United Kingdom",
    )


def test_ror_resolver_accepts_only_the_chosen_result() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            return {
                "items": [
                    {
                        "chosen": True,
                        "organization": {
                            "id": "https://ror.org/123",
                            "names": [
                                {
                                    "value": "Example University",
                                    "types": ["ror_display"],
                                }
                            ],
                            "locations": [
                                {"geonames_details": {"country_code": "BR"}}
                            ],
                            "types": ["education"],
                        },
                    }
                ]
            }

    affiliation = RorAffiliationResolver(FakeHttp()).resolve(
        "Dept., Example University, Brazil"
    )

    assert affiliation is not None
    assert affiliation.institution_id == "https://ror.org/123"
    assert affiliation.country_code == "BR"


def test_ror_resolver_rejects_lexically_unsupported_chosen_result() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            return {
                "items": [
                    {
                        "chosen": True,
                        "organization": {
                            "id": "https://ror.org/wrong",
                            "names": [
                                {
                                    "value": "New England Baptist Hospital",
                                    "types": ["ror_display"],
                                }
                            ],
                            "locations": [
                                {"geonames_details": {"country_code": "US"}}
                            ],
                            "types": ["healthcare"],
                        },
                    }
                ]
            }

    assert (
        RorAffiliationResolver(FakeHttp()).resolve(
            "Hong Kong Baptist University"
        )
        is None
    )


def test_reviewed_organization_aliases_resolve_without_ror() -> None:
    class NoHttp:
        def get_json(self, url: str) -> dict:
            raise AssertionError("known aliases should not call ROR")

    resolver = RorAffiliationResolver(NoHttp())
    assert resolver.resolve("Huawei Noah's Ark Lab").country_code == "CN"
    assert resolver.resolve("Origin Research").country_code == "US"
    assert resolver.resolve("EPFL").country_code == "CH"
    assert resolver.resolve("Salesforce").country_code == "US"
    assert resolver.resolve("CeRAI, IIT Madras").country_code == "IN"
    assert resolver.resolve("WSAI, IIT Madras").country_code == "IN"
    assert resolver.resolve("KAIST AI").country_code == "KR"
    assert resolver.resolve("NEC Laboratories Europe").country_code == "DE"
    assert resolver.resolve("IIIT Hyderabad").country_code == "IN"
    assert resolver.resolve("MBZUAI").country_code == "AE"
    assert resolver.resolve("LIACC, FEUP, University of Porto").country_code == "PT"
    assert resolver.resolve("Universitas Indonesia").country_code == "ID"
    assert resolver.resolve("Skoltech").country_code == "RU"
    assert resolver.resolve("AIRI").country_code == "RU"


def test_aaai_pdf_is_discovered_from_article_page() -> None:
    class FakeHttp:
        def get_text(self, url: str) -> str:
            assert url.endswith("/article/view/32241")
            return (
                '<a class="obj_galley_link pdf" '
                'href="/index.php/AAAI/article/download/32241/34396">PDF</a>'
            )

    paper = Paper(
        paper_id="10.1609/aaai.v39i3.32241",
        venue_key="aaai",
        year=2025,
        track="main",
        title="EchoMimic",
        authors=(),
        doi="10.1609/aaai.v39i3.32241",
        official_url="https://doi.org/10.1609/aaai.v39i3.32241",
        source_kind="fixture",
    )
    diagnostics = []

    discovered = OpenAlexReconciler(FakeHttp())._with_discovered_pdf(
        paper, diagnostics
    )

    assert discovered.official_url.endswith("/article/view/32241")
    assert discovered.pdf_url == (
        "https://ojs.aaai.org/index.php/AAAI/article/download/32241/34396"
    )
    assert diagnostics[0].stage == "aaai_pdf_discovery"
    assert diagnostics[0].outcome == "success"


def test_authors_are_discovered_from_official_citation_metadata() -> None:
    class FakeHttp:
        def get_text(self, url: str) -> str:
            return """
            <meta name="citation_author" content="Nicola Cancedda">
            <meta content="Tara Fowler" name="citation_author">
            """

    paper = sample_paper()
    paper = Paper(**{**paper.to_dict(), "authors": ()})
    diagnostics = []

    discovered = OpenAlexReconciler(FakeHttp())._with_discovered_authors(
        paper, diagnostics
    )

    assert discovered.authors == ("Nicola Cancedda", "Tara Fowler")
    assert diagnostics[0].outcome == "success"


def test_authors_are_discovered_from_unquoted_citation_metadata() -> None:
    class FakeHttp:
        def get_text(self, url: str) -> str:
            return """
            <meta content="Nicola Cancedda" name=citation_author>
            <meta content="Tara Fowler" name=citation_author>
            """

    paper = sample_paper()
    paper = Paper(**{**paper.to_dict(), "authors": ()})

    discovered = OpenAlexReconciler(FakeHttp())._with_discovered_authors(
        paper, []
    )

    assert discovered.authors == ("Nicola Cancedda", "Tara Fowler")


def test_ambiguous_bare_organization_names_do_not_use_ror() -> None:
    class NoHttp:
        def get_json(self, url: str) -> dict:
            raise AssertionError("ambiguous bare names should not call ROR")

    resolver = RorAffiliationResolver(NoHttp())

    assert resolver.resolve("Institute for Machine Learning") is None
    assert resolver.resolve("Google DeepMind") is None
    assert resolver.resolve("Centre for Artificial Intelligence and Robotics") is None


def test_named_international_campus_uses_its_local_country() -> None:
    resolved = RorAffiliationResolver.known_affiliation(
        "Macquarie University, New York University Abu Dhabi"
    )

    assert resolved is not None
    assert resolved.institution_name == "New York University Abu Dhabi"
    assert resolved.country_code == "AE"


def test_exact_ror_alias_is_used_when_affiliation_endpoint_declines() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            if "affiliation=" in url:
                return {"items": []}
            return {
                "items": [
                    {
                        "id": "https://ror.org/00d9ah105",
                        "names": [
                            {"value": "UC Merced", "types": ["alias"]},
                            {
                                "value": "University of California, Merced",
                                "types": ["ror_display"],
                            },
                        ],
                        "locations": [
                            {"geonames_details": {"country_code": "US"}}
                        ],
                        "types": ["education"],
                    }
                ]
            }

    resolved = RorAffiliationResolver(FakeHttp()).resolve("UC Merced")

    assert resolved is not None
    assert resolved.institution_name == "University of California, Merced"
    assert resolved.country_code == "US"


def test_ror_query_fallback_rejects_non_exact_name() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            if "affiliation=" in url:
                return {"items": []}
            return {
                "items": [
                    {
                        "id": "https://ror.org/wrong",
                        "names": [
                            {
                                "value": "American Friends of Tel Aviv University",
                                "types": ["ror_display"],
                            }
                        ],
                        "locations": [
                            {"geonames_details": {"country_code": "US"}}
                        ],
                        "types": ["education"],
                    }
                ]
            }

    assert RorAffiliationResolver(FakeHttp()).resolve("Tel Aviv University") is None


def test_ror_query_fallback_rejects_bare_acronym() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            return {"items": []}

    assert RorAffiliationResolver(FakeHttp()).resolve("UCLA") is None


def test_ror_query_fallback_rejects_ambiguous_exact_alias() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            if "affiliation=" in url:
                return {"items": []}
            return {
                "items": [
                    {
                        "id": f"https://ror.org/{index}",
                        "names": [
                            {
                                "value": "Vector Institute",
                                "types": ["alias"],
                            }
                        ],
                        "locations": [
                            {"geonames_details": {"country_code": country}}
                        ],
                        "types": ["facility"],
                    }
                    for index, country in (("one", "CA"), ("two", "RU"))
                ]
            }

    assert RorAffiliationResolver(FakeHttp()).resolve("Vector Institute") is None


def test_author_branch_registry_requires_author_and_organization() -> None:
    nvidia = Affiliation("", "NVIDIA AI Technology Center", None, None)

    resolved = RorAffiliationResolver.author_branch_affiliations(
        ("Simon See",),
        (nvidia,),
    )

    assert [(item.country_code, item.institution_name) for item in resolved] == [
        ("SG", "NVIDIA AI Technology Center, Singapore")
    ]
    assert not RorAffiliationResolver.author_branch_affiliations(
        ("Another Author",),
        (nvidia,),
    )
    assert not RorAffiliationResolver.author_branch_affiliations(
        ("Simon See",),
        (Affiliation("", "NVIDIA", None, None),),
    )


def test_paper_branch_registry_supports_missing_venue_authors() -> None:
    resolved = RorAffiliationResolver.author_branch_affiliations(
        (),
        (Affiliation("", "JPMorgan AI Research", None, None),),
        (),
        "CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal "
        "Text Generation",
    )

    assert [(item.country_code, item.institution_name) for item in resolved] == [
        ("US", "JPMorgan AI Research, New York")
    ]


def test_reviewed_author_branch_registry_preserves_branch_country() -> None:
    resolved = RorAffiliationResolver.author_branch_affiliations(
        ("Mingbao Lin",),
        (Affiliation("", "Skywork AI", None, None),),
    )

    assert [(item.country_code, item.institution_name) for item in resolved] == [
        ("SG", "Skywork AI, Singapore")
    ]


def test_reviewed_author_and_exact_paper_can_restore_lost_branch_line() -> None:
    resolved = RorAffiliationResolver.author_branch_affiliations(
        ("Alan Schelten", "Tara Fowler", "Nicola Cancedda"),
        (),
        paper_title="HalluLens: LLM Hallucination Benchmark",
    )

    assert {item.country_code for item in resolved} == {"GB", "US"}


def test_openalex_affiliation_requires_distinctive_pdf_support() -> None:
    region = """
    Key Laboratory, Ministry of Education of China, Xiamen University, P.R. China
    Skywork AI
    """

    assert RorAffiliationResolver.pdf_supports_affiliation(
        Affiliation("", "Xiamen University", "CN", "education"), region
    )
    assert not RorAffiliationResolver.pdf_supports_affiliation(
        Affiliation("", "Skyworks Solutions (United States)", "US", "company"),
        region,
    )
    assert not RorAffiliationResolver.pdf_supports_affiliation(
        Affiliation(
            "",
            "Ministry of Public Security of the People's Republic of China",
            "CN",
            "government",
        ),
        region,
    )
    assert RorAffiliationResolver.pdf_supports_affiliation(
        Affiliation("", "The University of Queensland", "AU", "education"),
        "University of Queensland, Australia",
    )


def test_generic_public_health_match_is_rejected() -> None:
    assert not RorAffiliationResolver._lexically_supported(
        "Department of Biostatistics, Johns Hopkins Bloomberg School of Public Health",
        "Hanoi School of Public Health",
    )


def test_ambiguous_generic_unit_rejects_ror_match_inside_richer_text() -> None:
    class FakeHttp:
        def get_json(self, url: str) -> dict:
            return {
                "items": [
                    {
                        "chosen": True,
                        "organization": {
                            "id": "https://ror.org/wrong",
                            "names": [
                                {
                                    "value": "Institute for Machine Learning",
                                    "types": ["ror_display"],
                                }
                            ],
                            "locations": [
                                {"geonames_details": {"country_code": "CH"}}
                            ],
                            "types": ["facility"],
                        },
                    }
                ]
            }

    assert (
        RorAffiliationResolver(FakeHttp()).resolve(
            "ELLIS Unit, Institute for Machine Learning, JKU Linz"
        )
        is None
    )


def test_unnumbered_known_company_is_found_in_front_matter() -> None:
    text = """Playmate: Flexible Control
Authors
Guangzhou Quwan Network Technology
Abstract
Paper body.
Correspondence to: author@example.org
"""
    assert PdfAffiliationExtractor.candidate_lines(text) == (
        "Guangzhou Quwan Network Technology",
    )


def test_unnumbered_salesforce_affiliation_is_found_in_front_matter() -> None:
    text = """Paper title
Authors
Salesforce
Abstract
Paper body.
"""
    assert PdfAffiliationExtractor.candidate_lines(text) == ("Salesforce",)


def test_acl_publisher_is_not_an_author_affiliation() -> None:
    text = """Proceedings of the Association for Computational Linguistics
Paper title
Authors
1Peking University
Abstract
"""

    assert PdfAffiliationExtractor.candidate_lines(text) == (
        "Peking University",
    )


def test_compact_numbered_affiliations_are_split_before_resolution() -> None:
    text = """Paper title
Authors
1EPFL2Northeastern University
Abstract
"""
    values = PdfAffiliationExtractor.candidate_lines(text)
    assert "EPFL" in values
    assert "Northeastern University" in values


def test_suffix_numbered_affiliations_with_spaces_are_split() -> None:
    text = """Paper title
Authors
Carnegie Mellon University1 KAIST AI2 University of Washington3
NEC Laboratories Europe4 Ss. Cyril and Methodius University of Skopje5
Abstract
"""

    values = PdfAffiliationExtractor.candidate_lines(text)

    assert "KAIST AI" in values
    assert "University of Washington" in values
    assert "Ss. Cyril and Methodius University of Skopje" in values


def test_country_mentions_survive_pdf_line_wrapping() -> None:
    text = """Paper title
Authors
1 University College London, London, United King-
dom 2 CEREA, Institut Polytechnique de Paris, France.
Correspondence to: author@example.org
Proceedings of ICML, Vancouver, Canada.
"""

    region = PdfAffiliationExtractor.affiliation_region(text)
    countries = CountryMentionExtractor.country_codes(region)

    assert countries == ("FR", "GB")
    assert "Canada" not in region


def test_affiliation_region_stops_before_abstract_country_mentions() -> None:
    text = """Paper title
Authors
1 University of Melbourne
2 King’s College London
Abstract
We compare English learners with Japanese and Korean L1 backgrounds.
1 Introduction
"""

    region = PdfAffiliationExtractor.affiliation_region(text)

    assert "University of Melbourne" in region
    assert "Japanese" not in region
    assert CountryMentionExtractor.country_codes(region) == ()


def test_affiliations_are_found_when_pdf_columns_place_them_after_introduction() -> None:
    text = """Paper title
Abstract
Abstract text.
1. Introduction
Introduction text from the left column.
*Equal contribution. 1Department of Computer Science,
University of California, United States
2Saarland University, Germany. Correspondence to: author@example.org
Proceedings, Vancouver, Canada.
"""

    region = PdfAffiliationExtractor.affiliation_region(text)

    assert "University of California" in region
    assert "Introduction text" not in region
    assert CountryMentionExtractor.country_codes(region) == ("DE", "US")


def test_country_aliases_do_not_match_inside_other_words() -> None:
    assert CountryMentionExtractor.country_codes("A campus usage study") == ()
    assert CountryMentionExtractor.country_codes("KAIST, South Korea") == ("KR",)
    assert CountryMentionExtractor.country_codes(
        "Indian Institute of Science, India"
    ) == ("IN",)
    assert CountryMentionExtractor.country_codes(
        "The Hong Kong University of Science and Technology (Guangzhou), China"
    ) == ("CN",)
    assert CountryMentionExtractor.country_codes(
        "KAUST, Saudi Arabia; Tel Aviv, Israel; Amsterdam, Netherlands"
    ) == ("IL", "NL", "SA")


def test_reviewed_affiliation_regression_cases() -> None:
    cases = json.loads(
        (ROOT / "tests/fixtures/reviewed_affiliation_cases.json").read_text(
            encoding="utf-8"
        )
    )
    for case in cases:
        region = PdfAffiliationExtractor.affiliation_region(case["text"])
        assert set(CountryMentionExtractor.country_codes(region)) == set(
            case["expected"]
        ), case["name"]
