from __future__ import annotations

from pathlib import Path

import anyio

from conference_pipeline.affiliation_search_mcp import create_server
from conference_pipeline.search_evidence import (
    BraveSearchProvider,
    EvidenceSearch,
    SearchCache,
    TavilySearchProvider,
    evaluate_affiliation_evidence,
    source_rank,
)


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        count: int,
        freshness: str | None = None,
        country: str = "",
        search_lang: str = "en",
    ) -> dict:
        self.calls.append(query)
        return {
            "web": {
                "results": [
                    {
                        "title": "Mingbao Lin",
                        "url": "https://example.edu/profile",
                        "description": "Skywork AI, Singapore, Apr. 2024-May 2025",
                    }
                ]
            }
        }


def test_web_search_is_cached(tmp_path: Path) -> None:
    provider = FakeProvider()
    cache = SearchCache(tmp_path / "search.sqlite3")
    search = EvidenceSearch(provider, cache)

    first = search.web_search("test query")
    second = search.web_search("test query")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert provider.calls == ["test query"]
    assert second["results"][0]["url"] == "https://example.edu/profile"


def test_affiliation_search_is_date_aware_and_deduplicates(tmp_path: Path) -> None:
    provider = FakeProvider()
    search = EvidenceSearch(provider, SearchCache(tmp_path / "search.sqlite3"))

    result = search.affiliation_evidence(
        "Mingbao Lin", "Skywork AI", "2025-04"
    )

    assert len(provider.calls) == 3
    assert any("2025" in query for query in result["queries"])
    assert len(result["results"]) == 1
    assert result["decision"] == "evidence_only"


def test_evidence_decision_requires_linkage_date_and_corroboration() -> None:
    payload = {
        "author": "Meera Hahn",
        "organization": "Google DeepMind",
        "paper_date": "2025",
        "results": [
            {
                "title": "Meera Hahn — Google DeepMind",
                "url": "https://research.google/people/meera-hahn",
                "snippet": "Meera Hahn, Google DeepMind, Atlanta, USA, 2025",
                "published_at": None,
                "source_rank": 2,
            },
            {
                "title": "Meera Hahn",
                "url": "https://example.edu/people/meera-hahn",
                "snippet": "Meera Hahn works at Google DeepMind in Atlanta, USA",
                "published_at": None,
                "source_rank": 0,
            },
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "auto_assign"
    assert decision.country_code == "US"
    assert decision.confidence == "high"
    assert len(decision.evidence) == 2


def test_evidence_decision_does_not_use_organization_headquarters() -> None:
    payload = {
        "author": "Meera Hahn",
        "organization": "Google DeepMind",
        "paper_date": "2025",
        "results": [
            {
                "title": "Google DeepMind",
                "url": "https://deepmind.google/about",
                "snippet": "Our headquarters is in London, United Kingdom, 2025",
                "published_at": None,
                "source_rank": 2,
            }
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "no_evidence"
    assert decision.country_code is None


def test_evidence_decision_defers_single_linked_source() -> None:
    payload = {
        "author": "A. Author",
        "organization": "Global Lab",
        "paper_date": "2025",
        "results": [
            {
                "title": "A. Author",
                "url": "https://global-lab.example/profile",
                "snippet": "A. Author joined Global Lab in Singapore in 2025",
                "published_at": None,
                "source_rank": 2,
            }
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "defer"
    assert decision.country_code is None


def test_recent_author_linkedin_profile_can_assign_at_medium_confidence() -> None:
    payload = {
        "author": "Meera Hahn",
        "organization": "Google DeepMind",
        "paper_date": "2025",
        "retrieved_at": "2026-07-28T12:00:00+00:00",
        "results": [
            {
                "title": "Meera Hahn",
                "url": "https://www.linkedin.com/in/meera-hahn",
                "snippet": "Meera Hahn works at Google DeepMind in Atlanta, USA",
                "published_at": None,
                "source_rank": 3,
            }
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "auto_assign"
    assert decision.country_code == "US"
    assert decision.confidence == "medium"
    assert decision.evidence[0]["author_controlled"] is True


def test_old_undated_linkedin_profile_still_defers() -> None:
    payload = {
        "author": "Meera Hahn",
        "organization": "Google DeepMind",
        "paper_date": "2022",
        "retrieved_at": "2026-07-28T12:00:00+00:00",
        "results": [
            {
                "title": "Meera Hahn",
                "url": "https://www.linkedin.com/in/meera-hahn",
                "snippet": "Meera Hahn works at Google DeepMind in Atlanta, USA",
                "published_at": None,
                "source_rank": 3,
            }
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "defer"
    assert decision.country_code is None


def test_weak_third_party_country_does_not_override_author_profile() -> None:
    payload = {
        "author": "Meera Hahn",
        "organization": "Google DeepMind",
        "paper_date": "2025",
        "retrieved_at": "2026-07-28T12:00:00+00:00",
        "results": [
            {
                "title": "Another author's page",
                "url": "https://example.com/another-author",
                "snippet": "Paper by Meera Hahn at Google DeepMind; site owner lives in India",
                "published_at": None,
                "source_rank": 2,
            },
            {
                "title": "Meera Hahn",
                "url": "https://www.linkedin.com/in/meera-hahn",
                "snippet": "Meera Hahn works at Google DeepMind in Atlanta, USA",
                "published_at": None,
                "source_rank": 3,
            },
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "auto_assign"
    assert decision.country_code == "US"


def test_single_token_author_cannot_use_linkedin_exception() -> None:
    payload = {
        "author": "Yash",
        "organization": "IIT Bombay",
        "paper_date": "2025",
        "retrieved_at": "2026-07-28T12:00:00+00:00",
        "results": [
            {
                "title": "Yash Nehra",
                "url": "https://www.linkedin.com/in/yash-nehra",
                "snippet": "Yash Nehra studies at IIT Bombay in India",
                "published_at": None,
                "source_rank": 3,
            }
        ],
    }

    decision = evaluate_affiliation_evidence(payload)

    assert decision.decision == "defer"
    assert decision.country_code is None


def test_affiliation_search_rejects_invalid_date(tmp_path: Path) -> None:
    search = EvidenceSearch(FakeProvider(), SearchCache(tmp_path / "search.sqlite3"))

    try:
        search.affiliation_evidence("A", "B", "April 2025")
    except ValueError as error:
        assert "paper_date" in str(error)
    else:
        raise AssertionError("invalid paper_date should fail")


def test_source_rank_prefers_academic_and_government_sources() -> None:
    assert source_rank("https://profile.example.edu/person") == 0
    assert source_rank("https://orcid.org/0000") == 1
    assert source_rank("https://example.com/about") == 2
    assert source_rank("https://linkedin.com/in/person") == 3


def test_mcp_server_registers_search_tools(tmp_path: Path) -> None:
    search = EvidenceSearch(FakeProvider(), SearchCache(tmp_path / "search.sqlite3"))
    server = create_server(search)

    tools = anyio.run(server.list_tools)

    assert {tool.name for tool in tools} == {
        "search_web",
        "search_affiliation_evidence",
    }


def test_brave_key_is_required_only_when_a_search_is_called() -> None:
    provider = BraveSearchProvider("")

    try:
        provider.search("query", count=1)
    except RuntimeError as error:
        assert "BRAVE_SEARCH_API_KEY" in str(error)
    else:
        raise AssertionError("search without a configured API key should fail")


class FakeHttpResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "results": [
                {
                    "title": "Profile",
                    "url": "https://example.edu/profile",
                    "content": "Skywork AI, Singapore",
                    "score": 0.9,
                }
            ],
            "usage": {"credits": 1},
            "request_id": "request-1",
        }


class FakeHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeHttpResponse:
        self.requests.append({"url": url, **kwargs})
        return FakeHttpResponse()


def test_tavily_provider_uses_basic_search_and_normalizes_results() -> None:
    client = FakeHttpClient()
    provider = TavilySearchProvider("tvly-test", client=client)

    payload = provider.search("Mingbao Lin", count=5, country="SG")

    request = client.requests[0]
    assert request["json"]["search_depth"] == "basic"
    assert request["json"]["country"] == "singapore"
    assert request["json"]["max_results"] == 5
    assert payload["usage"] == {"credits": 1}
    assert payload["web"]["results"][0]["description"] == "Skywork AI, Singapore"


def test_tavily_key_is_required_only_when_a_search_is_called() -> None:
    provider = TavilySearchProvider("")

    try:
        provider.search("query", count=1)
    except RuntimeError as error:
        assert "TAVILY_API_KEY" in str(error)
    else:
        raise AssertionError("search without a configured API key should fail")
