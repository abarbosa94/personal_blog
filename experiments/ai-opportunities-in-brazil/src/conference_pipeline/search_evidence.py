"""Live web search with a durable cache for affiliation evidence discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from .reconcile import CountryMentionExtractor


BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    published_at: str | None
    source_rank: int
    query: str


@dataclass(frozen=True)
class AffiliationEvidenceDecision:
    """A conservative, auditable decision over search-result snippets."""

    decision: str
    country_code: str | None
    confidence: str
    reason: str
    evidence: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = list(self.evidence)
        return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SearchCache:
    """SQLite cache so an evidence search can be reproduced and audited."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS searches (
                cache_key TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                retrieved_at TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get(
        self, request: dict[str, Any], max_age: timedelta
    ) -> tuple[dict[str, Any], str] | None:
        row = self.connection.execute(
            "SELECT response_json, retrieved_at FROM searches WHERE cache_key = ?",
            (self.key(request),),
        ).fetchone()
        if not row:
            return None
        retrieved_at = datetime.fromisoformat(row[1])
        if _utc_now() - retrieved_at > max_age:
            return None
        return json.loads(row[0]), row[1]

    def put(self, request: dict[str, Any], response: dict[str, Any]) -> str:
        retrieved_at = _utc_now().isoformat()
        self.connection.execute(
            """
            INSERT OR REPLACE INTO searches
                (cache_key, request_json, response_json, retrieved_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                self.key(request),
                json.dumps(request, sort_keys=True),
                json.dumps(response, ensure_ascii=False),
                retrieved_at,
            ),
        )
        self.connection.commit()
        return retrieved_at

    def close(self) -> None:
        self.connection.close()


class BraveSearchProvider:
    name = "brave"

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = BRAVE_WEB_SEARCH_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=20.0)

    def search(
        self,
        query: str,
        *,
        count: int,
        freshness: str | None = None,
        country: str = "",
        search_lang: str = "en",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "BRAVE_SEARCH_API_KEY is not configured for the MCP server"
            )
        params: dict[str, Any] = {
            "q": query,
            "count": count,
            "search_lang": search_lang,
            "text_decorations": "false",
            "result_filter": "web",
        }
        if country:
            params["country"] = country
        if freshness:
            params["freshness"] = freshness
        response = self.client.get(
            self.endpoint,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params=params,
        )
        response.raise_for_status()
        return response.json()


class TavilySearchProvider:
    """Tavily's free-tier, LLM-oriented web search API."""

    name = "tavily"
    COUNTRY_NAMES = {
        "AU": "australia",
        "BR": "brazil",
        "CA": "canada",
        "CN": "china",
        "DE": "germany",
        "FR": "france",
        "GB": "united kingdom",
        "HK": "hong kong",
        "IN": "india",
        "SG": "singapore",
        "US": "united states",
    }

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = TAVILY_SEARCH_URL,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=20.0)

    def search(
        self,
        query: str,
        *,
        count: int,
        freshness: str | None = None,
        country: str = "",
        search_lang: str = "en",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not configured for the MCP server"
            )
        body: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "max_results": count,
            "include_answer": False,
            "include_raw_content": False,
            "topic": "general",
        }
        # Tavily accepts country names rather than ISO codes. It currently
        # searches English by default; search_lang remains in the shared
        # provider interface for Brave compatibility.
        country_name = self.COUNTRY_NAMES.get(country.upper(), country.casefold())
        if country_name:
            body["country"] = country_name
        if freshness in {"day", "week", "month", "year", "d", "w", "m", "y"}:
            body["time_range"] = freshness
        elif freshness and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}", freshness
        ):
            body["start_date"], body["end_date"] = freshness.split("to", 1)
        response = self.client.post(
            self.endpoint,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
        # Normalize Tavily into the common provider response shape while
        # retaining usage and request IDs in the cached raw payload.
        return {
            "web": {
                "results": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "description": item.get("content"),
                        "published_at": item.get("published_date"),
                        "score": item.get("score"),
                    }
                    for item in payload.get("results") or []
                ]
            },
            "usage": payload.get("usage"),
            "request_id": payload.get("request_id"),
            "response_time": payload.get("response_time"),
        }


def source_rank(url: str) -> int:
    """Rank likely first-party evidence ahead of aggregators and social pages."""

    hostname = (urlparse(url).hostname or "").casefold()
    if hostname.endswith((".edu", ".ac.uk", ".edu.sg", ".gov", ".gov.sg")):
        return 0
    if hostname in {"orcid.org", "ror.org"}:
        return 1
    if hostname.endswith(("linkedin.com", "researchgate.net")):
        return 3
    return 2


def _normalized_words(value: str) -> tuple[str, ...]:
    import unicodedata

    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _mentions_entity(text: str, entity: str) -> bool:
    """Require all meaningful entity words, allowing punctuation differences."""

    ignored = {"and", "at", "for", "of", "research", "the"}
    entity_words = tuple(
        word for word in _normalized_words(entity) if word not in ignored
    )
    text_words = set(_normalized_words(text))
    return bool(entity_words) and all(word in text_words for word in entity_words)


def _is_first_party(url: str, organization: str) -> bool:
    hostname = (urlparse(url).hostname or "").casefold()
    if source_rank(url) <= 1:
        return True
    ignored = {
        "ai",
        "and",
        "company",
        "institute",
        "laboratory",
        "lab",
        "of",
        "research",
        "the",
    }
    organization_words = {
        word
        for word in _normalized_words(organization)
        if len(word) >= 4 and word not in ignored
    }
    hostname_words = set(_normalized_words(hostname))
    return bool(organization_words & hostname_words)


def _is_author_linkedin_profile(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").casefold()
    return hostname.endswith("linkedin.com") and parsed.path.casefold().startswith(
        "/in/"
    )


def evaluate_affiliation_evidence(
    payload: dict[str, Any],
) -> AffiliationEvidenceDecision:
    """Decide only from explicit author↔organization↔country evidence.

    Automatic assignment requires two independent pages agreeing on a country,
    at least one first-party/academic page, and at least one page whose text is
    temporally close to the paper. Everything else remains review evidence.
    """

    author = str(payload.get("author") or "")
    organization = str(payload.get("organization") or "")
    paper_year = int(str(payload.get("paper_date") or "0")[:4] or 0)
    retrieved_year_match = re.search(
        r"\b(?:19|20)\d{2}\b", str(payload.get("retrieved_at") or "")
    )
    retrieved_year = (
        int(retrieved_year_match.group()) if retrieved_year_match else None
    )
    candidates: list[dict[str, Any]] = []
    for result in payload.get("results") or []:
        text = " ".join(
            str(result.get(key) or "")
            for key in ("title", "snippet", "published_at")
        )
        if not _mentions_entity(text, author):
            continue
        if not _mentions_entity(text, organization):
            continue
        countries = CountryMentionExtractor.country_codes(text)
        if len(countries) != 1:
            continue
        years = {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", text)}
        temporal = any(abs(year - paper_year) <= 1 for year in years)
        candidates.append(
            {
                "url": result.get("url") or "",
                "title": result.get("title") or "",
                "snippet": result.get("snippet") or "",
                "country_code": countries[0],
                "temporal_support": temporal,
                "first_party": _is_first_party(
                    result.get("url") or "", organization
                ),
                "author_controlled": _is_author_linkedin_profile(
                    result.get("url") or ""
                ),
                "source_rank": result.get("source_rank"),
            }
        )
    if not candidates:
        return AffiliationEvidenceDecision(
            "no_evidence",
            None,
            "low",
            "No result explicitly linked the author, organization, and one country",
            (),
        )
    countries = {item["country_code"] for item in candidates}
    authoritative_countries = {
        item["country_code"]
        for item in candidates
        if item["first_party"] or item["author_controlled"]
    }
    if len(authoritative_countries) > 1:
        return AffiliationEvidenceDecision(
            "defer",
            None,
            "low",
            "Authoritative evidence conflicts across countries",
            tuple(candidates),
        )
    for country in countries:
        agreeing = [
            item for item in candidates if item["country_code"] == country
        ]
        hostnames = {
            (urlparse(item["url"]).hostname or "").casefold()
            for item in agreeing
        }
        has_first_party = any(item["first_party"] for item in agreeing)
        has_temporal = any(item["temporal_support"] for item in agreeing)
        if len(hostnames) >= 2 and has_first_party and has_temporal:
            return AffiliationEvidenceDecision(
                "auto_assign",
                country,
                "high",
                "Two independent sources agree; first-party and temporal evidence present",
                tuple(candidates),
            )
    author_profiles = [item for item in candidates if item["author_controlled"]]
    author_identity_is_specific = len(_normalized_words(author)) >= 2
    retrieval_is_recent = (
        retrieved_year is not None and abs(retrieved_year - paper_year) <= 1
    )
    profile_countries = {item["country_code"] for item in author_profiles}
    if (
        len(profile_countries) == 1
        and author_profiles
        and author_identity_is_specific
        and retrieval_is_recent
    ):
        country = next(iter(profile_countries))
        return AffiliationEvidenceDecision(
            "auto_assign",
            country,
            "medium",
            "Recent author-controlled LinkedIn profile explicitly links author, "
            "organization, and country",
            tuple(candidates),
        )
    return AffiliationEvidenceDecision(
        "defer",
        None,
        "medium",
        "Linked evidence found but corroboration, source quality, or date is insufficient",
        tuple(candidates),
    )


def normalize_results(payload: dict[str, Any], query: str) -> list[SearchResult]:
    values: list[SearchResult] = []
    for item in (payload.get("web") or {}).get("results") or []:
        url = item.get("url") or ""
        if not url:
            continue
        values.append(
            SearchResult(
                title=re.sub(r"<[^>]+>", "", item.get("title") or "").strip(),
                url=url,
                snippet=re.sub(
                    r"<[^>]+>", "", item.get("description") or ""
                ).strip(),
                published_at=(
                    item.get("page_age")
                    or item.get("age")
                    or item.get("published_at")
                ),
                source_rank=source_rank(url),
                query=query,
            )
        )
    return values


class EvidenceSearch:
    def __init__(
        self,
        provider: BraveSearchProvider | TavilySearchProvider,
        cache: SearchCache,
        *,
        cache_days: int = 30,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.max_age = timedelta(days=cache_days)
        self.clock = clock

    def web_search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,
        country: str = "",
        search_lang: str = "en",
        refresh: bool = False,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= count <= 20:
            raise ValueError("count must be between 1 and 20")
        request = {
            "provider": self.provider.name,
            "query": query.strip(),
            "count": count,
            "freshness": freshness,
            "country": country.upper(),
            "search_lang": search_lang,
        }
        cached = None if refresh else self.cache.get(request, self.max_age)
        if cached:
            payload, retrieved_at = cached
            cache_hit = True
        else:
            payload = self.provider.search(
                request["query"],
                count=count,
                freshness=freshness,
                country=request["country"],
                search_lang=search_lang,
            )
            retrieved_at = self.cache.put(request, payload)
            cache_hit = False
        results = normalize_results(payload, request["query"])
        return {
            "query": request["query"],
            "provider": self.provider.name,
            "retrieved_at": retrieved_at,
            "cache_hit": cache_hit,
            "results": [asdict(item) for item in results],
        }

    def affiliation_evidence(
        self,
        author: str,
        organization: str,
        paper_date: str,
        *,
        paper_title: str = "",
        count_per_query: int = 8,
        refresh: bool = False,
    ) -> dict[str, Any]:
        if not author.strip() or not organization.strip():
            raise ValueError("author and organization must not be empty")
        if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", paper_date):
            raise ValueError("paper_date must be YYYY, YYYY-MM, or YYYY-MM-DD")
        year = paper_date[:4]
        author_query = f'"{author.strip()}" "{organization.strip()}" location'
        dated_query = f'"{author.strip()}" "{organization.strip()}" {year}'
        organization_query = (
            f'"{organization.strip()}" address headquarters branch {year}'
        )
        paper_query = (
            f'"{paper_title.strip()}" "{author.strip()}" "{organization.strip()}"'
            if paper_title.strip()
            else ""
        )
        searches = [
            self.web_search(
                query,
                count=count_per_query,
                refresh=refresh,
            )
            for query in (
                author_query,
                dated_query,
                organization_query,
                paper_query,
            )
            if query
        ]
        deduplicated: dict[str, dict[str, Any]] = {}
        for search in searches:
            for item in search["results"]:
                previous = deduplicated.get(item["url"])
                if previous is None or item["source_rank"] < previous["source_rank"]:
                    deduplicated[item["url"]] = item
        results = sorted(
            deduplicated.values(),
            key=lambda item: (item["source_rank"], item["url"]),
        )
        return {
            "author": author.strip(),
            "organization": organization.strip(),
            "paper_title": paper_title.strip(),
            "paper_date": paper_date,
            "decision": "evidence_only",
            "guidance": (
                "Verify that evidence names both author and organization, states "
                "a location, and overlaps the paper date. Do not infer a branch "
                "from headquarters alone."
            ),
            "queries": [search["query"] for search in searches],
            "retrieved_at": max(search["retrieved_at"] for search in searches),
            "cache_hits": sum(bool(search["cache_hit"]) for search in searches),
            "results": results,
        }


def default_cache_path() -> Path:
    return Path(
        os.environ.get(
            "AFFILIATION_SEARCH_CACHE",
            "artifacts/affiliation-search/search-cache.sqlite3",
        )
    )
