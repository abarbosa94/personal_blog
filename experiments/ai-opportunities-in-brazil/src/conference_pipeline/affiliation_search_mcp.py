"""MCP stdio server for date-aware affiliation evidence search."""

from __future__ import annotations

import os
from typing import Any

from .search_evidence import (
    BraveSearchProvider,
    EvidenceSearch,
    SearchCache,
    TavilySearchProvider,
    default_cache_path,
)


def create_search() -> EvidenceSearch:
    provider_name = os.environ.get("AFFILIATION_SEARCH_PROVIDER", "").casefold()
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
    if provider_name == "brave" or (brave_key and not tavily_key):
        provider = BraveSearchProvider(brave_key)
    else:
        provider = TavilySearchProvider(tavily_key)
    return EvidenceSearch(
        provider,
        SearchCache(default_cache_path()),
        cache_days=int(os.environ.get("AFFILIATION_SEARCH_CACHE_DAYS", "30")),
    )


def create_server(search: EvidenceSearch | None = None) -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as error:
        raise RuntimeError(
            "Install the MCP dependencies with: pip install -e '.[mcp]'"
        ) from error

    evidence_search = search or create_search()
    server = FastMCP("affiliation-evidence-search")

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )

    @server.tool(annotations=read_only)
    def search_web(
        query: str,
        count: int = 10,
        freshness: str | None = None,
        country: str = "",
        search_lang: str = "en",
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Search the live web and return citable, cached structured results."""

        return evidence_search.web_search(
            query,
            count=count,
            freshness=freshness,
            country=country,
            search_lang=search_lang,
            refresh=refresh,
        )

    @server.tool(annotations=read_only)
    def search_affiliation_evidence(
        author: str,
        organization: str,
        paper_date: str,
        paper_title: str = "",
        count_per_query: int = 8,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Find evidence for an author's organization branch at a paper date.

        Results are evidence candidates, not an automatic country decision.
        Prefer first-party sources and require temporal overlap.
        """

        return evidence_search.affiliation_evidence(
            author,
            organization,
            paper_date,
            paper_title=paper_title,
            count_per_query=count_per_query,
            refresh=refresh,
        )

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
