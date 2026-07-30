from __future__ import annotations

from html.parser import HTMLParser
import re
from typing import Any, Iterable
from urllib.parse import quote, urljoin

from .http import HttpClient
from .models import Paper
from .parsing import normalize_space, split_authors


class _PmlrParser(HTMLParser):
    def __init__(self, venue_key: str, year: int, base_url: str) -> None:
        super().__init__()
        self.venue_key = venue_key
        self.year = year
        self.base_url = base_url
        self.depth = 0
        self.in_paper = False
        self.capture: str | None = None
        self.capture_tag: str | None = None
        self.buffer: list[str] = []
        self.current: dict[str, Any] = {}
        self.papers: list[Paper] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "div" and "paper" in classes and not self.in_paper:
            self.in_paper = True
            self.depth = 1
            self.current = {}
            return
        if not self.in_paper:
            return
        if tag == "div":
            self.depth += 1
        if "title" in classes or "authors" in classes:
            self.capture = "title" if "title" in classes else "authors"
            self.capture_tag = tag
            self.buffer = []
        if tag == "a":
            href = values.get("href") or ""
            if href.endswith(".html") and "official_url" not in self.current:
                self.current["official_url"] = urljoin(self.base_url, href)
            if href.endswith(".pdf") and "pdf_url" not in self.current:
                self.current["pdf_url"] = urljoin(self.base_url, href)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_paper:
            return
        if self.capture and tag == self.capture_tag:
            self.current[self.capture] = normalize_space("".join(self.buffer))
            self.capture = None
            self.capture_tag = None
            self.buffer = []
        if tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self._finish_paper()

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def _finish_paper(self) -> None:
        url = self.current.get("official_url", "")
        paper_id = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        title = self.current.get("title", "")
        if paper_id and title:
            self.papers.append(
                Paper(
                    paper_id=paper_id,
                    venue_key=self.venue_key,
                    year=self.year,
                    track="main",
                    title=title,
                    authors=split_authors(self.current.get("authors", "")),
                    doi=None,
                    official_url=url,
                    source_kind="official_pmlr",
                    pdf_url=self.current.get("pdf_url"),
                )
            )
        self.in_paper = False
        self.current = {}


class _AclParser(HTMLParser):
    def __init__(self, venue_key: str, year: int, volume_id: str, base_url: str) -> None:
        super().__init__()
        self.venue_key = venue_key
        self.year = year
        self.volume_id = volume_id
        self.base_url = base_url
        self.capture_title = False
        self.title_buffer: list[str] = []
        self.current_href: str | None = None
        self.papers: list[Paper] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        prefix = f"/{self.volume_id}."
        if href.startswith(prefix) and href.endswith("/") and not href.endswith(".0/"):
            self.current_href = href
            self.capture_title = True
            self.title_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.capture_title and self.current_href:
            paper_id = self.current_href.strip("/")
            title = normalize_space("".join(self.title_buffer))
            if title:
                self.papers.append(
                    Paper(
                        paper_id=paper_id,
                        venue_key=self.venue_key,
                        year=self.year,
                        track="main",
                        title=title,
                        authors=(),
                        doi=f"10.18653/v1/{paper_id}",
                        official_url=urljoin(self.base_url, self.current_href),
                        source_kind="official_acl_anthology",
                        pdf_url=urljoin(
                            self.base_url, self.current_href.rstrip("/") + ".pdf"
                        ),
                    )
                )
            self.capture_title = False
            self.current_href = None

    def handle_data(self, data: str) -> None:
        if self.capture_title:
            self.title_buffer.append(data)


class PmlrEnumerator:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def enumerate(self, venue_key: str, year: int, url: str) -> list[Paper]:
        return self.parse(self.http.get_text(url), venue_key, year, url)

    @staticmethod
    def parse(html_text: str, venue_key: str, year: int, url: str) -> list[Paper]:
        parser = _PmlrParser(venue_key, year, url)
        parser.feed(html_text)
        return parser.papers


class AclAnthologyEnumerator:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def enumerate(
        self, venue_key: str, year: int, volume_id: str, url: str
    ) -> list[Paper]:
        return self.parse(self.http.get_text(url), venue_key, year, volume_id, url)

    @staticmethod
    def parse(
        html_text: str, venue_key: str, year: int, volume_id: str, url: str
    ) -> list[Paper]:
        parser = _AclParser(venue_key, year, volume_id, url)
        parser.feed(html_text)
        seen: set[str] = set()
        unique: list[Paper] = []
        for paper in parser.papers:
            if paper.paper_id not in seen:
                unique.append(paper)
                seen.add(paper.paper_id)
        return unique


class CrossrefAaaaiEnumerator:
    """Enumerate AAAI papers from DOI metadata registered by AAAI in Crossref."""

    endpoint = "https://api.crossref.org/prefixes/10.1609/works"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def enumerate(self, year: int, volume: str, rows: int = 1000) -> list[Paper]:
        cursor = "*"
        items: list[dict[str, Any]] = []
        pages = 0
        while cursor and pages < 100:
            url = (
                f"{self.endpoint}?rows={rows}"
                "&select=DOI,title,author,container-title,volume,URL"
                f"&cursor={quote(cursor)}"
            )
            payload = self.http.get_json(url)["message"]
            batch = payload.get("items", [])
            items.extend(batch)
            next_cursor = payload.get("next-cursor")
            # Crossref may return the same cursor token while advancing its
            # server-side scroll context. An empty page is the terminator.
            cursor = next_cursor if batch and next_cursor else ""
            pages += 1
        if pages == 100 and cursor:
            raise RuntimeError("Crossref pagination exceeded the 100-page guard")
        return self.from_items(items, year, volume)

    @staticmethod
    def from_items(
        items: Iterable[dict[str, Any]], year: int, volume: str
    ) -> list[Paper]:
        papers: list[Paper] = []
        for item in items:
            container = " ".join(item.get("container-title", []))
            if "AAAI Conference on Artificial Intelligence" not in container:
                continue
            if str(item.get("volume", "")) != volume:
                continue
            doi = item.get("DOI")
            title = normalize_space(" ".join(item.get("title", [])))
            if not doi or not title:
                continue
            authors = tuple(
                normalize_space(
                    " ".join(
                        part
                        for part in (author.get("given"), author.get("family"))
                        if part
                    )
                )
                for author in item.get("author", [])
            )
            papers.append(
                Paper(
                    paper_id=doi.lower(),
                    venue_key="aaai",
                    year=year,
                    track="main",
                    title=title,
                    authors=tuple(author for author in authors if author),
                    doi=doi.lower(),
                    official_url=item.get("URL") or f"https://doi.org/{doi}",
                    source_kind="aaai_crossref_doi_metadata",
                )
            )
        return papers


class _NeuripsParser(HTMLParser):
    def __init__(self, year: int, base_url: str) -> None:
        super().__init__()
        self.year = year
        self.base_url = base_url
        self.in_item = False
        self.current: dict[str, str] = {}
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.papers: list[Paper] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "li" and values.get("data-track"):
            self.in_item = True
            self.current = {"track": values["data-track"] or "main"}
        if not self.in_item:
            return
        if tag == "a" and values.get("title") == "paper title":
            self.current["href"] = values.get("href") or ""
            self.capture = "title"
            self.buffer = []
        elif tag == "span" and "paper-authors" in classes:
            self.capture = "authors"
            self.buffer = []

    def handle_endtag(self, tag: str) -> None:
        if not self.in_item:
            return
        if self.capture == "title" and tag == "a":
            self.current["title"] = normalize_space("".join(self.buffer))
            self.capture = None
        elif self.capture == "authors" and tag == "span":
            self.current["authors"] = normalize_space("".join(self.buffer))
            self.capture = None
        if tag == "li":
            self._finish()

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def _finish(self) -> None:
        href = self.current.get("href", "")
        title = self.current.get("title", "")
        if href and title:
            filename = href.rsplit("/", 1)[-1]
            paper_id = filename.split("-Abstract-", 1)[0]
            pdf_href = href.replace("/hash/", "/file/").replace(
                "-Abstract-", "-Paper-"
            ).removesuffix(".html") + ".pdf"
            self.papers.append(
                Paper(
                    paper_id=paper_id,
                    venue_key="neurips",
                    year=self.year,
                    track=self.current.get("track", "main"),
                    title=title,
                    authors=split_authors(self.current.get("authors", "")),
                    doi=None,
                    official_url=urljoin(self.base_url, href),
                    source_kind="official_neurips",
                    pdf_url=urljoin(self.base_url, pdf_href),
                )
            )
        self.in_item = False
        self.current = {}


class NeuripsEnumerator:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def enumerate(self, year: int, url: str) -> list[Paper]:
        return self.parse(self.http.get_text(url), year, url)

    @staticmethod
    def parse(html_text: str, year: int, url: str) -> list[Paper]:
        parser = _NeuripsParser(year, url)
        parser.feed(html_text)
        return parser.papers


class IclrOpenReviewEnumerator:
    """Enumerate accepted ICLR submissions with the official OpenReview client."""

    base_url = "https://api2.openreview.net"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import openreview

            client = openreview.api.OpenReviewClient(baseurl=self.base_url)
        self.client = client

    def enumerate(self, year: int, venue_id: str) -> list[Paper]:
        notes = self.client.get_all_notes(content={"venueid": venue_id})
        return self.from_notes(notes, year)

    @staticmethod
    def _value(content: dict[str, Any], key: str, default: Any = None) -> Any:
        value = content.get(key, default)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    @classmethod
    def from_notes(cls, notes: Iterable[Any], year: int) -> list[Paper]:
        papers: list[Paper] = []
        seen: set[str] = set()
        for note in notes:
            paper_id = str(note.id)
            content = note.content
            title = normalize_space(str(cls._value(content, "title", "")))
            authors = cls._value(content, "authors", ()) or ()
            venue = normalize_space(str(cls._value(content, "venue", "main")))
            if not paper_id or paper_id in seen or not title:
                continue
            seen.add(paper_id)
            papers.append(
                Paper(
                    paper_id=paper_id,
                    venue_key="iclr",
                    year=year,
                    track=venue or "main",
                    title=title,
                    authors=tuple(normalize_space(str(author)) for author in authors),
                    doi=None,
                    official_url=f"https://openreview.net/forum?id={paper_id}",
                    source_kind="official_openreview",
                    pdf_url=f"https://openreview.net/pdf?id={paper_id}",
                )
            )
        return papers


class _KddAdsParser(HTMLParser):
    DOI_PATTERN = re.compile(r"https://doi\.org/(10\.1145/\d+\.\d+)", re.I)

    def __init__(self, year: int, source_url: str) -> None:
        super().__init__()
        self.year = year
        self.source_url = source_url
        self.table_index = 0
        self.in_table = False
        self.in_cell = False
        self.in_strong = False
        self.cell_text: list[str] = []
        self.title_text: list[str] = []
        self.pending: tuple[str, str, str] | None = None
        self.papers: list[Paper] = []
        self.affiliations_by_doi: dict[str, tuple[str, ...]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table_index += 1
            self.in_table = True
        elif self.in_table and tag == "td":
            self.in_cell = True
            self.cell_text = []
            self.title_text = []
        elif self.in_cell and tag == "strong":
            self.in_strong = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "strong":
            self.in_strong = False
        elif tag == "td" and self.in_cell:
            self._finish_cell()
            self.in_cell = False
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_text.append(data)
            if self.in_strong:
                self.title_text.append(data)

    def _finish_cell(self) -> None:
        text = normalize_space("".join(self.cell_text))
        title = normalize_space("".join(self.title_text))
        doi_match = self.DOI_PATTERN.search(text)
        if title and doi_match:
            cycle = "february" if self.table_index == 1 else "august"
            self.pending = (title, doi_match.group(1).lower(), cycle)
            return
        if not self.pending or not text:
            return
        title, doi, cycle = self.pending
        authors = tuple(
            normalize_space(author.split(" (", 1)[0])
            for author in text.split(";")
            if normalize_space(author.split(" (", 1)[0])
        )
        affiliations = tuple(
            normalize_space(author.partition(" (")[2][:-1])
            for author in text.split(";")
            if " (" in author and author.rstrip().endswith(")")
        )
        self.affiliations_by_doi[doi] = tuple(
            dict.fromkeys(value for value in affiliations if value)
        )
        self.papers.append(
            Paper(
                paper_id=doi,
                venue_key="kdd-ads",
                year=self.year,
                track=f"ads-{cycle}-cycle",
                title=title,
                authors=authors,
                doi=doi,
                official_url=f"https://doi.org/{doi}",
                source_kind="official_kdd_ads",
            )
        )
        self.pending = None


class KddAdsEnumerator:
    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def enumerate(self, year: int, url: str) -> list[Paper]:
        return self.parse(self.http.get_text(url), year, url)

    @staticmethod
    def parse(html_text: str, year: int, url: str) -> list[Paper]:
        parser = _KddAdsParser(year, url)
        parser.feed(html_text)
        seen: set[str] = set()
        papers: list[Paper] = []
        for paper in parser.papers:
            if paper.paper_id not in seen:
                papers.append(paper)
                seen.add(paper.paper_id)
        return papers
