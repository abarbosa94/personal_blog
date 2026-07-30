"""Enumerate the official AIES and FAccT 2025 accepted-paper universes."""

from __future__ import annotations

import argparse
import csv
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from .models import Paper
from .parsing import normalize_space, split_authors


class AiesIssueParser(HTMLParser):
    def __init__(self, issue_number: int) -> None:
        super().__init__()
        self.issue_number = issue_number
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.section = ""
        self.summary_depth = 0
        self.current: dict[str, str] = {}
        self.papers: list[Paper] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "h2":
            self.capture, self.buffer = "section", []
        elif tag == "div" and "obj_article_summary" in classes:
            self.summary_depth, self.current = 1, {}
        elif self.summary_depth:
            if tag == "div":
                self.summary_depth += 1
            if tag == "a" and values.get("id", "").startswith("article-"):
                self.capture, self.buffer = "title", []
                self.current["official_url"] = values.get("href") or ""
                self.current["paper_id"] = values["id"].removeprefix("article-")
            elif tag == "div" and "authors" in classes:
                self.capture, self.buffer = "authors", []
            elif tag == "a" and "pdf" in classes:
                self.current["pdf_url"] = values.get("href") or ""

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self.capture == "section":
            self.section = normalize_space("".join(self.buffer))
            self.capture = None
        elif tag == "a" and self.capture == "title":
            self.current["title"] = normalize_space("".join(self.buffer))
            self.capture = None
        elif tag == "div" and self.capture == "authors":
            self.current["authors"] = normalize_space("".join(self.buffer))
            self.capture = None
        if tag == "div" and self.summary_depth:
            self.summary_depth -= 1
            if self.summary_depth == 0:
                self._finish()

    def _finish(self) -> None:
        if self.section.startswith("Main Track") and self.current.get("title"):
            paper_id = self.current["paper_id"]
            self.papers.append(
                Paper(
                    paper_id=paper_id,
                    venue_key="aies",
                    year=2025,
                    track=self.section,
                    title=self.current["title"],
                    authors=split_authors(self.current.get("authors", "")),
                    doi=f"10.1609/aies.v8i{self.issue_number}.{paper_id}",
                    official_url=self.current["official_url"],
                    source_kind="official_aies_ojs",
                    pdf_url=self.current.get("pdf_url"),
                )
            )


def parse_aies_issue(text: str, issue_number: int) -> list[Paper]:
    parser = AiesIssueParser(issue_number)
    parser.feed(text)
    return parser.papers


def read_facct(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            doi = row["URL"].removeprefix("https://doi.org/")
            papers.append(
                Paper(
                    paper_id=doi,
                    venue_key="facct",
                    year=2025,
                    track=row["TYPE"],
                    title=normalize_space(row["TITLE"]),
                    authors=split_authors(row["AUTHOR"].replace(" and ", ", ")),
                    doi=doi,
                    official_url=row["URL"],
                    source_kind="official_facct_csv",
                    pdf_url=row["URL-OLD"] or None,
                )
            )
    return papers


def paper_record(paper: Paper) -> dict[str, object]:
    return {
        "paper": {
            "paper_id": paper.paper_id, "venue_key": paper.venue_key,
            "year": paper.year, "track": paper.track, "title": paper.title,
            "authors": list(paper.authors), "doi": paper.doi,
            "official_url": paper.official_url, "source_kind": paper.source_kind,
            "pdf_url": paper.pdf_url,
        },
        "openalex_id": None, "match_method": None, "affiliations": [],
        "diagnostics": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aies-dir", type=Path, required=True)
    parser.add_argument("--facct-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aies = []
    for path in sorted(args.aies_dir.glob("issue-*.html")):
        issue_number = int(path.stem.rsplit("-", 1)[-1]) - 676
        aies.extend(parse_aies_issue(path.read_text(encoding="utf-8"), issue_number))
    facct = read_facct(args.facct_csv)
    if len(aies) != 238:
        raise ValueError(f"Expected 238 AIES main-track papers, found {len(aies)}")
    if len(facct) != 217:
        raise ValueError(f"Expected 217 FAccT CSV entries, found {len(facct)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for venue, papers in (("aies", aies), ("facct", facct)):
        path = args.output_dir / f"{venue}-2025.jsonl"
        path.write_text(
            "".join(json.dumps(paper_record(p), ensure_ascii=False) + "\n" for p in papers),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
