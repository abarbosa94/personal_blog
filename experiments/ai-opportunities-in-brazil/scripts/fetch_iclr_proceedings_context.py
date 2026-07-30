"""Fetch all ICLR 2025 abstracts from the official proceedings pages."""

from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urljoin

import requests


INDEX_URL = "https://proceedings.iclr.cc/paper_files/paper/2025"
RENAMED_TITLES = {
    "SrGP0RQbYH": "Adaptive backtracking for faster optimization",
    "AsFxRSLtqR": "LR0.FM: LOW-RESOLUTION ZERO-SHOT CLASSIFICATION BENCHMARK FOR FOUNDATION MODELS",
    "SYmUS6qRub": "Denoising Levy Probabilistic Models",
    "bqv7M0wc4x": "TSVD: Bridging Theory and Practice in Continual Learning with Pre-trained Models",
}
_local = threading.local()


class AbstractLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href") or ""
            if "-Abstract-Conference.html" in href:
                self._href = href
                self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((clean(" ".join(self._text)), self._href))
            self._href = ""
            self._text = []


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def normalized(value: str) -> str:
    value = html.unescape(value).casefold()
    value = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / digest[:2] / f"{digest}.bin"


def get_bytes(cache_dir: Path, url: str) -> bytes:
    path = cache_path(cache_dir, url)
    if path.exists():
        return path.read_bytes()
    if not hasattr(_local, "session"):
        value = requests.Session()
        value.headers["User-Agent"] = "BrazilAIResearch/1.0 (RAI corpus audit)"
        _local.session = value
    response = _local.session.get(url, timeout=45)
    response.raise_for_status()
    value = response.content
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)
    return value


def extract_abstract(text: str) -> str:
    patterns = (
        r"<h[1-6][^>]*>\s*Abstract:?\s*</h[1-6]>\s*<p[^>]*>(.*?)</p>",
        r'<meta[^>]+(?:name|property)=["\']citation_abstract["\'][^>]+content=["\'](.*?)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = clean(re.sub(r"<[^>]+>", " ", match.group(1)))
            if value:
                return value
    return ""


def reconcile(
    papers: list[dict[str, str]], links: list[tuple[str, str]]
) -> tuple[dict[str, str], list[dict[str, object]]]:
    by_title: dict[str, list[tuple[str, str]]] = {}
    for title, href in links:
        by_title.setdefault(normalized(title), []).append((title, href))
    mapping: dict[str, str] = {}
    diagnostics: list[dict[str, object]] = []
    unused = {href: title for title, href in links}
    unresolved = []
    for paper in papers:
        candidates = by_title.get(normalized(paper["title"]), [])
        if len(candidates) == 1:
            _, href = candidates[0]
            mapping[paper["paper_id"]] = href
            unused.pop(href, None)
        else:
            unresolved.append(paper)
    for paper in unresolved:
        alias = RENAMED_TITLES.get(paper["paper_id"])
        if alias:
            candidates = by_title.get(normalized(alias), [])
            if len(candidates) == 1:
                _, href = candidates[0]
                mapping[paper["paper_id"]] = href
                unused.pop(href, None)
                diagnostics.append({
                    "paper_id": paper["paper_id"], "title": paper["title"],
                    "outcome": "documented_title_change",
                    "best_score": 1.0, "best_title": alias,
                })
                continue
        scored = sorted(
            (
                SequenceMatcher(
                    None, normalized(paper["title"]), normalized(title)
                ).ratio(),
                href,
                title,
            )
            for href, title in unused.items()
        )
        score, href, title = scored[-1]
        if score < 0.92:
            diagnostics.append({
                "paper_id": paper["paper_id"], "title": paper["title"],
                "outcome": "unresolved", "best_score": score, "best_title": title,
            })
            continue
        mapping[paper["paper_id"]] = href
        unused.pop(href, None)
        diagnostics.append({
            "paper_id": paper["paper_id"], "title": paper["title"],
            "outcome": "fuzzy", "best_score": score, "best_title": title,
        })
    return mapping, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()

    with args.input.open(encoding="utf-8") as handle:
        papers = [json.loads(line) for line in handle]
    papers = [row.get("paper", row) for row in papers]
    index = get_bytes(cache_dir, INDEX_URL).decode("utf-8", errors="replace")
    parser_ = AbstractLinkParser()
    parser_.feed(index)
    mapping, diagnostics = reconcile(papers, parser_.links)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(
        json.dumps({
            "index_url": INDEX_URL,
            "expected_papers": len(papers),
            "index_links": len(parser_.links),
            "matched": len(mapping),
            "unmatched": len(papers) - len(mapping),
            "reconciliation": diagnostics,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if len(mapping) / len(papers) < 0.99:
        raise RuntimeError("ICLR proceedings reconciliation below 99%")

    completed = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row["error"]:
                    completed.add(row["paper_id"])
    pending = [paper for paper in papers if paper["paper_id"] not in completed]

    def fetch(paper: dict[str, str]) -> dict[str, str]:
        href = mapping.get(paper["paper_id"], "")
        result = {
            "venue": "iclr", "paper_id": paper["paper_id"],
            "title": paper["title"], "source_url": "", "abstract": "", "error": "",
        }
        if not href:
            result["error"] = "proceedings title reconciliation unresolved"
            return result
        url = urljoin(INDEX_URL, href)
        result["source_url"] = url
        try:
            text = get_bytes(cache_dir, url).decode("utf-8", errors="replace")
            result["abstract"] = extract_abstract(text)
            if not result["abstract"]:
                raise ValueError("official proceedings page yielded no abstract")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    with args.output.open("a", encoding="utf-8", newline="") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, paper) for paper in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                state = "ERROR" if row["error"] else f"OK {len(row['abstract'])}"
                print(f"{index}/{len(pending)} {row['paper_id']} {state}", flush=True)


if __name__ == "__main__":
    main()
