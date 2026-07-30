"""Resumably fetch abstract context for the six frozen 2025 RAI venues."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from pypdf import PdfReader


_local = threading.local()
_cache_dir: Path | None = None


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        value = requests.Session()
        value.headers["User-Agent"] = "BrazilAIResearch/1.0 (RAI corpus audit)"
        _local.session = value
    return _local.session


def clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def html_abstract(text: str) -> str:
    patterns = (
        r'<div[^>]+id=["\']abstract["\'][^>]*>(.*?)</div>',
        r'<section[^>]+class=["\'][^"\']*\bitem\s+abstract\b[^"\']*["\'][^>]*>(.*?)</section>',
        r'<h[1-6][^>]*>\s*Abstract:?\s*</h[1-6]>\s*(.*?)(?:<h[1-6]|<hr|<section|<footer)',
        r'<meta[^>]+(?:name|property)=["\'](?:DC\.Description|citation_abstract)["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:name|property)=["\'](?:DC\.Description|citation_abstract)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = clean_html(match.group(1))
            if len(value) >= 80:
                return value
    return ""


def pdf_abstract(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text = re.sub(
        r"\s+", " ", "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    )
    marker = re.search(r"\babstract\b", text[:1500], flags=re.IGNORECASE)
    if marker:
        text = text[marker.end():]
    end = re.search(r"\b1\.?\s+introduction\b", text, flags=re.IGNORECASE)
    if end:
        text = text[:end.start()]
    return text.strip()[:8000]


def unwrap(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    return str(value or "").strip()


def load_papers(paths: list[Path]) -> list[dict[str, str]]:
    papers = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                paper = record.get("paper", record)
                papers.append({key: str(paper.get(key) or "") for key in (
                    "venue_key", "paper_id", "title", "official_url", "pdf_url"
                )})
    return papers


def cache_path(url: str) -> Path | None:
    if _cache_dir is None:
        return None
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return _cache_dir / digest[:2] / f"{digest}.bin"


def request_bytes(url: str) -> bytes:
    path = cache_path(url)
    if path and path.exists():
        return path.read_bytes()
    for attempt in range(5):
        response = session().get(url, timeout=60)
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            value = response.content
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(value)
                temporary.replace(path)
            return value
        retry_after = response.headers.get("Retry-After")
        delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
        time.sleep(delay + random.random())
    response.raise_for_status()
    return response.content


def request_text(url: str) -> str:
    value = request_bytes(url)
    if value.startswith(b"\x1f\x8b"):
        value = gzip.decompress(value)
    return value.decode("utf-8", errors="replace")


def fetch(
    paper: dict[str, str], facct: dict[str, str]
) -> dict[str, str]:
    venue, paper_id = paper["venue_key"], paper["paper_id"]
    result = {
        "venue": venue, "paper_id": paper_id, "title": paper["title"],
        "source_url": "", "abstract": "", "error": "",
    }
    try:
        if venue == "facct":
            result["abstract"] = facct.get(paper_id) or facct.get(paper["title"], "")
            result["source_url"] = "data/raw/external/facct/2025/facct2025-final.csv"
            if not result["abstract"]:
                raise ValueError("blank official FAccT abstract")
            return result
        if venue == "iclr":
            url = f"https://api2.openreview.net/notes?id={paper_id}"
            notes = json.loads(request_text(url)).get("notes", [])
            note = next(
                (n for n in notes if unwrap(n.get("content", {}).get("abstract"))),
                None,
            )
            if not note:
                raise ValueError("OpenReview note has no abstract")
            result["abstract"] = unwrap(note["content"]["abstract"])
            result["source_url"] = url
            return result
        url = paper["official_url"]
        if venue == "aaai":
            article_id = paper_id.rsplit(".", 1)[-1]
            url = f"https://ojs.aaai.org/index.php/AAAI/article/view/{article_id}"
        abstract = html_abstract(request_text(url))
        result["source_url"] = url
        if not abstract and paper["pdf_url"]:
            abstract = pdf_abstract(request_bytes(paper["pdf_url"]))
            result["source_url"] = paper["pdf_url"]
        if not abstract:
            raise ValueError("official page/PDF yielded no abstract")
        result["abstract"] = abstract
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--facct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    global _cache_dir
    _cache_dir = args.cache_dir.resolve()

    with args.facct.open(encoding="utf-8-sig", newline="") as handle:
        facct_rows = list(csv.DictReader(handle))
    facct = {}
    for row in facct_rows:
        abstract = row.get("ABSTRACT", "").strip()
        url = row.get("URL", "").strip()
        facct[url] = abstract
        facct[url.removeprefix("https://doi.org/")] = abstract
        facct[row.get("TITLE", "").strip()] = abstract
    papers = load_papers(args.input)

    completed = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row["error"]:
                    completed.add((row["venue"], row["paper_id"] or row["title"]))
    pending = [
        p for p in papers
        if (p["venue_key"], p["paper_id"] or p["title"]) not in completed
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, paper, facct): paper for paper in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                status = "ERROR" if row["error"] else f"OK {len(row['abstract'])}"
                print(f"{index}/{len(pending)} {row['venue']} {row['paper_id']} {status}")


if __name__ == "__main__":
    main()
