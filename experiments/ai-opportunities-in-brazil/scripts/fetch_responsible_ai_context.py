"""Fetch official abstract context for the frozen Responsible-AI validation sample."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import time
from pathlib import Path

import requests
from pypdf import PdfReader


ICLR_PROCEEDINGS_HASHES = {
    "YLIsIzC74j": "04c0399a47ee4107cd03b08f1f8c3eeb",
    "CSj72Rr2PB": "a6efa49c54bedf4411f1bcd32f15937a",
    "d2UrCGtntF": "ef62614753535977071395fb1f1435be",
    "KL8Sm4xRn7": "95b7a93e60fdfd10cc202f44fd6adf5f",
    "xNsIfzlefG": "97d289bf36ed0c388f27604b29325447",
    "uQnvYP7yX9": "2aa212d6f40c1cb19b777e83db00ec6a",
    "LO4MEPoqrG": "6c5da478b9d13f541993d67897a0bb30",
    "ptjrpEGrGg": "496b549556509bbb9770bf9d335c5800",
    "Oh8MuCacJW": "283dd122db4484cec1c2c8fa351d078a",
    "bDt5qc7TfO": "99d27a4939a1dfcdde8394b9137bd885",
    "INqLJwqUmc": "392d0d05e2f514063e6ce6f8b370834c",
    "xJXq6FkqEw": "01328d0767830e73a612f9073e9ff15f",
    "FEpAUnS7f7": "bef8e5620c699630405adafaa86cb038",
}


def clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def extract_html_abstract(text: str) -> str:
    patterns = (
        r'<section[^>]+class=["\'][^"\']*\bitem\s+abstract\b[^"\']*["\'][^>]*>.*?<p[^>]*>(.*?)</p>',
        r'<h[1-6][^>]*>\s*Abstract:?\s*</h[1-6]>\s*(.*?)(?:<h[1-6]|<hr|<section|<div[^>]+class=)',
        r'<meta[^>]+(?:name|property)=["\'](?:DC\.Description|citation_abstract|og:description)["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:name|property)=["\'](?:DC\.Description|citation_abstract|og:description)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            abstract = clean_html(match.group(1))
            if abstract:
                return abstract
    return ""


def extract_meta_abstract(text: str) -> str:
    for name in ("DC.Description", "citation_abstract", "og:description"):
        patterns = (
            rf'<meta[^>]+(?:name|property)=["\']{re.escape(name)}["\'][^>]+content=["\'](.*?)["\']',
            rf'<meta[^>]+content=["\'](.*?)["\'][^>]+(?:name|property)=["\']{re.escape(name)}["\']',
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                value = clean_html(match.group(1))
                if value:
                    return value
    return ""


def pdf_context(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    return re.sub(r"\s+", " ", text).strip()[:6000]


def unwrap(value: object) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    return str(value or "").strip()


def fetch_context(
    row: dict[str, str],
    session: requests.Session,
    facct: dict[str, dict[str, str]],
    pdf_urls: dict[tuple[str, str], str],
) -> tuple[str, str]:
    venue = row["venue"]
    paper_id = row["paper_id"]
    if venue == "facct":
        record = facct.get(paper_id) or facct.get(row["title"])
        abstract = (record or {}).get("ABSTRACT", "").strip()
        if abstract:
            return abstract, "official_facct_csv"
        if paper_id == "10.1145/3715275.3732218":
            url = "https://arxiv.org/abs/2506.12098"
            response = session.get(url, timeout=45)
            response.raise_for_status()
            return extract_meta_abstract(response.text), url
        return "", "official_facct_csv"
    if venue == "iclr":
        digest = ICLR_PROCEEDINGS_HASHES[paper_id]
        url = (
            "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
            f"{digest}-Abstract-Conference.html"
        )
        response = session.get(url, timeout=45)
        response.raise_for_status()
        abstract = extract_html_abstract(response.text)
        if len(abstract) >= 300:
            return abstract, url
        pdf_url = (
            "https://proceedings.iclr.cc/paper_files/paper/2025/file/"
            f"{digest}-Paper-Conference.pdf"
        )
        pdf_response = session.get(pdf_url, timeout=60)
        pdf_response.raise_for_status()
        return pdf_context(pdf_response.content), pdf_url
    if venue == "aaai":
        article_id = paper_id.rsplit(".", 1)[-1]
        url = f"https://ojs.aaai.org/index.php/AAAI/article/view/{article_id}"
    else:
        url = row["official_url"]
    response = session.get(url, timeout=45)
    response.raise_for_status()
    if venue in {"aaai", "aies"}:
        abstract = extract_meta_abstract(response.text)
    else:
        abstract = extract_html_abstract(response.text)
    if len(abstract) >= 300:
        return abstract, response.url
    pdf_url = pdf_urls.get((venue, paper_id), "")
    if pdf_url:
        pdf_response = session.get(pdf_url, timeout=60)
        pdf_response.raise_for_status()
        return pdf_context(pdf_response.content), pdf_url
    return abstract, response.url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--facct", type=Path, required=True)
    parser.add_argument("--raw", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.sample.open(encoding="utf-8-sig", newline="") as handle:
        sample = list(csv.DictReader(handle))
    with args.facct.open(encoding="utf-8-sig", newline="") as handle:
        facct_rows = list(csv.DictReader(handle))
    facct: dict[str, dict[str, str]] = {}
    for record in facct_rows:
        facct[record.get("ID", "")] = record
        facct[record.get("TITLE", "")] = record
    pdf_urls: dict[tuple[str, str], str] = {}
    for path in args.raw:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                paper = record.get("paper", record)
                pdf_url = paper.get("pdf_url", "")
                if pdf_url:
                    pdf_urls[(paper["venue_key"], paper["paper_id"])] = pdf_url

    session = requests.Session()
    session.headers["User-Agent"] = "BrazilAIResearch/1.0 (responsible-ai validation)"
    results: list[dict[str, str]] = []
    for index, row in enumerate(sample, start=1):
        result = {
            "venue": row["venue"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "source_url": "",
            "abstract": "",
            "error": "",
        }
        try:
            abstract, source_url = fetch_context(row, session, facct, pdf_urls)
            result["abstract"] = abstract
            result["source_url"] = source_url
        except Exception as exc:  # Preserve fetch failures for explicit review.
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        print(f"{index:02d}/{len(sample)} {row['venue']} {paper_id_status(result)}")
        if row["venue"] not in {"facct", "iclr"}:
            time.sleep(0.15)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def paper_id_status(result: dict[str, str]) -> str:
    if result["error"]:
        return f"ERROR {result['paper_id']} {result['error']}"
    return f"OK {result['paper_id']} abstract_chars={len(result['abstract'])}"


if __name__ == "__main__":
    main()
