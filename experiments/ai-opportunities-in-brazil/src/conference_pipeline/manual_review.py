"""Build and serve a resumable human-review queue for reconciled papers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from .affiliation_names import canonical_affiliation_name


STATUSES = {"needs_review", "pass", "fail", "defer"}
CONFIDENCE = {"", "low", "medium", "high"}
FAILURE_CODES = {
    "api_error",
    "not_found",
    "ambiguous_match",
    "wrong_work",
    "missing_affiliation",
    "incomplete_affiliations",
    "wrong_affiliation",
    "ror_mismatch",
    "other",
}
EDITABLE_FIELDS = {
    "review_status",
    "confidence",
    "failure_codes",
    "review_note",
    "add_to_regression",
}
FIELDNAMES = [
    "review_id",
    "venue",
    "paper_id",
    "track",
    "title",
    "authors",
    "official_url",
    "pdf_url",
    "openalex_id",
    "match_method",
    "affiliations_json",
    "system_signal",
    "priority",
    "priority_reason",
    "review_status",
    "confidence",
    "failure_codes",
    "review_note",
    "add_to_regression",
]

AMBIGUOUS_MULTINATIONAL_BRANCHES = {
    "google deepmind",
    "google research",
    "meta",
    "meta ai",
    "meta superintelligence labs",
    "microsoft research",
}


def _normalized_name(value: str) -> str:
    return canonical_affiliation_name(value)


def _signal(record: dict[str, Any]) -> tuple[str, str, str]:
    openalex_id = record.get("openalex_id")
    affiliations = record.get("affiliations") or []
    countries = [item.get("country_code") for item in affiliations if item.get("country_code")]
    method = record.get("match_method") or ""
    if not affiliations:
        return "unreconciled", "high", "No affiliation was resolved"
    if affiliations and not countries:
        return "country_missing", "high", "Affiliations exist but no country was resolved"
    resolved_names = {
        _normalized_name(item.get("institution_name") or "")
        for item in affiliations
        if item.get("country_code")
    }
    ambiguous = sorted(
        {
            item.get("institution_name") or ""
            for item in affiliations
            if not item.get("country_code")
            and _normalized_name(item.get("institution_name") or "")
            not in resolved_names
            and _normalized_name(item.get("institution_name") or "")
            in AMBIGUOUS_MULTINATIONAL_BRANCHES
        }
    )
    if ambiguous:
        return (
            "ambiguous_multinational_branch",
            "high",
            "Branch location is unresolved for: " + ", ".join(ambiguous),
        )
    if not openalex_id:
        return "fallback_only", "medium", "Country evidence comes only from PDF/ROR fallback"
    if "fallback" in method or "pdf-ner" in method:
        return "mixed_sources", "medium", "OpenAlex work and fallback affiliations are combined"
    return "automatic_pass", "low", "Automatic match has country-bearing affiliations"


def build_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for record in records:
            paper = record["paper"]
            signal, priority, reason = _signal(record)
            rows.append(
                {
                    "review_id": f"{paper['venue_key']}:{paper['paper_id']}",
                    "venue": paper["venue_key"],
                    "paper_id": paper["paper_id"],
                    "track": paper["track"],
                    "title": paper["title"],
                    "authors": " | ".join(paper.get("authors") or []),
                    "official_url": paper["official_url"],
                    "pdf_url": paper.get("pdf_url") or "",
                    "openalex_id": record.get("openalex_id") or "",
                    "match_method": record.get("match_method") or "",
                    "affiliations_json": json.dumps(
                        record.get("affiliations") or [], ensure_ascii=False
                    ),
                    "system_signal": signal,
                    "priority": priority,
                    "priority_reason": reason,
                    "review_status": "needs_review",
                    "confidence": "",
                    "failure_codes": "",
                    "review_note": "",
                    "add_to_regression": "false",
                }
            )
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        rows,
        key=lambda row: (rank[row["priority"]], row["venue"], row["paper_id"]),
    )


def write_queue(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ReviewStore:
    """Thread-safe CSV state with validation and atomic replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._lock = threading.RLock()
        self._revision = 0
        # Excel and PowerShell commonly add a UTF-8 BOM when resaving a CSV.
        # utf-8-sig accepts both BOM-bearing and ordinary UTF-8 queues.
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDNAMES:
                missing = sorted(set(FIELDNAMES) - set(reader.fieldnames or ()))
                extra = sorted(set(reader.fieldnames or ()) - set(FIELDNAMES))
                raise ValueError(
                    "Review queue has an unexpected schema "
                    f"(missing={missing}, extra={extra}). "
                    "Pass the generated review CSV, not a JSONL pipeline file."
                )
            self._rows = [{key: value or "" for key, value in row.items()} for row in reader]
        ids = [row["review_id"] for row in self._rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Review queue contains duplicate review_id values")
        self._by_id = {row["review_id"]: row for row in self._rows}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = [dict(row) for row in self._rows]
            counts = {
                status: sum(row["review_status"] == status for row in items)
                for status in sorted(STATUSES)
            }
            return {
                "items": items,
                "meta": {
                    "total": len(items),
                    "completed": counts["pass"] + counts["fail"],
                    "remaining": counts["needs_review"],
                    "deferred": counts["defer"],
                    "status_counts": counts,
                    "revision": self._revision,
                    "source": self.path.name,
                },
            }

    def update(self, review_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - EDITABLE_FIELDS
        if unknown or not changes:
            raise ValueError(f"Invalid editable fields: {sorted(unknown)}")
        normalized: dict[str, str] = {}
        for key, value in changes.items():
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            normalized[key] = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        with self._lock:
            row = self._by_id.get(review_id)
            if row is None:
                raise KeyError(review_id)
            candidate = {**row, **normalized}
            if candidate["review_status"] not in STATUSES:
                raise ValueError("review_status must be needs_review, pass, fail, or defer")
            if candidate["confidence"] not in CONFIDENCE:
                raise ValueError("confidence must be low, medium, high, or empty")
            if candidate["add_to_regression"] not in {"true", "false"}:
                raise ValueError("add_to_regression must be true or false")
            codes = {code for code in candidate["failure_codes"].split("|") if code}
            if not codes <= FAILURE_CODES:
                raise ValueError(f"Unknown failure codes: {sorted(codes - FAILURE_CODES)}")
            if candidate["review_status"] in {"pass", "fail"} and not candidate["confidence"]:
                raise ValueError("Completed reviews require a confidence level")
            if candidate["review_status"] == "fail" and not codes:
                raise ValueError("Failed reviews require at least one failure code")
            if candidate["review_status"] == "pass" and codes:
                raise ValueError("Passing reviews cannot retain failure codes")
            if ("other" in codes or candidate["review_status"] == "fail") and not candidate["review_note"]:
                raise ValueError("Failed and Other-coded reviews require a note")
            previous = dict(row)
            row.update(normalized)
            try:
                self._atomic_write()
            except Exception:
                row.clear()
                row.update(previous)
                raise
            self._revision += 1
            return {"item": dict(row), "revision": self._revision}

    def _atomic_write(self) -> None:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent, text=True
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
                writer.writeheader()
                writer.writerows(self._rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


def make_handler(store: ReviewStore, assets: Path) -> type[BaseHTTPRequestHandler]:
    files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/items":
                self._json(HTTPStatus.OK, store.snapshot())
                return
            if path not in files:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            filename, content_type = files[path]
            body = (assets / filename).read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PATCH(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            prefix = "/api/items/"
            if not path.startswith(prefix):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                changes = json.loads(self.rfile.read(size))
                result = store.update(unquote(path[len(prefix) :]), changes)
                self._json(HTTPStatus.OK, result)
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Unknown review item"})
            except (ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

        def log_message(self, *_: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("inputs", nargs="+", type=Path)
    build.add_argument("--output", required=True, type=Path)
    serve = subparsers.add_parser("serve")
    serve.add_argument("queue", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8765, type=int)
    serve.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        rows = build_rows(args.inputs)
        write_queue(rows, args.output)
        print(f"Wrote {len(rows)} review items to {args.output}")
        return
    assets = Path(__file__).with_name("review_ui")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ReviewStore(args.queue), assets))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Review UI: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
