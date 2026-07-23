"""Local browser UI for reviewing bilingual sentence-alignment proposals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


REVIEW_STATUSES = {"needs_review", "defer", "accept", "localized", "exclude"}
TERMINAL_REVIEW_STATUSES = {"accept", "localized", "exclude"}
EDITABLE_FIELDS = {
    "review_status",
    "reviewed_english",
    "reviewed_portuguese",
    "review_note",
}
REQUIRED_FIELDS = {
    "alignment_id",
    "english",
    "portuguese",
    "review_status",
    "reviewed_english",
    "reviewed_portuguese",
    "review_note",
}
MAX_REQUEST_BYTES = 1_000_000
HUMAN_REVIEW_STATUSES = {"needs_review", "defer", "reviewed"}
HUMAN_CHOICES = {"", "A", "B", "tie"}
HUMAN_CONFIDENCE = {"", "low", "medium", "high"}
HUMAN_EDITABLE_FIELDS = {
    "choice_A_B_or_tie",
    "review_status",
    "confidence",
    "failure_tags",
    "note",
    "add_to_golden",
}
HUMAN_REQUIRED_FIELDS = {
    "sample_id",
    "direction",
    "source",
    "human_reference",
    "candidate_A",
    "candidate_B",
    *HUMAN_EDITABLE_FIELDS,
}


class ReviewStore:
    """Thread-safe CSV-backed review state with atomic writes."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path.resolve()
        self._lock = threading.RLock()
        self._revision = 0
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, str]] = []
        self._by_id: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Review CSV does not exist: {self.csv_path}")
        with self.csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Review CSV has no header")
            missing = REQUIRED_FIELDS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Review CSV is missing fields: {sorted(missing)}")
            rows = [{key: value or "" for key, value in row.items()} for row in reader]
        identifiers = [row["alignment_id"] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Review CSV contains duplicate alignment_id values")
        invalid = {
            row["review_status"] for row in rows if row["review_status"] not in REVIEW_STATUSES
        }
        if invalid:
            raise ValueError(f"Review CSV contains invalid statuses: {sorted(invalid)}")
        self._fieldnames = list(reader.fieldnames)
        self._rows = rows
        self._by_id = {row["alignment_id"]: row for row in rows}

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = [dict(row) for row in self._rows]
            status_counts = {
                status: sum(row["review_status"] == status for row in self._rows)
                for status in sorted(REVIEW_STATUSES)
            }
            high_priority = sum(
                row.get("review_priority", "") == "high" for row in self._rows
            )
            terminal_count = sum(
                row["review_status"] in TERMINAL_REVIEW_STATUSES for row in self._rows
            )
            return {
                "items": items,
                "meta": {
                    "total": len(items),
                    "reviewed": terminal_count,
                    "remaining": len(items) - terminal_count,
                    "high_priority": high_priority,
                    "status_counts": status_counts,
                    "revision": self._revision,
                    "source": self.csv_path.name,
                },
            }

    def update(self, alignment_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"Fields are not editable: {sorted(unknown)}")
        if not changes:
            raise ValueError("No review changes were supplied")
        normalized: dict[str, str] = {}
        for key, value in changes.items():
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            normalized[key] = value.replace("\r\n", "\n").replace("\r", "\n")
        if "review_status" in normalized:
            normalized["review_status"] = normalized["review_status"].strip()
            if normalized["review_status"] not in REVIEW_STATUSES:
                raise ValueError(
                    "review_status must be needs_review, defer, accept, localized, or exclude"
                )
        with self._lock:
            row = self._by_id.get(alignment_id)
            if row is None:
                raise KeyError(alignment_id)
            candidate = {**row, **normalized}
            note = candidate["review_note"].strip()
            if candidate["review_status"] == "exclude" and not note:
                raise ValueError("A reviewer note is required when excluding an alignment")
            has_override = bool(
                candidate["reviewed_english"].strip()
                or candidate["reviewed_portuguese"].strip()
            )
            if has_override and not note:
                raise ValueError("A reviewer note is required when overriding either text")

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
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.csv_path.name}.",
            suffix=".tmp",
            dir=self.csv_path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._fieldnames,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(self._rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.csv_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


class HumanReviewStore(ReviewStore):
    """CSV-backed state for blinded translation preference review."""

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Review CSV does not exist: {self.csv_path}")
        with self.csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Review CSV has no header")
            missing = HUMAN_REQUIRED_FIELDS - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Human review CSV is missing fields: {sorted(missing)}")
            rows = [{key: value or "" for key, value in row.items()} for row in reader]
        identifiers = [row["sample_id"] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Human review CSV contains duplicate sample_id values")
        invalid = {row["review_status"] for row in rows} - HUMAN_REVIEW_STATUSES
        if invalid:
            raise ValueError(f"Human review CSV contains invalid statuses: {sorted(invalid)}")
        self._fieldnames = list(reader.fieldnames)
        self._rows = rows
        self._by_id = {row["sample_id"]: row for row in rows}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            items = [dict(row) for row in self._rows]
            reviewed = sum(row["review_status"] == "reviewed" for row in self._rows)
            deferred = sum(row["review_status"] == "defer" for row in self._rows)
            status_counts = {
                status: sum(row["review_status"] == status for row in self._rows)
                for status in sorted(HUMAN_REVIEW_STATUSES)
            }
            return {
                "items": items,
                "meta": {
                    "total": len(items),
                    "reviewed": reviewed,
                    "remaining": len(items) - reviewed,
                    "deferred": deferred,
                    "status_counts": status_counts,
                    "revision": self._revision,
                    "source": self.csv_path.name,
                    "mode": "human_preference",
                },
            }

    def update(self, sample_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        unknown = set(changes) - HUMAN_EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"Fields are not editable: {sorted(unknown)}")
        if not changes:
            raise ValueError("No review changes were supplied")
        normalized: dict[str, str] = {}
        for key, value in changes.items():
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            normalized[key] = value.replace("\r\n", "\n").replace("\r", "\n")
        for field in ("choice_A_B_or_tie", "review_status", "confidence", "add_to_golden"):
            if field in normalized:
                normalized[field] = normalized[field].strip()
        if normalized.get("choice_A_B_or_tie", "") not in HUMAN_CHOICES:
            raise ValueError("choice_A_B_or_tie must be A, B, tie, or empty")
        if "review_status" in normalized and normalized["review_status"] not in HUMAN_REVIEW_STATUSES:
            raise ValueError("review_status must be needs_review, defer, or reviewed")
        if normalized.get("confidence", "") not in HUMAN_CONFIDENCE:
            raise ValueError("confidence must be low, medium, high, or empty")
        if "add_to_golden" in normalized and normalized["add_to_golden"] not in {"true", "false"}:
            raise ValueError("add_to_golden must be true or false")
        with self._lock:
            row = self._by_id.get(sample_id)
            if row is None:
                raise KeyError(sample_id)
            candidate = {**row, **normalized}
            if candidate["review_status"] == "reviewed":
                if candidate["choice_A_B_or_tie"] not in {"A", "B", "tie"}:
                    raise ValueError("Choose A, B, or tie before completing this review")
                if candidate["confidence"] not in {"low", "medium", "high"}:
                    raise ValueError("Choose a confidence level before completing this review")
            tags = {tag for tag in candidate["failure_tags"].split("|") if tag}
            if "other" in tags and not candidate["note"].strip():
                raise ValueError("A note is required when using the Other failure tag")
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


def make_handler(
    store: ReviewStore | HumanReviewStore, assets_dir: Path
) -> type[BaseHTTPRequestHandler]:
    assets = {
        "/": (assets_dir / "index.html", "text/html; charset=utf-8"),
        "/index.html": (assets_dir / "index.html", "text/html; charset=utf-8"),
        "/app.js": (assets_dir / "app.js", "text/javascript; charset=utf-8"),
        "/styles.css": (assets_dir / "styles.css", "text/css; charset=utf-8"),
    }

    class ReviewRequestHandler(BaseHTTPRequestHandler):
        server_version = "TranslationReview/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, message: str) -> None:
            self._json_response(status, {"error": message})

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json_response(
                    HTTPStatus.OK,
                    {"status": "ok", "revision": store.revision},
                )
                return
            if path == "/api/items":
                self._json_response(HTTPStatus.OK, store.snapshot())
                return
            asset = assets.get(path)
            if asset is None or not asset[0].is_file():
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = asset[0].read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", asset[1])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler API
            path = urlparse(self.path).path
            prefix = "/api/items/"
            if not path.startswith(prefix):
                self._error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._error(HTTPStatus.BAD_REQUEST, "Invalid request size")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._error(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON")
                return
            if not isinstance(payload, dict):
                self._error(HTTPStatus.BAD_REQUEST, "Request body must be an object")
                return
            alignment_id = unquote(path[len(prefix) :])
            try:
                updated = store.update(alignment_id, payload)
            except KeyError:
                self._error(HTTPStatus.NOT_FOUND, "Unknown alignment_id")
                return
            except ValueError as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self._json_response(HTTPStatus.OK, updated)

    return ReviewRequestHandler


def create_server(
    csv_path: Path,
    assets_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "alignment",
) -> tuple[ThreadingHTTPServer, ReviewStore | HumanReviewStore]:
    if mode not in {"alignment", "human"}:
        raise ValueError("mode must be alignment or human")
    store = HumanReviewStore(csv_path) if mode == "human" else ReviewStore(csv_path)
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Review UI assets do not exist: {assets_dir}")
    server = ThreadingHTTPServer((host, port), make_handler(store, assets_dir))
    server.daemon_threads = True
    return server, store


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["alignment", "human"], default="alignment")
    parser.add_argument(
        "--csv",
        type=Path,
    )
    parser.add_argument(
        "--assets",
        type=Path,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        help="defaults to 8765 for alignment review and 8766 for human preference review",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("For safety, the review server may bind only to localhost.")
    try:
        csv_path = args.csv or repo / "posts" / "data" / (
            "translation-eval-human-review.csv"
            if args.mode == "human"
            else "translation-eval-alignment-review.csv"
        )
        assets_dir = args.assets or repo / (
            "human_review_ui" if args.mode == "human" else "review_ui"
        )
        port = args.port if args.port is not None else (8766 if args.mode == "human" else 8765)
        server, store = create_server(
            csv_path, assets_dir, args.host, port, mode=args.mode
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from None
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"
    snapshot = store.snapshot()["meta"]
    item_name = "translation comparisons" if args.mode == "human" else "alignments"
    print(
        f"Reviewing {snapshot['total']} {item_name} from {snapshot['source']} at {url}",
        flush=True,
    )
    print("Press Ctrl+C to stop. Changes auto-save to the CSV.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
