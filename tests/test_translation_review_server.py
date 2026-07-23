from __future__ import annotations

import csv
import json
import os
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from translation_review_server import HumanReviewStore, ReviewStore, create_server  # noqa: E402


FIELDNAMES = [
    "alignment_id",
    "pair_id",
    "english_cell",
    "portuguese_cell",
    "english_sentence_ids",
    "portuguese_sentence_ids",
    "english",
    "portuguese",
    "alignment_type",
    "labse_similarity",
    "transition_score",
    "review_priority",
    "automatic_warning",
    "review_status",
    "reviewed_english",
    "reviewed_portuguese",
    "review_note",
]


def review_row(index: int = 1, **overrides: str) -> dict[str, str]:
    row = {
        "alignment_id": f"p01-a{index:02d}",
        "pair_id": "1",
        "english_cell": "3",
        "portuguese_cell": "3",
        "english_sentence_ids": f"p01-en{index:02d}",
        "portuguese_sentence_ids": f"p01-pt{index:02d}",
        "english": f"English source {index}.",
        "portuguese": f"Fonte em português {index}.",
        "alignment_type": "1:1",
        "labse_similarity": "0.82",
        "transition_score": "0.81",
        "review_priority": "high" if index == 1 else "normal",
        "automatic_warning": "low similarity" if index == 1 else "",
        "review_status": "needs_review",
        "reviewed_english": "",
        "reviewed_portuguese": "",
        "review_note": "",
    }
    row.update(overrides)
    return row


def write_review_csv(path: Path, count: int = 3) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_row(index) for index in range(1, count + 1))
    return path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_human_review_csv(path: Path) -> Path:
    fields = [
        "sample_id", "direction", "source", "human_reference", "candidate_A",
        "candidate_B", "choice_A_B_or_tie", "review_status", "confidence",
        "failure_tags", "note", "add_to_golden",
    ]
    row = {
        "sample_id": "human-01", "direction": "en -> pt-BR", "source": "Source.",
        "human_reference": "ReferÃªncia.", "candidate_A": "Candidato A.",
        "candidate_B": "Candidato B.", "choice_A_B_or_tie": "",
        "review_status": "needs_review", "confidence": "", "failure_tags": "",
        "note": "", "add_to_golden": "false",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return path


def test_store_persists_unicode_and_preserves_source_fields(tmp_path: Path) -> None:
    csv_path = write_review_csv(tmp_path / "review.csv")
    store = ReviewStore(csv_path)

    result = store.update(
        "p01-a01",
        {"review_status": "accept", "review_note": "Tradução equivalente."},
    )

    assert result["revision"] == 1
    assert result["item"]["review_status"] == "accept"
    persisted = read_rows(csv_path)[0]
    assert persisted["review_note"] == "Tradução equivalente."
    assert persisted["english"] == "English source 1."
    assert persisted["portuguese"] == "Fonte em português 1."


def test_store_requires_rationale_for_exclusion_and_overrides(tmp_path: Path) -> None:
    store = ReviewStore(write_review_csv(tmp_path / "review.csv"))

    with pytest.raises(ValueError, match="note is required when excluding"):
        store.update("p01-a01", {"review_status": "exclude"})
    with pytest.raises(ValueError, match="note is required when overriding"):
        store.update("p01-a01", {"reviewed_portuguese": "Texto corrigido."})

    assert store.snapshot()["items"][0]["review_status"] == "needs_review"


def test_defer_is_recorded_but_remains_unresolved(tmp_path: Path) -> None:
    store = ReviewStore(write_review_csv(tmp_path / "review.csv"))

    store.update("p01-a01", {"review_status": "defer"})
    meta = store.snapshot()["meta"]

    assert meta["status_counts"]["defer"] == 1
    assert meta["reviewed"] == 0
    assert meta["remaining"] == 3


def test_human_store_requires_choice_and_confidence_then_persists(tmp_path: Path) -> None:
    csv_path = write_human_review_csv(tmp_path / "human.csv")
    store = HumanReviewStore(csv_path)

    with pytest.raises(ValueError, match="Choose A, B, or tie"):
        store.update("human-01", {"review_status": "reviewed"})
    with pytest.raises(ValueError, match="confidence"):
        store.update(
            "human-01", {"choice_A_B_or_tie": "A", "review_status": "reviewed"}
        )

    result = store.update(
        "human-01",
        {
            "choice_A_B_or_tie": "A",
            "review_status": "reviewed",
            "confidence": "high",
            "failure_tags": "terminology|locale",
            "add_to_golden": "true",
        },
    )
    assert result["item"]["choice_A_B_or_tie"] == "A"
    assert store.snapshot()["meta"]["reviewed"] == 1
    assert read_rows(csv_path)[0]["failure_tags"] == "terminology|locale"


def test_human_review_api_uses_dedicated_assets_and_mode(tmp_path: Path) -> None:
    csv_path = write_human_review_csv(tmp_path / "human.csv")
    server, _store = create_server(
        csv_path, REPO_ROOT / "human_review_ui", port=0, mode="human"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, payload = request_json(f"{base_url}/api/items")
        assert status == 200
        assert payload["meta"]["mode"] == "human_preference"
        assert payload["items"][0]["candidate_A"] == "Candidato A."
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request_json(url: str, *, method: str = "GET", payload: dict[str, str] | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=5) as response:  # noqa: S310 - localhost fixture
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_http_api_loads_and_updates_review_data(tmp_path: Path) -> None:
    csv_path = write_review_csv(tmp_path / "review.csv")
    server, _store = create_server(csv_path, REPO_ROOT / "review_ui", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, payload = request_json(f"{base_url}/api/items")
        assert status == 200
        assert payload["meta"]["total"] == 3
        assert payload["items"][0]["portuguese"] == "Fonte em português 1."

        status, payload = request_json(
            f"{base_url}/api/items/p01-a01",
            method="PATCH",
            payload={"review_status": "localized", "review_note": "Adapted example."},
        )
        assert status == 200
        assert payload["item"]["review_status"] == "localized"
        assert read_rows(csv_path)[0]["review_note"] == "Adapted example."

        status, payload = request_json(
            f"{base_url}/api/items/p01-a01",
            method="PATCH",
            payload={"pair_id": "changed"},
        )
        assert status == 400
        assert "not editable" in payload["error"]

        status, payload = request_json(
            f"{base_url}/api/items/missing",
            method="PATCH",
            payload={"review_status": "accept"},
        )
        assert status == 404
        assert payload["error"] == "Unknown alignment_id"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def installed_chrome() -> Path | None:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ]
    return next((path for path in candidates if path.is_file()), None)


def test_browser_review_workflow(tmp_path: Path) -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    browser_path = installed_chrome()
    if browser_path is None:
        pytest.skip("Chrome or Edge is required for the browser workflow test")

    csv_path = write_review_csv(tmp_path / "review.csv")
    server, _store = create_server(csv_path, REPO_ROOT / "review_ui", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=str(browser_path),
                headless=True,
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            console_errors: list[str] = []
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.goto(base_url, wait_until="networkidle")

            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a01")
            sync_api.expect(page.locator("#englishText")).to_contain_text("English source 1")
            sync_api.expect(page.locator("#portugueseText")).to_contain_text("Fonte em português 1")
            sync_api.expect(page.locator("#progressLabel")).to_have_text("0 of 3 reviewed")

            page.locator(".decision-exclude").click()
            sync_api.expect(page.locator("#validationMessage")).to_contain_text("rationale")
            assert read_rows(csv_path)[0]["review_status"] == "needs_review"

            page.locator("#reviewNote").fill("Navigation text; exclude from the benchmark.")
            page.locator(".decision-exclude").click()
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a02")
            assert read_rows(csv_path)[0]["review_status"] == "exclude"

            page.keyboard.press("1")
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a03")
            assert read_rows(csv_path)[1]["review_status"] == "accept"

            page.keyboard.press("u")
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a02")
            sync_api.expect(page.locator("#statusBadge")).to_have_text("Unreviewed")
            assert read_rows(csv_path)[1]["review_status"] == "needs_review"

            page.keyboard.press("1")
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a03")
            page.keyboard.press("d")
            sync_api.expect(page.locator("#statusBadge")).to_have_text("Deferred")
            sync_api.expect(page.locator("#progressLabel")).to_have_text("2 of 3 reviewed")
            sync_api.expect(page.locator("#remainingLabel")).to_have_text("1 remaining")
            assert read_rows(csv_path)[2]["review_status"] == "defer"

            page.reload(wait_until="networkidle")
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a03")
            sync_api.expect(page.locator("#statusBadge")).to_have_text("Deferred")

            page.locator("#jumpInput").fill("p01-a01")
            page.locator("#jumpButton").click()
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a01")
            sync_api.expect(page.locator("#reviewNote")).to_have_value(
                "Navigation text; exclude from the benchmark."
            )
            page.reload(wait_until="networkidle")
            sync_api.expect(page.locator("#statusBadge")).to_have_text("Excluded")

            page.locator("#nextButton").click()
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a02")
            page.keyboard.press("ArrowRight")
            sync_api.expect(page.locator("#alignmentBadge")).to_have_text("p01-a03")
            sync_api.expect(page.locator("#positionLabel")).to_have_text("3 of 3")

            page.locator("#shortcutsButton").click()
            sync_api.expect(page.locator("#shortcutsDialog")).to_be_visible()
            page.locator("#shortcutsDialog .dialog-close").click()
            sync_api.expect(page.locator("#shortcutsDialog")).not_to_be_visible()
            if screenshot_directory := os.environ.get("REVIEW_UI_SCREENSHOT_DIR"):
                destination = Path(screenshot_directory)
                destination.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=destination / "review-wide.png", full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.screenshot(path=destination / "review-mobile.png", full_page=True)
            assert console_errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
