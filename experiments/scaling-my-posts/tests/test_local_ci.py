from __future__ import annotations

import json
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from local_ci import (  # noqa: E402
    DEPLOYMENT_TESTS,
    build_evidence,
    build_public_summary,
    directory_size,
    run_stage,
    sha256_file,
)


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "draft.ipynb"
    source.write_bytes(b'{"cells":[]}')

    assert sha256_file(source) == sha256_file(source)


def test_directory_size_counts_nested_files(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "one.txt").write_bytes(b"123")
    (nested / "two.txt").write_bytes(b"4567")

    assert directory_size(tmp_path) == 7


def test_directory_size_does_not_double_count_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"1234")
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(source)
    except OSError:
        return

    assert directory_size(tmp_path) == 4


def test_deployment_tests_exclude_screening_artifacts() -> None:
    selected = {path.name for path in DEPLOYMENT_TESTS}

    assert selected == {"test_local_ci.py", "test_translation_eval.py"}
    assert "test_screening.py" not in selected


def test_run_stage_records_success(tmp_path: Path) -> None:
    result = run_stage(
        "smoke",
        [sys.executable, "-c", "print('ready')"],
        cwd=tmp_path,
        timeout_seconds=10,
    )

    assert result.status == "PASS"
    assert result.output_tail == "ready"


def test_evidence_is_json_serializable(tmp_path: Path) -> None:
    notebook = Path("post.ipynb")
    (tmp_path / notebook).write_text('{"cells":[]}', encoding="utf-8")
    stage = run_stage(
        "smoke",
        [sys.executable, "-c", "print('ready')"],
        cwd=tmp_path,
        timeout_seconds=10,
    )

    evidence = build_evidence(
        repo=tmp_path,
        notebook=notebook,
        stages=[stage],
        started_at="2026-07-23T00:00:00+00:00",
        elapsed_seconds=1.0,
        model_smoke=None,
    )

    encoded = json.dumps(evidence)
    assert evidence["status"] == "PASS"
    assert '"model_smoke": {"status": "SKIPPED"}' in encoded


def test_public_summary_omits_commands_and_output(tmp_path: Path) -> None:
    notebook = Path("post.ipynb")
    (tmp_path / notebook).write_text('{"cells":[]}', encoding="utf-8")
    stage = run_stage(
        "smoke",
        [sys.executable, "-c", "print('private output')"],
        cwd=tmp_path,
        timeout_seconds=10,
    )
    evidence = build_evidence(
        repo=tmp_path,
        notebook=notebook,
        stages=[stage],
        started_at="2026-07-24T00:00:00+00:00",
        elapsed_seconds=1.0,
        model_smoke=None,
    )

    summary = build_public_summary(evidence)
    serialized = json.dumps(summary)

    assert summary["input"]["notebook_sha256"] == evidence["input"]["notebook_sha256"]
    assert summary["stages"] == [
        {"name": "smoke", "status": "PASS", "elapsed_seconds": stage.elapsed_seconds}
    ]
    assert "command" not in serialized
    assert "private output" not in serialized
