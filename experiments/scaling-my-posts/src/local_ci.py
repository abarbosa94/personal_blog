"""Run the translation experiment in a runner-like environment and record evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import resource
except ImportError:  # pragma: no cover - Windows development host
    resource = None


EXPERIMENT = Path("experiments/scaling-my-posts")
DEFAULT_NOTEBOOK = Path("posts/2026-07-23-Scaling-MyPost-WithAIAgents.ipynb")
DEFAULT_OUTPUT = EXPERIMENT / "artifacts/local-ci/execution-evidence.json"
TOWER_MODEL = "Unbabel/Tower-Plus-2B"
TOWER_REVISION = "4d779ca939174189c0677d4a75642d36d6a33b66"
DEPLOYMENT_TESTS = (
    EXPERIMENT / "tests/test_local_ci.py",
    EXPERIMENT / "tests/test_translate_notebook.py",
    EXPERIMENT / "tests/test_translate_notebook_bdd.py",
    EXPERIMENT / "tests/test_translation_eval.py",
)


@dataclass(frozen=True)
class StageResult:
    name: str
    command: list[str]
    status: str
    elapsed_seconds: float
    output_tail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cgroup_value(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def directory_size(
    path: Path,
    *,
    excluded_directory_names: frozenset[str] = frozenset(),
) -> int | None:
    if not path.exists():
        return None
    total = 0
    try:
        for root, directories, files in os.walk(path):
            directories[:] = [
                name
                for name in directories
                if name not in excluded_directory_names
                and not (Path(root) / name).is_symlink()
            ]
            for name in files:
                item = Path(root) / name
                if not item.is_symlink():
                    total += item.stat().st_size
    except OSError:
        return None
    return total


def run_stage(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
) -> StageResult:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout or ""
    tail = "\n".join(output.splitlines()[-40:])
    result = StageResult(
        name=name,
        command=list(command),
        status="PASS" if completed.returncode == 0 else "FAIL",
        elapsed_seconds=round(elapsed, 3),
        output_tail=tail,
    )
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(asdict(result), ensure_ascii=False))
    return result


def run_tower_smoke(repo: Path) -> dict[str, object]:
    sys.path.insert(0, str((repo / EXPERIMENT / "src").resolve()))
    from translation_benchmark import TowerTranslator

    translator = TowerTranslator("cpu")
    cases = [
        ("en", "pt-BR", "The translation workflow runs locally before deployment."),
        ("pt-BR", "en", "A tradução é revisada antes da publicação."),
    ]
    outputs = [
        {
            "source_language": source,
            "target_language": target,
            "source": text,
            "prediction": translator.translate([text], source, target)[0],
        }
        for source, target, text in cases
    ]
    if any(not str(case["prediction"]).strip() for case in outputs):
        raise RuntimeError("Tower+ returned an empty smoke-test translation")
    return {
        "status": "PASS",
        "model": TOWER_MODEL,
        "revision": TOWER_REVISION,
        "parameters": translator.parameter_count,
        "cases": outputs,
    }


def build_evidence(
    *,
    repo: Path,
    notebook: Path,
    stages: list[StageResult],
    started_at: str,
    elapsed_seconds: float,
    model_smoke: dict[str, object] | None,
) -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    child_usage = (
        resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
    )
    disk = shutil.disk_usage(repo)
    model_cache = Path(os.getenv("HF_HOME", "/model-cache"))
    return {
        "schema_version": 1,
        "status": "PASS",
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count_visible": os.cpu_count(),
            "declared_cpu_limit": os.getenv("CI_CPU_LIMIT"),
            "declared_memory_limit_bytes": os.getenv("CI_MEMORY_LIMIT_BYTES"),
            "declared_disk_budget_bytes": os.getenv("CI_DISK_BUDGET_BYTES"),
            "cgroup_memory_max": read_cgroup_value(
                Path("/sys/fs/cgroup/memory.max")
            ),
            "cgroup_cpu_max": read_cgroup_value(Path("/sys/fs/cgroup/cpu.max")),
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "repository_worktree_bytes_excluding_caches": directory_size(
                repo,
                excluded_directory_names=frozenset(
                    {
                        ".git",
                        ".pytest_cache",
                        ".quarto",
                        ".venv-benchmark",
                        ".venv-xcomet",
                        "__pycache__",
                        "_site",
                        "artifacts",
                    }
                ),
            ),
            "model_cache_bytes": directory_size(model_cache),
            "disk_budget_enforced": False,
            "peak_rss_self_kib": usage.ru_maxrss if usage else None,
            "peak_rss_children_kib": child_usage.ru_maxrss if child_usage else None,
        },
        "input": {
            "notebook": notebook.as_posix(),
            "notebook_sha256": sha256_file(repo / notebook),
        },
        "stages": [asdict(stage) for stage in stages],
        "model_smoke": model_smoke or {"status": "SKIPPED"},
        "limitations": [
            "This run does not exercise GitHub token permissions.",
            "This run does not create a real branch, pull request, or Pages preview.",
            "The 14 GB disk budget is reported but not enforced by Docker Compose.",
            "A successful local run is evidence of runner compatibility, not proof of deployment.",
        ],
    }


def build_public_summary(evidence: dict[str, object]) -> dict[str, object]:
    environment = evidence["environment"]
    assert isinstance(environment, dict)
    stages = evidence["stages"]
    assert isinstance(stages, list)
    return {
        "schema_version": 1,
        "status": evidence["status"],
        "generated_at": evidence["finished_at"],
        "input": evidence["input"],
        "resource_envelope": {
            "declared_cpu_limit": environment["declared_cpu_limit"],
            "cgroup_cpu_max": environment["cgroup_cpu_max"],
            "declared_memory_limit_bytes": environment[
                "declared_memory_limit_bytes"
            ],
            "cgroup_memory_max": environment["cgroup_memory_max"],
            "declared_disk_budget_bytes": environment[
                "declared_disk_budget_bytes"
            ],
            "disk_budget_enforced": environment["disk_budget_enforced"],
        },
        "measurements": {
            "elapsed_seconds": evidence["elapsed_seconds"],
            "repository_worktree_bytes_excluding_caches": environment[
                "repository_worktree_bytes_excluding_caches"
            ],
            "model_cache_bytes": environment["model_cache_bytes"],
            "peak_rss_self_kib": environment["peak_rss_self_kib"],
            "peak_rss_children_kib": environment["peak_rss_children_kib"],
        },
        "stages": [
            {
                "name": stage["name"],
                "status": stage["status"],
                "elapsed_seconds": stage["elapsed_seconds"],
            }
            for stage in stages
        ],
        "model_smoke": evidence["model_smoke"],
        "limitations": evidence["limitations"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--public-summary-output",
        type=Path,
        help="write a sanitized, portable evidence summary suitable for version control",
    )
    parser.add_argument("--with-model-smoke", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    notebook = args.notebook
    if not (repo / notebook).is_file():
        raise SystemExit(f"Notebook not found: {repo / notebook}")

    started_at = utc_now()
    started = time.perf_counter()
    stages = [
        run_stage(
            "experiment-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *(str(path) for path in DEPLOYMENT_TESTS),
            ],
            cwd=repo,
            timeout_seconds=args.timeout_seconds,
        ),
    ]
    if not args.skip_render:
        with tempfile.TemporaryDirectory(prefix="translation-ci-preview-") as preview:
            stages.append(
                run_stage(
                    "quarto-render",
                    [
                        "quarto",
                        "render",
                        notebook.as_posix(),
                        "--to",
                        "html",
                        "--output-dir",
                        preview,
                        "-M",
                        "draft:false",
                    ],
                    cwd=repo,
                    timeout_seconds=args.timeout_seconds,
                )
            )

    model_smoke = run_tower_smoke(repo) if args.with_model_smoke else None
    evidence = build_evidence(
        repo=repo,
        notebook=notebook,
        stages=stages,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        model_smoke=model_smoke,
    )
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.public_summary_output:
        summary_output = repo / args.public_summary_output
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(
                build_public_summary(evidence),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Public evidence summary: {summary_output}")
    print(f"Local CI rehearsal passed. Evidence: {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"Local CI rehearsal failed: {error}") from None
