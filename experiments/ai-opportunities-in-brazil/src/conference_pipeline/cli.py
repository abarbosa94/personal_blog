from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
import random
from pathlib import Path

from .enumerators import (
    AclAnthologyEnumerator,
    CrossrefAaaaiEnumerator,
    IclrOpenReviewEnumerator,
    KddAdsEnumerator,
    NeuripsEnumerator,
    PmlrEnumerator,
)
from .http import HttpClient
from .io import (
    read_papers,
    read_reconciled,
    write_jsonl,
    write_reconciled_jsonl,
)
from .quality import build_quality_report
from .reconcile import OpenAlexReconciler


DEFAULTS = {
    "aaai": {"year": 2025, "volume": "39"},
    "icml": {"year": 2025, "url": "https://proceedings.mlr.press/v267/"},
    "neurips": {
        "year": 2025,
        "url": "https://proceedings.neurips.cc/paper_files/paper/2025",
    },
    "acl": {
        "year": 2025,
        "volume_id": "2025.acl-long",
        "url": "https://aclanthology.org/volumes/2025.acl-long/",
    },
    "emnlp": {
        "year": 2025,
        "volume_id": "2025.emnlp-main",
        "url": "https://aclanthology.org/volumes/2025.emnlp-main/",
    },
    "iclr": {"year": 2025, "venue_id": "ICLR.cc/2025/Conference"},
    "kdd-ads": {
        "year": 2025,
        "url": "https://www.kdd.org/kdd2025/applied-data-science-ads-track-papers-2/",
    },
}


def reconcile_bounded(
    papers: list[Paper],
    reconciler: OpenAlexReconciler,
    workers: int,
):
    """Yield reconciliations while keeping only a small bounded queue in flight."""

    iterator = iter(papers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        in_flight = set()
        for _ in range(min(len(papers), workers * 2)):
            in_flight.add(executor.submit(reconciler.reconcile, next(iterator)))
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                try:
                    paper = next(iterator)
                except StopIteration:
                    continue
                in_flight.add(executor.submit(reconciler.reconcile, paper))


def select_validation_sample(
    papers: list,
    sample_size: int | None,
    seed: int,
) -> list:
    """Select a deterministic random sample while preserving sampled order."""
    if sample_size is None:
        return papers
    if sample_size < 1:
        raise ValueError("--sample-size must be greater than zero")
    if sample_size > len(papers):
        raise ValueError(
            f"--sample-size {sample_size} exceeds the {len(papers)} available papers"
        )
    indexes = sorted(random.Random(seed).sample(range(len(papers)), sample_size))
    return [papers[index] for index in indexes]


def reviewed_paper_ids(queues: list[Path]) -> set[tuple[str, str]]:
    """Read venue/paper identifiers from one or more manual-review queues."""

    values: set[tuple[str, str]] = set()
    for queue in queues:
        with queue.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                venue = (row.get("venue") or "").strip()
                paper_id = (row.get("paper_id") or "").strip()
                if venue and paper_id:
                    values.add((venue, paper_id))
    return values


def api_error_papers(records: list) -> list:
    return [
        record.paper
        for record in records
        if any(diagnostic.outcome == "api_error" for diagnostic in record.diagnostics)
    ]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Enumerate official conference papers")
    commands = value.add_subparsers(dest="command", required=True)

    enumerate_parser = commands.add_parser("enumerate")
    enumerate_parser.add_argument("--venue", choices=sorted(DEFAULTS), required=True)
    enumerate_parser.add_argument("--output", type=Path, required=True)

    reconcile_parser = commands.add_parser("reconcile")
    reconcile_parser.add_argument("--input", type=Path, required=True)
    reconcile_parser.add_argument("--output", type=Path, required=True)
    reconcile_parser.add_argument(
        "--limit",
        type=int,
        help="Reconcile only the first N papers. Intended for smoke tests.",
    )
    reconcile_parser.add_argument(
        "--sample-size",
        type=int,
        help="Reconcile a deterministic random validation sample.",
    )
    reconcile_parser.add_argument(
        "--seed",
        type=int,
        default=20250727,
        help="Random seed used with --sample-size (default: 20250727).",
    )
    reconcile_parser.add_argument(
        "--exclude-review-queue",
        action="append",
        type=Path,
        default=[],
        help="Exclude paper IDs found in this review CSV; may be repeated.",
    )
    reconcile_parser.add_argument(
        "--http-attempts",
        type=int,
        default=3,
        help="Attempts per uncached HTTP request (default: 3).",
    )
    reconcile_parser.add_argument(
        "--http-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds per uncached HTTP attempt (default: 60).",
    )
    reconcile_parser.add_argument(
        "--http-backoff",
        type=float,
        default=1.0,
        help="Initial exponential retry delay in seconds (default: 1).",
    )
    reconcile_parser.add_argument(
        "--http-max-backoff",
        type=float,
        default=3600.0,
        help="Maximum retry delay, including Retry-After (default: 3600).",
    )
    reconcile_parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Skip OpenAlex and rerun only the PDF/ROR fallback.",
    )
    reconcile_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output, skipping completed paper IDs.",
    )
    reconcile_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Rewrite a resumable output after every N new papers.",
    )
    reconcile_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded concurrent reconciliation workers (default: 1).",
    )
    reconcile_parser.add_argument(
        "--no-affiliation-fallback",
        action="store_true",
        help="Fast OpenAlex census pass; defer PDF/GROBID fallback.",
    )

    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--input", type=Path, required=True)
    audit_parser.add_argument("--official-total", type=int, required=True)
    audit_parser.add_argument("--output", type=Path)
    retry_parser = commands.add_parser("extract-api-error-papers")
    retry_parser.add_argument("--input", required=True, type=Path)
    retry_parser.add_argument("--output", required=True, type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "reconcile":
        if args.limit is not None and args.sample_size is not None:
            raise ValueError("--limit and --sample-size cannot be used together")
        papers = read_papers(args.input)
        excluded = reviewed_paper_ids(args.exclude_review_queue)
        if excluded:
            papers = [
                paper
                for paper in papers
                if (paper.venue_key, paper.paper_id) not in excluded
            ]
        if args.limit is not None:
            papers = papers[: args.limit]
        else:
            papers = select_validation_sample(papers, args.sample_size, args.seed)
        if args.checkpoint_every < 0:
            raise ValueError("--checkpoint-every must not be negative")
        if args.workers < 1:
            raise ValueError("--workers must be at least one")
        reconciler = OpenAlexReconciler(
            HttpClient(
                cache_dir=Path("artifacts/http-cache"),
                attempts=args.http_attempts,
                backoff_seconds=args.http_backoff,
                max_backoff_seconds=args.http_max_backoff,
                timeout_seconds=args.http_timeout,
            ),
            api_key=os.environ.get("OPENALEX_API_KEY"),
            use_openalex=not args.pdf_only,
            use_affiliation_fallback=not args.no_affiliation_fallback,
        )
        reconciled = (
            read_reconciled(args.output)
            if args.resume and args.output.exists()
            else []
        )
        completed = {
            (record.paper.venue_key, record.paper.paper_id)
            for record in reconciled
        }
        pending = [
            paper
            for paper in papers
            if (paper.venue_key, paper.paper_id) not in completed
        ]
        def checkpoint(index: int) -> None:
            if args.checkpoint_every and index % args.checkpoint_every == 0:
                write_reconciled_jsonl(reconciled, args.output)
                print(
                    f"Checkpointed {len(reconciled)}/{len(papers)} papers",
                    flush=True,
                )

        if args.workers == 1:
            for index, paper in enumerate(pending, start=1):
                reconciled.append(reconciler.reconcile(paper))
                checkpoint(index)
        else:
            for index, result in enumerate(
                reconcile_bounded(pending, reconciler, args.workers),
                start=1,
            ):
                reconciled.append(result)
                checkpoint(index)
        count = write_reconciled_jsonl(reconciled, args.output)
        print(f"Wrote {count} reconciled papers to {args.output}")
        return 0

    if args.command == "audit":
        report = build_quality_report(read_reconciled(args.input), args.official_total)
        value = {
            "official_total": report.official_total,
            "enumerated_total": report.enumerated_total,
            "reconciled_total": report.reconciled_total,
            "with_country_total": report.with_country_total,
            "enumeration_rate": report.enumeration_rate,
            "reconciliation_rate": report.reconciliation_rate,
            "country_coverage": report.country_coverage,
            "passes": report.passes(),
        }
        rendered = json.dumps(value, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    if args.command == "extract-api-error-papers":
        papers = api_error_papers(read_reconciled(args.input))
        count = write_jsonl(papers, args.output)
        print(f"Wrote {count} API-error papers to {args.output}")
        return 0

    config = DEFAULTS[args.venue]
    http = HttpClient(cache_dir=Path("artifacts/http-cache"))
    if args.venue == "aaai":
        papers = CrossrefAaaaiEnumerator(http).enumerate(
            config["year"], config["volume"]
        )
    elif args.venue == "icml":
        papers = PmlrEnumerator(http).enumerate(
            "icml", config["year"], config["url"]
        )
    elif args.venue == "neurips":
        papers = NeuripsEnumerator(http).enumerate(config["year"], config["url"])
    elif args.venue == "iclr":
        papers = IclrOpenReviewEnumerator().enumerate(
            config["year"], config["venue_id"]
        )
    elif args.venue == "kdd-ads":
        papers = KddAdsEnumerator(http).enumerate(config["year"], config["url"])
    else:
        papers = AclAnthologyEnumerator(http).enumerate(
            args.venue, config["year"], config["volume_id"], config["url"]
        )
    count = write_jsonl(papers, args.output)
    print(f"Wrote {count} {args.venue.upper()} papers to {args.output}")
    return 0
