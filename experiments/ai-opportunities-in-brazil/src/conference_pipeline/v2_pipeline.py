"""Build, augment, and sample untouched reconciliation records for v2 review."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import json
import random
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Iterable

from .affiliation_names import canonical_affiliation_name
from .http import HttpClient
from .io import read_reconciled, write_reconciled_jsonl
from .models import Affiliation, ReconciledPaper, ReconciliationDiagnostic
from .ner_pilot import spacy_organization_candidates
from .reconcile import (
    OpenAlexReconciler,
    PdfAffiliationExtractor,
    RorAffiliationResolver,
)
if TYPE_CHECKING:
    from .search_evidence import EvidenceSearch


def untouched_ids(queue: Path) -> set[str]:
    with queue.open(encoding="utf-8", newline="") as handle:
        return {
            row["review_id"]
            for row in csv.DictReader(handle)
            if row["review_status"] == "needs_review"
        }


def queue_ids(queue: Path, statuses: set[str] | None = None) -> set[str]:
    with queue.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["review_id"]
            for row in csv.DictReader(handle)
            if statuses is None or row["review_status"] in statuses
        }


def select_records(
    queue: Path,
    sources: Iterable[Path],
    *,
    statuses: set[str] | None = None,
    complement: bool = False,
) -> list[ReconciledPaper]:
    selected = queue_ids(queue, statuses)
    records: dict[str, ReconciledPaper] = {}
    seen: set[str] = set()
    for source in sources:
        for record in read_reconciled(source):
            review_id = f"{record.paper.venue_key}:{record.paper.paper_id}"
            seen.add(review_id)
            if (review_id in selected) != complement:
                records[review_id] = record
    if not complement:
        missing = selected - seen
        if missing:
            raise ValueError(f"Selected records missing from sources: {sorted(missing)}")
    return sorted(
        records.values(),
        key=lambda record: (record.paper.venue_key, record.paper.paper_id),
    )


def build_untouched(
    queue: Path, sources: Iterable[Path]
) -> list[ReconciledPaper]:
    wanted = untouched_ids(queue)
    records: dict[str, ReconciledPaper] = {}
    for source in sources:
        for record in read_reconciled(source):
            review_id = f"{record.paper.venue_key}:{record.paper.paper_id}"
            if review_id in wanted:
                records[review_id] = record
    missing = wanted - set(records)
    if missing:
        raise ValueError(f"Untouched records missing from sources: {sorted(missing)}")
    return sorted(
        records.values(),
        key=lambda record: (record.paper.venue_key, record.paper.paper_id),
    )


def _merge_affiliations(values: Iterable[Affiliation]) -> tuple[Affiliation, ...]:
    merged: dict[tuple[str, str], Affiliation] = {}
    for value in values:
        if PdfAffiliationExtractor._is_non_author_organization(
            value.institution_name
        ):
            continue
        canonical_name = canonical_affiliation_name(value.institution_name)
        if not canonical_name:
            continue
        key = (canonical_name, value.country_code or "")
        previous = merged.get(key)
        if previous and previous.country_code and not value.country_code:
            continue
        if (
            previous
            and not previous.country_code
            and not value.country_code
            and len(previous.institution_name) <= len(value.institution_name)
        ):
            # Equivalent front-matter fragments should retain the least noisy
            # display label, which also produces a more precise search query.
            continue
        merged[key] = value
    resolved_names = {
        name for name, country in merged if country
    }
    return tuple(
        value
        for (name, country), value in merged.items()
        if country or name not in resolved_names
    )


def augment_search_evidence(
    records: Iterable[ReconciledPaper],
    evidence_search: "EvidenceSearch",
    search_max_authors: int = 1,
) -> list[ReconciledPaper]:
    """Add only web-evidence decisions to an already enriched frozen version."""

    from .search_evidence import evaluate_affiliation_evidence

    output: list[ReconciledPaper] = []
    for record in records:
        affiliations = list(record.affiliations)
        diagnostics = list(record.diagnostics)
        unresolved = [
            item
            for item in _merge_affiliations(affiliations)
            if not item.country_code and item.institution_name
        ]
        for affiliation in unresolved:
            for author in record.paper.authors[:search_max_authors]:
                try:
                    evidence = evidence_search.affiliation_evidence(
                        author,
                        affiliation.institution_name,
                        str(record.paper.year),
                        paper_title=record.paper.title,
                    )
                    decision = evaluate_affiliation_evidence(evidence)
                except Exception as error:
                    diagnostics.append(
                        ReconciliationDiagnostic(
                            "search_affiliation_evidence",
                            "api_error",
                            f"{type(error).__name__}:{affiliation.institution_name[:80]}",
                        )
                    )
                    continue
                detail = {
                    "author": author,
                    "organization": affiliation.institution_name,
                    "country_code": decision.country_code,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "retrieved_at": evidence.get("retrieved_at"),
                    "evidence": list(decision.evidence),
                }
                diagnostics.append(
                    ReconciliationDiagnostic(
                        "search_affiliation_evidence",
                        decision.decision,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    )
                )
                if decision.decision == "auto_assign":
                    affiliations.append(
                        Affiliation(
                            affiliation.institution_id,
                            affiliation.institution_name,
                            decision.country_code,
                            affiliation.institution_type,
                        )
                    )
                    break
        output.append(
            ReconciledPaper(
                paper=record.paper,
                openalex_id=record.openalex_id,
                match_method="+".join(
                    filter(None, (record.match_method, "v9-search-evidence"))
                ),
                affiliations=_merge_affiliations(affiliations),
                diagnostics=tuple(diagnostics),
            )
        )
    return output


def augment_records(
    records: Iterable[ReconciledPaper],
    http: HttpClient,
    model: str = "en_core_web_sm",
    evidence_search: EvidenceSearch | None = None,
    search_max_authors: int = 1,
    nlp=None,
) -> list[ReconciledPaper]:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("Install the optional 'ner' dependencies") from error
    nlp = nlp or spacy.load(model)
    pdf_reconciler = OpenAlexReconciler(http, use_openalex=False)
    ror = RorAffiliationResolver(http)
    output: list[ReconciledPaper] = []
    for baseline in records:
        pdf_result = pdf_reconciler.reconcile(baseline.paper)
        working_paper = pdf_result.paper
        # Preserve only affiliations obtained directly from a successful
        # OpenAlex work. Old PDF/ROR fallback values must be recomputed, or a
        # later precision fix can never remove earlier false positives.
        baseline_method = baseline.match_method or ""
        trusted_baseline = (
            bool(baseline.openalex_id)
            and "fallback" not in baseline_method
            and "pdf+ror" not in baseline_method
        )
        baseline_affiliations = (
            list(baseline.affiliations) if trusted_baseline else []
        )
        affiliations = list(pdf_result.affiliations)
        diagnostics = list(baseline.diagnostics) + list(pdf_result.diagnostics)
        candidates: tuple[str, ...] = ()
        region = ""
        chosen = 0
        if working_paper.pdf_url:
            try:
                pdf = http.get_bytes(working_paper.pdf_url)
                region = PdfAffiliationExtractor.affiliation_region(
                    PdfAffiliationExtractor.extract_text(pdf)
                )
                candidates = spacy_organization_candidates(region, model, nlp)
                for candidate in candidates:
                    try:
                        affiliation = ror.resolve(candidate)
                    except Exception as error:
                        diagnostics.append(
                            ReconciliationDiagnostic(
                                "spacy_ror",
                                "api_error",
                                f"{type(error).__name__}:{candidate[:80]}",
                            )
                        )
                        continue
                    if affiliation:
                        affiliations.append(affiliation)
                        chosen += 1
            except Exception as error:
                diagnostics.append(
                    ReconciliationDiagnostic(
                        "spacy_ner", "api_error", type(error).__name__
                    )
                )
        if trusted_baseline:
            named_pdf_evidence = any(
                not affiliation.institution_name.startswith(
                    "Country explicitly stated"
                )
                for affiliation in pdf_result.affiliations
            )
            if region and named_pdf_evidence:
                support_text = f"{region}\n" + "\n".join(candidates)
                accepted = [
                    affiliation
                    for affiliation in baseline_affiliations
                    if RorAffiliationResolver.pdf_supports_affiliation(
                        affiliation, support_text
                    )
                ]
                rejected = [
                    affiliation.institution_name
                    for affiliation in baseline_affiliations
                    if affiliation not in accepted
                ]
                affiliations = accepted + affiliations
                diagnostics.append(
                    ReconciliationDiagnostic(
                        "openalex_affiliation_validation",
                        "rejected" if rejected else "success",
                        "; ".join(rejected),
                    )
                )
            else:
                # Preserve OpenAlex recall when the authoritative PDF cannot be
                # read; validation is impossible rather than negative.
                affiliations = baseline_affiliations + affiliations
                diagnostics.append(
                    ReconciliationDiagnostic(
                        "openalex_affiliation_validation",
                        "skipped",
                        "PDF organization evidence unavailable",
                    )
                )
        diagnostics.append(
            ReconciliationDiagnostic(
                "spacy_ner",
                "success" if candidates else "no_organization_detected",
                f"{len(candidates)} candidates; {chosen} ROR chosen",
            )
        )
        branch_affiliations = RorAffiliationResolver.author_branch_affiliations(
            working_paper.authors,
            affiliations,
            candidates,
            working_paper.title,
        )
        affiliations.extend(branch_affiliations)
        if branch_affiliations:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "author_branch_registry",
                    "success",
                    ",".join(
                        sorted(
                            item.country_code
                            for item in branch_affiliations
                            if item.country_code
                        )
                    ),
                )
            )
        enriched = ReconciledPaper(
            paper=working_paper,
            openalex_id=baseline.openalex_id,
            match_method="+".join(
                filter(None, [baseline.match_method, "v2-pdf-ner"])
            ),
            affiliations=_merge_affiliations(affiliations),
            diagnostics=tuple(diagnostics),
        )
        if evidence_search:
            enriched = augment_search_evidence(
                [enriched], evidence_search, search_max_authors
            )[0]
        output.append(
            enriched
        )
    return output


def sample_records(
    records: list[ReconciledPaper], size: int, seed: int
) -> list[ReconciledPaper]:
    if not 0 < size <= len(records):
        raise ValueError("sample size must be between one and the population size")
    indexes = sorted(random.Random(seed).sample(range(len(records)), size))
    return [records[index] for index in indexes]


def complement_records(
    records: list[ReconciledPaper], size: int, seed: int
) -> list[ReconciledPaper]:
    sampled = {
        (record.paper.venue_key, record.paper.paper_id)
        for record in sample_records(records, size, seed)
    }
    return [
        record
        for record in records
        if (record.paper.venue_key, record.paper.paper_id) not in sampled
    ]


def missing_country_records(
    records: Iterable[ReconciledPaper],
) -> list[ReconciledPaper]:
    return [record for record in records if not record.countries]


def overlay_records(
    baseline: Iterable[ReconciledPaper],
    updates: Iterable[ReconciledPaper],
) -> list[ReconciledPaper]:
    replacements = {
        (record.paper.venue_key, record.paper.paper_id): record
        for record in updates
    }
    values = []
    for record in baseline:
        key = (record.paper.venue_key, record.paper.paper_id)
        values.append(replacements.pop(key, record))
    if replacements:
        raise ValueError(
            f"Update records are absent from baseline: {sorted(replacements)}"
        )
    return values


def merge_evidence_records(
    baseline: Iterable[ReconciledPaper],
    updates: Iterable[ReconciledPaper],
) -> list[ReconciledPaper]:
    """Union evidence from update records into the complete baseline universe."""

    update_by_key = {
        (record.paper.venue_key, record.paper.paper_id): record
        for record in updates
    }
    values: list[ReconciledPaper] = []
    for record in baseline:
        key = (record.paper.venue_key, record.paper.paper_id)
        update = update_by_key.pop(key, None)
        if update is None:
            values.append(record)
            continue
        methods = list(
            dict.fromkeys(
                filter(
                    None,
                    (
                        *(record.match_method or "").split("+"),
                        *(update.match_method or "").split("+"),
                    ),
                )
            )
        )
        diagnostics = tuple(
            dict.fromkeys((*record.diagnostics, *update.diagnostics))
        )
        values.append(
            ReconciledPaper(
                paper=record.paper,
                openalex_id=update.openalex_id or record.openalex_id,
                match_method="+".join(methods) or None,
                affiliations=_merge_affiliations(
                    (*record.affiliations, *update.affiliations)
                ),
                diagnostics=diagnostics,
            )
        )
    if update_by_key:
        raise ValueError(
            f"Update records are absent from baseline: {sorted(update_by_key)}"
        )
    return values


def augment_pdf_country_record(
    record: ReconciledPaper,
    http: HttpClient,
) -> ReconciledPaper:
    diagnostics = list(record.diagnostics)
    affiliations = list(record.affiliations)
    if not record.paper.pdf_url:
        diagnostics.append(
            ReconciliationDiagnostic(
                "pdf_country_census", "not_available", "No official PDF URL"
            )
        )
    else:
        try:
            pdf = http.get_bytes(record.paper.pdf_url)
            codes = PdfAffiliationExtractor.country_codes(pdf)
            existing = {item.country_code for item in affiliations}
            for code in codes:
                if code in existing:
                    continue
                affiliations.append(
                    Affiliation(
                        "",
                        f"Country explicitly stated in PDF affiliation block ({code})",
                        code,
                        None,
                    )
                )
            diagnostics.append(
                ReconciliationDiagnostic(
                    "pdf_country_census",
                    "success" if codes else "unresolved",
                    ",".join(codes),
                )
            )
        except Exception as error:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "pdf_country_census", "api_error", type(error).__name__
                )
            )
    return ReconciledPaper(
        paper=record.paper,
        openalex_id=record.openalex_id,
        match_method="+".join(
            filter(None, (record.match_method, "v9-pdf-country"))
        ),
        affiliations=_merge_affiliations(affiliations),
        diagnostics=tuple(diagnostics),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-input")
    build.add_argument("queue", type=Path)
    build.add_argument("sources", nargs="+", type=Path)
    build.add_argument("--output", required=True, type=Path)
    select = commands.add_parser("select")
    select.add_argument("queue", type=Path)
    select.add_argument("sources", nargs="+", type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--status", action="append")
    select.add_argument("--complement", action="store_true")
    augment = commands.add_parser("augment")
    augment.add_argument("input", type=Path)
    augment.add_argument("--output", required=True, type=Path)
    augment.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    augment.add_argument("--model", default="en_core_web_sm")
    augment.add_argument("--http-attempts", type=int, default=3)
    augment.add_argument("--http-timeout", type=float, default=60.0)
    search_mode = augment.add_mutually_exclusive_group()
    search_mode.add_argument(
        "--search-evidence",
        dest="search_evidence",
        action="store_true",
        help="Use cached web evidence search (the V9 default)",
    )
    search_mode.add_argument(
        "--no-search-evidence",
        dest="search_evidence",
        action="store_false",
        help="Disable V9 web evidence search for an offline/PDF-only run",
    )
    augment.set_defaults(search_evidence=True)
    augment.add_argument("--search-max-authors", type=int, default=1)
    augment.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded concurrent augmentation workers (default: 1).",
    )
    augment.add_argument(
        "--resume",
        action="store_true",
        help="Resume augmentation from an existing output.",
    )
    augment.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Persist output after every N newly augmented records.",
    )
    search_augment = commands.add_parser("search-augment")
    search_augment.add_argument("input", type=Path)
    search_augment.add_argument("--output", required=True, type=Path)
    search_augment.add_argument("--search-max-authors", type=int, default=1)
    sample = commands.add_parser("sample")
    sample.add_argument("input", type=Path)
    sample.add_argument("--output", required=True, type=Path)
    sample.add_argument("--size", required=True, type=int)
    sample.add_argument("--seed", default=20250727, type=int)
    sample.add_argument("--complement", action="store_true")
    missing = commands.add_parser("missing-country")
    missing.add_argument("input", type=Path)
    missing.add_argument("--output", required=True, type=Path)
    overlay = commands.add_parser("overlay")
    overlay.add_argument("baseline", type=Path)
    overlay.add_argument("updates", type=Path)
    overlay.add_argument("--output", required=True, type=Path)
    merge_evidence = commands.add_parser("merge-evidence")
    merge_evidence.add_argument("baseline", type=Path)
    merge_evidence.add_argument("updates", type=Path)
    merge_evidence.add_argument("--output", required=True, type=Path)
    pdf_country = commands.add_parser("pdf-country-augment")
    pdf_country.add_argument("input", type=Path)
    pdf_country.add_argument("--output", required=True, type=Path)
    pdf_country.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    pdf_country.add_argument("--http-attempts", type=int, default=2)
    pdf_country.add_argument("--http-timeout", type=float, default=30.0)
    pdf_country.add_argument("--workers", type=int, default=5)
    pdf_country.add_argument("--checkpoint-every", type=int, default=100)
    pdf_country.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "build-input":
        records = build_untouched(args.queue, args.sources)
    elif args.command == "select":
        records = select_records(
            args.queue,
            args.sources,
            statuses=set(args.status) if args.status else None,
            complement=args.complement,
        )
    elif args.command == "augment":
        evidence_search = None
        if args.search_evidence and args.workers == 1:
            from .affiliation_search_mcp import create_search

            evidence_search = create_search()
        if args.checkpoint_every < 0 or args.workers < 1:
            raise ValueError("checkpoint-every must be non-negative and workers positive")
        source_records = read_reconciled(args.input)
        records = (
            read_reconciled(args.output)
            if args.resume and args.output.exists()
            else []
        )
        completed = {
            (record.paper.venue_key, record.paper.paper_id)
            for record in records
        }
        pending = [
            record
            for record in source_records
            if (record.paper.venue_key, record.paper.paper_id) not in completed
        ]
        if args.workers == 1:
            batch_size = args.checkpoint_every or max(1, len(pending))
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                records.extend(
                    augment_records(
                        batch,
                        HttpClient(
                            cache_dir=args.cache_dir,
                            attempts=args.http_attempts,
                            timeout_seconds=args.http_timeout,
                        ),
                        args.model,
                        evidence_search,
                        args.search_max_authors,
                    )
                )
                if args.checkpoint_every:
                    write_reconciled_jsonl(records, args.output)
                    print(
                        f"Checkpointed {len(records)}/{len(source_records)} records",
                        flush=True,
                    )
        else:
            worker_state = threading.local()

            def augment_one(record: ReconciledPaper) -> ReconciledPaper:
                if not hasattr(worker_state, "http"):
                    import spacy

                    worker_state.http = HttpClient(
                        cache_dir=args.cache_dir,
                        attempts=args.http_attempts,
                        timeout_seconds=args.http_timeout,
                    )
                    worker_state.nlp = spacy.load(args.model)
                    worker_state.search = None
                    if args.search_evidence:
                        from .affiliation_search_mcp import create_search

                        worker_state.search = create_search()
                return augment_records(
                    [record],
                    worker_state.http,
                    args.model,
                    worker_state.search,
                    args.search_max_authors,
                    nlp=worker_state.nlp,
                )[0]

            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                in_flight = set()
                for _ in range(min(len(pending), args.workers * 2)):
                    in_flight.add(executor.submit(augment_one, next(iterator)))
                completed_now = 0
                while in_flight:
                    done, in_flight = wait(
                        in_flight, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        records.append(future.result())
                        completed_now += 1
                        if (
                            args.checkpoint_every
                            and completed_now % args.checkpoint_every == 0
                        ):
                            write_reconciled_jsonl(records, args.output)
                            print(
                                f"Checkpointed {len(records)}/{len(source_records)} records",
                                flush=True,
                            )
                        try:
                            record = next(iterator)
                        except StopIteration:
                            continue
                        in_flight.add(executor.submit(augment_one, record))
    elif args.command == "search-augment":
        from .affiliation_search_mcp import create_search

        records = augment_search_evidence(
            read_reconciled(args.input),
            create_search(),
            args.search_max_authors,
        )
    elif args.command == "missing-country":
        records = missing_country_records(read_reconciled(args.input))
    elif args.command == "overlay":
        records = overlay_records(
            read_reconciled(args.baseline),
            read_reconciled(args.updates),
        )
    elif args.command == "merge-evidence":
        records = merge_evidence_records(
            read_reconciled(args.baseline),
            read_reconciled(args.updates),
        )
    elif args.command == "pdf-country-augment":
        if args.workers < 1 or args.checkpoint_every < 1:
            raise ValueError("workers and checkpoint-every must be positive")
        source_records = read_reconciled(args.input)
        records = (
            read_reconciled(args.output)
            if args.resume and args.output.exists()
            else []
        )
        completed = {
            (record.paper.venue_key, record.paper.paper_id)
            for record in records
        }
        pending = [
            record
            for record in source_records
            if (record.paper.venue_key, record.paper.paper_id) not in completed
        ]
        http = HttpClient(
            cache_dir=args.cache_dir,
            attempts=args.http_attempts,
            timeout_seconds=args.http_timeout,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(augment_pdf_country_record, record, http): record
                for record in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if index % args.checkpoint_every == 0:
                    write_reconciled_jsonl(records, args.output)
                    print(
                        f"Checkpointed {len(records)}/{len(source_records)} records",
                        flush=True,
                    )
    else:
        population = list(read_reconciled(args.input))
        records = (
            complement_records(population, args.size, args.seed)
            if args.complement
            else sample_records(population, args.size, args.seed)
        )
    count = write_reconciled_jsonl(records, args.output)
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
