"""Recover KDD ADS countries from affiliations on the official paper page."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .enumerators import _KddAdsParser
from .http import HttpClient
from .io import read_reconciled, write_reconciled_jsonl
from .models import Affiliation, ReconciledPaper, ReconciliationDiagnostic
from .reconcile import CountryMentionExtractor, RorAffiliationResolver


def parse_official_affiliations(
    html_text: str, year: int, url: str
) -> dict[str, tuple[str, ...]]:
    parser = _KddAdsParser(year, url)
    parser.feed(html_text)
    return parser.affiliations_by_doi


def resolve_official_affiliation(
    raw: str, http: HttpClient
) -> Affiliation | None:
    country = CountryMentionExtractor.single_country_code(raw)
    if country:
        return Affiliation("", raw, country, None)
    try:
        return RorAffiliationResolver(http).resolve(raw)
    except Exception:
        return None


def augment_record(
    record: ReconciledPaper,
    evidence: dict[str, tuple[str, ...]],
    resolved: dict[str, Affiliation | None],
) -> ReconciledPaper:
    doi = (record.paper.doi or "").lower()
    raw_values = evidence.get(doi, ())
    affiliations = tuple(
        affiliation
        for raw in raw_values
        if (affiliation := resolved.get(raw)) is not None
    )
    countries = sorted(
        {affiliation.country_code for affiliation in affiliations if affiliation.country_code}
    )
    diagnostic = ReconciliationDiagnostic(
        "kdd_official_affiliations",
        "success" if countries else "unresolved",
        ",".join(countries),
    )
    return replace(
        record,
        match_method="+".join(
            filter(None, (record.match_method, "v9-kdd-official-affiliations"))
        ),
        affiliations=tuple({(a.institution_id, a.institution_name): a for a in (
            *record.affiliations, *affiliations
        )}.values()),
        diagnostics=(*record.diagnostics, diagnostic),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--official-html", required=True, type=Path)
    parser.add_argument("--official-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    records = read_reconciled(args.input)
    evidence = parse_official_affiliations(
        args.official_html.read_text(encoding="utf-8"), 2025, args.official_url
    )
    raw_values = sorted({raw for values in evidence.values() for raw in values})
    http = HttpClient(cache_dir=Path("artifacts/http-cache"), attempts=4)
    resolved: dict[str, Affiliation | None] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(resolve_official_affiliation, raw, http): raw
            for raw in raw_values
        }
        for future in as_completed(futures):
            resolved[futures[future]] = future.result()

    augmented = [augment_record(record, evidence, resolved) for record in records]
    count = write_reconciled_jsonl(augmented, args.output)
    resolved_count = sum(value is not None for value in resolved.values())
    print(
        f"Wrote {count} records; resolved {resolved_count}/{len(raw_values)} "
        "unique official affiliations"
    )


if __name__ == "__main__":
    main()
