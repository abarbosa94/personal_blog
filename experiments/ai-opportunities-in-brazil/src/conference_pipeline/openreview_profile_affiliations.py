"""Recover paper countries from accepted authors' OpenReview profiles."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Iterable

from .http import HttpClient
from .io import read_reconciled, write_reconciled_jsonl
from .models import Affiliation, ReconciliationDiagnostic
from .reconcile import CountryMentionExtractor, RorAffiliationResolver


BASE_URL = "https://api2.openreview.net"
VENUE_ID = "ICLR.cc/2025/Conference"


def value(content: dict[str, Any], key: str, default: Any = None) -> Any:
    item = content.get(key, default)
    return item.get("value", default) if isinstance(item, dict) else item


def active_institution_names(history: Iterable[dict[str, Any]], year: int) -> tuple[str, ...]:
    names: list[str] = []
    for item in history or ():
        start = item.get("start")
        end = item.get("end")
        if start and int(start) > year:
            continue
        if end and int(end) < year:
            continue
        institution = item.get("institution") or {}
        name = institution.get("name") or institution.get("domain") or ""
        if name and name not in names:
            names.append(name)
    return tuple(names)


def resolve_name(raw: str, http: HttpClient) -> Affiliation | None:
    country = CountryMentionExtractor.single_country_code(raw)
    if country:
        return Affiliation("", raw, country, None)
    try:
        return RorAffiliationResolver(http).resolve(raw)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise RuntimeError("OpenReview credentials are required")

    import openreview

    client = openreview.api.OpenReviewClient(
        baseurl=BASE_URL, username=username, password=password
    )
    notes = client.get_all_notes(content={"venueid": VENUE_ID})
    author_ids_by_paper = {
        note.id: tuple(value(note.content, "authorids", ()) or ())
        for note in notes
    }
    author_ids = sorted(
        {
            author_id
            for ids in author_ids_by_paper.values()
            for author_id in ids
            if str(author_id).startswith("~")
        }
    )
    profiles = client.search_profiles(ids=author_ids)
    institution_names_by_author: dict[str, tuple[str, ...]] = {}
    for profile in profiles:
        content = profile.content or {}
        institution_names_by_author[profile.id] = active_institution_names(
            content.get("history") or (), args.year
        )

    raw_names = sorted(
        {
            name
            for names in institution_names_by_author.values()
            for name in names
        }
    )
    http = HttpClient(cache_dir=Path("artifacts/http-cache"), attempts=4)
    resolved: dict[str, Affiliation | None] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(resolve_name, name, http): name for name in raw_names
        }
        for future in as_completed(futures):
            resolved[futures[future]] = future.result()

    records = read_reconciled(args.input)
    output = []
    for record in records:
        affiliations = list(record.affiliations)
        for author_id in author_ids_by_paper.get(record.paper.paper_id, ()):
            for name in institution_names_by_author.get(author_id, ()):
                affiliation = resolved.get(name)
                if affiliation:
                    affiliations.append(affiliation)
        unique = {
            (item.institution_id, item.institution_name, item.country_code): item
            for item in affiliations
        }
        countries = sorted(
            {item.country_code for item in unique.values() if item.country_code}
        )
        diagnostic = ReconciliationDiagnostic(
            "openreview_profile_affiliations",
            "success" if countries else "unresolved",
            ",".join(countries),
        )
        output.append(
            replace(
                record,
                match_method="+".join(
                    filter(
                        None,
                        (
                            record.match_method,
                            "v9-openreview-profile-affiliations",
                        ),
                    )
                ),
                affiliations=tuple(unique.values()),
                diagnostics=(*record.diagnostics, diagnostic),
            )
        )
    count = write_reconciled_jsonl(output, args.output)
    with_country = sum(bool(record.countries) for record in output)
    print(
        f"Wrote {count} records; {with_country} with country; "
        f"{len(profiles)}/{len(author_ids)} profiles fetched; "
        f"{sum(value is not None for value in resolved.values())}/{len(raw_names)} "
        "unique institutions resolved",
        flush=True,
    )


if __name__ == "__main__":
    main()
