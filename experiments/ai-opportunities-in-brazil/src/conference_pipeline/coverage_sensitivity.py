"""Quantify how missing country affiliations affect conference indicators."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .conference_indicators import read_panel
from .io import read_reconciled
from .models import ReconciledPaper


def competition_ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    previous: float | None = None
    rank = 0
    for position, (code, value) in enumerate(ordered, start=1):
        if previous is None or value != previous:
            rank = position
            previous = value
        ranks[code] = rank
    return ranks


def sensitivity_rows(
    records: Iterable[ReconciledPaper],
    panel: dict[str, dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_venue: dict[str, list[ReconciledPaper]] = defaultdict(list)
    for record in records:
        by_venue[record.paper.venue_key].append(record)

    rows: list[dict[str, object]] = []
    coverage: dict[str, dict[str, object]] = {}
    for venue, venue_records in sorted(by_venue.items()):
        universe = len(venue_records)
        observed = sum(bool(record.countries) for record in venue_records)
        missing = universe - observed
        rate = observed / universe if universe else 0.0
        full: dict[str, float] = {}
        fractional: dict[str, float] = {}
        for code in panel:
            full[code] = sum(code in record.countries for record in venue_records)
            fractional[code] = sum(
                1.0 / len(record.countries)
                for record in venue_records
                if code in record.countries
            )
        full_ranks = competition_ranks(full)
        fractional_ranks = competition_ranks(fractional)
        for code, country in panel.items():
            count = full[code]
            fractional_count = fractional[code]
            rows.append(
                {
                    "venue": venue,
                    "country_code": code,
                    "country_name": country["country_name"],
                    "comparison_group": country["comparison_group"],
                    "paper_universe": universe,
                    "papers_with_country": observed,
                    "papers_missing_country": missing,
                    "country_coverage": round(rate, 8),
                    "papers_full_count": int(count),
                    "papers_fractional_count": round(fractional_count, 6),
                    "observed_full_share": round(
                        count / observed if observed else 0.0, 8
                    ),
                    "observed_fractional_share": round(
                        fractional_count / observed if observed else 0.0, 8
                    ),
                    "mar_estimated_full_count": round(
                        count / rate if rate else 0.0, 6
                    ),
                    "population_share_lower_bound": round(
                        count / universe if universe else 0.0, 8
                    ),
                    "population_share_upper_bound": round(
                        (count + missing) / universe if universe else 0.0, 8
                    ),
                    "full_rank_in_panel": full_ranks[code],
                    "fractional_rank_in_panel": fractional_ranks[code],
                    "rank_change_full_to_fractional": (
                        fractional_ranks[code] - full_ranks[code]
                    ),
                }
            )
        coverage[venue] = {
            "paper_universe": universe,
            "papers_with_country": observed,
            "papers_missing_country": missing,
            "country_coverage": rate,
            "coverage_warning": rate < 0.9,
        }
    metadata = {
        "venues": coverage,
        "assumptions": {
            "observed_shares": (
                "Denominator is papers with at least one resolved country."
            ),
            "mar_estimate": (
                "Missing papers have the same country distribution as observed papers."
            ),
            "lower_bound": "No missing paper belongs to the country.",
            "upper_bound": "Every missing paper belongs to the country.",
            "rank_scope": "Frozen 16-country comparison panel only.",
        },
    }
    return rows, metadata


def write_outputs(
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    output: Path,
    metadata_output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()
    records = [
        record for path in args.inputs for record in read_reconciled(path)
    ]
    rows, metadata = sensitivity_rows(records, read_panel(args.panel))
    write_outputs(rows, metadata, args.output, args.metadata_output)
    print(f"Wrote {len(rows)} sensitivity rows")


if __name__ == "__main__":
    main()
