"""Build frozen equal-venue, paper-weighted, and leave-one-out conference pools."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .conference_indicators import read_panel
from .coverage_sensitivity import competition_ranks
from .io import read_reconciled
from .models import ReconciledPaper


def pooled_rows(
    by_venue: dict[str, list[ReconciledPaper]],
    panel: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    observed = {
        venue: sum(bool(record.countries) for record in records)
        for venue, records in by_venue.items()
    }
    fractional: dict[str, dict[str, float]] = {}
    for venue, records in by_venue.items():
        fractional[venue] = {
            code: sum(
                1 / len(record.countries)
                for record in records
                if code in record.countries
            )
            for code in panel
        }

    scenarios = [("all_venues", tuple(sorted(by_venue)))]
    scenarios.extend(
        (
            f"leave_out_{excluded}",
            tuple(venue for venue in sorted(by_venue) if venue != excluded),
        )
        for excluded in sorted(by_venue)
    )
    rows: list[dict[str, object]] = []
    for scenario, venues in scenarios:
        estimates: dict[str, dict[str, float]] = {
            "equal_venue_fractional_share": {},
            "paper_weighted_fractional_share": {},
        }
        for code in panel:
            shares = [
                fractional[venue][code] / observed[venue]
                for venue in venues
                if observed[venue]
            ]
            estimates["equal_venue_fractional_share"][code] = (
                sum(shares) / len(shares) if shares else 0.0
            )
            denominator = sum(observed[venue] for venue in venues)
            estimates["paper_weighted_fractional_share"][code] = (
                sum(fractional[venue][code] for venue in venues) / denominator
                if denominator
                else 0.0
            )
        for estimator, values in estimates.items():
            ranks = competition_ranks(values)
            for code, country in panel.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "estimator": estimator,
                        "country_code": code,
                        "country_name": country["country_name"],
                        "comparison_group": country["comparison_group"],
                        "venue_count": len(venues),
                        "venues": "|".join(venues),
                        "estimate": round(values[code], 10),
                        "rank_in_panel": ranks[code],
                    }
                )
    return rows


def gate_metadata(
    by_venue: dict[str, list[ReconciledPaper]],
    official_totals: dict[str, int],
) -> dict[str, object]:
    venues: dict[str, dict[str, object]] = {}
    rates: list[float] = []
    for venue, records in sorted(by_venue.items()):
        official = official_totals[venue]
        with_country = sum(bool(record.countries) for record in records)
        rate = with_country / official
        rates.append(rate)
        venues[venue] = {
            "official_total": official,
            "enumerated_records": len(records),
            "papers_with_country": with_country,
            "papers_without_country": official - with_country,
            "country_coverage": rate,
            "passes_90_percent": rate >= 0.9,
        }
    spread = max(rates) - min(rates) if rates else 0.0
    return {
        "contract_version": "3.0",
        "venues": venues,
        "coverage_floor": 0.9,
        "coverage_spread": spread,
        "coverage_spread_limit": 0.15,
        "all_venues_pass_coverage": all(
            item["passes_90_percent"] for item in venues.values()
        ),
        "passes_coverage_spread": spread <= 0.15,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--official-total", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    args = parser.parse_args()
    totals = {
        venue: int(total)
        for venue, total in (value.split("=", 1) for value in args.official_total)
    }
    by_venue: dict[str, list[ReconciledPaper]] = defaultdict(list)
    for path in args.inputs:
        for record in read_reconciled(path):
            by_venue[record.paper.venue_key].append(record)
    if set(by_venue) != set(totals):
        raise ValueError(
            f"Venue mismatch: records={sorted(by_venue)}, totals={sorted(totals)}"
        )
    rows = pooled_rows(dict(by_venue), read_panel(args.panel))
    metadata = gate_metadata(dict(by_venue), totals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} pooled rows; "
        f"coverage_gate={metadata['all_venues_pass_coverage']}; "
        f"spread_gate={metadata['passes_coverage_spread']}"
    )


if __name__ == "__main__":
    main()
