"""Derive the frozen Epoch AI model indicators for the country panel."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from .conference_indicators import read_panel


COUNTRY_CODES = {
    "Argentina": "AR",
    "Brazil": "BR",
    "Canada": "CA",
    "Chile": "CL",
    "China": "CN",
    "Colombia": "CO",
    "France": "FR",
    "Germany": "DE",
    "India": "IN",
    "Indonesia": "ID",
    "Mexico": "MX",
    "South Africa": "ZA",
    "Turkey": "TR",
    "United Arab Emirates": "AE",
    "United Kingdom of Great Britain and Northern Ireland": "GB",
    "United States of America": "US",
}


def split_field(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def country_sectors(row: dict[str, str]) -> dict[str, str]:
    """Return one non-exclusive sector classification for each represented country."""
    countries = split_field(row.get("Country (of organization)", ""))
    categories = split_field(row.get("Organization categorization", ""))
    by_country: dict[str, set[str]] = defaultdict(set)
    for index, country in enumerate(countries):
        if country not in COUNTRY_CODES:
            continue
        category = categories[index] if index < len(categories) else ""
        if category in {"Industry", "Academia"}:
            by_country[COUNTRY_CODES[country]].add(category.lower())
        elif category:
            by_country[COUNTRY_CODES[country]].add("other")
    result: dict[str, str] = {}
    for code, sectors in by_country.items():
        if {"industry", "academia"} <= sectors:
            result[code] = "mixed"
        elif "industry" in sectors:
            result[code] = "industry"
        elif "academia" in sectors:
            result[code] = "academia"
        else:
            result[code] = "other"
    return result


def derive_rows(
    records: list[dict[str, str]],
    panel: dict[str, dict[str, str]],
    start: date,
    end: date,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    counts = {code: defaultdict(int) for code in panel}
    eligible = 0
    missing_country = 0
    for row in records:
        raw_date = row.get("Publication date", "")
        try:
            published = date.fromisoformat(raw_date)
        except ValueError:
            continue
        if not start <= published <= end or not row.get("Notability criteria", "").strip():
            continue
        eligible += 1
        sectors = country_sectors(row)
        if not sectors:
            missing_country += 1
        for code, sector in sectors.items():
            counts[code]["notable_models"] += 1
            counts[code][sector] += 1

    output: list[dict[str, object]] = []
    for code, details in panel.items():
        total = counts[code]["notable_models"]
        for sector in ("industry", "academia", "mixed", "other"):
            value = counts[code][sector]
            output.append(
                {
                    "period_start": start.isoformat(),
                    "period_end": end.isoformat(),
                    "country_code": code,
                    "country_name": details["country_name"],
                    "comparison_group": details["comparison_group"],
                    "indicator": "notable_ai_models",
                    "sector": sector,
                    "model_count": value,
                    "country_total": total,
                    "sector_share": round(value / total, 10) if total else None,
                }
            )
    metadata = {
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "eligible_notable_models": eligible,
        "eligible_models_without_mapped_country": missing_country,
        "country_attribution": "full count; one presence per model-country",
        "sector_rule": "industry, academia, mixed, or other among organizations paired to that country",
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    rows, metadata = derive_rows(records, read_panel(args.panel), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
