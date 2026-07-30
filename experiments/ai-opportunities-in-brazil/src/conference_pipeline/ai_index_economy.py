"""Build the frozen 2025 AI Index/Quid country aggregates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pypdf import PdfReader

from .conference_indicators import read_panel
from .coverage_sensitivity import competition_ranks


COUNTRY_CODES = {
    "Argentina": "AR", "Brazil": "BR", "Canada": "CA", "Chile": "CL",
    "China": "CN", "Colombia": "CO", "France": "FR", "Germany": "DE",
    "India": "IN", "Indonesia": "ID", "Mexico": "MX", "South Africa": "ZA",
    "Turkey": "TR", "United Arab Emirates": "AE", "United Kingdom": "GB",
    "United States": "US",
}


def read_investment(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["Country"]: float(row["Total Investment (in Billions of U.S. Dollars)"])
            for row in rows
        }


def read_new_companies_chart(path: Path) -> dict[str, float]:
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    axis = next(i for i, line in enumerate(lines) if line.startswith("0 100 200"))
    label_end = lines.index("Number of companies")
    values = [float(line.replace(",", "")) for line in lines[:axis]]
    countries = lines[axis + 1 : label_end]
    if len(values) != len(countries):
        raise ValueError("Figure 4.2.9 values and country labels do not align")
    return dict(zip(countries, values, strict=True))


def indicator_rows(
    values: dict[str, float],
    panel: dict[str, dict[str, str]],
    indicator: str,
    unit: str,
) -> list[dict[str, object]]:
    reported = {
        COUNTRY_CODES[country]: value
        for country, value in values.items()
        if country in COUNTRY_CODES and COUNTRY_CODES[country] in panel
    }
    ranks = competition_ranks(reported)
    upper_bound = min(values.values())
    return [
        {
            "year": 2025,
            "country_code": code,
            "country_name": details["country_name"],
            "comparison_group": details["comparison_group"],
            "indicator": indicator,
            "unit": unit,
            "value": reported.get(code),
            "observation_status": "reported_top15" if code in reported else "not_reported_top15",
            "upper_bound_if_unreported": None if code in reported else upper_bound,
            "rank_among_reported_panel": ranks.get(code),
        }
        for code, details in panel.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--investment-csv", type=Path, required=True)
    parser.add_argument("--companies-pdf", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    panel = read_panel(args.panel)
    investment = read_investment(args.investment_csv)
    companies = read_new_companies_chart(args.companies_pdf)
    rows = indicator_rows(
        investment, panel, "total_ai_private_investment", "billions_current_usd"
    )
    rows += indicator_rows(
        companies, panel, "newly_funded_ai_companies", "companies"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "year": 2025,
        "source": "2026 AI Index report / Quid 2025",
        "investment_figure": "4.2.8",
        "new_companies_figure": "4.2.9",
        "reporting_rule": "top-15 absence is censored, not zero",
        "investment_top15_floor_billions_usd": min(investment.values()),
        "new_companies_top15_floor": min(companies.values()),
    }
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
