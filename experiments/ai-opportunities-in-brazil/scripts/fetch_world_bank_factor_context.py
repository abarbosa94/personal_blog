"""Freeze World Bank factor indicators for the preregistered country panel."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "analysis-specs" / "country-comparison-panel.csv"
RAW = ROOT / "data" / "raw" / "external" / "world-bank" / str(date.today())
OUTPUT = ROOT / "artifacts" / "analysis" / "world-bank-factor-context.csv"
MANIFEST = ROOT / "artifacts" / "analysis" / "world-bank-factor-context-manifest.json"
INDICATORS = {
    "GB.XPD.RSDV.GD.ZS": "research_and_development_expenditure_percent_gdp",
    "SP.POP.TOTL": "population",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "BrazilAIAdvantageResearch/1.0"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def main() -> None:
    with PANEL.open(encoding="utf-8-sig", newline="") as handle:
        panel = list(csv.DictReader(handle))
    countries = ";".join(row["country_code"] for row in panel)
    names = {row["country_code"]: row["country_name"] for row in panel}
    RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    for indicator, label in INDICATORS.items():
        query = urlencode({"format": "json", "per_page": 20000, "date": "2018:2025"})
        url = f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?{query}"
        payload = download(url)
        raw_path = RAW / f"{indicator}.json"
        raw_path.write_bytes(payload)
        files.append({
            "path": str(raw_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "url": url,
        })
        parsed = json.loads(payload)
        records = parsed[1] if len(parsed) > 1 and parsed[1] else []
        by_country: dict[str, list[dict[str, object]]] = {}
        for record in records:
            # The frozen panel uses ISO alpha-2; the API exposes it in country.id.
            code = record["country"]["id"]
            if code in names and record["value"] is not None:
                by_country.setdefault(code, []).append(record)
        for code in names:
            available = sorted(
                by_country.get(code, []), key=lambda item: int(item["date"]), reverse=True
            )
            latest = available[0] if available else None
            output_rows.append({
                "country_code": code,
                "country_name": names[code],
                "indicator_code": indicator,
                "indicator": label,
                "value": latest["value"] if latest else "",
                "year": latest["date"] if latest else "",
                "observation_status": "observed" if latest else "missing_2018_2025",
            })

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    MANIFEST.write_text(
        json.dumps(
            {
                "retrieved_on": str(date.today()),
                "source": "World Bank Indicators API",
                "files": files,
                "processed_output": str(OUTPUT.relative_to(ROOT)),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
