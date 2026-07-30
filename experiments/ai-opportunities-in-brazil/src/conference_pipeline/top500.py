"""Parse frozen official TOP500 pages and aggregate systems and Rmax by country."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

from .conference_indicators import read_panel


COUNTRY_CODES = {
    "Argentina": "AR", "Brazil": "BR", "Canada": "CA", "Chile": "CL",
    "China": "CN", "Colombia": "CO", "France": "FR", "Germany": "DE",
    "India": "IN", "Indonesia": "ID", "Mexico": "MX", "South Africa": "ZA",
    "Türkiye": "TR", "Turkey": "TR", "United Arab Emirates": "AE",
    "United Kingdom": "GB", "United States": "US",
}


class Top500Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = self.in_row = self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[list[str]] = []
        self.rows: list[list[list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class", "")
        if tag == "table" and "table-condensed" in classes:
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row, self.row = True, []
        elif self.in_row and tag == "td":
            self.in_cell, self.cell_parts = True, []

    def handle_data(self, data: str) -> None:
        if self.in_cell and data.strip():
            self.cell_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell:
            self.row.append(self.cell_parts)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if len(self.row) == 6:
                self.rows.append(self.row)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False


def parse_page(text: str) -> list[dict[str, object]]:
    parser = Top500Parser()
    parser.feed(text)
    rows: list[dict[str, object]] = []
    for cells in parser.rows:
        rows.append(
            {
                "rank": int(cells[0][0]),
                "country": cells[1][-1],
                "cores": int(cells[2][0].replace(",", "")),
                "rmax_pflops": float(cells[3][0].replace(",", "")),
            }
        )
    return rows


def aggregate(
    records: list[dict[str, object]],
    panel: dict[str, dict[str, str]],
    snapshot: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    values = {code: defaultdict(float) for code in panel}
    unmapped: dict[str, int] = defaultdict(int)
    for row in records:
        country = str(row["country"])
        code = COUNTRY_CODES.get(country)
        if code not in panel:
            unmapped[country] += 1
            continue
        values[code]["systems"] += 1
        values[code]["rmax_pflops"] += float(row["rmax_pflops"])
    output = [
        {
            "snapshot": snapshot,
            "country_code": code,
            "country_name": details["country_name"],
            "comparison_group": details["comparison_group"],
            "systems": int(values[code]["systems"]),
            "rmax_pflops": round(values[code]["rmax_pflops"], 6),
        }
        for code, details in panel.items()
    ]
    return output, {
        "snapshot": snapshot,
        "records": len(records),
        "unique_ranks": len({row["rank"] for row in records}),
        "rmax_unit": "PFlop/s (HPL)",
        "unmapped_non_panel_countries": dict(sorted(unmapped.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.input_dir.glob("page-*.html")):
        records.extend(parse_page(path.read_text(encoding="utf-8")))
    if len(records) != 500 or len({row["rank"] for row in records}) != 500:
        raise ValueError("TOP500 census must contain exactly 500 unique ranks")
    rows, metadata = aggregate(records, read_panel(args.panel), args.snapshot)
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
