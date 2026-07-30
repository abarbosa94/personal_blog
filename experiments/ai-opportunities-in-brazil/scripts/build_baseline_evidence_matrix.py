"""Build the frozen seven-indicator baseline evidence matrix."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "analysis"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rank(rows: list[dict[str, str]], field: str, code: str = "BR") -> int:
    values = sorted(
        ((float(row[field]), row["country_code"]) for row in rows),
        reverse=True,
    )
    return next(index for index, (_, country) in enumerate(values, 1) if country == code)


def main() -> None:
    epoch_2025 = read_csv("epoch-notable-models-2025.csv")
    epoch_2026 = read_csv("epoch-notable-models-2026-ytd.csv")
    top_2025 = read_csv("top500-2025-11.csv")
    top_2026 = read_csv("top500-2026-06.csv")
    economy = read_csv("ai-index-economy-2025.csv")
    conferences = read_csv("conference-presence-2025-seven-venues-pooled.csv")

    br_epoch_2025 = next(row for row in epoch_2025 if row["country_code"] == "BR")
    br_epoch_2026 = next(row for row in epoch_2026 if row["country_code"] == "BR")
    br_top_2025 = next(row for row in top_2025 if row["country_code"] == "BR")
    br_top_2026 = next(row for row in top_2026 if row["country_code"] == "BR")
    br_investment = next(
        row for row in economy
        if row["country_code"] == "BR"
        and row["indicator"] == "total_ai_private_investment"
    )
    br_companies = next(
        row for row in economy
        if row["country_code"] == "BR"
        and row["indicator"] == "newly_funded_ai_companies"
    )
    br_conference = next(
        row for row in conferences
        if row["country_code"] == "BR"
        and row["scenario"] == "all_venues"
        and row["estimator"] == "equal_venue_fractional_share"
    )
    rows = [
        {
            "indicator_id": "BASE-2025-NOTABLE-MODELS",
            "family": "frontier_production",
            "period": "2025; 2026 YTD",
            "brazil_observation": (
                f"{br_epoch_2025['country_total']} models in 2025; "
                f"{br_epoch_2026['country_total']} in 2026 YTD"
            ),
            "panel_position": "no observed model",
            "evidence_direction": "challenges",
            "interpretation": "No Brazilian organization is attributed a notable model in the frozen Epoch universe.",
            "limitation": "Dataset absence is evidence about observed frontier presence, not proof of no domestic capability.",
            "source_artifact": "epoch-notable-models-2025.csv|epoch-notable-models-2026-ytd.csv",
        },
        {
            "indicator_id": "BASE-2025-MODEL-MIX",
            "family": "frontier_production",
            "period": "2025; 2026 YTD",
            "brazil_observation": "undefined because country_total=0",
            "panel_position": "not applicable",
            "evidence_direction": "inconclusive",
            "interpretation": "Academia–industry composition cannot be estimated without an observed model.",
            "limitation": "Do not convert an undefined composition into zero industry participation.",
            "source_artifact": "epoch-notable-models-2025.csv|epoch-notable-models-2026-ytd.csv",
        },
        {
            "indicator_id": "BASE-2025-2026-AI-CONF-COUNTRY",
            "family": "scientific_presence",
            "period": "2025",
            "brazil_observation": f"{100 * float(br_conference['estimate']):.4f}% equal-venue fractional share",
            "panel_position": f"rank {br_conference['rank_in_panel']} of 16",
            "evidence_direction": "challenges",
            "interpretation": "Observed presence is small, second in Latin America, and partly dependent on ICML.",
            "limitation": "Measures selected accepted-paper venues, not total national scientific output or impact.",
            "source_artifact": "conference-presence-2025-seven-venues-pooled.csv",
        },
        {
            "indicator_id": "BASE-2025-2026-SUPERCOMPUTERS",
            "family": "infrastructure",
            "period": "2025-11; 2026-06",
            "brazil_observation": f"{br_top_2025['systems']} systems; {br_top_2026['systems']} systems",
            "panel_position": f"2025 panel rank {rank(top_2025, 'systems')}; 2026 panel rank {rank(top_2026, 'systems')}",
            "evidence_direction": "mixed",
            "interpretation": "Brazil has visible and persistent regional HPC scale but remains far below the global frontier.",
            "limitation": "TOP500 HPC count is not equivalent to accessible AI training compute.",
            "source_artifact": "top500-2025-11.csv|top500-2026-06.csv",
        },
        {
            "indicator_id": "BASE-2025-2026-RMAX",
            "family": "infrastructure",
            "period": "2025-11; 2026-06",
            "brazil_observation": f"{br_top_2025['rmax_pflops']} PFlop/s; {br_top_2026['rmax_pflops']} PFlop/s",
            "panel_position": f"2025 panel rank {rank(top_2025, 'rmax_pflops')}; 2026 panel rank {rank(top_2026, 'rmax_pflops')}",
            "evidence_direction": "mixed",
            "interpretation": "Capacity increased, supports a regional infrastructure signal, but the frontier gap remains material.",
            "limitation": "Rmax is benchmark capacity, not observed availability, suitability, utilization, or cost for AI.",
            "source_artifact": "top500-2025-11.csv|top500-2026-06.csv",
        },
        {
            "indicator_id": "BASE-2025-PRIVATE-INVESTMENT",
            "family": "economic_conversion",
            "period": "2025",
            "brazil_observation": f"< US$ {float(br_investment['upper_bound_if_unreported']):.3f}B",
            "panel_position": "outside published global top 15",
            "evidence_direction": "challenges",
            "interpretation": "The published aggregate does not show investment scale near the leading markets.",
            "limitation": "Censored upper bound; the value is not zero and cannot be precisely ranked.",
            "source_artifact": "ai-index-economy-2025.csv",
        },
        {
            "indicator_id": "BASE-2025-FUNDED-COMPANIES",
            "family": "economic_conversion",
            "period": "2025",
            "brazil_observation": f"< {int(float(br_companies['upper_bound_if_unreported']))} companies",
            "panel_position": "outside published global top 15",
            "evidence_direction": "challenges",
            "interpretation": "The published aggregate does not show broad first-funding conversion at leading-market scale.",
            "limitation": "Censored upper bound; private-company coverage and the definition of AI company remain source-dependent.",
            "source_artifact": "ai-index-economy-2025.csv",
        },
    ]
    csv_path = ANALYSIS / "baseline-seven-indicator-evidence-matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Initial seven-indicator baseline evidence matrix",
        "",
        "| Family | Indicator | Brazil | Panel position | Direction |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['family']} | {row['indicator_id']} | "
            f"{row['brazil_observation']} | {row['panel_position']} | "
            f"{row['evidence_direction']} |"
        )
    md.extend([
        "",
        "No aggregate score is calculated. Responsible AI is deferred and absent",
        "from this matrix after its blind validation failure.",
    ])
    (ANALYSIS / "baseline-seven-indicator-evidence-matrix.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
