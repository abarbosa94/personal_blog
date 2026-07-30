"""Generate reproducible figures for the existing blog notebook."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "analysis"
OUTPUT = ROOT.parents[1] / "posts" / "images" / "competitive-advantage-ai"
BLUE = "#2563eb"
DARK = "#172033"
GRAY = "#a8b0bd"
LIGHT = "#e5e7eb"


def rows(name: str) -> list[dict[str, str]]:
    with (ANALYSIS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def color(code: str) -> str:
    return BLUE if code == "BR" else GRAY


def finish(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def conference_figure() -> None:
    values = [
        row for row in rows("conference-presence-2025-seven-venues-pooled.csv")
        if row["scenario"] == "all_venues"
        and row["estimator"] == "equal_venue_fractional_share"
    ]
    values.sort(key=lambda row: float(row["estimate"]))
    fig, ax = plt.subplots(figsize=(9, 7))
    estimates = [float(row["estimate"]) for row in values]
    ax.barh(
        [row["country_name"] for row in values],
        estimates,
        color=[color(row["country_code"]) for row in values],
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.set_xlabel("Participação fracionária média por conferência")
    ax.set_title(
        "Presença nas sete conferências selecionadas, 2025",
        loc="left", fontsize=15, weight="bold", color=DARK, pad=24,
    )
    ax.text(
        0, 1.005,
        "Média com peso igual por venue; o Brasil aparece em azul.",
        transform=ax.transAxes, color="#4b5563",
    )
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    finish(fig, "baseline-conference-presence-2025.png")


def frontier_infrastructure_figure() -> None:
    epoch = rows("epoch-notable-models-2025.csv")
    totals = {}
    names = {}
    for row in epoch:
        totals[row["country_code"]] = int(row["country_total"])
        names[row["country_code"]] = row["country_name"]
    ordered = sorted(totals, key=totals.get)
    top = rows("top500-2026-06.csv")
    top.sort(key=lambda row: float(row["rmax_pflops"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    ax.barh(
        [names[code] for code in ordered],
        [totals[code] for code in ordered],
        color=[color(code) for code in ordered],
    )
    ax.set_title("Modelos notáveis do Epoch, 2025", loc="left", weight="bold")
    ax.set_xlabel("Modelos atribuídos a organizações do país")
    ax.grid(axis="x", color=LIGHT)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.barh(
        [row["country_name"] for row in top],
        [float(row["rmax_pflops"]) for row in top],
        color=[color(row["country_code"]) for row in top],
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_title("Capacidade Rmax no TOP500, jun. 2026", loc="left", weight="bold")
    ax.set_xlabel("PFlop/s (escala simétrica logarítmica)")
    ax.grid(axis="x", color=LIGHT)
    ax.set_axisbelow(True)
    for ax in axes:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "Fronteira de modelos e infraestrutura contam histórias diferentes",
        x=0.06, ha="left", fontsize=16, weight="bold", color=DARK,
    )
    fig.subplots_adjust(top=0.86, wspace=0.38)
    finish(fig, "baseline-frontier-infrastructure.png")


def economy_figure() -> None:
    economy = rows("ai-index-economy-2025.csv")
    definitions = (
        ("total_ai_private_investment", "Investimento privado em IA", "US$ bilhões"),
        ("newly_funded_ai_companies", "Empresas financiadas pela primeira vez", "Empresas"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (indicator, title, xlabel) in zip(axes, definitions):
        observed = [
            row for row in economy
            if row["indicator"] == indicator and row["value"]
        ]
        observed.sort(key=lambda row: float(row["value"]))
        ax.barh(
            [row["country_name"] for row in observed],
            [float(row["value"]) for row in observed],
            color=GRAY,
        )
        brazil = next(
            row for row in economy
            if row["indicator"] == indicator and row["country_code"] == "BR"
        )
        bound = float(brazil["upper_bound_if_unreported"])
        # Brazil is censored outside the published top 15. Give the bound its
        # own row instead of drawing a reference line over every observed bar.
        brazil_y = -0.72
        ax.hlines(brazil_y, 0, bound, color=BLUE, linewidth=5)
        ax.plot(bound, brazil_y, marker="<", color=BLUE, markersize=8)
        bound_label = (
            f"Brasil: menos de US$ {bound:.3f} bi".replace(".", ",")
            if indicator == "total_ai_private_investment"
            else f"Brasil: menos de {bound:.0f}"
        )
        ax.text(
            0, brazil_y - 0.20, bound_label,
            ha="left", va="top", color=BLUE, fontsize=9, weight="bold",
        )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_ylim(-1.45, len(observed) - 0.35)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=LIGHT)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "O Brasil está fora do top 15 econômico publicado para 2025",
        x=0.06, ha="left", fontsize=16, weight="bold", color=DARK,
    )
    fig.subplots_adjust(top=0.84, wspace=0.42)
    finish(fig, "baseline-economic-conversion-2025.png")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": DARK,
        "xtick.color": "#4b5563",
        "ytick.color": "#374151",
    })
    conference_figure()
    frontier_infrastructure_figure()
    economy_figure()


if __name__ == "__main__":
    main()
