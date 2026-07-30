"""Screen accepted papers for manual Responsible AI thematic validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


DIMENSIONS = {
    "privacy_data_governance": (
        r"\bprivacy\b", r"\bprivate data\b", r"\bdifferential(?:ly)? private\b",
        r"\bdata governance\b", r"\bmachine unlearning\b",
        r"\bmembership inference\b", r"\bdata deletion\b",
    ),
    "transparency_explainability": (
        r"\bexplainab", r"\binterpretability\b", r"\binterpretable\b",
        r"\bmodel transparency\b", r"\btransparent ai\b",
        r"\bfeature attribution\b", r"\bcounterfactual explanation",
    ),
    "security_safety": (
        r"\badversarial attack", r"\badversarial example", r"\bbackdoor",
        r"\bdata poisoning\b", r"\bprompt injection\b", r"\bjailbreak",
        r"\bai safety\b", r"\bmodel safety\b", r"\balignment\b",
        r"\bharmless", r"\bharmful content\b", r"\btoxicity\b",
        r"\bred team", r"\bmisuse\b",
    ),
    "fairness": (
        r"\bfairness\b", r"\balgorithmic fairness\b", r"\bdiscrimination\b",
        r"\bequity\b", r"\bdemographic parity\b", r"\bequalized odds\b",
        r"\bsocial bias\b", r"\bgender bias\b", r"\bracial bias\b",
    ),
}

AMBIGUOUS = (
    r"\bbias(?:es|ed)?\b", r"\bsafe(?:ty)?\b", r"\bsecure\b", r"\brobust(?:ness)?\b",
    r"\btrustworthy\b", r"\baccountab", r"\bethic",
)


def screen_title(title: str) -> tuple[tuple[str, ...], bool]:
    normalized = title.casefold()
    dimensions = tuple(
        dimension
        for dimension, patterns in DIMENSIONS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    )
    ambiguous = bool(any(re.search(pattern, normalized) for pattern in AMBIGUOUS))
    return dimensions, ambiguous


def read_candidates(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                paper = record["paper"]
                dimensions, ambiguous = screen_title(paper["title"])
                if not dimensions and not ambiguous:
                    continue
                rows.append(
                    {
                        "venue": paper["venue_key"],
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "dimensions": "|".join(dimensions),
                        "ambiguous_trigger": ambiguous,
                        "screen_status": (
                            "dimension_candidate" if dimensions else "ambiguous_context_review"
                        ),
                        "manual_label": "",
                        "manual_dimensions": "",
                        "review_notes": "",
                    }
                )
    return rows


def read_screened(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                paper = record["paper"]
                dimensions, ambiguous = screen_title(paper["title"])
                rows.append(
                    {
                        "venue": paper["venue_key"],
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "official_url": paper["official_url"],
                        "dimensions": "|".join(dimensions),
                        "ambiguous_trigger": ambiguous,
                        "screen_status": (
                            "dimension_candidate"
                            if dimensions
                            else "ambiguous_context_review"
                            if ambiguous
                            else "screen_negative"
                        ),
                        "manual_label": "",
                        "manual_dimensions": "",
                        "review_notes": "",
                    }
                )
    return rows


def validation_sample(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select a stable sample spanning dimensions, ambiguity, venues, and negatives."""
    def stable(row: dict[str, object]) -> str:
        value = f"{row['venue']}|{row['paper_id']}".encode()
        return hashlib.sha256(value).hexdigest()

    selected: dict[tuple[str, str], dict[str, object]] = {}
    for dimension in DIMENSIONS:
        eligible = [r for r in rows if dimension in str(r["dimensions"]).split("|")]
        for row in sorted(eligible, key=stable)[:8]:
            selected[(str(row["venue"]), str(row["paper_id"]))] = row
    venues = sorted({str(row["venue"]) for row in rows})
    for venue in venues:
        for status, count in (
            ("ambiguous_context_review", 4),
            ("screen_negative", 5),
        ):
            eligible = [
                row for row in rows
                if row["venue"] == venue and row["screen_status"] == status
            ]
            for row in sorted(eligible, key=stable)[:count]:
                selected[(str(row["venue"]), str(row["paper_id"]))] = row
    return sorted(selected.values(), key=lambda row: (str(row["venue"]), stable(row)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-sample", action="store_true")
    args = parser.parse_args()
    rows = (
        validation_sample(read_screened(args.input))
        if args.validation_sample
        else read_candidates(args.input)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
