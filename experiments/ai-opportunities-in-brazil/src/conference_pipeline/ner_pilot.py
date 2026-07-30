"""Evaluate spaCy-assisted organization extraction on frozen residual cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from .http import HttpClient
from .io import read_reconciled
from .reconcile import (
    CountryMentionExtractor,
    PdfAffiliationExtractor,
    RorAffiliationResolver,
)


def numbered_candidates(text: str) -> tuple[str, ...]:
    """Split compact ``1Org 2Org`` affiliation lists into organization strings."""

    normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    normalized = re.sub(r"\s+", " ", normalized)
    matches = list(re.finditer(r"(?<!\d)(\d+)(?=[A-Z])", normalized))
    values: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        value = normalized[match.end() : end]
        value = re.split(r"(?:Correspondence|Proceedings|Abstract)\b", value, maxsplit=1)[0]
        value = value.strip(" ,.;:*†‡#")
        if 2 <= len(value) <= 180:
            values.append(value)
    return tuple(dict.fromkeys(values))


def spacy_organization_candidates(
    text: str, model: str, nlp: Any | None = None
) -> tuple[str, ...]:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError("Install the optional 'ner' dependencies") from error
    nlp = nlp or spacy.load(model)
    numbered = numbered_candidates(text)
    candidates = list(numbered)
    entities = [
        entity.text.strip(" ,.;:*†‡#0123456789")
        for entity in nlp(text).ents
        if entity.label_ == "ORG"
    ]
    if numbered:
        numbered_text = " | ".join(numbered).casefold()
        entities = [
            entity
            for entity in entities
            if entity.casefold() in numbered_text
            and not any(
                entity.casefold() != value.casefold()
                and entity.casefold() in value.casefold()
                for value in numbered
            )
        ]
    candidates.extend(entities)
    return tuple(
        dict.fromkeys(
            value
            for value in candidates
            if 2 <= len(value) <= 180 and "\nAbstract" not in value
        )
    )


def read_expectations(path: Path) -> dict[str, set[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["review_id"]: set(filter(None, row["expected_countries"].split("|")))
            for row in csv.DictReader(handle)
        }


def run_pilot(
    v2_path: Path,
    expectations_path: Path,
    cache_dir: Path,
    output: Path,
    model: str = "en_core_web_sm",
) -> dict[str, Any]:
    expectations = read_expectations(expectations_path)
    http = HttpClient(cache_dir=cache_dir)
    resolver = RorAffiliationResolver(http)
    rows: list[dict[str, Any]] = []
    for record in read_reconciled(v2_path):
        review_id = f"{record.paper.venue_key}:{record.paper.paper_id}"
        expected = expectations.get(review_id, set())
        baseline = set(record.countries)
        if baseline == expected:
            continue
        pdf = http.get_bytes(record.paper.pdf_url or "")
        region = PdfAffiliationExtractor.affiliation_region(
            PdfAffiliationExtractor.extract_text(pdf)
        )
        candidates = spacy_organization_candidates(region, model)
        resolutions = []
        added = set(CountryMentionExtractor.country_codes(region))
        for candidate in candidates:
            try:
                affiliation = resolver.resolve(candidate)
            except Exception as error:
                resolutions.append(
                    {
                        "candidate": candidate,
                        "country": None,
                        "outcome": f"error:{type(error).__name__}",
                    }
                )
                continue
            country = affiliation.country_code if affiliation else None
            if country:
                added.add(country)
            resolutions.append(
                {
                    "candidate": candidate,
                    "country": country,
                    "outcome": "chosen" if affiliation else "unresolved",
                }
            )
        predicted = baseline | added
        rows.append(
            {
                "review_id": review_id,
                "expected": sorted(expected),
                "baseline": sorted(baseline),
                "predicted": sorted(predicted),
                "missing": sorted(expected - predicted),
                "extra": sorted(predicted - expected),
                "exact": predicted == expected,
                "candidates": list(candidates),
                "resolutions": resolutions,
            }
        )
    baseline_exact = sum(set(row["baseline"]) == set(row["expected"]) for row in rows)
    pilot_exact = sum(row["exact"] for row in rows)
    result = {
        "model": model,
        "cases": len(rows),
        "baseline_exact": baseline_exact,
        "pilot_exact": pilot_exact,
        "introduced_extra_country_cases": sum(bool(row["extra"]) for row in rows),
        "accepted": pilot_exact > baseline_exact
        and not any(row["extra"] for row in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v2", type=Path)
    parser.add_argument("expectations", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/http-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()
    result = run_pilot(
        args.v2,
        args.expectations,
        args.cache_dir,
        args.output,
        args.model,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
