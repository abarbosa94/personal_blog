from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import (
    Affiliation,
    Paper,
    ReconciledPaper,
    ReconciliationDiagnostic,
)


def write_jsonl(papers: Iterable[Paper], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for paper in papers:
            stream.write(json.dumps(paper.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_papers(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            value["authors"] = tuple(value.get("authors", []))
            papers.append(Paper(**value))
    return papers


def write_reconciled_jsonl(
    papers: Iterable[ReconciledPaper], output: Path
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for record in papers:
            value = {
                "paper": record.paper.to_dict(),
                "openalex_id": record.openalex_id,
                "match_method": record.match_method,
                "affiliations": [
                    {
                        "institution_id": item.institution_id,
                        "institution_name": item.institution_name,
                        "country_code": item.country_code,
                        "institution_type": item.institution_type,
                    }
                    for item in record.affiliations
                ],
                "diagnostics": [
                    {
                        "stage": item.stage,
                        "outcome": item.outcome,
                        "detail": item.detail,
                    }
                    for item in record.diagnostics
                ],
            }
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_reconciled(path: Path) -> list[ReconciledPaper]:
    records: list[ReconciledPaper] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            paper_value = value["paper"]
            paper_value["authors"] = tuple(paper_value.get("authors", []))
            records.append(
                ReconciledPaper(
                    paper=Paper(**paper_value),
                    openalex_id=value.get("openalex_id"),
                    match_method=value.get("match_method"),
                    affiliations=tuple(
                        Affiliation(**item)
                        for item in value.get("affiliations", [])
                    ),
                    diagnostics=tuple(
                        ReconciliationDiagnostic(**item)
                        for item in value.get("diagnostics", [])
                    ),
                )
            )
    return records
