from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    paper_id: str
    venue_key: str
    year: int
    track: str
    title: str
    authors: tuple[str, ...]
    doi: str | None
    official_url: str
    source_kind: str
    pdf_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authors"] = list(self.authors)
        return value


@dataclass(frozen=True)
class Affiliation:
    institution_id: str
    institution_name: str
    country_code: str | None
    institution_type: str | None


@dataclass(frozen=True)
class ReconciledPaper:
    paper: Paper
    openalex_id: str | None
    match_method: str | None
    affiliations: tuple[Affiliation, ...]
    diagnostics: tuple["ReconciliationDiagnostic", ...] = ()

    @property
    def countries(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    affiliation.country_code
                    for affiliation in self.affiliations
                    if affiliation.country_code
                }
            )
        )

    @property
    def institutions(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    affiliation.institution_id
                    for affiliation in self.affiliations
                    if affiliation.institution_id
                }
            )
        )


@dataclass(frozen=True)
class ReconciliationDiagnostic:
    stage: str
    outcome: str
    detail: str = ""
