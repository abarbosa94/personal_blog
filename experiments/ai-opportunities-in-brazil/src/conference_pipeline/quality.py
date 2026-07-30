from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import ReconciledPaper


@dataclass(frozen=True)
class QualityReport:
    official_total: int
    enumerated_total: int
    reconciled_total: int
    with_country_total: int

    @property
    def enumeration_rate(self) -> float:
        return self.enumerated_total / self.official_total if self.official_total else 0.0

    @property
    def reconciliation_rate(self) -> float:
        return self.reconciled_total / self.enumerated_total if self.enumerated_total else 0.0

    @property
    def country_coverage(self) -> float:
        return self.with_country_total / self.enumerated_total if self.enumerated_total else 0.0

    def passes(self) -> bool:
        return (
            self.enumeration_rate >= 0.95
            and self.reconciliation_rate >= 0.90
            and self.country_coverage >= 0.85
        )


def build_quality_report(
    papers: Iterable[ReconciledPaper], official_total: int
) -> QualityReport:
    values = list(papers)
    return QualityReport(
        official_total=official_total,
        enumerated_total=len(values),
        reconciled_total=sum(paper.openalex_id is not None for paper in values),
        with_country_total=sum(bool(paper.countries) for paper in values),
    )

