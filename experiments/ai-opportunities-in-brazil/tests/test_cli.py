from pathlib import Path

from conference_pipeline.cli import (
    api_error_papers,
    reconcile_bounded,
    reviewed_paper_ids,
)
from conference_pipeline.models import Paper, ReconciledPaper, ReconciliationDiagnostic


def test_reviewed_paper_ids_combines_multiple_queues(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    first.write_text("venue,paper_id\nneurips,paper-1\n", encoding="utf-8")
    second = tmp_path / "second.csv"
    second.write_text(
        "venue,paper_id\nneurips,paper-2\nacl,paper-3\n",
        encoding="utf-8",
    )

    assert reviewed_paper_ids([first, second]) == {
        ("neurips", "paper-1"),
        ("neurips", "paper-2"),
        ("acl", "paper-3"),
    }


def test_api_error_papers_selects_only_retryable_records() -> None:
    def record(paper_id: str, outcome: str) -> ReconciledPaper:
        return ReconciledPaper(
            Paper(
                paper_id,
                "icml",
                2025,
                "main",
                paper_id,
                (),
                None,
                f"https://example.test/{paper_id}",
                "fixture",
            ),
            None,
            None,
            (),
            (ReconciliationDiagnostic("openalex_title", outcome),),
        )

    assert [paper.paper_id for paper in api_error_papers(
        [record("retry", "api_error"), record("keep", "not_found")]
    )] == ["retry"]


def test_bounded_reconciliation_returns_every_paper() -> None:
    papers = [
        Paper(str(index), "icml", 2025, "main", str(index), (), None,
              f"https://example.test/{index}", "fixture")
        for index in range(20)
    ]

    class FakeReconciler:
        def reconcile(self, paper: Paper) -> ReconciledPaper:
            return ReconciledPaper(paper, None, None, ())

    results = list(reconcile_bounded(papers, FakeReconciler(), workers=4))

    assert {result.paper.paper_id for result in results} == {
        paper.paper_id for paper in papers
    }
