from pathlib import Path

from conference_pipeline.openreview_pdf_cache import valid_pdf


def test_valid_pdf_requires_signature_and_content(tmp_path: Path) -> None:
    path = tmp_path / "paper.bin"
    assert not valid_pdf(path)
    path.write_bytes(b"html error")
    assert not valid_pdf(path)
    path.write_bytes(b"%PDF-1.7")
    assert valid_pdf(path)
