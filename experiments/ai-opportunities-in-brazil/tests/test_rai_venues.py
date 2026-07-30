from pathlib import Path

from conference_pipeline.rai_venues import parse_aies_issue, read_facct


def test_aies_parser_excludes_student_abstracts() -> None:
    html = """
    <h2>Main Track III</h2><div class="obj_article_summary">
    <h3 class="title"><a id="article-10" href="/article/view/10">Paper A</a></h3>
    <div class="authors">A. Author, B. Author</div>
    <a class="obj_galley_link pdf" href="/article/view/10/20">PDF</a></div>
    <h2>Student Abstracts 25</h2><div class="obj_article_summary">
    <h3 class="title"><a id="article-11" href="/article/view/11">Student</a></h3>
    <div class="authors">C. Author</div></div>
    """
    papers = parse_aies_issue(html, 3)
    assert len(papers) == 1
    assert papers[0].title == "Paper A"


def test_facct_parser_preserves_nonarchival_entries(tmp_path: Path) -> None:
    source = tmp_path / "facct.csv"
    source.write_text(
        "TYPE,ID,ABSTRACT,AUTHOR,TITLE,URL,URL-OLD\n"
        "nonarchival,1,Abstract,\"Doe, Jane\",Title,"
        "https://doi.org/10.1145/example,https://example.test/p.pdf\n",
        encoding="utf-8",
    )
    paper = read_facct(source)[0]
    assert paper.track == "nonarchival"
    assert paper.doi == "10.1145/example"
