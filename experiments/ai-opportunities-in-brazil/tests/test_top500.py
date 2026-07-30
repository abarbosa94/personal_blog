from conference_pipeline.top500 import aggregate, parse_page


def test_parse_page_extracts_country_and_rmax() -> None:
    html = """<table class="table table-condensed table-striped"><tr>
    <td>1</td><td><b>Machine</b><br/><a>Site</a><br>Brazil</td>
    <td>1,024</td><td>12.50</td><td>15.00</td><td>100</td></tr></table>"""
    assert parse_page(html) == [
        {"rank": 1, "country": "Brazil", "cores": 1024, "rmax_pflops": 12.5}
    ]


def test_aggregate_includes_panel_zeros() -> None:
    panel = {
        "BR": {"country_name": "Brazil", "comparison_group": "focus"},
        "US": {"country_name": "United States", "comparison_group": "frontier"},
    }
    rows, metadata = aggregate(
        [{"rank": 1, "country": "Brazil", "cores": 1, "rmax_pflops": 2.5}],
        panel,
        "2025-11",
    )
    assert rows[0]["systems"] == 1
    assert rows[0]["rmax_pflops"] == 2.5
    assert rows[1]["systems"] == 0
    assert metadata["unique_ranks"] == 1
