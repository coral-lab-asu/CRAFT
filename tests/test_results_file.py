"""Tests for the human-readable results file written by the terminal app."""

from craft_tabqa.tui.results_file import append_query_results, start_results_file


def test_results_file_is_readable_and_appends(tmp_path):
    path = tmp_path / "results.txt"
    start_results_file(str(path), dataset="custom", scope="Stages 1+2")

    append_query_results(
        str(path),
        question="Which planet has the most moons?",
        results=[
            {
                "rank": 1,
                "score": 0.83,
                "table_id": "planets_0",
                "title": "Planets",
                "headers": ["Planet", "Moons"],
                "rows": [["Saturn", "146"], ["Jupiter", "95"]],
            }
        ],
    )
    append_query_results(str(path), question="Second question?", results=[])

    text = path.read_text(encoding="utf-8")
    assert "CRAFT retrieval results" in text
    assert "Dataset : custom" in text
    assert "Q: Which planet has the most moons?" in text
    assert "#1" in text and "planets_0" in text
    assert "columns: Planet, Moons" in text
    assert "Saturn | 146" in text
    # second query with no results is still recorded
    assert "Q: Second question?" in text
    assert "(no tables retrieved)" in text
