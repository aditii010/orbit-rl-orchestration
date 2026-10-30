"""Smoke test: the page loads and a replay run completes with no exception (no model needed)."""
from streamlit.testing.v1 import AppTest


def test_replay_run_and_compare():
    at = AppTest.from_file("app/dashboard.py", default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Replay (recorded, offline)").run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    assert any("Final answer" in s.value for s in at.subheader)
    at.button(key="cmp").click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1
