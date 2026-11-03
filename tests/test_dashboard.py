"""Smoke tests: the page loads, the replay run and compare work, and the replay presets set the controls and run."""
from streamlit.testing.v1 import AppTest


def fresh():
    at = AppTest.from_file("app/dashboard.py", default_timeout=90).run()
    assert not at.exception
    return at


def test_replay_run_and_compare():
    at = fresh()
    at.radio(key="mode").set_value("Replay (recorded, offline)").run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    assert any("Final answer" in m.value for m in at.markdown)
    at.button(key="cmp").click().run()
    assert not at.exception, at.exception
    tables = [d.value for d in at.dataframe if "Controller" in d.value.columns]
    assert len(tables) == 1 and len(tables[0]) == 5


def test_retry_preset_sets_controls_and_shows_a_retry():
    at = fresh()
    at.button(key="preset3").click().run()          # "Failed check, then retry"
    assert not at.exception, at.exception
    assert at.radio(key="mode").value.startswith("Replay") and "code02" in at.selectbox(key="task").value
    html = " ".join(m.value for m in at.markdown)
    assert "Solver (retry)" in html and "FAIL" in html and "Correct" in " ".join(str(m.value) for m in at.metric)


def test_tight_budget_preset_ends_cheaply():
    at = fresh()
    at.button(key="preset4").click().run()          # "Tight budget"
    assert not at.exception, at.exception
    assert at.slider(key="budget").value == 0.15
    cost = [float(m.value) for m in at.metric if m.label == "Cost"][0]
    assert cost < 0.15
