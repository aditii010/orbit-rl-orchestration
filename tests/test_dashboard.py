import os

os.environ["ORBIT_FAST"] = "1"

from streamlit.testing.v1 import AppTest

R = "DQN (refit on real runs)"


def fresh():
    at = AppTest.from_file("app/dashboard.py", default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def html(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_page_loads_with_five_shortcuts_and_two_corners():
    at = fresh()
    assert at.selectbox(key="left").value == "Fixed pipeline" and at.selectbox(key="right").value == R
    assert all(f"preset{i}" in [b.key for b in at.button] for i in range(5))


def test_race_shows_both_corners_and_a_winner():
    at = fresh()
    at.radio(key="mode").set_value("Replay (recorded, offline)").run()
    at.button(key="race").click().run()
    assert not at.exception, at.exception
    h = html(at)
    assert "LEFT CORNER" in h and "RIGHT CORNER" in h and "WINNER" in h and "wins." in h


def test_run_one_shows_the_right_corner_alone():
    at = fresh()
    at.radio(key="mode").set_value("Replay (recorded, offline)").run()
    at.button(key="run").click().run()
    assert not at.exception, at.exception
    assert "Final answer" in html(at) and "In plain words" in html(at)


def test_retry_shortcut_sets_everything_and_dqn_wins():
    at = fresh()
    at.button(key="preset2").click().run()
    assert not at.exception, at.exception
    assert at.radio(key="mode").value.startswith("Replay") and "code02" in at.selectbox(key="task").value
    assert at.selectbox(key="left").value == "Single agent"
    h = html(at)
    assert "Solver (retry)" in h and "FAIL" in h and f"<b>{R} wins.</b>" in h


def test_tight_budget_shortcut_sets_budget_and_dqn_wins():
    at = fresh()
    at.button(key="preset3").click().run()
    assert not at.exception, at.exception
    assert at.slider(key="budget").value == 0.15
    assert f"<b>{R} wins.</b>" in html(at)


def test_scoreboard_shortcut_counts_wins_over_unseen_tasks():
    at = fresh()
    at.button(key="preset4").click().run()
    assert not at.exception, at.exception
    h = html(at)
    assert "24 unseen tasks" in h and "won by right corner" in h and "wins the scoreboard" in h


def test_scoreboard_against_the_hand_tuned_rule_is_reported_honestly():
    at = fresh()
    at.radio(key="mode").set_value("Replay (recorded, offline)").run()
    at.selectbox(key="left").set_value("Tuned rule").run()
    at.button(key="board").click().run()
    assert not at.exception, at.exception
    h = html(at)
    assert f"<b>{R} wins the scoreboard.</b>" not in h
