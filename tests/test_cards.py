"""Trace cards must be a single line of HTML (no blank lines), or Streamlit's markdown cuts them in half."""
from app.cards import step_card


def ev(detail, q=True):
    return {"kind": "step", "step": 2, "action": "INVOKE_SOLVER", "cost": 0.09, "detail": detail, "tin": 100, "tout": 80,
            "seconds": 1.2, "spent": 0.1, "money_left": 0.7, "q": [0.1, 0.5, 0.0, 0.2, 0.3, -0.1] if q else None,
            "legal": [True, True, False, False, False, True]}


def test_code_with_blank_lines_stays_one_block():
    code = "```python\ndef f(x):\n    return x\n\n\nresult = f(1)\n```"
    card = step_card(ev(code), has_q=True)
    assert "\n" not in card
    assert "&#10;" in card and "def f(x):" in card


def test_html_in_model_output_is_escaped():
    card = step_card(ev("<script>alert(1)</script> & more"), has_q=False)
    assert "<script>" not in card and "&lt;script&gt;" in card


def test_scores_fold_out_only_for_dqn():
    assert "Why this action" in step_card(ev("x"), has_q=True)
    assert "Why this action" not in step_card(ev("x", q=False), has_q=False)


def test_plain_words_describe_each_step_and_the_outcome():
    from app.cards import narrate
    steps = [dict(ev("x"), action="INVOKE_SOLVER"), dict(ev("PASS"), action="INVOKE_VERIFIER", step=3), dict(ev("."), action="STOP", step=4, cost=0.0)]
    final = {"kind": "final", "answer": "x", "correct": True, "cost": 0.11, "steps": 3, "forced_end": False}
    out = narrate(steps, final)
    assert "Solver</b> to write an answer" in out and "it <b>passed</b>" in out and "<b>correct</b>" in out and "0.110" in out
    assert "\n" not in out                      # single line, so markdown cannot cut it


def _steps(actions_costs):
    out, spent = [], 0.0
    for i, (a, c) in enumerate(actions_costs, 1):
        spent += c
        out.append(dict(ev("x"), action=a, step=i, cost=c, spent=spent))
    return out


def test_race_chart_is_one_line_valid_svg_with_both_flags():
    import xml.etree.ElementTree as ET
    from app.cards import race_chart
    a = _steps([("INVOKE_SOLVER", 0.1), ("STOP", 0.0)])
    b = _steps([("INVOKE_RESEARCHER", 0.02), ("INVOKE_SOLVER", 0.1), ("INVOKE_VERIFIER", 0.02), ("STOP", 0.0)])
    fa = {"answer": "x", "correct": False, "cost": 0.1}
    fb = {"answer": "y", "correct": True, "cost": 0.14}
    svg = race_chart(("Single agent", "DQN (refit)"), (a, b), (fa, fb), (-0.125, 0.825), "B", 0.8)
    assert "\n" not in svg
    ET.fromstring(svg.replace("&#10003;", "ok").replace("&#10007;", "x").replace("&#9733;", "*"))    # well-formed XML
    assert "correct" in svg and "wrong" in svg and "WINNER" in svg and "Single agent" in svg


def test_board_chart_is_one_line_valid_svg_and_sorted_best_first():
    import xml.etree.ElementTree as ET
    from app.cards import board_chart
    rows = [("t1", "coding", 0.5, 0.4), ("t2", "coding", 0.1, 0.9), ("t3", "research", 0.3, 0.3)]
    svg = board_chart(rows, ("Fixed pipeline", "DQN"))
    assert "\n" not in svg
    ET.fromstring(svg)
    assert svg.index("t2") < svg.index("t3") < svg.index("t1")          # biggest right-corner advantage first
