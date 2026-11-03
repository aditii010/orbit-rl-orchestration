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
