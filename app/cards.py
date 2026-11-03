"""HTML for the trace cards. Kept separate so it can be unit-tested without starting Streamlit."""
import html

AGENT = {"INVOKE_RESEARCHER": ("Researcher", "#5E6FA8"), "INVOKE_SOLVER": ("Solver", "#47565A"),
         "INVOKE_VERIFIER": ("Verifier", "#7C8120"), "INVOKE_SUMMARIZER": ("Summarizer", "#7A6A8F"),
         "RETRY_SOLVER": ("Solver (retry)", "#A5601F"), "STOP": ("STOP", "#7D8A8E")}


def _one_line(text):
    """Escape text and turn newlines into &#10; so the card stays ONE line. Markdown ends an HTML block at a blank line,
    which would cut a card in half whenever the model's code contains one."""
    return html.escape(text).replace("\r", "").replace("\n", "&#10;")


def q_bars(ev):
    """Small HTML bar chart: the controller's score (Q-value) for each action; illegal actions greyed out."""
    legal = [q for q, ok in zip(ev["q"], ev["legal"]) if ok]
    lo, hi = min(legal), max(legal)
    rows = []
    for (key, (name, color)), q, ok in zip(AGENT.items(), ev["q"], ev["legal"]):
        chosen = key == ev["action"]
        if not ok:
            rows.append(f'<div class="qrow qoff"><span class="qname">{name}</span>not allowed now</div>')
            continue
        w = 6 + 94 * (q - lo) / (hi - lo + 1e-9)
        bold = "font-weight:700;" if chosen else ""
        rows.append(f'<div class="qrow" style="{bold}"><span class="qname">{name}</span>'
                    f'<span class="qbar" style="width:{w * 2.2:.0f}px;background:{color}"></span>{q:+.2f}'
                    f'{"  &larr; chosen" if chosen else ""}</div>')
    return "".join(rows)


def step_card(ev, has_q):
    """One trace card as HTML: coloured left edge, badge, cost line, output, and a fold-out with the DQN's scores."""
    name, color = AGENT[ev["action"]]
    cls = "out code" if ("```" in ev["detail"] or "def " in ev["detail"]) else "out"
    why = (f'<details><summary>Why this action? (the DQN\'s scores)</summary>{q_bars(ev)}</details>'
           if has_q and ev["q"] else "")
    return (f'<div class="step" style="border-left-color:{color}"><span class="badge" style="background:{color}">'
            f'Step {ev["step"]} &middot; {name}</span><span class="meta">cost <b>{ev["cost"]:.3f}</b> &middot; '
            f'{ev["tin"] + ev["tout"]} tokens &middot; {ev["seconds"]:.1f}s</span>'
            f'<div class="{cls}">{_one_line(ev["detail"][:600])}</div>{why}</div>')


_PLAIN = {"INVOKE_RESEARCHER": "Asked the <b>Researcher</b> to look things up.",
          "INVOKE_SOLVER": "Asked the <b>Solver</b> to write an answer.",
          "RETRY_SOLVER": "Asked the <b>Solver</b> to try again.",
          "INVOKE_SUMMARIZER": "Asked the <b>Summarizer</b> to tidy the answer.",
          "STOP": "Decided it had done enough and <b>handed in</b> the answer."}


def narrate(steps, final=None):
    """The run in plain words: one numbered line per step, plus a closing sentence once the run is over."""
    lines = []
    for ev in steps:
        if ev["action"] == "INVOKE_VERIFIER":
            text = "Ran the <b>Verifier</b> to check the answer: it <b>%s</b>." % ("passed" if ev["detail"].startswith("PASS") else "failed")
        else:
            text = _PLAIN[ev["action"]]
        if final is not None and final["forced_end"] and ev is steps[-1] and ev["action"] != "STOP":
            text += " <i>The money or step limit ran out, so the run had to end here.</i>"
        cost = "" if ev["action"] == "STOP" else f' <span class="pc">cost {ev["cost"]:.3f}</span>'
        lines.append(f"<li>{text}{cost}</li>")
    out = '<ol class="plain">' + "".join(lines) + "</ol>"
    if final is not None:
        verdict = {True: "the answer is <b>correct</b>", False: "the answer is <b>wrong</b>", None: "there is no answer key for your own task"}[final["correct"]]
        if final["answer"] is None:
            verdict = "it never produced an answer"
        out += f'<div class="plainend">So: {verdict}, and the whole run cost <b>{final["cost"]:.3f}</b> in {final["steps"]} steps.</div>'
    return out
