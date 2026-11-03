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
