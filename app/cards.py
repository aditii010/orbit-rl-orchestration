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


# ---------------------------------------------------------------- charts drawn as inline SVG (one line, so markdown cannot cut them)
LEFT_COLOR, RIGHT_COLOR = "#6C7DB0", "#2B373B"
_SHORT = {"INVOKE_RESEARCHER": "Research", "INVOKE_SOLVER": "Solve", "RETRY_SOLVER": "Retry", "INVOKE_VERIFIER": "Verify",
          "INVOKE_SUMMARIZER": "Tidy", "STOP": "Stop"}


def _flag(final):
    if final["answer"] is None:
        return "&#10007; no answer", "#A5601F"
    if final["correct"] is None:
        return "not graded", "#47565A"
    return ("&#10003; correct", "#5E6B00") if final["correct"] else ("&#10007; wrong", "#A5601F")


def race_chart(names, steps, finals, scores, winner, budget):
    """Money spent so far, step by step, for both corners on one picture. Each dot is one helper call (helper colour);
    the flag at the end of each line says whether the answer was right and what the run scored."""
    W, H, L, R, T, B = 1000, 400, 70, 170, 64, 58
    pw, ph = W - L - R, H - T - B
    nmax = max(3, max(len(x) for x in steps))
    spent_max = max([ev["spent"] for x in steps for ev in x] + [0.05])
    ymax = spent_max * 1.2
    show_budget = budget <= spent_max * 1.8
    if show_budget:
        ymax = max(ymax, budget * 1.12)
    X = lambda i: L + pw * i / nmax
    Y = lambda v: T + ph * (1 - v / ymax)
    colors = (LEFT_COLOR, RIGHT_COLOR)
    ends = [None, None]
    f = 'font-family="Inter, sans-serif"'
    o = [f'<svg class="rchart" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">']
    for k in range(5):
        v = ymax * k / 4
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}" stroke="#2B373B" stroke-opacity="0.12"/>'
                 f'<text x="{L - 10}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="12" fill="#47565A" {f}>{v:.2f}</text>')
    for i in range(nmax + 1):
        o.append(f'<text x="{X(i):.1f}" y="{T + ph + 22}" text-anchor="middle" font-size="12" fill="#47565A" {f}>{"start" if i == 0 else f"step {i}"}</text>')
    o.append(f'<text x="16" y="{T + ph / 2:.0f}" transform="rotate(-90 16 {T + ph / 2:.0f})" text-anchor="middle" font-size="12" fill="#47565A" {f}>money spent so far</text>')
    if show_budget:
        o.append(f'<line x1="{L}" y1="{Y(budget):.1f}" x2="{L + pw}" y2="{Y(budget):.1f}" stroke="#A5601F" stroke-width="1.5" stroke-dasharray="6 5"/>'
                 f'<text x="{L + pw}" y="{Y(budget) - 6:.1f}" text-anchor="end" font-size="12" fill="#A5601F" {f}>budget {budget:.2f}</text>')
    for k in (0, 1):
        pts = [(X(0), Y(0))] + [(X(ev["step"]), Y(ev["spent"])) for ev in steps[k]]
        win = winner == "AB"[k]
        o.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{colors[k]}" '
                 f'stroke-width="{4 if win else 2.5}" stroke-linejoin="round" stroke-linecap="round"/>')
        for ev, (x, y) in zip(steps[k], pts[1:]):
            o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{AGENT[ev["action"]][1]}" stroke="#fff" stroke-width="2"/>'
                     + ("" if ev["action"] == "STOP" else
                        f'<text x="{x:.1f}" y="{y + (-13 if k == 0 else 24):.1f}" text-anchor="middle" font-size="11" fill="{colors[k]}" {f}>{_SHORT[ev["action"]]}</text>'))
        ends[k] = pts[-1]
    fy = [ends[0][1], ends[1][1]]
    if abs(fy[0] - fy[1]) < 44:                       # two flags would collide: spread them above and below the middle
        mid = (fy[0] + fy[1]) / 2
        fy = [mid - 22, mid + 22] if fy[0] <= fy[1] else [mid + 22, mid - 22]
    for k in (0, 1):
        x, y = ends[k]
        word, col = _flag(finals[k])
        fx = L + pw + 26
        o.append(f'<line x1="{x + 9:.1f}" y1="{y:.1f}" x2="{fx - 6}" y2="{fy[k]:.1f}" stroke="{colors[k]}" stroke-width="1" stroke-dasharray="2 4"/>'
                 f'<text x="{fx}" y="{fy[k] + 5:.1f}" font-size="15" font-weight="700" fill="{col}" {f}>{word}</text>')
        if scores[k] is not None:
            o.append(f'<text x="{fx}" y="{fy[k] + 22:.1f}" font-size="12" fill="#2B373B" {f}>score {scores[k]:+.2f}</text>')
    for k in (0, 1):
        x0 = L + k * 380
        mark = " &#9733; WINNER" if winner == "AB"[k] else ""
        o.append(f'<line x1="{x0}" y1="26" x2="{x0 + 34}" y2="26" stroke="{colors[k]}" stroke-width="{4 if winner == "AB"[k] else 2.5}" stroke-linecap="round"/>'
                 f'<text x="{x0 + 44}" y="31" font-size="14" font-weight="700" fill="{colors[k]}" {f}>{html.escape(names[k])}{mark}</text>')
    o.append("</svg>")
    return "".join(o)


def board_chart(rows, names):
    """One row per task: where the left corner scored (blue dot) and where the right corner scored (dark dot), joined by a line.
    Olive line = right corner better, copper = left corner better, grey = a tie. rows = (task id, type, left score, right score)."""
    rows = sorted(rows, key=lambda r: r[3] - r[2], reverse=True)
    W, L, R, T, rh = 1000, 96, 40, 62, 22
    H = T + rh * len(rows) + 44
    lo = min(min(r[2], r[3]) for r in rows) - 0.06
    hi = max(max(r[2], r[3]) for r in rows) + 0.06
    X = lambda v: L + (W - L - R) * (v - lo) / (hi - lo)
    f = 'font-family="Inter, sans-serif"'
    o = [f'<svg class="rchart" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img">']
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        o.append(f'<line x1="{X(v):.1f}" y1="{T - 8}" x2="{X(v):.1f}" y2="{T + rh * len(rows)}" stroke="#2B373B" stroke-opacity="0.12"/>'
                 f'<text x="{X(v):.1f}" y="{T + rh * len(rows) + 20}" text-anchor="middle" font-size="12" fill="#47565A" {f}>{v:+.2f}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle" font-size="12" fill="#47565A" {f}>score (higher is better)</text>')
    for i, (tid, typ, a, b) in enumerate(rows):
        y = T + rh * i + rh / 2
        d = b - a
        col = "#7C8120" if d > 0.01 else "#A5601F" if d < -0.01 else "#9AA3AA"
        o.append(f'<text x="{L - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#2B373B" {f}>{tid}</text>'
                 f'<line x1="{X(a):.1f}" y1="{y:.1f}" x2="{X(b):.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="3.5" stroke-linecap="round"/>'
                 f'<circle cx="{X(a):.1f}" cy="{y:.1f}" r="5" fill="{LEFT_COLOR}" stroke="#fff" stroke-width="1.5"/>'
                 f'<circle cx="{X(b):.1f}" cy="{y:.1f}" r="6" fill="{RIGHT_COLOR}" stroke="#fff" stroke-width="1.5"/>')
    o.append(f'<circle cx="{L}" cy="22" r="6" fill="{LEFT_COLOR}"/><text x="{L + 14}" y="27" font-size="13" font-weight="700" fill="{LEFT_COLOR}" {f}>{html.escape(names[0])} (left)</text>'
             f'<circle cx="{L + 330}" cy="22" r="6" fill="{RIGHT_COLOR}"/><text x="{L + 344}" y="27" font-size="13" font-weight="700" fill="{RIGHT_COLOR}" {f}>{html.escape(names[1])} (right)</text>'
             f'<line x1="{L + 640}" y1="22" x2="{L + 668}" y2="22" stroke="#7C8120" stroke-width="3.5"/><text x="{L + 676}" y="27" font-size="12" fill="#47565A" {f}>right better</text>'
             f'<line x1="{L + 760}" y1="22" x2="{L + 788}" y2="22" stroke="#A5601F" stroke-width="3.5"/><text x="{L + 796}" y="27" font-size="12" fill="#47565A" {f}>left better</text>')
    o.append("</svg>")
    return "".join(o)
