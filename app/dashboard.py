"""ORBIT dashboard. Run from the project root:  streamlit run app/dashboard.py
Desktop layout: slate hero band, a left rail with the controls, a right stage with the live run and the comparison."""
import html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from app.engine import build_controllers, custom_task, run_live, run_replay
from llm.client import LLM
from llm.tasks import all_tasks
from llm.trace_env import load_cache

st.set_page_config(page_title="ORBIT", layout="wide", initial_sidebar_state="collapsed")
# Palette (user's 'Rich Beige Complementary'): Moist Gold #DBDE70, Cuddle #BECAE9, Orchid Hue #E2E99C, Asagi Koi #47565A, Old World #ADB6CE.
# Fonts: Bodoni Moda for headings only (free lookalike of the user's Amethyst); Inter for all other text.
AGENT = {"INVOKE_RESEARCHER": ("Researcher", "#5E6FA8"), "INVOKE_SOLVER": ("Solver", "#47565A"),
         "INVOKE_VERIFIER": ("Verifier", "#7C8120"), "INVOKE_SUMMARIZER": ("Summarizer", "#7A6A8F"),
         "RETRY_SOLVER": ("Solver (retry)", "#A5601F"), "STOP": ("STOP", "#7D8A8E")}
HELPERS = [("Researcher", "#5E6FA8", "looks up facts"), ("Solver", "#47565A", "writes the answer"),
           ("Verifier", "#7C8120", "checks it"), ("Summarizer", "#7A6A8F", "tidies it"), ("STOP", "#7D8A8E", "hands it in")]
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,500;0,6..96,700;1,6..96,500&family=Cormorant+Garamond:ital,wght@1,500;1,600&family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea, .stApp summary {
  font-family: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important; }
.stApp span[data-testid="stIconMaterial"], .stApp .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
/* palette: Asagi Koi #47565A (slate), Cuddle #BECAE9, Old World #ADB6CE, Orchid Hue #E2E99C, Moist Gold #DBDE70 */
.stApp { background: linear-gradient(135deg, #AEBCE6 0%, #C9D3EE 28%, #E2E99C 68%, #D3D75A 100%) fixed; color: #2B373B; font-size: 1rem; }
.block-container { max-width: none !important; padding: 1.2rem 2.6rem 3rem !important; }
header[data-testid="stHeader"] { background: transparent; } footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { background: #47565A; }
[data-testid="stSidebar"] * { color: #E2E99C !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp .hero-title, .stApp .ph-t { font-family: 'Bodoni Moda', Georgia, serif !important; }
.stApp h4 { color: #2B373B; }
[data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { font-size: 0.98rem !important; font-weight: 600; }
[data-testid="stCaptionContainer"] p { font-size: 0.9rem !important; color: #47565A; }
/* hero: full-width slate band, centred heading, flowchart below */
.hero { display: flex; flex-direction: column; align-items: center; text-align: center; color: #E2E99C;
  background: linear-gradient(135deg, #36444A 0%, #47565A 50%, #63777C 100%);
  padding: 38px 46px 34px; border-radius: 4px; border-bottom: 5px solid #DBDE70; margin-bottom: 22px; }
.eyebrow { letter-spacing: 0.34em; text-transform: uppercase; font-size: 0.9rem; color: #DBDE70; font-weight: 600; }
.hero-title { font-size: 8.5rem; font-weight: 700; letter-spacing: 0.22em; padding-left: 0.22em; line-height: 1.0; margin: 8px 0 0; color: #E2E99C; }
.gold-rule { width: 220px; height: 2px; background: #DBDE70; margin: 18px auto 18px; }
.stApp .tag { font-family: 'Cormorant Garamond', Georgia, serif !important; font-style: italic; font-weight: 600; font-size: 2.1rem;
  color: #E2E99C; line-height: 1.25; max-width: 820px; letter-spacing: 0.01em; }
.flow { display: flex; align-items: stretch; justify-content: center; gap: 0; margin-top: 30px; width: 100%; max-width: 1100px; }
.node { flex: 1; background: rgba(190,202,233,0.16); border: 1px solid #BECAE9; border-top: 4px solid #DBDE70; border-radius: 3px;
  padding: 16px 20px; text-align: center; }
.node b { display: block; font-size: 1.2rem; color: #E2E99C; margin-bottom: 4px; }
.node span { color: #E4E9F7; font-size: 0.95rem; line-height: 1.35; }
.arrow { flex: none; width: 74px; display: flex; align-items: center; justify-content: center; }
/* panels: slate header strip + white body */
div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,0.9); border: 1px solid #ADB6CE !important; border-radius: 4px; }
.panelhead { margin: -1rem -1rem 0.9rem -1rem; background: linear-gradient(135deg, #36444A, #5A6C71); padding: 10px 18px;
  display: flex; align-items: center; gap: 14px; border-bottom: 4px solid #DBDE70; border-radius: 3px 3px 0 0; }
.num { width: 34px; height: 34px; border-radius: 50%; background: #DBDE70; color: #2B373B; display: flex; align-items: center;
  justify-content: center; font-weight: 700; flex: none; }
.ph-t { font-size: 1.5rem; font-weight: 700; line-height: 1.1; color: #E2E99C; } .ph-s { font-size: 0.86rem; color: #BECAE9; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 0 0 12px; font-size: 0.92rem; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 3px; color: #fff; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em; }
.step { background: #fff; border: 1px solid #C4CCE0; border-left: 6px solid #47565A; border-radius: 3px; padding: 10px 16px; margin: 0 0 10px; }
.step .meta { margin-left: 10px; color: #47565A; font-size: 0.9rem; }
.step pre { margin: 8px 0 4px; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, Menlo, monospace !important;
  font-size: 0.92rem; background: #F1F4FA; padding: 8px 10px; border-radius: 2px; color: #2B373B; }
.step summary { cursor: pointer; color: #47608F; font-weight: 700; margin-top: 4px; }
.qrow { display: flex; align-items: center; margin: 3px 0; font-size: 0.88rem; }
.qname { width: 120px; } .qbar { height: 12px; border-radius: 2px; margin-right: 8px; } .qoff { color: #7D8A8E; font-style: italic; }
div[data-testid="stMetric"] { background: #F1F4FA; border: 1px solid #C4CCE0; border-top: 4px solid #DBDE70; padding: 8px 14px; border-radius: 3px; }
div[data-testid="stMetricValue"] { color: #2B373B; font-size: 1.6rem; font-weight: 700; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
button[kind="primary"] { background: #DBDE70; border: 1px solid #47565A; color: #2B373B; letter-spacing: 0.08em; font-weight: 700; font-size: 0.95rem; padding: 4px 26px; }
button[kind="primary"]:hover { background: #47565A; border-color: #47565A; color: #E2E99C; }
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem !important; font-weight: 600; }
.stTabs [aria-selected="true"] p { color: #2B373B; }
.stTabs [data-baseweb="tab-highlight"] { background: #7C8120; height: 3px; }
.stApp [data-testid="stSelectbox"] div { font-size: 0.95rem; }
div[data-testid="stProgress"] > div > div > div { background: #8E9A2A; }
.stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"], .stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"] * {
  color: #47565A !important; opacity: 1 !important; font-size: 0.9rem !important; }
.stApp [data-testid="stCode"] code, .stApp [data-testid="stCode"] pre { font-size: 0.95rem !important; }
div[data-testid="stProgress"] p { font-size: 0.9rem !important; }
</style>""", unsafe_allow_html=True)


def panel_head(num, title, sub):
    return (f'<div class="panelhead"><div class="num">{num}</div><div><div class="ph-t">{title}</div>'
            f'<div class="ph-s">{sub}</div></div></div>')


@st.cache_resource
def controllers():
    return build_controllers()


@st.cache_resource
def records():
    return {r["id"]: r for r in load_cache("runs/cache_qwen3b.jsonl")}


@st.cache_resource
def tasks():
    return {t["id"]: t for t in all_tasks()}


def load_groq_key():
    """Read GROQ_API_KEY from .env into the environment. The value is never displayed or logged."""
    if os.environ.get("GROQ_API_KEY") or not os.path.exists(".env"):
        return bool(os.environ.get("GROQ_API_KEY"))
    for line in open(".env"):
        if line.startswith("GROQ_API_KEY="):
            os.environ["GROQ_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    return bool(os.environ.get("GROQ_API_KEY"))


def short(t):
    txt = t["spec"] if t["type"] == "coding" else t["question"]
    return f"[{t['type']}] {t['id']}: {txt[:70]}"


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
    why = (f'<details><summary>Why this action? (the DQN\'s scores)</summary>{q_bars(ev)}</details>'
           if has_q and ev["q"] else "")
    return (f'<div class="step" style="border-left-color:{color}"><span class="badge" style="background:{color}">'
            f'Step {ev["step"]} &middot; {name}</span><span class="meta">cost <b>{ev["cost"]:.3f}</b> &middot; '
            f'{ev["tin"] + ev["tout"]} tokens &middot; {ev["seconds"]:.1f}s</span>'
            f'<pre>{html.escape(ev["detail"][:600])}</pre>{why}</div>')


def run_episode(task, ctrl, mode, budget, draw):
    """Returns a generator of events for the chosen mode."""
    if mode == "Replay (recorded, offline)":
        return run_replay(records()[task["id"]], task, ctrl, budget=budget, seed=draw)
    if mode.startswith("Live - Groq"):
        llm = LLM("groq", "openai/gpt-oss-20b", max_requests=200, min_tokens=1500, reasoning_effort="low")
    else:
        llm = LLM("ollama", "qwen2.5:3b")
    return run_live(task, ctrl, llm, budget=budget)


# ---------------------------------------------------------------- page
CS = controllers()
T = tasks()
has_groq = load_groq_key()

with st.sidebar:
    st.markdown("### In plain words")
    st.markdown("**Controller**: the decision-maker. After every step it chooses the next helper, or chooses to stop.  \n"
                "**Helpers**: small AI agents (researcher, solver, verifier, summarizer). Each call costs money.  \n"
                "**Budget**: how much money it may spend on one task.  \n"
                "**Scores**: how good the learned controller thinks each next step is. Higher is better.  \n"
                "**DQN**: the reinforcement-learning method. It learns by trial and error, rewarded for correct answers and "
                "charged for cost.")

st.markdown("""<div class="hero"><div class="eyebrow">Reinforcement learning &middot; Multi-agent LLM orchestration</div>
<div class="hero-title">ORBIT</div><div class="gold-rule"></div>
<div class="tag">A learned controller that decides which AI helper to call next, and when to stop.</div>
<div class="flow">
<div class="node"><b>Pick a task</b><span>A coding, maths or research question.</span></div><div class="arrow"><svg width="56" height="22" viewBox="0 0 56 22"><line x1="2" y1="11" x2="44" y2="11" stroke="#DBDE70" stroke-width="2.5"/><polygon points="42,3 55,11 42,19" fill="#DBDE70"/></svg></div>
<div class="node"><b>Pick who decides</b><span>The learned controller, or a simple baseline.</span></div><div class="arrow"><svg width="56" height="22" viewBox="0 0 56 22"><line x1="2" y1="11" x2="44" y2="11" stroke="#DBDE70" stroke-width="2.5"/><polygon points="42,3 55,11 42,19" fill="#DBDE70"/></svg></div>
<div class="node"><b>Watch it work</b><span>See each helper it calls, what it costs and why.</span></div>
</div></div>""", unsafe_allow_html=True)

rail, stage = st.columns([1, 2.1], gap="large")

with rail:
    with st.container(border=True):
        st.markdown(panel_head("1", "Pick a task", "What should the helpers work on?"), unsafe_allow_html=True)
        pick = st.selectbox("Task", ["(write your own)"] + [short(t) for t in T.values()], index=1, label_visibility="collapsed")
        typed = ""
        if pick == "(write your own)":
            typed = st.text_input("Your task (maths or research; research only knows our 16 invented documents)",
                                  "Who founded Norvane Labs, and in which year?")
    with st.container(border=True):
        st.markdown(panel_head("2", "Pick who decides", "Who chooses the next helper?"), unsafe_allow_html=True)
        ctrl_name = st.radio("Controller", list(CS), index=list(CS).index("DQN (refit on real runs)"), key="ctrl",
                             captions=[c.blurb for c in CS.values()], label_visibility="collapsed")
        st.markdown("---")
        modes = ["Live - Ollama qwen2.5:3b (local)", "Replay (recorded, offline)"]
        if has_groq:
            modes.insert(1, "Live - Groq gpt-oss-20b")
        mode = st.radio("Where do the helpers run?", modes, key="mode",
                        help="Live = a real AI model answers now. Replay = recorded answers, instant, works offline.")
        if mode.startswith("Live - Groq"):
            st.caption("The controllers were trained on qwen costs; costs are measured the same way (tokens) but not re-trained.")
        budget = st.slider("Money budget", 0.2, 1.2, 0.8, 0.05, key="budget",
                           help="One helper call costs about 0.02 to 0.15. When the money runs out, the task ends.")
        draw = 0
        if mode.startswith("Replay"):
            draw = st.number_input("Replay draw (which recorded answers are used)", 0, 999, 0)


def get_task():
    if pick == "(write your own)":
        return custom_task(typed)
    return T[pick.split("] ")[1].split(":")[0]]


with stage:
    with st.container(border=True):
        st.markdown(panel_head("3", "Run it", "Press the button and watch the helpers being called."), unsafe_allow_html=True)
        st.markdown('<div class="legend">' + "".join(
            f'<div><span class="badge" style="background:{c}">{n}</span> {d}</div>' for n, c, d in HELPERS) + '</div>',
            unsafe_allow_html=True)
        tab_live, tab_cmp = st.tabs(["Live run: watch one controller", "Compare: all controllers, same task"])

        with tab_live:
            if st.button("RUN", type="primary", key="run"):
                try:
                    task = get_task()
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                if mode.startswith("Replay") and task["id"] not in records():
                    st.error("Replay only has the 50 built-in tasks. Pick one from the list or switch to a live mode.")
                    st.stop()
                ctrl = CS[ctrl_name]
                cols = st.columns(4)
                ph = [c.empty() for c in cols]
                meter, answer = st.empty(), st.empty()
                st.markdown(f"#### Agent trace &nbsp;<small>&mdash; {task['type']} task &middot; {ctrl.name}</small>", unsafe_allow_html=True)
                trace = st.empty()
                cards, n_tok, final = [], 0, None
                ph[0].metric("Result", "Running...")
                for ev in run_episode(task, ctrl, mode, budget, draw):
                    if ev["kind"] == "step":
                        cards.append(step_card(ev, ctrl.agent is not None))
                        trace.markdown("".join(cards), unsafe_allow_html=True)
                        n_tok += ev["tin"] + ev["tout"]
                        ph[1].metric("Cost", f"{ev['spent']:.3f}")
                        ph[2].metric("Tokens", n_tok)
                        ph[3].metric("Steps", ev["step"])
                        meter.progress(min(1.0, ev["money_left"] / budget), text=f"Money left {ev['money_left']:.2f} of {budget:.2f}")
                    else:
                        final = ev
                ph[0].metric("Result", {True: "Correct", False: "Wrong", None: "Not graded"}[final["correct"]])
                with answer.container():
                    st.markdown("#### Final answer")
                    if final["answer"] is None:
                        st.warning("The controller stopped without producing any answer.")
                    else:
                        st.code(final["answer"], language=None)
                    if final["forced_end"]:
                        st.caption("The episode ended because the budget or step limit ran out, not because the controller chose to stop.")
            else:
                st.caption("Runs the chosen controller once. Results and each helper call appear here as they happen.")

        with tab_cmp:
            st.write("Runs every controller on the same task. In live mode each run samples the model again, so results can vary run to run.")
            if st.button("RUN ALL CONTROLLERS", type="primary", key="cmp"):
                try:
                    task = get_task()
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                if mode.startswith("Replay") and task["id"] not in records():
                    st.error("Replay only has the 50 built-in tasks.")
                    st.stop()
                rows = []
                with st.spinner("Running..."):
                    for name, ctrl in CS.items():
                        final = [e for e in run_episode(task, ctrl, mode, budget, draw) if e["kind"] == "final"][0]
                        c = final["calls"]
                        rows.append({"Controller": name,
                                     "Result": {True: "Correct", False: "Wrong", None: "n/a"}[final["correct"]],
                                     "Cost": round(final["cost"], 3), "Tokens": final["tokens"],
                                     "LLM seconds": round(final["llm_seconds"], 1), "Steps": final["steps"],
                                     "Calls (res/sol/ver/sum)": f"{c['researcher']}/{c['solver']}/{c['verifier']}/{c['summarizer']}"})
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.bar_chart(df, x="Controller", y="Cost", color="#47565A", horizontal=True)
