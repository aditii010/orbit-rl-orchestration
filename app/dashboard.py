"""ORBIT dashboard. Run from the project root:  streamlit run app/dashboard.py
Tabs: Live run (watch one controller handle a task), Compare (all controllers on the same task)."""
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
# Palette: navy #0B0B52, soft beige #F6E4C4, clear blue #7B93CC, gold #9B7B2E. Fonts: Bodoni Moda (display), Cormorant Garamond (text).
AGENT = {"INVOKE_RESEARCHER": ("Researcher", "#5673B5"), "INVOKE_SOLVER": ("Solver", "#14146B"),
         "INVOKE_VERIFIER": ("Verifier", "#9B7B2E"), "INVOKE_SUMMARIZER": ("Summarizer", "#5B4B8A"),
         "RETRY_SOLVER": ("Solver (retry)", "#A5601F"), "STOP": ("STOP", "#6B6B6B")}
HELPERS = [("Researcher", "#5673B5", "Looks up facts in the documents."), ("Solver", "#14146B", "Writes the answer."),
           ("Verifier", "#9B7B2E", "Checks the answer."), ("Summarizer", "#5B4B8A", "Tidies the final answer."),
           ("STOP", "#6B6B6B", "Hands in the answer.")]
ACTION_ORDER = list(AGENT)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,500;0,6..96,700;1,6..96,500&family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&display=swap');
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea {
  font-family: 'Cormorant Garamond', Georgia, serif !important; }
.stApp span[data-testid="stIconMaterial"], .stApp .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
[data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { font-size: 1.2rem !important; font-weight: 600; }
[data-testid="stCaptionContainer"] p { font-size: 1.08rem !important; color: #4a4a7a; }
.stTabs [data-baseweb="tab"] p { font-size: 1.3rem !important; font-weight: 700; }
.stApp [data-testid="stSelectbox"] div { font-size: 1.15rem; }
.stApp { background: #FBF5E8; color: #0E0E4A; font-size: 1.2rem; }
.block-container { max-width: 1180px; padding-top: 1.6rem; }
header[data-testid="stHeader"] { background: transparent; } footer { visibility: hidden; }
h1, h2, h3, h4, .stApp .hero-title, .stApp .sec-t, .stApp .num, .stApp .how b, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Bodoni Moda', 'Playfair Display', Georgia, serif !important; color: #0B0B52; }
/* hero: framed like the palette card */
.hero { text-align: center; padding: 34px 20px 26px; margin-bottom: 18px; background: #FFFAF0;
  border: 1px solid #9B7B2E; outline: 1px solid #D9C48F; outline-offset: 6px; border-radius: 2px; }
.eyebrow { letter-spacing: 0.28em; text-transform: uppercase; font-size: 0.82rem; color: #9B7B2E; font-weight: 700; }
.hero-title { font-size: 4.4rem; font-weight: 700; letter-spacing: 0.18em; line-height: 1.05; margin: 8px 0 4px; }
.gold-rule { width: 120px; height: 2px; background: linear-gradient(90deg, transparent, #9B7B2E, transparent); margin: 10px auto 14px; }
.tag { font-size: 1.55rem; font-style: italic; color: #14146B; max-width: 720px; margin: 0 auto; line-height: 1.3; }
.how { display: flex; gap: 14px; margin: 6px 0 26px; }
.how > div { flex: 1; background: #F6E4C4; border-top: 3px solid #9B7B2E; padding: 12px 16px; border-radius: 2px; }
.how b { font-family: 'Bodoni Moda', serif; font-size: 1.05rem; color: #0B0B52; }
.how span { display: block; font-size: 1.08rem; color: #2b2b6b; margin-top: 2px; }
.sec { display: flex; align-items: center; gap: 14px; margin: 26px 0 8px; border-bottom: 1px solid #D9C48F; padding-bottom: 8px; }
.num { width: 38px; height: 38px; border: 1.5px solid #9B7B2E; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font-family: 'Bodoni Moda', serif; font-weight: 700; color: #0B0B52; background: #FFFAF0; flex: none; }
.sec-t { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.sec-s { font-size: 1.12rem; font-style: italic; color: #5a5a86; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 22px; margin: 4px 0 14px; }
.legend div { font-size: 1.05rem; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 3px; color: #fff; font-weight: 700; font-size: 0.98rem; letter-spacing: 0.03em; }
.qrow { display: flex; align-items: center; margin: 3px 0; font-size: 1rem; }
.qname { width: 120px; } .qbar { height: 12px; border-radius: 2px; margin-right: 8px; } .qoff { color: #8a8aa8; font-style: italic; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #D9C48F !important; background: #FFFCF4; border-radius: 3px; }
div[data-testid="stMetric"] { background: #FFFAF0; border: 1px solid #D9C48F; border-top: 3px solid #9B7B2E; padding: 10px 14px; border-radius: 2px; }
.stApp div[data-testid="stMetricValue"], .stApp div[data-testid="stMetricValue"] div { font-family: 'Bodoni Moda', serif !important; color: #0B0B52; font-size: 1.7rem; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
button[kind="primary"] { background: #14146B; border: 1px solid #9B7B2E; letter-spacing: 0.08em; font-weight: 700; font-size: 1.1rem; }
button[kind="primary"]:hover { background: #9B7B2E; border-color: #14146B; color: #fff; }
.stTabs [data-baseweb="tab"] { font-size: 1.2rem; font-weight: 700; } .stTabs [aria-selected="true"] { color: #14146B; }
pre, code, .stText { font-family: ui-monospace, Menlo, monospace !important; font-size: 0.92rem !important; }
</style>""", unsafe_allow_html=True)


def section(num, title, sub):
    st.markdown(f'<div class="sec"><div class="num">{num}</div><div><div class="sec-t">{title}</div>'
                f'<div class="sec-s">{sub}</div></div></div>', unsafe_allow_html=True)


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
                    f'<span class="qbar" style="width:{w * 1.6:.0f}px;background:{color}"></span>{q:+.2f}'
                    f'{"  &larr; chosen" if chosen else ""}</div>')
    return "".join(rows)


def render_step(box, ev, has_q):
    name, color = AGENT[ev["action"]]
    with box.container(border=True):
        st.markdown(f'<span class="badge" style="background:{color}">Step {ev["step"]} &middot; {name}</span>'
                    f'&nbsp; cost **{ev["cost"]:.3f}** &middot; {ev["tin"] + ev["tout"]} tokens &middot; {ev["seconds"]:.1f}s',
                    unsafe_allow_html=True)
        st.text(ev["detail"][:500])
        if has_q and ev["q"]:
            with st.expander("Why this action? (the DQN's scores)"):
                st.markdown(q_bars(ev), unsafe_allow_html=True)


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
<div class="tag">A learned controller that decides which AI helper to call next, and when to stop.</div></div>
<div class="how"><div><b>1 &nbsp;Pick a task</b><span>A coding, maths or research question.</span></div>
<div><b>2 &nbsp;Pick who decides</b><span>The learned controller, or a simple baseline.</span></div>
<div><b>3 &nbsp;Watch it work</b><span>See every helper it calls, what it costs and why.</span></div></div>""",
            unsafe_allow_html=True)

section("1", "Pick a task", "What should the AI helpers work on?")
pick = st.selectbox("Task", ["(write your own)"] + [short(t) for t in T.values()], index=1, label_visibility="collapsed")
typed = ""
if pick == "(write your own)":
    typed = st.text_input("Your task (maths or research; research only knows our 16 invented documents)",
                          "Who founded Norvane Labs, and in which year?")

section("2", "Pick who decides", "Who chooses the next helper? Try the learned one against the others.")
col_c, col_s = st.columns([3, 2], gap="large")
with col_c:
    ctrl_name = st.radio("Controller", list(CS), index=list(CS).index("DQN (refit on real runs)"), key="ctrl",
                         captions=[c.blurb for c in CS.values()])
with col_s:
    with st.container(border=True):
        st.markdown("**Settings**")
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

section("3", "Run it", "Press the button and watch the helpers being called.")
st.markdown('<div class="legend">' + "".join(
    f'<div><span class="badge" style="background:{c}">{n}</span>&nbsp; {d}</div>' for n, c, d in HELPERS) + '</div>',
    unsafe_allow_html=True)
tab_live, tab_cmp = st.tabs(["Live run: watch one controller", "Compare: all controllers, same task"])


def get_task():
    if pick == "(write your own)":
        return custom_task(typed)
    return T[pick.split("] ")[1].split(":")[0]]


with tab_live:
    st.caption("Runs the chosen controller once. Each helper call appears below as it happens.")
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
        left, right = st.columns([3, 2])
        with right:
            st.markdown("#### Meter")
            meter, stats = st.empty(), st.empty()
        with left:
            st.markdown(f"#### Agent trace &nbsp;<small>({task['type']} task)</small>", unsafe_allow_html=True)
            trace = st.container()
        final, n_tok, n_sec = None, 0, 0.0
        for ev in run_episode(task, ctrl, mode, budget, draw):
            if ev["kind"] == "step":
                render_step(trace.empty(), ev, ctrl.agent is not None)
                n_tok += ev["tin"] + ev["tout"]
                n_sec += ev["seconds"]
                meter.progress(min(1.0, ev["money_left"] / budget), text=f"Money left {ev['money_left']:.2f} of {budget:.2f}")
                stats.markdown(f"**Spent** {ev['spent']:.3f} &nbsp; **Tokens** {n_tok} &nbsp; **LLM time** {n_sec:.1f}s")
            else:
                final = ev
        with left:
            st.markdown("#### Final answer")
            if final["answer"] is None:
                st.warning("The controller stopped without producing any answer.")
            else:
                st.code(final["answer"], language=None)
            verdict = {True: "Correct", False: "Wrong", None: "Not graded (custom task)"}[final["correct"]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Result", verdict)
            c2.metric("Cost", f"{final['cost']:.3f}")
            c3.metric("Tokens", final["tokens"])
            c4.metric("Steps", final["steps"])
            if final["forced_end"]:
                st.caption("The episode ended because the budget or step limit ran out, not because the controller chose to stop.")

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
        st.bar_chart(df, x="Controller", y="Cost", color="#14146B", horizontal=True)
