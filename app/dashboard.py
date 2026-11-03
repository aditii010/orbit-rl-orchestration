"""ORBIT dashboard. Run from the project root:  streamlit run app/dashboard.py
Desktop layout: frosted-glass cards on a gradient, a left rail with the controls, a right stage with the live run and the comparison."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from app.cards import AGENT, step_card
from app.engine import build_controllers, custom_task, run_live, run_replay
from llm.client import LLM
from llm.tasks import all_tasks
from llm.trace_env import load_cache

st.set_page_config(page_title="ORBIT", layout="wide", initial_sidebar_state="collapsed")
# Palette (user's 'Rich Beige Complementary'): Moist Gold #DBDE70, Cuddle #BECAE9, Orchid Hue #E2E99C, Asagi Koi #47565A, Old World #ADB6CE.
# Fonts: Bodoni Moda for headings only (free lookalike of the user's Amethyst); Inter for all other text.
HELPERS = [("Researcher", "#5E6FA8", "looks up facts"), ("Solver", "#47565A", "writes the answer"),
           ("Verifier", "#7C8120", "checks it"), ("Summarizer", "#7A6A8F", "tidies it"), ("STOP", "#7D8A8E", "hands it in")]
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,500;0,6..96,700;1,6..96,500&family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea, .stApp summary {
  font-family: 'Inter', -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif !important; }
.stApp span[data-testid="stIconMaterial"], .stApp .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
/* palette: Asagi Koi #47565A (slate), Cuddle #BECAE9, Old World #ADB6CE, Orchid Hue #E2E99C, Moist Gold #DBDE70 */
.stApp { background: linear-gradient(135deg, #AEBCE6 0%, #C9D3EE 28%, #E2E99C 68%, #D3D75A 100%) fixed; color: #2B373B; font-size: 1rem; }
.block-container { max-width: none !important; padding: 0.6rem 2.6rem 3rem !important; }
header[data-testid="stHeader"] { background: transparent; } footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { background: #47565A; } [data-testid="stSidebar"] * { color: #E2E99C !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp .hero-title, .stApp .ph-t { font-family: 'Bodoni Moda', Georgia, serif !important; }
.stApp h4 { color: #2B373B; }
[data-testid="stRadio"] label p, [data-testid="stWidgetLabel"] p { font-size: 0.98rem !important; font-weight: 600; }
[data-testid="stCaptionContainer"] p { font-size: 0.9rem !important; color: #47565A; }
/* hero: no band, text straight on the gradient, animated orbit behind the title */
.hero { position: relative; display: flex; flex-direction: column; align-items: center; text-align: center; padding: 30px 20px 10px; margin-bottom: 18px; }
.hero > * { position: relative; z-index: 1; }
.orbit { position: absolute !important; left: 0; top: 18px; width: 100%; height: 260px; z-index: 0 !important; opacity: 0.55; pointer-events: none; }
.chip { display: inline-block; letter-spacing: 0.28em; text-transform: uppercase; font-size: 0.78rem; font-weight: 600; color: #36444A;
  background: rgba(255,255,255,0.5); border: 1px solid rgba(71,86,90,0.25); padding: 6px 18px; border-radius: 999px; backdrop-filter: blur(8px); }
.hero-title { font-size: 9.5rem; font-weight: 700; letter-spacing: 0.22em; padding-left: 0.22em; line-height: 1.0; margin: 14px 0 0;
  color: #36444A; background: linear-gradient(135deg, #2B373B 10%, #47565A 55%, #5E6FA8 100%); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent; }
.stApp .tag { font-family: 'Instrument Serif', Georgia, serif !important; font-style: italic; font-weight: 400; font-size: 1.85rem;
  color: #36444A; line-height: 1.2; max-width: 980px; margin-top: 8px; letter-spacing: 0.012em; text-wrap: balance; }
.flow { display: flex; align-items: stretch; justify-content: center; margin-top: 30px; width: 100%; max-width: 1150px; }
.node { flex: 1; background: rgba(255,255,255,0.55); border: 1px solid rgba(255,255,255,0.8); border-radius: 18px; padding: 18px 22px;
  text-align: center; box-shadow: 0 10px 30px rgba(71,86,90,0.14); backdrop-filter: blur(12px); transition: transform .2s, box-shadow .2s; }
.node:hover { transform: translateY(-3px); box-shadow: 0 16px 36px rgba(71,86,90,0.2); }
.ico { width: 76px; height: 76px; margin: 0 auto 10px; border-radius: 50%; background: rgba(255,255,255,0.65); border: 1px solid rgba(71,86,90,0.22); display: flex; align-items: center; justify-content: center; }
.node b { display: block; font-size: 1.15rem; color: #2B373B; margin-bottom: 3px; }
.node span { color: #47565A; font-size: 0.93rem; line-height: 1.35; }
.arrow { flex: none; width: 70px; display: flex; align-items: center; justify-content: center; }
/* glass cards */
div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,0.55); backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,0.8) !important; border-radius: 20px; box-shadow: 0 12px 36px rgba(71,86,90,0.13); }
.ph { display: flex; align-items: center; gap: 12px; padding-bottom: 10px; margin-bottom: 12px; border-bottom: 1px solid rgba(71,86,90,0.18); }
.num { width: 32px; height: 32px; border-radius: 10px; background: #47565A; color: #E2E99C; display: flex; align-items: center;
  justify-content: center; font-weight: 700; flex: none; }
.ph-t { font-size: 1.55rem; font-weight: 700; line-height: 1.1; color: #2B373B; } .ph-s { font-size: 0.86rem; color: #5a6a6f; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 0 0 12px; font-size: 0.92rem; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 999px; color: #fff; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.02em; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.step { background: rgba(255,255,255,0.92); border-left: 6px solid #47565A; border-radius: 12px; padding: 10px 16px; margin: 0 0 10px;
  box-shadow: 0 4px 14px rgba(71,86,90,0.10); }
.step.new { animation: rise .4s ease both; }
.step .meta { margin-left: 10px; color: #47565A; font-size: 0.9rem; }
.step .out { margin: 8px 0 4px; white-space: pre-wrap; word-break: break-word; font-size: 0.95rem; color: #2B373B; line-height: 1.45; }
.step .out.code { font-family: ui-monospace, Menlo, monospace !important; font-size: 0.9rem; background: #F1F4FA; padding: 8px 10px; border-radius: 8px; }
.step summary { cursor: pointer; color: #47608F; font-weight: 700; margin-top: 4px; }
.qrow { display: flex; align-items: center; margin: 3px 0; font-size: 0.88rem; }
.qname { width: 120px; } .qbar { height: 12px; border-radius: 6px; margin-right: 8px; } .qoff { color: #7D8A8E; font-style: italic; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.75); border: 1px solid rgba(255,255,255,0.9); padding: 10px 16px; border-radius: 14px;
  box-shadow: 0 4px 14px rgba(71,86,90,0.08); }
div[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.74rem !important; color: #5a6a6f; }
div[data-testid="stMetricValue"] { color: #2B373B; font-size: 1.7rem; font-weight: 700; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
button[kind="primary"] { background: #47565A; border: none; color: #E2E99C; letter-spacing: 0.1em; font-weight: 700; font-size: 0.95rem;
  padding: 6px 34px; border-radius: 999px; box-shadow: 0 6px 16px rgba(71,86,90,0.28); transition: transform .15s, background .15s; }
button[kind="primary"]:hover { background: #DBDE70; color: #2B373B; transform: translateY(-2px); }
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem !important; font-weight: 600; }
.stTabs [aria-selected="true"] p { color: #2B373B; }
.stTabs [data-baseweb="tab-highlight"] { background: #47565A; height: 3px; border-radius: 3px; }
.stApp [data-testid="stSelectbox"] div { font-size: 0.95rem; }
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius: 12px; }
div[data-testid="stProgress"] > div > div > div { background: #7C8120; border-radius: 6px; }
.stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"], .stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"] * {
  color: #47565A !important; opacity: 1 !important; font-size: 0.9rem !important; }
.stApp [data-testid="stCode"] code, .stApp [data-testid="stCode"] pre { font-size: 0.95rem !important; }
div[data-testid="stProgress"] p { font-size: 0.9rem !important; }
</style>""", unsafe_allow_html=True)


def panel_head(num, title, sub):
    return (f'<div class="ph"><div class="num">{num}</div><div><div class="ph-t">{title}</div>'
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

st.markdown("""<div class="hero"><svg class="orbit" viewBox="0 0 1000 260" preserveAspectRatio="xMidYMid meet" height="260"><g transform="rotate(-7 500 130)"><ellipse cx="500" cy="130" rx="470" ry="62" fill="none" stroke="#47565A" stroke-opacity="0.5" stroke-width="1.2"/><ellipse cx="500" cy="130" rx="380" ry="40" fill="none" stroke="#47565A" stroke-opacity="0.3" stroke-width="1" stroke-dasharray="3 6"/><circle r="9" fill="#DBDE70" stroke="#47565A" stroke-width="1.5"><animateMotion dur="14s" repeatCount="indefinite" path="M30,130 a470,62 0 1,0 940,0 a470,62 0 1,0 -940,0"/></circle><circle r="6" fill="#5E6FA8"><animateMotion dur="9s" repeatCount="indefinite" path="M120,130 a380,40 0 1,1 760,0 a380,40 0 1,1 -760,0"/></circle></g></svg><div class="chip">Reinforcement learning &middot; Multi-agent LLM orchestration</div>
<div class="hero-title">ORBIT</div>
<div class="tag">A learned controller that decides which AI helper to call next, and when to stop.</div>
<div class="flow">
<div class="node"><div class="ico"><svg width="54" height="54" viewBox="0 0 64 64" fill="none"><g transform="rotate(-18 32 32)"><path d="M5 32 A27 8.5 0 0 1 59 32" stroke="#2B373B" stroke-width="1.5"/></g><circle cx="32" cy="32" r="13" fill="#DBDE70" stroke="#2B373B" stroke-width="1.5"/><path d="M20.5 27 Q32 23 43.5 27 M19.5 33 Q32 37 44.5 33" stroke="#2B373B" stroke-width="1" opacity="0.7"/><g transform="rotate(-18 32 32)"><path d="M5 32 A27 8.5 0 0 0 59 32" stroke="#2B373B" stroke-width="1.5"/></g><circle cx="55" cy="14" r="1.6" fill="#2B373B"/><circle cx="10" cy="52" r="1.2" fill="#2B373B"/></svg></div><b>Pick a task</b><span>A coding, maths or research question.</span></div><div class="arrow"><svg width="54" height="22" viewBox="0 0 54 22"><line x1="2" y1="11" x2="42" y2="11" stroke="#47565A" stroke-width="2.4" stroke-dasharray="1 5" stroke-linecap="round"/><polygon points="40,4 53,11 40,18" fill="#47565A"/></svg></div>
<div class="node"><div class="ico"><svg width="54" height="54" viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="25" stroke="#2B373B" stroke-width="1.5"/><circle cx="32" cy="32" r="21" stroke="#2B373B" stroke-width="0.6" stroke-dasharray="1.5 3"/><path d="M32 4v6M32 54v6M4 32h6M54 32h6" stroke="#2B373B" stroke-width="1.5" stroke-linecap="round"/><g transform="rotate(38 32 32)"><polygon points="32,12 37.5,32 26.5,32" fill="#DBDE70" stroke="#2B373B" stroke-width="1.4" stroke-linejoin="round"/><polygon points="26.5,32 37.5,32 32,52" fill="#5E6FA8" fill-opacity="0.85" stroke="#2B373B" stroke-width="1.4" stroke-linejoin="round"/></g><circle cx="32" cy="32" r="2.4" fill="#2B373B"/></svg></div><b>Pick who decides</b><span>The learned controller, or a simple baseline.</span></div><div class="arrow"><svg width="54" height="22" viewBox="0 0 54 22"><line x1="2" y1="11" x2="42" y2="11" stroke="#47565A" stroke-width="2.4" stroke-dasharray="1 5" stroke-linecap="round"/><polygon points="40,4 53,11 40,18" fill="#47565A"/></svg></div>
<div class="node"><div class="ico"><svg width="54" height="54" viewBox="0 0 64 64" fill="none"><path d="M9 46 L23 27 L37 39 L54 15" stroke="#2B373B" stroke-width="1.3" stroke-dasharray="2 3" stroke-linecap="round"/><circle cx="9" cy="46" r="3.2" fill="#2B373B"/><circle cx="23" cy="27" r="3.2" fill="#2B373B"/><circle cx="37" cy="39" r="3.2" fill="#5E6FA8" stroke="#2B373B" stroke-width="1.2"/><circle cx="54" cy="15" r="8.5" stroke="#2B373B" stroke-width="1.2"/><circle cx="54" cy="15" r="4.6" fill="#DBDE70" stroke="#2B373B" stroke-width="1.2"/><circle cx="14" cy="14" r="1.2" fill="#2B373B"/><circle cx="46" cy="54" r="1.2" fill="#2B373B"/><circle cx="58" cy="40" r="1" fill="#2B373B"/></svg></div><b>Watch it work</b><span>See each helper it calls, what it costs and why.</span></div>
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
                        trace.markdown("".join(cards[:-1]) + cards[-1].replace('class="step"', 'class="step new"', 1), unsafe_allow_html=True)
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
