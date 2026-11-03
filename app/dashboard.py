"""ORBIT dashboard. Run from the project root:  streamlit run app/dashboard.py
Desktop layout: frosted-glass cards on a gradient, a left rail with the controls, a right stage with the live run and the comparison."""
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.cards import AGENT, narrate, step_card
from app.engine import board_scores, build_controllers, custom_task, lockstep, run_live, run_replay, score, verdict
from app.results_data import ABLATE, REAL, SHIFT, SIM, SIM_BLIND
from llm.client import LLM
from llm.tasks import all_tasks
from llm.trace_env import load_cache, split_tasks

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
.flow { display: flex; align-items: flex-start; justify-content: center; margin-top: 30px; width: 100%; max-width: 1150px; }
.node { flex: 1; text-align: center; padding: 16px 26px 4px; border-top: 1px solid #2B373B; }
.ico { width: 64px; height: 64px; margin: 0 auto 6px; display: flex; align-items: center; justify-content: center; }
.node b { display: block; font-size: 1.1rem; color: #2B373B; margin-bottom: 3px; letter-spacing: 0.02em; }
.node span { color: #47565A; font-size: 0.93rem; line-height: 1.4; }
.arrow { flex: none; width: 70px; display: flex; align-items: center; justify-content: center; margin-top: 38px; }
/* flat editorial sections: hairline rules, big numerals, no boxes */
div[data-testid="stVerticalBlockBorderWrapper"] { background: transparent; border: none !important; box-shadow: none; border-radius: 0; padding: 0 !important; }
.ph { display: flex; align-items: baseline; gap: 14px; border-top: 2px solid #2B373B; padding-top: 8px; margin-bottom: 16px; }
.num { font-family: 'Bodoni Moda', Georgia, serif !important; font-size: 3rem; font-weight: 500; line-height: 0.9; color: #47565A; flex: none; }
.ph-t { font-size: 1.5rem; font-weight: 700; line-height: 1.1; color: #2B373B; } .ph-s { font-size: 0.86rem; color: #47565A; margin-top: 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 0 0 12px; font-size: 0.92rem; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 2px; color: #fff; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.02em; }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.step { background: rgba(255,255,255,0.72); border: 1px solid rgba(71,86,90,0.28); border-left: 5px solid #47565A; border-radius: 2px; padding: 10px 16px; margin: 0 0 8px; }
.step.new { animation: rise .4s ease both; }
.step .meta { margin-left: 10px; color: #47565A; font-size: 0.9rem; }
.step .out { margin: 8px 0 4px; white-space: pre-wrap; word-break: break-word; font-size: 0.95rem; color: #2B373B; line-height: 1.45; }
.step .out.code { font-family: ui-monospace, Menlo, monospace !important; font-size: 0.9rem; background: #F1F4FA; padding: 8px 10px; border-radius: 8px; }
.step summary { cursor: pointer; color: #47608F; font-weight: 700; margin-top: 4px; }
.qrow { display: flex; align-items: center; margin: 3px 0; font-size: 0.88rem; }
.qname { width: 120px; } .qbar { height: 12px; border-radius: 1px; margin-right: 8px; } .qoff { color: #7D8A8E; font-style: italic; }
div[data-testid="stMetric"] { background: transparent; border: none; border-top: 2px solid #2B373B; padding: 8px 0 0; border-radius: 0; box-shadow: none; }
div[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.74rem !important; color: #5a6a6f; }
div[data-testid="stMetricValue"] { color: #2B373B; font-size: 2.1rem; font-weight: 500; }
div[data-testid="stMetricValue"] > div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
button[kind="primary"] { background: #2B373B; border: 1px solid #2B373B; color: #E2E99C; letter-spacing: 0.14em; font-weight: 600; font-size: 0.85rem;
  padding: 6px 22px; border-radius: 2px; box-shadow: none; transition: background .15s, color .15s; }
button[kind="primary"]:hover { background: #DBDE70; border-color: #2B373B; color: #2B373B; }
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem !important; font-weight: 600; }
.stTabs [aria-selected="true"] p { color: #2B373B; }
.stTabs [data-baseweb="tab-highlight"] { background: #47565A; height: 3px; border-radius: 3px; }
.stApp [data-testid="stSelectbox"] div { font-size: 0.95rem; }
div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius: 2px; }
div[data-testid="stProgress"] > div > div > div { background: #7C8120; border-radius: 1px; }
.stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"], .stApp [data-testid="stRadio"] [data-testid="stCaptionContainer"] * {
  color: #47565A !important; opacity: 1 !important; font-size: 0.9rem !important; }
.stApp [data-testid="stCode"], .stApp [data-testid="stCode"] pre { border-radius: 2px !important; }
.stApp [data-testid="stCode"] code, .stApp [data-testid="stCode"] pre { font-size: 0.95rem !important; }
div[data-testid="stProgress"] p { font-size: 0.9rem !important; }
.preview { border-left: 2px solid #2B373B; padding: 2px 0 2px 14px; margin-top: 14px; font-size: 0.95rem; line-height: 1.5; }
.preview .small { margin-top: 8px; font-size: 0.85rem; color: #47565A; } .preview code { font-size: 0.82rem; background: rgba(255,255,255,0.6); padding: 1px 5px; border-radius: 2px; }
.tchip { display: inline-block; text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.68rem; font-weight: 700; color: #2B373B; background: #DBDE70;
  padding: 2px 8px; border-radius: 2px; margin-bottom: 8px; }
.callout { border-left: 2px solid #5E6FA8; padding: 2px 0 2px 14px; margin-top: 12px; font-size: 0.93rem; line-height: 1.5; color: #2B373B; }
.empty { text-align: center; color: #47565A; font-size: 1rem; padding: 26px 20px 30px; border: 1px dashed rgba(43,55,59,0.45); border-radius: 2px; }
button[kind="secondary"] { border: 1px solid rgba(43,55,59,0.45); border-radius: 2px; background: rgba(255,255,255,0.45); color: #2B373B; font-size: 0.82rem;
  font-weight: 600; padding: 4px 8px; line-height: 1.25; min-height: 2.6rem; }
button[kind="secondary"]:hover { background: #DBDE70; border-color: #2B373B; color: #2B373B; }
.qd { letter-spacing: 0.22em; text-transform: uppercase; font-size: 0.74rem; font-weight: 700; color: #47565A; margin: 4px 0 6px; }
.bars { margin: 8px 0 14px; } .brow { display: flex; align-items: center; margin: 7px 0; }
.blab { width: 38%; text-align: right; padding-right: 16px; font-size: 0.93rem; line-height: 1.25; color: #2B373B; }
.btrack { flex: 1; display: flex; align-items: center; } .bfill { height: 26px; border-radius: 1px; }
.bval { margin-left: 10px; font-size: 0.92rem; font-weight: 600; color: #2B373B; }
.banner { display: flex; align-items: center; gap: 12px; border: 1px solid #2B373B; border-left-width: 5px; background: rgba(255,255,255,0.65);
  padding: 10px 16px; margin: 0 0 14px; font-size: 0.95rem; }
.banner.done { border-left-color: #7C8120; }
.pulse { width: 12px; height: 12px; border-radius: 50%; background: #7C8120; animation: pulse 1s ease-in-out infinite; flex: none; }
@keyframes pulse { 0%, 100% { transform: scale(0.7); opacity: 0.5; } 50% { transform: scale(1.25); opacity: 1; } }
.plain { margin: 4px 0 8px 18px; padding: 0; font-size: 1rem; line-height: 1.7; } .plain li { padding-left: 4px; }
.pc { color: #47565A; font-size: 0.85rem; margin-left: 6px; }
.plainend { border-left: 3px solid #2B373B; padding: 4px 0 4px 12px; margin: 6px 0 14px; font-size: 1rem; }
.corner { border-top: 2px solid #2B373B; padding-top: 6px; margin-bottom: 8px; }
.ctag { letter-spacing: 0.2em; font-size: 0.7rem; font-weight: 700; color: #47565A; }
.cname { font-family: 'Bodoni Moda', Georgia, serif !important; font-size: 1.5rem; font-weight: 700; color: #2B373B; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.corner.win { border-top: 4px solid #7C8120; }
.crown { font-family: 'Inter', sans-serif !important; font-size: 0.68rem; letter-spacing: 0.18em; background: #DBDE70; color: #2B373B; padding: 3px 9px; border-radius: 2px; }
.winner { border: 1px solid #2B373B; border-left: 8px solid #7C8120; background: rgba(255,255,255,0.7); padding: 14px 18px; margin: 16px 0 10px; font-size: 1.15rem; line-height: 1.45; }
.wsub { font-size: 0.88rem; color: #47565A; margin-top: 6px; }
.tally { display: flex; gap: 28px; flex-wrap: wrap; font-size: 1.05rem; padding: 6px 0 10px; } .tally .of { margin-left: auto; color: #47565A; }
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


def safe_events(task, ctrl, mode, budget, draw):
    """run_episode with a friendly message instead of a traceback when a model is unreachable."""
    try:
        yield from run_episode(task, ctrl, mode, budget, draw)
    except Exception as e:                                        # noqa: BLE001 (show any failure to the user)
        if type(e).__name__ in ("StopException", "RerunException"):
            raise
        if isinstance(e, urllib.error.URLError):
            st.error("Could not reach the model. If you chose Ollama, open the Ollama app and try again. "
                     "Or switch to **Replay** in box 03, which needs no model.")
        elif type(e).__name__ == "BudgetExceeded" or "429" in str(e):
            st.error("The Groq free limit was hit. Switch to Ollama or Replay in box 03.")
        else:
            st.error(f"Something went wrong: {e}. Switching to **Replay** in box 03 always works.")
        st.stop()


def bars(rows, fmt="{:+.3f}"):
    """Horizontal bars as HTML. rows = (label, value, highlight). The learned controller is dark slate, the rest soft grey-blue.
    Bar length is proportional to |value|; the printed value keeps its sign."""
    mx = max(abs(v) for _, v, _ in rows) or 1.0
    out = []
    for name, v, hl in rows:
        w = max(1.5, 82 * abs(v) / mx)
        out.append(f'<div class="brow"><div class="blab">{name}</div><div class="btrack">'
                   f'<div class="bfill" style="width:{w:.1f}%;background:{"#47565A" if hl else "#ADB6CE"}"></div>'
                   f'<span class="bval">{fmt.format(v)}</span></div></div>')
    return '<div class="bars">' + "".join(out) + "</div>"


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
<div class="node"><div class="ico"><svg width="54" height="54" viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="25" stroke="#2B373B" stroke-width="1.5"/><circle cx="32" cy="32" r="21" stroke="#2B373B" stroke-width="0.6" stroke-dasharray="1.5 3"/><path d="M32 4v6M32 54v6M4 32h6M54 32h6" stroke="#2B373B" stroke-width="1.5" stroke-linecap="round"/><g transform="rotate(38 32 32)"><polygon points="32,12 37.5,32 26.5,32" fill="#DBDE70" stroke="#2B373B" stroke-width="1.4" stroke-linejoin="round"/><polygon points="26.5,32 37.5,32 32,52" fill="#5E6FA8" fill-opacity="0.85" stroke="#2B373B" stroke-width="1.4" stroke-linejoin="round"/></g><circle cx="32" cy="32" r="2.4" fill="#2B373B"/></svg></div><b>Pick two corners</b><span>A baseline on the left, the learned controller on the right.</span></div><div class="arrow"><svg width="54" height="22" viewBox="0 0 54 22"><line x1="2" y1="11" x2="42" y2="11" stroke="#47565A" stroke-width="2.4" stroke-dasharray="1 5" stroke-linecap="round"/><polygon points="40,4 53,11 40,18" fill="#47565A"/></svg></div>
<div class="node"><div class="ico"><svg width="54" height="54" viewBox="0 0 64 64" fill="none"><path d="M9 46 L23 27 L37 39 L54 15" stroke="#2B373B" stroke-width="1.3" stroke-dasharray="2 3" stroke-linecap="round"/><circle cx="9" cy="46" r="3.2" fill="#2B373B"/><circle cx="23" cy="27" r="3.2" fill="#2B373B"/><circle cx="37" cy="39" r="3.2" fill="#5E6FA8" stroke="#2B373B" stroke-width="1.2"/><circle cx="54" cy="15" r="8.5" stroke="#2B373B" stroke-width="1.2"/><circle cx="54" cy="15" r="4.6" fill="#DBDE70" stroke="#2B373B" stroke-width="1.2"/><circle cx="14" cy="14" r="1.2" fill="#2B373B"/><circle cx="46" cy="54" r="1.2" fill="#2B373B"/><circle cx="58" cy="40" r="1" fill="#2B373B"/></svg></div><b>Watch the race</b><span>See every helper each one calls, what it costs and who wins.</span></div>
</div></div>""", unsafe_allow_html=True)

def get_task():
    if pick == "(write your own)":
        return custom_task(typed)
    return T[pick.split("] ")[1].split(":")[0]]


LIVE_Q, REPLAY = "Live - Ollama qwen2.5:3b (local)", "Replay (recorded, offline)"
R_NAME = "DQN (refit on real runs)"
LABEL = {t["id"]: short(t) for t in T.values()}
_, TEST_RECS = split_tasks(list(records().values()))        # the 24 tasks the retrained DQN never saw
REPLAY_DELAY = 0.0 if os.environ.get("ORBIT_FAST") else 0.7   # replay is instant, so pause a little to make it watchable
# One-click shortcuts: (button text, task id, left corner, right corner, mode, budget, replay draw, what to start).
# The four races were checked in advance: the retrained DQN wins each one (see docs/DEMO.md).
PRESETS = [("Race: a coding task", "code04", "Fixed pipeline", R_NAME, LIVE_Q, 0.8, 0, "race"),
           ("Race: why we retrain", "res01", "DQN (simulator only)", R_NAME, LIVE_Q, 0.8, 0, "race"),
           ("Race: failed check, retry", "code02", "Single agent", R_NAME, REPLAY, 0.8, 3, "race"),
           ("Race: tight budget", "res01", "Fixed pipeline", R_NAME, REPLAY, 0.15, 0, "race"),
           ("Scoreboard: 24 unseen tasks", "code04", "Fixed pipeline", R_NAME, REPLAY, 0.8, 0, "board")]


def apply_preset(i):
    _, tid, left, right, mode_, bud, drw, act = PRESETS[i]
    st.session_state.update(task=LABEL[tid], left=left, right=right, mode=mode_, budget=bud, draw=drw, autorun=act)


def preview_html():
    """What the chosen task actually says, so the user sees it before running."""
    if pick == "(write your own)":
        return f'<div class="preview"><span class="tchip">your own</span><div>{typed}</div></div>'
    t = T[pick.split("] ")[1].split(":")[0]]
    if t["type"] == "coding":
        tests = "<br>".join(f"<code>assert {a_}</code>" for a_ in t["visible"])
        body = f'<div>Write <code>{t["fn"]}</code>. {t["spec"]}</div><div class="small">Example checks the verifier can run:<br>{tests}</div>'
    else:
        body = f'<div>{t["question"]}</div>'
    return f'<div class="preview"><span class="tchip">{t["type"]}</span>{body}</div>'


st.session_state.setdefault("task", LABEL["code04"])
st.session_state.setdefault("left", "Fixed pipeline")
st.session_state.setdefault("right", R_NAME)
st.session_state.setdefault("mode", LIVE_Q)
st.session_state.setdefault("budget", 0.8)
st.session_state.setdefault("draw", 0)

st.markdown('<div class="qd">Quick demo &mdash; one click sets everything and starts it</div>', unsafe_allow_html=True)
for col, (i, preset) in zip(st.columns(len(PRESETS), gap="small"), enumerate(PRESETS)):
    col.button(preset[0], key=f"preset{i}", on_click=apply_preset, args=(i,), use_container_width=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    with st.container():
        st.markdown(panel_head("01", "Pick a task", "What should they work on?"), unsafe_allow_html=True)
        pick = st.selectbox("Task", ["(write your own)"] + [short(t) for t in T.values()], key="task", label_visibility="collapsed")
        typed = ""
        if pick == "(write your own)":
            typed = st.text_input("Your task (maths or research; research only knows our 16 invented documents)",
                                  "Who founded Norvane Labs, and in which year?")
        st.markdown(preview_html(), unsafe_allow_html=True)
with c2:
    with st.container():
        st.markdown(panel_head("02", "Pick the two corners", "Who races whom?"), unsafe_allow_html=True)
        left_name = st.selectbox("Left corner (the challenger)", list(CS), key="left")
        right_name = st.selectbox("Right corner (the learned controller)", list(CS), key="right")
        st.markdown(f'<div class="callout"><b>{left_name}</b><br>{CS[left_name].blurb}</div>'
                    f'<div class="callout"><b>{right_name}</b><br>{CS[right_name].blurb}</div>', unsafe_allow_html=True)
with c3:
    with st.container():
        st.markdown(panel_head("03", "Set the rules and race", "Where it runs, how much it may spend."), unsafe_allow_html=True)
        modes = [LIVE_Q, REPLAY]
        if has_groq:
            modes.insert(1, "Live - Groq gpt-oss-20b")
        mode = st.radio("Where do the helpers run?", modes, key="mode",
                        help="Live = a real AI model answers now. Replay = recorded answers, instant, works offline.")
        if mode.startswith("Live - Groq"):
            st.caption("The controllers were trained on qwen costs; costs are measured the same way (tokens) but not re-trained.")
        budget = st.slider("Money budget", min_value=0.1, max_value=1.2, step=0.05, key="budget",
                           help="One helper call costs about 0.02 to 0.15. When the money runs out, the task ends.")
        draw = 0
        if mode.startswith("Replay"):
            draw = st.number_input("Replay draw (which recorded answers are used)", min_value=0, max_value=999, key="draw")
        race_clicked = st.button("RACE", type="primary", key="race", use_container_width=True)
        b1, b2 = st.columns(2)
        board_clicked = b1.button("SCOREBOARD", key="board", use_container_width=True)
        one_clicked = b2.button("RUN ONE", key="run", use_container_width=True)
action = "race" if race_clicked else "board" if board_clicked else "one" if one_clicked else st.session_state.pop("autorun", None)


def corner(name, side, crown=False):
    tag = "LEFT CORNER" if side == "A" else "RIGHT CORNER"
    mark = '<span class="crown">WINNER</span>' if crown else ""
    return f'<div class="corner{" win" if crown else ""}"><div class="ctag">{tag}</div><div class="cname">{name}{mark}</div></div>'


def result_word(final):
    return {True: "Correct", False: "Wrong", None: "Not graded"}[final["correct"]] if final["answer"] is not None else "No answer"


def do_race(task):
    """Two controllers, same task, live and side by side. Steps alternate, so both columns fill in together."""
    names = {"A": left_name, "B": right_name}
    ctrls = {"A": CS[left_name], "B": CS[right_name]}
    ui = {}
    for side, col in zip("AB", st.columns(2, gap="large")):
        with col:
            head = st.empty()
            head.markdown(corner(names[side], side), unsafe_allow_html=True)
            meter = st.empty()
            ph = [c.empty() for c in st.columns(3)]
            plain = st.empty()
        ph[0].metric("Cost", "0.000")
        ph[1].metric("Steps", 0)
        ph[2].metric("Result", "Waiting...")
        ui[side] = dict(head=head, meter=meter, ph=ph, plain=plain, steps=[], final=None)
    gens = lockstep(*(safe_events(task, ctrls[sd], mode, budget, draw) for sd in "AB"))
    for side, ev in gens:
        u = ui[side]
        if ev["kind"] == "step":
            u["steps"].append(ev)
            u["plain"].markdown(narrate(u["steps"]), unsafe_allow_html=True)
            u["ph"][0].metric("Cost", f"{ev['spent']:.3f}")
            u["ph"][1].metric("Steps", ev["step"])
            u["ph"][2].metric("Result", "Running...")
            u["meter"].progress(min(1.0, ev["money_left"] / budget), text=f"Money left {ev['money_left']:.2f} of {budget:.2f}")
            if REPLAY_DELAY:
                time.sleep(REPLAY_DELAY)
        else:
            u["final"] = ev
            u["ph"][2].metric("Result", result_word(ev))
            u["plain"].markdown(narrate(u["steps"], ev), unsafe_allow_html=True)
    fa, fb = ui["A"]["final"], ui["B"]["final"]
    sa, sb = score(fa), score(fb)
    v = verdict(sa, sb)
    for side in "AB":
        ui[side]["head"].markdown(corner(names[side], side, crown=(v == side)), unsafe_allow_html=True)
    if sa is None:
        cheaper = left_name if fa["cost"] < fb["cost"] else right_name
        msg = (f"Your own task has no answer key, so I cannot tell who was right. The cheaper run was <b>{cheaper}</b> "
               f"({min(fa['cost'], fb['cost']):.3f}).")
    elif v == "tie":
        msg = f"<b>Tie.</b> Both scored {sa:+.2f}."
    else:
        win = names[v]
        msg = (f"<b>{win} wins.</b> Score = answer quality minus a cost penalty, the reward the DQN was trained on: "
               f"{left_name} {sa:+.2f}, {right_name} {sb:+.2f}.")
    st.markdown(f'<div class="winner">{msg}<div class="wsub">One task is a small sample. Press SCOREBOARD to race the same two '
                f'on {len(TEST_RECS)} tasks the controller never trained on.</div></div>', unsafe_allow_html=True)
    if sa is not None:
        st.markdown(bars([(left_name, sa, v == "A"), (right_name, sb, v == "B")], fmt="{:+.2f}"), unsafe_allow_html=True)
    with st.expander("See every call in detail (and the DQN's scores)"):
        for side, col in zip("AB", st.columns(2, gap="large")):
            with col:
                st.markdown(f"**{names[side]}**")
                st.markdown("".join(step_card(e, ctrls[side].agent is not None) for e in ui[side]["steps"]), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def board(bud):
    return board_scores(TEST_RECS, CS, bud, range(10))


def do_board():
    """Race the two corners on every unseen task (recorded runs, 10 draws each) and keep a running tally."""
    sc = board(budget)
    a_sc, a_co = sc[left_name]
    b_sc, b_co = sc[right_name]
    n = len(TEST_RECS)
    st.markdown(f"#### Scoreboard &nbsp;<small>&mdash; {right_name} vs {left_name} on {n} unseen tasks</small>", unsafe_allow_html=True)
    st.caption("These tasks were never used to train the controller. Each task is raced 10 times with different recorded model answers.")
    prog, tally = st.progress(0.0), st.empty()
    wins = ties = losses = 0
    for i in range(n):
        d = b_sc[i] - a_sc[i]
        wins, ties, losses = wins + (d > 0.01), ties + (abs(d) <= 0.01), losses + (d < -0.01)
        tally.markdown(f'<div class="tally"><span><b>{wins}</b> won by right corner</span><span><b>{ties}</b> ties</span>'
                       f'<span><b>{losses}</b> won by left corner</span><span class="of">{i + 1} of {n} tasks</span></div>',
                       unsafe_allow_html=True)
        prog.progress((i + 1) / n)
        if not os.environ.get("ORBIT_FAST"):
            time.sleep(0.08)
    prog.empty()
    adv = float((b_sc - a_sc).mean())
    if left_name == right_name:
        verdict_txt = "Same controller on both sides, so of course it is a tie."
    elif adv > 0.02 and wins > losses:
        verdict_txt = f"<b>{right_name} wins the scoreboard.</b>"
    elif adv < -0.02 and losses > wins:
        verdict_txt = f"<b>{left_name} wins the scoreboard.</b>"
    elif adv > 0.02:
        verdict_txt = f"<b>{right_name} is ahead on average</b>, but it wins fewer tasks than it loses, so call it close."
    else:
        verdict_txt = "<b>Too close to call.</b>"
    st.markdown(f'<div class="winner">{verdict_txt}<div class="wsub">Average advantage of the right corner: <b>{adv:+.3f}</b> per task '
                f'({wins} wins, {ties} ties, {losses} losses).</div></div>', unsafe_allow_html=True)
    st.markdown("**Average score per task** (higher is better)")
    st.markdown(bars([(left_name, float(a_sc.mean()), False), (right_name, float(b_sc.mean()), True)]), unsafe_allow_html=True)
    st.markdown("**Average cost per task** (lower is cheaper)")
    st.markdown(bars([(left_name, float(a_co.mean()), False), (right_name, float(b_co.mean()), True)], fmt="{:.3f}"), unsafe_allow_html=True)
    with st.expander("Task by task"):
        st.dataframe(pd.DataFrame({"Task": [r["id"] for r in TEST_RECS], "Type": [r["type"] for r in TEST_RECS],
                                   left_name: a_sc.round(3), right_name: b_sc.round(3),
                                   "Winner": ["right" if x > 0.01 else "left" if x < -0.01 else "tie" for x in (b_sc - a_sc)]}),
                     hide_index=True, use_container_width=True)


def do_one(task):
    """One controller alone: its steps in plain words, the details of each call, and the result."""
    ctrl = CS[right_name]
    cols = st.columns(4)
    ph = [c.empty() for c in cols]
    meter, answer = st.empty(), st.empty()
    st.markdown(f"#### In plain words &nbsp;<small>&mdash; {task['type']} task &middot; {ctrl.name}</small>", unsafe_allow_html=True)
    plain = st.empty()
    st.markdown("#### Agent trace &nbsp;<small>&mdash; the details of each call</small>", unsafe_allow_html=True)
    trace = st.empty()
    steps, cards, n_tok, final = [], [], 0, None
    ph[0].metric("Result", "Running...")
    for ev in safe_events(task, ctrl, mode, budget, draw):
        if ev["kind"] == "step":
            steps.append(ev)
            plain.markdown(narrate(steps), unsafe_allow_html=True)
            cards.append(step_card(ev, ctrl.agent is not None))
            trace.markdown("".join(cards[:-1]) + cards[-1].replace('class="step"', 'class="step new"', 1), unsafe_allow_html=True)
            n_tok += ev["tin"] + ev["tout"]
            ph[1].metric("Cost", f"{ev['spent']:.3f}")
            ph[2].metric("Tokens", n_tok)
            ph[3].metric("Steps", ev["step"])
            meter.progress(min(1.0, ev["money_left"] / budget), text=f"Money left {ev['money_left']:.2f} of {budget:.2f}")
            if REPLAY_DELAY:
                time.sleep(REPLAY_DELAY)
        else:
            final = ev
    ph[0].metric("Result", result_word(final))
    plain.markdown(narrate(steps, final), unsafe_allow_html=True)
    with answer.container():
        st.markdown("#### Final answer")
        if final["answer"] is None:
            st.warning("The controller stopped without producing any answer.")
        else:
            st.code(final["answer"], language=None)


st.markdown("<div id='results' style='height:6px'></div>", unsafe_allow_html=True)
with st.container():
    st.markdown(panel_head("04", "The race", "Same task, two decision-makers, live and side by side."), unsafe_allow_html=True)
    st.markdown('<div class="legend">' + "".join(
        f'<div><span class="badge" style="background:{c}">{n}</span> {d}</div>' for n, c, d in HELPERS) + '</div>',
        unsafe_allow_html=True)
    banner = st.empty()
    if action is None:
        st.markdown('<div class="empty">Pick a task and two corners above, then press <b>RACE</b> to watch them work side by side. '
                    '<b>SCOREBOARD</b> races them on 24 tasks they never trained on. <b>RUN ONE</b> shows only the right corner.</div>',
                    unsafe_allow_html=True)
    else:
        components.html("<script>const e = window.parent.document.getElementById('results');"
                        "if (e) e.scrollIntoView({behavior: 'smooth', block: 'start'});</script>", height=0)
        banner.markdown('<div class="banner"><span class="pulse"></span>Running now. Each step appears as soon as it finishes '
                        '(live mode can take up to 20 seconds).</div>', unsafe_allow_html=True)
        t0 = time.time()
        task = None
        if action in ("race", "one"):
            try:
                task = get_task()
            except ValueError as e:
                st.error(str(e))
                st.stop()
            if mode.startswith("Replay") and task["id"] not in records():
                st.error("Replay only has the 50 built-in tasks. Pick one from the list or switch to a live mode.")
                st.stop()
        {"race": lambda: do_race(task), "one": lambda: do_one(task), "board": do_board}[action]()
        banner.markdown(f'<div class="banner done">Finished in {time.time() - t0:.1f} seconds.</div>', unsafe_allow_html=True)


st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
with st.container():
    st.markdown(panel_head("05", "What we measured", "Every number comes from a test written down before it was run. Logs are in docs/logs."),
                unsafe_allow_html=True)
    t_sim, t_rob, t_real, t_lim = st.tabs(["In the simulator", "Robustness and ablations", "On real language models", "Limitations and future work"])
    with t_sim:
        st.caption("10,000 unseen tasks. Return = answer quality minus a cost penalty.")
        st.markdown(bars([(n, v, h) for n, v, h in SIM]), unsafe_allow_html=True)
        st.markdown("**Take away:** calling every helper every time is the worst choice, and the learned controller beats the best "
                    "hand-written rule by about 9%.")
        st.markdown("##### Without difficulty and confidence (the signals real language models cannot give us)")
        st.markdown(bars(SIM_BLIND), unsafe_allow_html=True)
        st.markdown("Even blind, the learned controller beats the best hand-written rule by about 16%, and all 5 training runs agree.")
    with t_rob:
        st.markdown("##### Still ahead when the world changes")
        st.dataframe(pd.DataFrame([(c, a, b, d, round(d - b, 3)) for c, a, b, d in SHIFT],
                                  columns=["World", "Single agent", "Best rule", "DQN", "DQN minus rule"]),
                     hide_index=True, use_container_width=True)
        st.markdown("##### What each ingredient is worth (change in return vs the full DQN)")
        st.markdown(bars([(n, v, False) for n, v in ABLATE]), unsafe_allow_html=True)
        st.markdown("**Take away:** the cost penalty is by far the most important part. Without it the controller learns to call "
                    "everything and the return collapses. The other signals each add only a little, because they overlap.")
    with t_real:
        st.caption("Recorded runs of two real models on 50 tasks (24 held out for testing). Higher is better.")
        for model, (single, fixed, rule, dqn) in REAL.items():
            st.markdown(f"##### {model}")
            st.markdown(bars([("Single agent", single, False), ("Fixed pipeline", fixed, False),
                                  ("Hand-tuned rule", rule, False), ("DQN retrained on real runs", dqn, True)]), unsafe_allow_html=True)
        st.markdown("**Take away:** calling the right helper matters. On research questions, one or two researcher calls lift success "
                    "from 0% to 43% (qwen) and to 93% (gpt-oss). The always-use-everyone pipeline loses on both models.")
    with t_lim:
        st.markdown("""**What this study does not show**
- The real-model results come from recorded runs on 50 tasks (24 for testing) and two models, so the error bars are wide.
- A well-tuned hand rule is as good as the retrained DQN on qwen (+0.576 vs +0.580) and slightly better on gpt-oss (+0.819 vs +0.770).
- A DQN trained only in the simulator does not transfer to research questions: it never calls the researcher. Retraining on real runs fixes this.
- The live demo and its scoreboard use one saved model (training run 0, picked in advance), which scores a little lower than the 5-run average in the charts. On the 24 unseen tasks it beats the fixed pipeline clearly (+0.196 per task), beats the single agent only slightly on average (+0.068, but wins fewer tasks than it loses), and is behind the hand-tuned rule (-0.056).
- A retry replays the next recorded answer; the effect of telling the solver why it was rejected is not captured.
- The conversion from tokens to cost units is a choice we calibrated; a different choice would shift the best policy.

**Future work**
- More tasks and more models, to tighten the error bars.
- Learn online from live model calls instead of recordings.
- Make the simulator more realistic (randomise skill and cost) so a simulator-trained controller transfers without retraining.
- A learned difficulty estimate, so the controller can use difficulty with real models.
- Add an LLM-as-router baseline, with its routing calls counted in the cost.""")
