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

st.set_page_config(page_title="ORBIT", layout="wide")
AGENT = {"INVOKE_RESEARCHER": ("Researcher", "#2f6fb0"), "INVOKE_SOLVER": ("Solver", "#2e8b57"),
         "INVOKE_VERIFIER": ("Verifier", "#b8860b"), "INVOKE_SUMMARIZER": ("Summarizer", "#7b4fa0"),
         "RETRY_SOLVER": ("Solver (retry)", "#c8602a"), "STOP": ("STOP", "#666666")}
ACTION_ORDER = list(AGENT)
st.markdown("""<style>
.badge{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-weight:600;font-size:0.85rem}
.qrow{display:flex;align-items:center;margin:2px 0;font-size:0.82rem}
.qname{width:110px}.qbar{height:12px;border-radius:3px;margin-right:8px}.qoff{color:#999;font-style:italic}
</style>""", unsafe_allow_html=True)


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


# ---------------------------------------------------------------- sidebar
CS = controllers()
T = tasks()
has_groq = load_groq_key()
with st.sidebar:
    st.title("ORBIT")
    st.caption("An RL controller decides which LLM agent to call next, or when to stop.")
    modes = ["Live - Ollama qwen2.5:3b (local)", "Replay (recorded, offline)"]
    if has_groq:
        modes.insert(1, "Live - Groq gpt-oss-20b")
    mode = st.radio("Where do the agents run?", modes)
    if mode.startswith("Live - Groq"):
        st.caption("The DQNs were trained on qwen costs; costs here are measured the same way (tokens) but not re-trained.")
    ctrl_name = st.radio("Controller", list(CS), index=list(CS).index("DQN (refit on real runs)"))
    st.caption(CS[ctrl_name].blurb)
    budget = st.slider("Money budget (cost units)", 0.2, 1.2, 0.8, 0.05,
                       help="A call costs about 0.02 to 0.15. When the budget runs out the episode ends.")
    draw = 0
    if mode.startswith("Replay"):
        draw = st.number_input("Replay draw (which recorded samples are used)", 0, 999, 0)

# ---------------------------------------------------------------- task picker (main area)
st.header("Orchestrating LLM agents with reinforcement learning")
pick = st.selectbox("Task", ["(write your own)"] + [short(t) for t in T.values()], index=1)
typed = ""
if pick == "(write your own)":
    typed = st.text_input("Your task (analytical or research; research only knows our 16 invented documents)",
                          "Who founded Norvane Labs, and in which year?")
tab_live, tab_cmp = st.tabs(["Live run", "Compare controllers"])


def get_task():
    if pick == "(write your own)":
        return custom_task(typed)
    return T[pick.split("] ")[1].split(":")[0]]


with tab_live:
    if st.button("Run", type="primary", key="run"):
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
            st.subheader("Meter")
            meter, stats = st.empty(), st.empty()
        with left:
            st.subheader(f"Agent trace  ({task['type']} task)")
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
            st.subheader("Final answer")
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
    if st.button("Run all controllers", key="cmp"):
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
        st.bar_chart(df.set_index("Controller")["Cost"])
