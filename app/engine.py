"""Runs ONE episode of the orchestrator and yields an event after every step, so a UI can draw it live.
Two modes with the same event format:
  run_live   : real agents (llm/agents.py) on a real LLM
  run_replay : recorded outputs from a trace cache (instant, offline)
Observation, action mask, cost rule and stopping rules are the same as in llm/trace_env.py (what the DQN was trained on)."""
import re

import numpy as np
import torch

from baselines.policies import (fixed_pipeline, make_typed_rule, single_agent)
from env.agents_sim import TASK_TYPES
from env.orbit_env import A, ACTIONS, DEFAULT_CONFIG, OBS_DIM
from llm.agents import final_answer, researcher, solver, strip_confidence, verifier
from llm.tasks import grade
from llm.trace_env import CALL_OVERHEAD, TraceEnv, _cost
from rl.dqn import DoubleDQN

MAX_STEPS = DEFAULT_CONFIG["max_steps"]
HIDE = (3, 14)
# Typed rule tuned on qwen dev tasks (docs/logs/typed_qwen3b.log).
QWEN_RULE = dict(res_calls=(0, 0, 1), verify=(True, False, False), max_retries=2, summarize=False)


class Controller:
    """A named policy (obs, mask) -> action. DQN controllers can also report their Q-values."""
    def __init__(self, name, policy, agent=None, blurb=""):
        self.name, self.policy, self.agent, self.blurb = name, policy, agent, blurb

    def q_values(self, obs, mask):
        if self.agent is None:
            return None
        with torch.no_grad():
            q = self.agent.q(torch.as_tensor(obs * self.agent.keep).unsqueeze(0))[0].numpy()
        return np.where(mask, q, np.nan)          # NaN = illegal action (masked out)


def build_controllers(refit_path="checkpoints/refit_qwen_seed0.pt", sim_path="checkpoints/blind_seed0.pt"):
    sim = DoubleDQN(seed=0, hide=HIDE).load(sim_path)
    cs = [Controller("Single agent", single_agent, blurb="One solver call, then stop."),
          Controller("Fixed pipeline", fixed_pipeline, blurb="Researcher, solver, verifier, summarizer, always."),
          Controller("Tuned rule", make_typed_rule(**QWEN_RULE),
                     blurb="Hand-written, tuned on the dev tasks: researcher on research tasks, test-check on coding.")]
    try:
        refit = DoubleDQN(seed=0, hide=HIDE).load(refit_path)
        cs.append(Controller("DQN (refit on real runs)", refit.policy(), refit,
                             blurb="Double DQN retrained on recorded qwen2.5:3b runs of the 26 dev tasks."))
    except FileNotFoundError:
        pass
    cs.append(Controller("DQN (simulator only)", sim.policy(), sim,
                         blurb="Double DQN trained only in the simulator, never saw a real LLM."))
    return {c.name: c for c in cs}


def classify(text):
    """Tiny keyword router for free-text tasks (the DQN needs a task type). A real system would use a classifier."""
    t = text.lower()
    if any(k in t for k in ("write a function", "write a python", "def ", "implement a function", "code")):
        return "coding"
    if re.search(r"\d", t) and any(k in t for k in ("how much", "how many", "total", "calculate", "cost", "price",
                                                    "per ", "%", "profit", "average", "rupees", "left")):
        return "analytical"
    return "research"


def custom_task(text):
    kind = classify(text)
    if kind == "coding":
        raise ValueError("Free-text coding tasks have no tests to run. Pick a coding example from the list.")
    return {"id": "custom", "type": kind, "question": text}


def _observe(ttype, st):
    o = np.zeros(OBS_DIM, dtype=np.float32)
    o[TASK_TYPES.index(ttype)] = 1.0
    o[4] = float(st["cand"] is not None)
    o[5 + st["last_verify"]] = 1.0
    for i, k in enumerate(["researcher", "solver", "verifier", "summarizer"]):
        o[8 + i] = st["calls"][k] / MAX_STEPS
    o[12] = max(0.0, (st["budget"] - st["spent"]) / st["budget"])
    o[13] = 1.0 - st["steps"] / MAX_STEPS
    o[15] = float(st["summarized"])
    return o


def _mask(st):
    has = st["cand"] is not None
    m = np.ones(len(ACTIONS), dtype=bool)
    m[A["INVOKE_SOLVER"]] = not has
    m[A["INVOKE_VERIFIER"]] = has
    m[A["INVOKE_SUMMARIZER"]] = has and not st["summarized"]
    m[A["RETRY_SOLVER"]] = has
    return m


def _event(st, name, cost, detail, usage, q, mask):
    return {"kind": "step", "step": st["steps"], "action": name, "cost": cost, "detail": detail,
            "tin": usage["tin"], "tout": usage["tout"], "seconds": usage["seconds"],
            "spent": st["spent"], "money_left": max(0.0, st["budget"] - st["spent"]),
            "q": None if q is None else q.tolist(), "legal": mask.tolist()}


def _final(task, st, answer, wall):
    ok = grade(task, answer) if (answer is not None and task["id"] != "custom") else None
    return {"kind": "final", "answer": answer, "correct": ok, "cost": st["spent"], "steps": st["steps"],
            "tokens": st["tokens"], "llm_seconds": st["seconds"], "wall_seconds": wall,
            "calls": dict(st["calls"]), "forced_end": st["forced"]}


def run_live(task, ctrl, llm, budget=0.8, max_steps=MAX_STEPS):
    """Real agents. Yields one 'step' event per action and a 'final' event at the end."""
    import time
    t0 = time.time()
    st = dict(cand=None, last_verify=0, summarized=False, calls={k: 0 for k in
              ("researcher", "solver", "verifier", "summarizer")}, budget=budget, spent=0.0, steps=0,
              tokens=0, seconds=0.0, forced=False)
    evidence, seen, answer = "", [], None
    zero = {"tin": 0, "tout": 0, "seconds": 0.0}
    while True:
        obs, mask = _observe(task["type"], st), _mask(st)
        action = ctrl.policy(obs, mask)
        q = ctrl.q_values(obs, mask)
        name, usage, detail = ACTIONS[action], zero, ""
        if name == "INVOKE_RESEARCHER":
            ev, new, usage = researcher(llm, task, seen)
            st["calls"]["researcher"] += 1
            if task["type"] == "research":
                evidence, seen = evidence + "\n" + ev, seen + new
            else:
                evidence = ev
            detail = ev
            cost = _cost(usage["tin"], usage["tout"])
        elif name in ("INVOKE_SOLVER", "RETRY_SOLVER"):
            s = solver(llm, task, evidence=evidence)
            usage = {k: s[k] for k in ("tin", "tout", "seconds")}
            st["cand"], st["last_verify"], st["summarized"] = s["text"], 0, False
            st["calls"]["solver"] += 1
            detail = strip_confidence(s["text"])
            cost = _cost(usage["tin"], usage["tout"])
        elif name == "INVOKE_VERIFIER":
            passed, fb, usage = verifier(llm, task, st["cand"], evidence)
            st["last_verify"] = 1 if passed else 2
            st["calls"]["verifier"] += 1
            detail = ("PASS - " if passed else "FAIL - ") + fb
            cost = _cost(usage["tin"], usage["tout"])
        elif name == "INVOKE_SUMMARIZER":
            st["summarized"] = True
            st["calls"]["summarizer"] += 1
            detail = final_answer(task, st["cand"])
            cost = CALL_OVERHEAD
        else:
            cost = 0.0
            detail = "Controller decided to stop."
        st["spent"] += cost
        st["steps"] += 1
        st["tokens"] += usage["tin"] + usage["tout"]
        st["seconds"] += usage["seconds"]
        yield _event(st, name, cost, detail, usage, q, mask)
        if name == "STOP" or st["spent"] >= budget or st["steps"] >= max_steps:
            st["forced"] = name != "STOP"
            break
    if st["cand"] is not None:
        answer = final_answer(task, st["cand"]) if st["summarized"] else strip_confidence(st["cand"])
    yield _final(task, st, answer, time.time() - t0)


def run_replay(record, task, ctrl, budget=0.8, seed=0, max_steps=MAX_STEPS):
    """Same events from recorded outputs (llm/trace_env.py). Instant and offline. 'task' is the llm.tasks dict."""
    env = TraceEnv([record], {"max_steps": max_steps})
    env.reset(seed)
    env.budget = budget                                     # let the UI set the budget
    tokens, seconds, done = 0, 0.0, False
    st = {"steps": 0, "spent": 0.0, "budget": budget}
    while not done:
        obs, mask = env._observe(), env.action_mask()
        action = ctrl.policy(obs, mask)
        q = ctrl.q_values(obs, mask)
        before = env.spent
        _, _, done, info = env.step(action)
        name = ACTIONS[action]
        c = env.cand
        if name == "INVOKE_RESEARCHER":
            note = record["notes"][min(env.calls["researcher"] - 1, len(record["notes"]) - 1)]
            usage = {k: note[k] for k in ("tin", "tout", "seconds")}
            detail = note["text"]
        elif name in ("INVOKE_SOLVER", "RETRY_SOLVER"):
            usage = {"tin": c["tin"], "tout": c["tout"], "seconds": c["sec"]}
            detail = strip_confidence(c["text"])
        elif name == "INVOKE_VERIFIER":
            usage = {"tin": c["v_tin"], "tout": c["v_tout"], "seconds": c["v_sec"]}
            detail = "PASS" if c["v_pass"] else "FAIL"
        else:
            usage = {"tin": 0, "tout": 0, "seconds": 0.0}
            detail = "Summarizer formatted the answer." if name == "INVOKE_SUMMARIZER" else "Controller decided to stop."
        tokens += usage["tin"] + usage["tout"]
        seconds += usage["seconds"]
        st.update(steps=env.steps, spent=env.spent)
        yield _event(st, name, env.spent - before, detail, usage, q, mask)
    answer = None
    if env.has_candidate:
        answer = final_answer(task, env.cand["text"]) if env.summarized else strip_confidence(env.cand["text"])
    fin = _final(task, {**st, "tokens": tokens, "seconds": seconds, "calls": env.calls,
                        "forced": info["forced_end"]}, answer, seconds)
    yield fin


# ---------------------------------------------------------------- racing two controllers
def score(final):
    """The reward the DQN was trained on (llm/trace_env.py): 1 for a correct answer, 0 for a wrong one, minus lambda*cost/reference,
    and a flat penalty when no answer was produced. None when the task has no answer key (a custom task)."""
    c = DEFAULT_CONFIG
    cost_term = c["lambda_cost"] * final["cost"] / c["reference_cost"]
    if final["answer"] is None:
        return -c["no_candidate_penalty"] - cost_term
    if final["correct"] is None:
        return None
    return float(final["correct"]) - cost_term


def lockstep(gen_a, gen_b):
    """Alternate between two episode generators so a page can show both live. Yields ('A' or 'B', event) until both are exhausted."""
    gens = {"A": gen_a, "B": gen_b}
    while gens:
        for side in list(gens):
            try:
                yield side, next(gens[side])
            except StopIteration:
                del gens[side]


def verdict(score_a, score_b, tol=0.005):
    """'A', 'B' or 'tie' (None if a score is missing)."""
    if score_a is None or score_b is None:
        return None
    return "tie" if abs(score_a - score_b) <= tol else ("A" if score_a > score_b else "B")


def replay_score(record, ctrl, budget, draw):
    """Fast score of one recorded episode using the training environment's own reward (no grading subprocess). Returns (reward, cost)."""
    env = TraceEnv([record])
    env.reset(draw)
    env.budget = budget
    total, done = 0.0, False
    while not done:
        _, r, done, info = env.step(ctrl.policy(env._observe(), env.action_mask()))
        total += r
    return total, info["total_cost"]


def board_scores(records, controllers, budget=0.8, draws=range(10)):
    """Mean score and cost of every controller on every record, averaged over several recorded draws.
    Returns {name: (scores array, costs array)}, arrays in the order of `records`."""
    out = {}
    for name, ctrl in controllers.items():
        sc, co = [], []
        for rec in records:
            runs = [replay_score(rec, ctrl, budget, d) for d in draws]
            sc.append(sum(r for r, _ in runs) / len(runs))
            co.append(sum(c for _, c in runs) / len(runs))
        out[name] = (np.array(sc), np.array(co))
    return out
