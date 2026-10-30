"""Engine tests with a fake LLM (no network): events, cost accounting, budget stop, replay agrees with TraceEnv."""
from app.engine import Controller, build_controllers, classify, custom_task, run_live, run_replay
from baselines.policies import fixed_pipeline, single_agent
from llm.tasks import all_tasks
from llm.trace_env import TraceEnv, load_cache

TASKS = {t["id"]: t for t in all_tasks()}
REC = {r["id"]: r for r in load_cache("runs/cache_qwen3b.jsonl")}


class FakeLLM:
    def chat(self, user, system=None, max_tokens=400, temperature=0.7):
        return {"text": "Answer: 16\nCONFIDENCE: 90", "tin": 100, "tout": 50, "seconds": 0.1}


def test_single_agent_live_has_two_events_and_a_final():
    ev = list(run_live(TASKS["ana00"], Controller("s", single_agent), FakeLLM(), budget=1.0))
    assert [e["kind"] for e in ev] == ["step", "step", "final"]
    assert ev[0]["action"] == "INVOKE_SOLVER" and ev[1]["action"] == "STOP"
    assert abs(ev[0]["cost"] - (150 / 2500 + 0.02)) < 1e-9
    assert ev[-1]["answer"] == "Answer: 16" and ev[-1]["tokens"] == 150


def test_tiny_budget_forces_the_end():
    ev = list(run_live(TASKS["ana00"], Controller("f", fixed_pipeline), FakeLLM(), budget=0.05))
    assert ev[-1]["forced_end"] and ev[-1]["steps"] <= 2


def test_replay_matches_trace_env_reward_outcome():
    t = TASKS["code00"]
    ev = list(run_replay(REC["code00"], t, Controller("f", fixed_pipeline), budget=1.0, seed=3))
    env = TraceEnv([REC["code00"]])
    obs, done = env.reset(3), False
    env.budget = 1.0
    while not done:
        obs, r, done, info = env.step(fixed_pipeline(env._observe(), env.action_mask()))
    assert ev[-1]["correct"] == info["success"] and abs(ev[-1]["cost"] - info["total_cost"]) < 1e-9


def test_router_and_custom_tasks():
    assert classify("Write a function that sums a list") == "coding"
    assert classify("A shirt costs 500 rupees with 20% off. How much do I pay?") == "analytical"
    assert classify("Who founded Veltrix Corp?") == "research"
    assert custom_task("Who founded Veltrix Corp?")["type"] == "research"


def test_controllers_build_and_dqn_reports_masked_q():
    cs = build_controllers()
    assert "DQN (simulator only)" in cs and "Single agent" in cs
    ev = list(run_replay(REC["res00"], TASKS["res00"], cs["DQN (simulator only)"], budget=1.0, seed=0))
    first = ev[0]
    assert first["q"] is not None and len(first["q"]) == 6
