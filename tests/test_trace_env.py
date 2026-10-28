"""Replay environment: same outcome for same seed, correct reward arithmetic, no leak of difficulty/confidence."""
import numpy as np
import pytest

from baselines.policies import fixed_pipeline, single_agent
from env.orbit_env import A
from llm.trace_env import TraceEnv, load_cache, split_tasks

REC = load_cache("runs/cache_qwen3b.jsonl")


def run(env, policy, seed):
    obs, done, tot = env.reset(seed), False, 0.0
    while not done:
        obs, r, done, info = env.step(policy(obs, env.action_mask()))
        tot += r
    return tot, info


def test_split_is_disjoint_and_stratified():
    dev, test = split_tasks(REC)
    assert not {r["id"] for r in dev} & {r["id"] for r in test}
    assert len(dev) + len(test) == 50 and abs(len(dev) - len(test)) <= 3


def test_same_seed_same_result():
    env = TraceEnv(REC)
    assert run(env, fixed_pipeline, 7)[0] == run(env, fixed_pipeline, 7)[0]


def test_observation_hides_difficulty_and_confidence():
    env = TraceEnv(REC)
    for s in range(20):
        o = env.reset(s)
        assert o[3] == 0 and o[14] == 0


def test_reward_matches_hand_computation():
    env = TraceEnv(REC)
    env.reset(0)
    _, r1, d1, _ = env.step(A["INVOKE_SOLVER"])
    assert not d1 and r1 < 0                                  # a call only costs
    _, r2, d2, info = env.step(A["STOP"])
    assert d2 and r2 == float(info["success"])                # stop is free; reward = correctness


def test_stop_without_candidate_is_penalised():
    env = TraceEnv(REC)
    env.reset(0)
    _, r, d, info = env.step(A["STOP"])
    assert d and r == pytest.approx(-0.3) and not info["success"]


def test_research_tasks_need_the_researcher():
    """In the cache, no sample is correct on research tasks with zero researcher calls."""
    for rec in REC:
        if rec["type"] == "research":
            assert not any(s["ok_final"] for s in rec["levels"]["0"])
