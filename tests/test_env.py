import numpy as np
import pytest

from env.agents_sim import solver_p_correct
from env.orbit_env import A, OBS_DIM, OrbitEnv


def run_script(actions, seed):
    env = OrbitEnv()
    env.reset(seed=seed)
    total, done = 0.0, False
    for a in actions + [A["STOP"]]:
        if done:
            break
        if not env.action_mask()[a]:
            continue
        _, r, done, _ = env.step(a)
        total += r
    return total


def mean_return(actions, n=1500):
    return float(np.mean([run_script(actions, s) for s in range(n)]))


def test_solver_gets_less_useful_with_repeats():
    p1 = solver_p_correct("coding", 0.5, 0)
    p3 = solver_p_correct("coding", 0.5, 2)
    assert p3 < p1


def test_harder_means_lower_probability():
    assert solver_p_correct("analytical", 0.8, 0) < solver_p_correct("analytical", 0.2, 0)


def test_observation_shape_and_range():
    env = OrbitEnv()
    o = env.reset(seed=0)
    assert o.shape == (OBS_DIM,) and np.all(o >= 0) and np.all(o <= 1)


def test_masking():
    env = OrbitEnv()
    env.reset(seed=1)
    m = env.action_mask()
    assert not m[A["INVOKE_VERIFIER"]] and not m[A["RETRY_SOLVER"]] and m[A["INVOKE_SOLVER"]]
    with pytest.raises(ValueError):
        env.step(A["INVOKE_VERIFIER"])


def test_hidden_truth_not_in_observation():
    env = OrbitEnv()
    env.reset(seed=3)
    o, *_ = env.step(A["INVOKE_SOLVER"])
    assert o.shape == (OBS_DIM,)


SOLVE, VERIFY, SUM = A["INVOKE_SOLVER"], A["INVOKE_VERIFIER"], A["INVOKE_SUMMARIZER"]
RESEARCH, STOP = A["INVOKE_RESEARCHER"], A["STOP"]


def test_1_stopping_immediately_is_worse_than_trying():
    assert mean_return([SOLVE, SUM]) > mean_return([])


def test_2_useless_repeat_calls_score_lower():
    assert mean_return([SOLVE, VERIFY, SUM]) > mean_return([SOLVE, VERIFY, VERIFY, VERIFY, SUM])


def test_3_wasted_calls_reduce_return_for_same_task():
    for s in range(20):
        assert run_script([SOLVE, SUM], s) > run_script([SOLVE, SUM, RESEARCH, RESEARCH], s) - 1e-9


def test_4_budget_exhausted_without_candidate_is_worst():
    burn = mean_return([RESEARCH] * 8)
    assert burn < mean_return([]) < mean_return([SOLVE, SUM])
