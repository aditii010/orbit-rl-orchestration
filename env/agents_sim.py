TASK_TYPES = ["coding", "analytical", "research"]

SOLVER_SKILL = {"coding": 0.95, "analytical": 0.85, "research": 0.70}

VERIFIER_RELIABILITY = {"coding": 0.95, "analytical": 0.85, "research": 0.70}

BASE_COST = {
    "researcher": {"coding": 0.08, "analytical": 0.08, "research": 0.10},
    "solver": {"coding": 0.10, "analytical": 0.12, "research": 0.15},
    "verifier": {"coding": 0.03, "analytical": 0.05, "research": 0.08},
    "summarizer": {"coding": 0.03, "analytical": 0.03, "research": 0.03},
}

REPEAT_DECAY = 0.85
RETRY_DISCOUNT = 0.8
FAILED_VERIFY_BONUS = 1.10
EVIDENCE_PER_CALL = 0.35


def solver_p_correct(task_type, difficulty, n_prev_solver_calls, evidence=0.0,
                     after_failed_verify=False):
    p = SOLVER_SKILL[task_type] * (1.0 - difficulty) * REPEAT_DECAY ** n_prev_solver_calls
    if task_type == "research":
        p *= 0.6 + 0.4 * evidence
    else:
        p *= 0.9 + 0.1 * evidence
    if after_failed_verify:
        p *= FAILED_VERIFY_BONUS
    return max(0.0, min(1.0, p))


def call_cost(agent, task_type, rng, is_retry=False):
    c = BASE_COST[agent][task_type] * rng.uniform(0.8, 1.2)
    return c * RETRY_DISCOUNT if is_retry else c


def verifier_verdict(task_type, candidate_correct, rng):
    right = rng.random() < VERIFIER_RELIABILITY[task_type]
    return candidate_correct if right else (not candidate_correct)


def confidence_score(candidate_correct, rng):
    mean = 0.70 if candidate_correct else 0.50
    return float(min(1.0, max(0.0, rng.normal(mean, 0.15))))
