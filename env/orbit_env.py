"""ORBIT orchestration environment (Gym-style: reset / step).

Hidden: true difficulty, candidate correctness, evidence level.
Seen by the controller: the 16-number observation from _observe().
"""
import numpy as np

from env.agents_sim import (TASK_TYPES, call_cost, confidence_score,
                            solver_p_correct, verifier_verdict)

ACTIONS = ["INVOKE_RESEARCHER", "INVOKE_SOLVER", "INVOKE_VERIFIER",
           "INVOKE_SUMMARIZER", "RETRY_SOLVER", "STOP"]
A = {name: i for i, name in enumerate(ACTIONS)}
OBS_DIM = 16

DEFAULT_CONFIG = {
    "max_steps": 8,
    "budget_range": (0.5, 1.2),     # money budget per episode, sampled uniformly
    "type_probs": (1 / 3, 1 / 3, 1 / 3),
    "difficulty_beta": (2.0, 2.0),  # difficulty ~ Beta(a, b)
    "lambda_cost": 0.5,
    "reference_cost": 0.4,
    "no_candidate_penalty": 0.3,
    "unsummarized_quality": 0.9,    # a correct answer that was never summarized is worth 0.9
    "difficulty_noise": 0.15,       # noise in the controller's difficulty estimate
}


class OrbitEnv:
    def __init__(self, config=None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.rng = np.random.default_rng()

    # ---------- episode control ----------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c, r = self.cfg, self.rng
        self.task_type = TASK_TYPES[r.choice(3, p=c["type_probs"])]
        self.difficulty = float(r.beta(*c["difficulty_beta"]))
        self.est_difficulty = float(np.clip(
            self.difficulty + r.normal(0, c["difficulty_noise"]), 0, 1))  # fixed per episode
        self.budget = float(r.uniform(*c["budget_range"]))
        # hidden state
        self.evidence = 0.0
        self.has_candidate = False
        self.candidate_correct = False
        self.last_verify = 0            # 0 none, 1 pass, 2 fail (for the current candidate)
        self.confidence = 0.0
        self.summarized = False
        # bookkeeping
        self.calls = {"researcher": 0, "solver": 0, "verifier": 0, "summarizer": 0}
        self.spent = 0.0
        self.steps = 0
        self.done = False
        self.log = []
        return self._observe()

    def action_mask(self):
        """True where an action is allowed."""
        m = np.ones(len(ACTIONS), dtype=bool)
        m[A["INVOKE_SOLVER"]] = not self.has_candidate      # first draft only
        m[A["INVOKE_VERIFIER"]] = self.has_candidate
        m[A["INVOKE_SUMMARIZER"]] = self.has_candidate and not self.summarized
        m[A["RETRY_SOLVER"]] = self.has_candidate
        return m

    # ---------- one step ----------
    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over, call reset()")
        if not self.action_mask()[action]:
            raise ValueError(f"illegal action {ACTIONS[action]}")
        c = self.cfg
        name = ACTIONS[action]
        cost = 0.0

        if name == "INVOKE_RESEARCHER":
            cost = call_cost("researcher", self.task_type, self.rng)
            self.calls["researcher"] += 1
            self.evidence = min(1.0, self.evidence + 0.35)
        elif name in ("INVOKE_SOLVER", "RETRY_SOLVER"):
            retry = name == "RETRY_SOLVER"
            cost = call_cost("solver", self.task_type, self.rng, is_retry=retry)
            p = solver_p_correct(self.task_type, self.difficulty, self.calls["solver"],
                                 self.evidence, after_failed_verify=retry and self.last_verify == 2)
            self.calls["solver"] += 1
            self.candidate_correct = bool(self.rng.random() < p)
            self.has_candidate = True
            self.last_verify = 0                              # new candidate, not yet checked
            self.confidence = confidence_score(self.candidate_correct, self.rng)
            self.summarized = False
        elif name == "INVOKE_VERIFIER":
            cost = call_cost("verifier", self.task_type, self.rng)
            self.calls["verifier"] += 1
            passed = verifier_verdict(self.task_type, self.candidate_correct, self.rng)
            self.last_verify = 1 if passed else 2
        elif name == "INVOKE_SUMMARIZER":
            cost = call_cost("summarizer", self.task_type, self.rng)
            self.calls["summarizer"] += 1
            self.summarized = True

        self.spent += cost
        self.steps += 1
        reward = -c["lambda_cost"] * cost / c["reference_cost"]

        out_of_budget = self.spent >= self.budget
        out_of_steps = self.steps >= c["max_steps"]
        stopped = name == "STOP"
        self.done = stopped or out_of_budget or out_of_steps
        info = {"cost": cost, "forced_end": self.done and not stopped}
        if self.done:
            reward += self._terminal_reward()
            info.update(success=self._success(), total_cost=self.spent, steps=self.steps,
                        calls=dict(self.calls), task_type=self.task_type)
        self.log.append((name, cost, reward))
        return self._observe(), reward, self.done, info

    # ---------- helpers ----------
    def _success(self):
        return self.has_candidate and self.candidate_correct

    def _terminal_reward(self):
        c = self.cfg
        if not self.has_candidate:
            return -c["no_candidate_penalty"]
        if not self.candidate_correct:
            return 0.0
        return 1.0 if self.summarized else c["unsummarized_quality"]

    def _observe(self):
        c = self.cfg
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[TASK_TYPES.index(self.task_type)] = 1.0                       # 0-2 task type
        o[3] = self.est_difficulty                                      # 3 noisy difficulty
        o[4] = float(self.has_candidate)                                # 4 has candidate
        o[5 + self.last_verify] = 1.0                                   # 5-7 last verify
        for i, k in enumerate(["researcher", "solver", "verifier", "summarizer"]):
            o[8 + i] = self.calls[k] / c["max_steps"]                   # 8-11 call counts
        o[12] = max(0.0, (self.budget - self.spent) / self.budget)      # 12 money left
        o[13] = 1.0 - self.steps / c["max_steps"]                       # 13 steps left
        o[14] = self.confidence                                         # 14 noisy confidence
        o[15] = float(self.summarized)                                  # 15 summarized
        return o
