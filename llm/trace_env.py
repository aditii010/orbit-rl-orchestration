import json

import numpy as np

from env.orbit_env import A, ACTIONS, DEFAULT_CONFIG, OBS_DIM
from env.agents_sim import TASK_TYPES

TOKENS_PER_COST = 2500.0
CALL_OVERHEAD = 0.02


def load_cache(path):
    return [json.loads(l) for l in open(path)]


def split_tasks(records):
    dev, test, seen = [], [], {}
    for r in records:
        k = seen.get(r["type"], 0)
        seen[r["type"]] = k + 1
        (dev if k % 2 == 0 else test).append(r)
    return dev, test


def _cost(tin, tout):
    return (tin + tout) / TOKENS_PER_COST + CALL_OVERHEAD


class TraceEnv:
    def __init__(self, records, config=None):
        self.records = records
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        c = self.cfg
        self.task = self.records[int(seed) % len(self.records)]
        self.task_type = self.task["type"]
        self.budget = float(rng.uniform(*c["budget_range"]))
        self.order = {lv: list(rng.permutation(len(rows))) for lv, rows in self.task["levels"].items()}
        self.n_res = 0
        self.has_candidate = False
        self.cand = None
        self.last_verify = 0
        self.summarized = False
        self.calls = {"researcher": 0, "solver": 0, "verifier": 0, "summarizer": 0}
        self.used = {lv: 0 for lv in self.order}
        self.spent, self.steps, self.done = 0.0, 0, False
        return self._observe()

    def _level(self):
        cap = 3 if self.task_type == "research" else 1
        return str(min(self.n_res, cap))

    def action_mask(self):
        m = np.ones(len(ACTIONS), dtype=bool)
        m[A["INVOKE_SOLVER"]] = not self.has_candidate
        m[A["INVOKE_VERIFIER"]] = self.has_candidate
        m[A["INVOKE_SUMMARIZER"]] = self.has_candidate and not self.summarized
        m[A["RETRY_SOLVER"]] = self.has_candidate
        return m

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over, call reset()")
        if not self.action_mask()[action]:
            raise ValueError(f"illegal action {ACTIONS[action]}")
        name, c, cost = ACTIONS[action], self.cfg, 0.0
        if name == "INVOKE_RESEARCHER":
            note = self.task["notes"][min(self.calls["researcher"], len(self.task["notes"]) - 1)]
            cost = _cost(note["tin"], note["tout"])
            self.calls["researcher"] += 1
            self.n_res += 1
        elif name in ("INVOKE_SOLVER", "RETRY_SOLVER"):
            lv = self._level()
            i = self.order[lv][self.used[lv] % len(self.order[lv])]
            self.used[lv] += 1
            self.cand = self.task["levels"][lv][i]
            cost = _cost(self.cand["tin"], self.cand["tout"])
            self.calls["solver"] += 1
            self.has_candidate, self.last_verify, self.summarized = True, 0, False
        elif name == "INVOKE_VERIFIER":
            cost = _cost(self.cand["v_tin"], self.cand["v_tout"])
            self.calls["verifier"] += 1
            self.last_verify = 1 if self.cand["v_pass"] else 2
        elif name == "INVOKE_SUMMARIZER":
            cost = CALL_OVERHEAD
            self.calls["summarizer"] += 1
            self.summarized = True
        self.spent += cost
        self.steps += 1
        reward = -c["lambda_cost"] * cost / c["reference_cost"]
        stopped = name == "STOP"
        self.done = stopped or self.spent >= self.budget or self.steps >= c["max_steps"]
        info = {"cost": cost, "forced_end": self.done and not stopped}
        if self.done:
            ok = self._correct()
            reward += -c["no_candidate_penalty"] if not self.has_candidate else float(ok)
            info.update(success=bool(ok), total_cost=self.spent, steps=self.steps,
                        calls=dict(self.calls), task_type=self.task_type, task_id=self.task["id"])
        return self._observe(), reward, self.done, info

    def _correct(self):
        if not self.has_candidate:
            return False
        return self.cand["ok_final"] if self.summarized else self.cand["ok_raw"]

    def _observe(self):
        c = self.cfg
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[TASK_TYPES.index(self.task_type)] = 1.0
        o[4] = float(self.has_candidate)
        o[5 + self.last_verify] = 1.0
        for i, k in enumerate(["researcher", "solver", "verifier", "summarizer"]):
            o[8 + i] = self.calls[k] / c["max_steps"]
        o[12] = max(0.0, (self.budget - self.spent) / self.budget)
        o[13] = 1.0 - self.steps / c["max_steps"]
        o[15] = float(self.summarized)
        return o
