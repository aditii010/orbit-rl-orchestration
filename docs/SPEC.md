# ORBIT spec (frozen before the second feasibility test)

## Task types
coding, analytical, research. Each has its own agent costs and verifier reliability.
Coding verifier (unit tests): very reliable. Research verifier: noisy.

## Actions (6)
INVOKE_RESEARCHER, INVOKE_SOLVER, INVOKE_VERIFIER, INVOKE_SUMMARIZER, RETRY_SOLVER, STOP.
Masking: verifier and retry need a candidate. Summarizer needs a candidate. Anything is
masked once the budget is spent. STOP is always allowed.

## State (7 groups, all kept)
1. task type (one-hot)
2. estimated difficulty (noisy, 0 to 1)
3. has candidate (0/1)
4. last verification result (none / pass / fail)
5. calls per agent (4 counts)
6. budget left (money and steps, normalized)
7. confidence in current candidate (0 to 1, noisy, not trusted blindly)

## Reward
R_terminal = 1.0 * correct  -  lambda * (total_cost / reference_cost)  -  0.3 * (stopped with no candidate)
Step reward: only the cost of the action taken, -lambda * (step_cost / reference_cost).
No bonus for "more text". Quality counts only at the end (hidden ground truth).
lambda is chosen so a typical successful episode still has clearly positive reward.

## Reward loophole tests (must pass before training)
1. Stop immediately must score below a sensible solve-then-stop policy.
2. A useless repeat call must score below stopping.
3. Correct answer found by verification must beat the same answer with extra wasted calls.
4. Budget exhausted without a candidate must score worst.

## Evaluation rule set in advance
- Methods: single-agent, fixed pipeline, rule-based, bandit, Double DQN.
- Metrics: success rate, cost per task, cost per successful task, calls per task.
- Several seeds, report mean and confidence interval.
- Train and test use different task parameters (held-out families and budgets).
- Second feasibility test: GO if exact/best-found policy beats best hand rule by 10% or more.
  If not, report honestly: RL matches the expert rule and beats fixed pipelines.
