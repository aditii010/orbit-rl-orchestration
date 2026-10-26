# Decisions log

## 2026-10-26 Setup
- Own Double DQN in PyTorch (no Stable-Baselines3), plain Python orchestration (no LangGraph).
- Simulator first, small real-LLM check later, Streamlit dashboard last.
- 3 ablations only: no cost penalty, no history, no uncertainty features.

## 2026-10-26 Feasibility test (experiments/feasibility.py, round2.py)
Question: is there room for a learned policy over hand-made rules?
Rule fixed in advance: GO if exact optimal reward >= 1.10 x best hand rule.
- Round 1 (weak, pricey verifier): best rule within 0 to 1.3% of optimal. No-go.
- Round 2 (sharper verifier, varying budget): within 0 to 1.2%. No-go.
- Why: in a simple world, "solve, verify, stop on pass, retry on fail" is already near optimal.
- Fixed pipeline lost to the single agent in most settings, so beating it proves little.
- Next: design a richer world (continuous uncertainty, task-specific costs and verifier
  reliability, correlated errors) BEFORE testing it, and test once. If the gap is still tiny,
  report "RL matches the expert rule, beats fixed pipelines".

## 2026-10-26 Milestone 2/3: simulator and baselines (first numbers, 2000 TEST tasks)
- single agent: return +0.152, success 0.34, cost 0.123
- fixed pipeline: return -0.012, success 0.36, cost 0.294
- rule router (grid-searched on DEV seeds 0-999): return +0.137, success 0.48, cost 0.273
- Finding: the tuned rule has much higher success but the SAME return as the single agent.
  With lambda=0.5 and reference_cost=0.4, extra calls cost about as much as the success they buy.
  Dev return 0.177 vs test 0.137 shows noise of this size, so the two are statistically tied.
- Not changing the world yet. Next: second feasibility test (DQN vs best rule) on this exact setup.
  Any change to lambda/skills must be decided and logged BEFORE looking at RL results.
