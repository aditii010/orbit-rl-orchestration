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

## 2026-10-27 Milestone 4a / second feasibility test: Double DQN vs baselines (10,000 TEST tasks)
- Two setup mistakes of mine, found before claiming anything:
  1. The rule search could not skip the summarizer, so the "tuned" rule looked worse than single agent.
     Fixed by adding a summarize on/off option. Winner did not change.
  2. 300 rule settings tuned on 1,000 dev tasks gave a lucky winner (dev +0.177, test +0.137).
     Fixed by tuning on 5,000 dev tasks and testing on 10,000 fresh tasks.
- Final numbers (return, mean): single +0.153, rule router +0.154 (tie), fixed pipeline +0.003.
- DQN, 5 seeds, final checkpoint (no cherry-picking), 20k episodes, ~30 s each:
  +0.185, +0.183, +0.181, +0.181, +0.136. Mean +0.173 = +12.7% over best simple baseline.
- Paired on identical tasks: seeds 0-3 beat both rule and single by +0.03 (CI +-0.008).
  Seed 4 is WORSE by -0.017. Training is not yet stable across seeds.
- Cost: DQN gets success 0.47 at cost ~0.217 vs rule 0.49 at cost 0.271 (about 20% cheaper).
- Verdict: WEAK GO. Passes the pre-registered 10% rule on the mean, but the gain is small in
  absolute terms (about 0.03 reward) and one seed fails. Next: find out why seed 4 differs,
  and what the DQN actually does differently (action distribution).

## 2026-10-27 Milestone 4b: what does the DQN do? (eval/analyze.py, 2000 test tasks)
- Caveat on reading the table: each row counts every decision step, so "stop 50%" after PASS
  mostly means "summarize first, then stop on the next step". Not two competing choices.
- Both DQN seeds learned the skeleton: always solve first, stop/summarize after PASS, retry after FAIL.
- Rule router verifies 100% of low-confidence candidates. DQN seed 0 verifies only 27% there and
  instead retries (30%) or stops (36%). Hypothesis (NOT yet tested): when confidence already
  says "probably wrong", a check adds little, so skipping it saves cost. Test by splitting on
  estimated difficulty and money left.
- DQN seed 0 re-verifies after a FAIL (40%). A second noisy opinion can be useful, but costs money. Untested.
- DQN seed 4 wastes calls: researcher after a candidate already exists (16% at high confidence,
  12% after PASS). Research only helps FUTURE solver calls, so these are mostly useless. Likely
  the cause of its lower return. Not yet confirmed by an ablation.
- DQN choices look "mushy" (many actions per situation) because the greedy policy depends on
  other inputs (task type, difficulty, money) that the table averages over, not because it is random.

## 2026-10-27 Milestone 4c: hypothesis test "skip the check at low confidence" (eval/hypothesis.py)
- Part C: P(truly wrong | unchecked, confidence<0.5) = 0.92 (n=6214). Matches my rough math.
- Part B (same hand rule, only the low-confidence action changes, 10,000 paired tasks):
  verify +0.154 | retry at once -0.0004 +-0.0084 (tie) | stop +0.0075 +-0.0063 (not significant;
  success 0.49 -> 0.38, cost 0.271 -> 0.170).
- Verdict: hypothesis NOT supported. Skipping the check saves only ~0.005 cost; the verifier is cheap.
  Giving up at low confidence is about as good as retrying: the reward is FLAT in that region.
- Part A: DQN seeds 0 and 1 agree on one pattern: low confidence + HARD task + money left -> stop
  (56% and 61%); easy task -> retry. Giving up on hard tasks, retrying easy ones.
- Fairness problem found: my rule family NEVER reads estimated difficulty (obs[3]). The DQN's edge
  may come only from using a feature the rules ignore. Must add difficulty-aware rules and re-tune
  on DEV before claiming RL beats rules.

## 2026-10-27 Milestone 4d: FAIR re-test with a difficulty-aware rule (eval/compare2.py)
- Rule family extended: giveup_diff (on hard tasks skip checks and retries) and lowconf_retry.
  960 rules tuned on 5,000 DEV tasks. Winner: verify, 3 retries, trust conf>=0.6, give up when
  est. difficulty >= 0.6-0.7 (several near-ties at dev +0.186).
- TEST (10,000 tasks): single +0.153 | old rule +0.154 | NEW rule +0.173 | DQN mean +0.173
  (seeds +0.185 +0.183 +0.181 +0.181 +0.136).
- Paired DQN - NEW rule: +0.012, +0.009, +0.007, +0.008, -0.037 (CIs about +-0.007).
- Verdict: the pre-registered GO test now FAILS (gain 0.0% vs needed 10%). The earlier +12.7%
  came from the rule ignoring difficulty. RL independently discovered the same "give up on hard
  tasks" structure the better hand rule uses.
- Honest story for the report: RL matches a well-tuned expert rule without being told it, clearly
  beats a fixed pipeline and an undesigned rule, and is somewhat unstable across seeds.
  Four seeds are marginally above the rule (+0.01), one clearly below.

## 2026-10-27 Plan A pre-declared (BEFORE running): stabilize the DQN, short fixed budget
- S1: 60k episodes, lr 1e-3. S2: 60k, lr 3e-4, target sync 1000. S3: S2 + exploration decay to 80%.
- 5 seeds each, final checkpoint only, no seed picking. Choose the setting by DEV mean only.
- Success: chosen setting beats the difficulty-aware rule (+0.173 test) clearly beyond noise AND no
  seed clearly below it. Otherwise Plan A failed: go to Plan B (honest story) with the original setting.
- Whatever happens, Plan B follows (held-out budgets/tasks, ablations, real-LLM check).

## 2026-10-27 Plan A result (log: docs/logs/plan_a_tune_dqn.log)
- DEV means: S1 +0.181, S2 +0.203, S3 +0.198. Chosen by dev: S2 (60k episodes, lr 3e-4, target sync 1000).
- TEST (10,000 tasks), S2, final checkpoints: +0.183 +0.194 +0.195 +0.184 +0.191, mean +0.189.
  Difficulty-aware rule: +0.173. Paired DQN - rule: +0.009 to +0.022, every seed above the rule beyond its CI.
- Seed instability is gone: worst seed is now above the rule (before: -0.037).
- Plan A success criterion (beats rule beyond noise, no seed below): MET.
  Original feasibility bar (+10% relative): +9.3%, just under. Report both, do not round up.
- Caveats: rule family is limited (960 hand-built variants); the DQN had 3x more training than before;
  S2 was picked from 3 settings using dev only. rl/train.py defaults now equal S2.
- Plan B starts: held-out budgets / task families, ablations, real-LLM check.

## 2026-10-27 Plan B step 1 pre-declared (BEFORE running): held-out budgets and task mixes (eval/shift.py)
- Train the 5 S2 DQNs and tune the rule ONCE on the default world. Test in 6 worlds (5,000 tasks each):
  train world, tight budget (0.3-0.6), loose budget (1.2-2.0), research-heavy mix (10/20/70%),
  harder tasks (Beta 3,2), easier tasks (Beta 2,3).
- Reported per world: single agent, fixed rule, RETUNED rule (reference: human who knew the shift),
  DQN mean/min over seeds, paired DQN minus fixed rule.
- Question: does the DQN's small edge survive a shift, and does it lose more or less than the fixed rule?
- Not allowed afterwards: changing the shifts, dropping a world, or tuning anything on these test tasks.
