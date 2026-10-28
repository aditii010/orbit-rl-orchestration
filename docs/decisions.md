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

## 2026-10-27 Plan B step 1 result: held-out budgets and task mixes (log: docs/logs/shift.log)
Return, 5,000 tasks per world. DQN = 5 S2 seeds trained once on the default world; "DQN-rule" is
the seed-AVERAGED per-task return minus the fixed rule (its CI hides seed-to-seed spread).
  world            single  rule(fixed) rule(retuned) DQN mean (min)   DQN-rule
  train world      +0.153  +0.166      +0.167        +0.187 (+0.181)  +0.020 +-0.008
  tight budget     +0.153  +0.172      +0.172        +0.167 (+0.161)  -0.005 +-0.008
  loose budget     +0.153  +0.166      +0.167        +0.181 (+0.170)  +0.016 +-0.008
  research-heavy   +0.067  +0.029      +0.072        +0.077 (+0.072)  +0.048 +-0.009
  harder tasks     +0.088  +0.085      +0.093        +0.103 (+0.090)  +0.018 +-0.008
  easier tasks     +0.213  +0.246      +0.241        +0.263 (+0.257)  +0.016 +-0.009
- The fixed rule BREAKS under research-heavy (+0.029, below single agent +0.067). The DQN does not (+0.077)
  and lands on the same level as a rule a human retuned for that world (+0.072).
- DQN >= retuned rule in 5 of 6 worlds; it loses only under TIGHT budget (+0.167 vs +0.172, a tie within noise).
- Caveats: retuned rule used only 2,000 dev tasks and 960 settings, so it is itself a bit noisy;
  the retuned column has no CI. Tight budget (0.3-0.6) partly overlaps the train range (0.5-1.2).
- H4 (generalization): mostly supported. The DQN keeps its edge over the fixed rule in 5 of 6 worlds
  and matches human retuning without any retuning. Biggest value is robustness, not peak reward.

## 2026-10-27 Plan B step 2 pre-declared (BEFORE running): ablations (eval/ablate.py)
- Full DQN = the 5 S2 checkpoints from eval/shift.py. Each variant: retrain 5 seeds with S2 settings.
  A no cost penalty (train with lambda=0, evaluate with the TRUE reward), B no history (zero last-verify,
  call counts, summarized flag), C1 no difficulty estimate, C2 no confidence score.
- Tested in the train world and the research-heavy world, 10,000 tasks, paired vs the full DQN.
- Predictions written first: C1 (difficulty) hurts most, because the DQN's "give up on hard tasks"
  pattern needs it. A makes the agent spend more (higher cost, more steps) and lowers true return.
  B hurts a little. C2 hurts little. If wrong, we say so.

## 2026-10-27 Plan B step 2 result: ablations (log: docs/logs/ablate.log), 10,000 paired tasks
Full DQN: train world +0.189, research-heavy +0.076. Change vs full (5-seed average):
  A no cost penalty  train -0.312, research-heavy -0.406  (calls 7.2 per task, cost 0.54, success 0.57)
  B no history       train -0.0205, research-heavy -0.004 (calls 2.4: stops verifying)
  C1 no difficulty   train -0.0104, research-heavy -0.0084
  C2 no confidence   train -0.0075, research-heavy -0.0065
- Predictions check: A as predicted (very strong). "C1 hurts most among features": TRUE only in
  research-heavy; in the train world B (history) hurt most, about 2x C1. "B hurts a little": wrong in
  train world. C2 small: roughly right.
- Every state-feature ablation still lands at or above the fixed rule (+0.173 train world), so the DQN's
  small edge is spread over several overlapping features; removing one leaves another that carries
  similar information (difficulty and confidence both signal "probably wrong").
- Without verify results (B) the agent stops verifying and drops to single-agent-like behaviour: the
  value of verification depends entirely on being able to read its result.
- Not tested: removing difficulty AND confidence together. Would show if the two are redundant.

## 2026-10-28 Real-LLM step 1: task set built (llm/tasks.py, tests/test_tasks.py)
- Qwen2.5-3b via Ollama works locally (~14 tok/s). In a smoke test it got a change-making problem wrong
  (said 21, correct 16), so it fails sometimes: good, orchestration has something to fix.
- 50 tasks: 20 coding (2 visible + 5-6 hidden asserts each, reference solution for each),
  15 analytical (exact number), 15 research (invented facts, 16 documents, keyword retrieval; 8 of 15
  questions need 2-3 documents, so a second researcher call can matter).
- Grader tests (48): every reference solution passes; a wrong function and a non-Python answer fail;
  an infinite loop is cut off at 10 s; every research answer is derivable from its gold documents;
  extra retrieval calls reach more gold docs.
- Safety note: model-written code is run in a subprocess (python -I, temp dir, 10 s timeout). No sandbox beyond
  that, fine for small functions on a personal machine; do not run on untrusted input at scale.

## 2026-10-28 Real-LLM step 2: client, agents, pilot on qwen2.5:3b (docs/logs/pilot_qwen3b.log)
Pilot (12 tasks x 3 samples). FIRST run had a design flaw, fixed before any real result was used:
- Research tasks scored 0/9. Cause: the 2nd researcher call retrieved irrelevant passages and the LLM extraction
  wrote "None of the provided passages mention ...", which was pasted into the evidence; solver then said "unknown".
  An unnecessary extra agent call poisoned the answer (a real instance of the failure mode in the project doc).
  Fix: researcher on research tasks is pure retrieval (no LLM call, evidence = the passages). The log file keeps the
  flawed first run for the record.
- After the fix: single-hop questions are answered correctly; 2-3 hop questions still return "unknown" even when
  all gold documents are in the evidence (qwen 3B cannot chain facts). So for this model research success has a
  ceiling near the 6 of 15 single-hop questions. A stronger model (Groq gpt-oss-20b) should chain them.
- Coding: solver correct 0.80 (4 tasks always right, 1 always wrong), verifier (visible tests) agrees with truth 0.93.
  Analytical: solver 0.92. So qwen is already strong on 2 of 3 task types: little room for orchestration there.
- CONFIDENCE is useless: mean 0.98 when correct vs 1.00 when wrong (and missing in 2 of 36). Cannot feed the
  controller. The simulator's confidence (gap ~0.2) is NOT realistic for this model.
- Speed: 2-3 s per coding call, 8-9 s per analytical call. Full cache is feasible.
- Consequence: the real controller cannot use observation 14 (confidence) and has no non-oracle difficulty
  estimate either (3). Plan: test "hide both" in the simulator first, then use that variant (and a rule without
  those features) for the real traces.

## 2026-10-28 PRE-DECLARED (before running): C3 "blind" test = hide difficulty (3) AND confidence (14) together
Why: real LLMs give us no usable confidence and no non-oracle difficulty, so the realistic controller is blind to both.
- DQN-blind: S2 settings, 5 seeds, hide=(3, 14), checkpoints saved to checkpoints/blind_seed{0..4}.pt.
- Blind rule: grid of 256 rules (research_on in {(), (2,), (1,2), (0,1,2)}, verify in {T,F}, max_retries 0..3,
  min_money_for_retry in {.05,.1,.15,.25}, summarize in {T,F}); no giveup_diff, no confidence. The policy wrapper
  also zeroes obs[3], obs[14] so it cannot cheat. Chosen on DEV (seeds 0-4999) only, reported on TEST (100000+).
- My prediction: the blind DQN loses about half of its 9% edge over the full-information rule, but still beats
  the blind rule by a few percent (history features carry most of the value). 
- Decision rule: "DQN keeps its edge" if blind-DQN mean TEST return > blind rule by >= 3% and at least 4 of 5 seeds
  are above the rule beyond the 95% CI. Otherwise report a tie (or a loss) as it is.

## 2026-10-28 RESULT of pre-declared C3 blind test (docs/logs/ablate_blind.log)
- Blind rule (chosen on DEV): single agent in disguise (no research, no check, no retry, no summarize): TEST +0.1528.
- Blind DQN, 5 seeds: +0.171, +0.184, +0.186, +0.173, +0.170; mean +0.1768. Every seed beats the blind rule beyond the 95% CI
  (+0.017 to +0.033). Gain +15.7%. Pre-declared bar (>=3%, >=4 of 5 seeds) PASSED.
- My prediction was wrong in direction of size: I expected the DQN to lose about half its edge; it lost only 0.012
  (+0.189 -> +0.177), while the best RULE lost 0.020 (+0.173 -> +0.153). So blindness hurts the rules more than the DQN.
- Caveat: the blind rule family is small (no way to use "how many calls so far" cleverly except max_retries), so part of
  the gap may be a weak baseline. Honest reading: with no difficulty or confidence, a learned policy still beats every
  hand rule we tried by ~16%, and beats the single agent by the same margin.
- Checkpoints: checkpoints/blind_seed{0..4}.pt (these are what we run zero-shot on real traces).

## 2026-10-29 PRE-DECLARED (before running): real-LLM comparison on the qwen2.5:3b trace cache (eval/real_eval.py)
Cache is complete: 50 tasks x (4 samples per evidence level). Facts seen BEFORE writing this (cache summary): coding 68% -> 79%
with the plan; analytical 97%; research 0% with no researcher call, 40% after one, 43% after 2-3; verifier agrees with truth
70-92%. (I looked at these aggregates, so the predictions below are informed by them. I have NOT run any policy on the cache.)
Setup (llm/trace_env.py):
- Replay env, same 16-number observation, difficulty (3) and confidence (14) always 0. Outcomes are recorded model outputs.
- Cost = (tokens in + out)/2500 + 0.02 per call. 2500 is chosen so an average solver call (~300 tokens) costs ~0.12 like
  the simulator. This choice fixes what lambda=0.5 means; with another scale the optimum would change. Reported as a limitation.
- Reward: 1 if the final answer is correct (summarized or raw text graded as it is, no 0.9 factor), -0.3 if none, minus
  0.5*cost/0.4. Retry = next unused recorded sample (feedback effect not recorded). Verifier verdict is fixed per sample.
- Split: tasks stratified by type, even position -> dev (26 tasks), odd -> test (24 tasks). Tune the rule (256-rule grid) on dev
  only; refit the DQN (20k episodes, 5 seeds, from scratch, hide 3 and 14, no checkpoint picking) on dev tasks only.
  Zero-shot DQN = the 5 sim-trained blind checkpoints (checkpoints/blind_seed*.pt), nothing changed.
- Unit of analysis is the TASK (24 test tasks, 100 episodes each averaged): CI = 2.07*sd/sqrt(24) of per-task paired differences.
  Episodes of one task are not independent, so per-episode CIs would be too narrow.
Predictions (to be scored against the result):
 P1 fixed pipeline beats single agent overall (research 0% -> ~40% outweighs its extra cost).
 P2 zero-shot DQN is about equal to the single agent or slightly above it, and clearly below the tuned rule; it will rarely call the
    researcher on research tasks (sim taught it the researcher is rarely useful), so its research success stays near 0.
 P3 tuned rule has research_on containing 2, and probably no summarizer (summarizer adds cost, the real raw text already grades fine).
 P4 refit DQN lands within error of the tuned rule (26 training tasks is tiny); I do NOT expect a clear win.
Claim rule: "method X beats Y" only if the paired per-task CI excludes 0; for DQN rows also >=4 of 5 seeds above Y's mean.

## 2026-10-29 RESULT: real qwen2.5:3b traces (docs/logs/real_eval_qwen3b.log), 24 test tasks, task-level paired CIs
Return (success, cost): single +0.462 (.59, .106) | fixed pipeline +0.336 (.75, .334) | tuned rule +0.569 (.72, .120) |
DQN zero-shot +0.438 (.64, .162) | DQN refit +0.580 (.77, .154).
Tuned rule (chosen on dev): researcher on research tasks, one solver call, no check, no retry, no summarizer.
Paired differences: tuned rule - fixed pipeline +0.233 ± 0.134 (only clear result); tuned rule - single +0.107 ± 0.136;
zero-shot DQN - single -0.024 ± 0.055; zero-shot DQN - tuned rule -0.132 ± 0.144; refit DQN - tuned rule +0.011 ± 0.044;
refit DQN - single +0.118 ± 0.137.
Scoring my predictions:
 P1 WRONG: fixed pipeline did NOT beat the single agent (-0.126, CI includes 0). It gets 75% success but pays 3x the cost
    (verifier on analytical costs as much as the solver, and the summarizer/researcher add calls on tasks that did not need them).
 P2 mostly RIGHT: zero-shot DQN ~ single agent; research success 0.00 (never calls the researcher: 0.00 calls). It does retry/verify
    coding (success .86 vs .74) but pays for it. "Clearly below the tuned rule": point estimate yes, but CI includes 0, so NOT claimable.
 P3 RIGHT (research_on has 2, no summarizer), plus no verify and no retry at all.
 P4 RIGHT: refit DQN equals the tuned rule within error (+0.011 ± 0.044). No win for RL over a tuned rule on these traces.
Honest reading:
 - The sim-to-real gap is real and visible: the sim-trained controller skips the researcher, so it scores 0% on research tasks.
   After refitting on real traces it calls the researcher (1.40 per research episode) and fixes it (success .46).
 - Refit DQN beats the tuned rule on coding only slightly (verifies coding with the cheap test run: success .86 vs .74), but my rule
   grid cannot express "verify only coding", so the rule is a bit handicapped. A type-specific rule would probably close that gap.
   NOT tested yet; if tested it must be pre-declared (tuned on dev only), because I now know this from the test result.
 - Everything is based on 24 test tasks, one 3B model, 4 samples per setting; most CIs are wide. Treat as evidence of direction only.
 - Headline for the report: RL is clearly better than a naive fixed pipeline and robust to blindness in the simulator, but on real
   traces from this small model a tuned rule is as good; the DQN's value is that it finds that rule automatically (refit),
   while the zero-shot transfer fails on research tasks.

## 2026-10-29 PRE-DECLARED: stronger-model check, Groq openai/gpt-oss-20b (reasoning_effort=low, max_tokens>=1500)
Test call: ~70-80 output tokens, 0.7 s, so cheap. Why this run: qwen 3B cannot chain 2-3 hop research questions; a stronger model might,
which would make the researcher and verifier matter more. Same pipeline, same trace-cache format, same eval script (cache path argument).
- Request cap raised from 600 to 650 (design said 600; the planned cache needs about 605). Samples per level fewer than qwen's 4:
  coding 3+3, analytical 3+2, research 2+3+3+2 (levels 0..3). The replay env handles uneven counts (wraps when exhausted).
- Output stays on disk in runs/cache_gptoss20b.jsonl (text only). Key is read from .env, never printed or logged.
- Predictions: (a) single agent is already strong on coding/analytical (>=90%), so little orchestration room there;
  (b) with 2-3 researcher calls, multi-hop research is answered mostly correctly (>=60% at level 3), 0% at level 0;
  (c) tuned rule again = researcher on research tasks, solver once; (d) zero-shot DQN again loses on research (never calls researcher);
  (e) refit DQN ties the tuned rule. If instead everything is ~100% except research, orchestration gain is small and I will say so.

## 2026-10-29 RESULT: gpt-oss-20b traces (docs/logs/real_eval_gptoss20b.log), 24 test tasks
Return (success, cost): single +0.563 (.71,.116) | fixed pipeline +0.282 (.75,.374) | tuned rule +0.670 (.83,.131) |
DQN zero-shot +0.530 (.71,.143) | DQN refit +0.770 (.96,.153; per seed .656 .791 .800 .793 .812).
Paired: tuned rule - fixed +0.388 ± 0.140 (clear); fixed - single -0.281 ± 0.217; zero-shot - single -0.033 ± 0.009 (CLEAR: zero-shot
is worse than doing nothing clever); zero-shot - tuned rule -0.140 ± 0.132; refit - tuned rule +0.100 ± 0.111 (not clear);
refit - single +0.207 ± 0.142 (just clear).
Predictions: (a) right (coding 100%, analytical 100%); (b) right (research 0% at level 0, 93% at level 3); (c) right;
(d) right (zero-shot never calls the researcher, research success 0.00); (e) tie with tuned rule: point estimate +0.10 but CI includes 0.
What drives refit's +0.10: it calls the researcher 3.33 times on research tasks (success .87) while the rule's grid can only call it ONCE
(success .43). So part of the refit "advantage" is a limit of my rule family, not intelligence. Same flag as for qwen (rule cannot be
type-specific about verify).
Next (pre-declared below): a fairer, richer rule family tuned on dev, then compare with refit DQN on both caches.

## 2026-10-29 PRE-DECLARED: typed rule family (eval/real_eval_typed.py)
Rule options per task type: number of researcher calls before solving (research: 0-3; coding and analytical: 0-1), verify once
per candidate (T/F per type), global max retries 0-2 (retry only after a failed check), summarize T/F. 768 rules, chosen on dev tasks
only, evaluated on test tasks. The DQN refit is retrained exactly as before (same seeds, 20k episodes, dev tasks only).
Prediction: the typed rule closes the gap: refit DQN - typed rule has a CI that includes 0 on BOTH caches, i.e. the honest conclusion
is "the DQN finds a rule as good as the best hand-tuned typed rule, automatically". If the DQN is still clearly better (CI excludes 0)
or clearly worse, I report that instead. Zero-shot DQN is unchanged, no need to rerun.

## 2026-10-29 RESULT: typed rule vs refit DQN (docs/logs/typed_qwen3b.log, typed_gptoss20b.log)
qwen 3B: typed rule (chosen on dev: researcher once on research tasks, verify coding only, up to 2 retries, no summarizer) test +0.576;
  refit DQN +0.580; difference +0.005 ± 0.018 -> TIE, as predicted.
gpt-oss-20b: typed rule (researcher twice on research tasks, no verify, no retry, no summarizer) test +0.819 (success 1.00);
  refit DQN +0.770 (success .96; seeds .656 .791 .800 .793 .812); difference -0.048 ± 0.041 -> the hand-tuned typed rule is
  slightly but CLEARLY better. My prediction (CI includes 0 on both) was WRONG for gpt-oss.
Reading: once the rule family may be type-specific, the earlier "refit DQN +0.10 over the tuned rule" disappears (the rule family was
the handicap). On real traces the DQN at best ties the best hand rule; it did slightly worse on gpt-oss (seed 0 under-researches,
and the DQN calls the researcher 3.3 times where 2 is enough, paying extra cost). Caveat: tiny training set (26 tasks, 20k episodes).
Honest final claim for the report:
 1. Simulator: DQN > best rule (+9.3% full info, +15.7% blind), robust to task-mix shift; cost penalty essential.
 2. Real traces: orchestration matters (researcher calls turn 0% into 43-93% on research tasks; fixed pipeline wastes cost and
    loses), but a tuned typed rule is as good as or slightly better than the refit DQN. Zero-shot sim->real transfer fails
    (clearly below single agent on gpt-oss: -0.033 ± 0.009).
 3. The RL value is automatic discovery of the policy (refit) and the ability to handle a cost trade-off without hand-writing rules.
Not done: LLM-router baseline (counted-cost) - optional, only if time remains.
