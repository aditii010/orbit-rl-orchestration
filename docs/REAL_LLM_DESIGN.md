# Real-LLM experiment design (written before any real-LLM result)

## Models
- Main: local Ollama qwen2.5:3b. Weak enough to fail often, so orchestration has room to matter. Free, no rate limit.
- Second: Groq openai/gpt-oss-20b (reasoning model, ~90% of tokens are hidden reasoning, so set max_tokens high).
  Limits seen: 8000 tokens/min, 1000 requests/day. Used as a stronger-model check (Risk 5 in the project doc).
- Not used: OpenAI key (risk, cost). Groq key lives only in .env.

## Tasks (50, all with checkable answers)
- 20 coding: small functions, HIDDEN unit tests decide success. Visible tests are what the verifier may run.
- 15 analytical: generated word problems with an exact numeric answer.
- 15 research: invented facts about fictional entities in a small local document set (model cannot know them
  from memory), keyword retrieval, exact-answer check.

## Agents (same 4 as the simulator)
Researcher = retrieval + one LLM call to extract evidence. Solver = LLM call. Verifier = visible tests (coding),
LLM check (others). Summarizer = LLM formatting call. Cost = real tokens converted with a fixed price table.

## Trace cache (makes the comparison fair and cheap)
For each task pre-sample N=4 solver outputs and the verifier verdict for each. Every policy replays the SAME cache,
so differences come from the policy, not from LLM luck. Simplification to report: a cached retry does not see the
verifier feedback that a live retry would.

## Controller inputs from real runs
task type (known), confidence (LLM self-report, checked for calibration before use), difficulty (open question:
must be a cheap predictor, NOT the ground truth), counts/budget (exact). Sim-to-real gap is reported, not hidden.

## Policies compared
single agent, fixed pipeline, tuned rule, DQN trained in simulation (zero-shot), then DQN re-fit on real traces
as a SEPARATE experiment. Spending cap on Groq: 600 requests total.
