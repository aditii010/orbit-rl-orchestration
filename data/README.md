# ORBIT dataset

Two parts: a 50-task benchmark with automatic graders, and recorded runs of two real LLMs on every task.

## 1. Task set

[`orbit_tasks.csv`](orbit_tasks.csv): 50 tasks, one per row.

| Type | Dev | Test | Total | How it is graded |
|---|---:|---:|---:|---|
| Coding | 10 | 10 | 20 | Hidden unit tests run on the generated function |
| Analytical | 8 | 7 | 15 | Exact numeric answer (tolerance 0.011) |
| Research | 8 | 7 | 15 | Accepted answer strings, found in the document set |
| **Total** | **26** | **24** | **50** | |

The dev split is used to tune the rules and refit the controller. The test split is held out for the final comparison.

| Column | Meaning |
|---|---|
| `id` | Task id (`code00`, `ana05`, `res10`, ...) |
| `type` | `coding`, `analytical` or `research` |
| `split` | `dev` or `test` |
| `question` | Task text given to the solver |
| `answer` | Ground truth (analytical: number; research: accepted strings separated by `\|`) |
| `visible_tests` | Coding only: tests the verifier agent is allowed to run |
| `hidden_tests` | Coding only: tests that decide correctness, never shown to any agent |
| `reference_solution` | Coding only: a correct implementation |
| `evidence_docs` | Research only: `doc_id`s that contain the answer |

[`research_documents.csv`](research_documents.csv): the 16 short documents the researcher agent retrieves from.
The people, companies and products in them are invented, so a model cannot answer from memory and has to look the facts up.

## 2. Recorded LLM traces

| File | Model | Tasks | Solver samples per task and evidence level |
|---|---|---:|---:|
| [`../runs/cache_qwen3b.jsonl`](../runs/cache_qwen3b.jsonl) | qwen2.5:3b (local, Ollama) | 50 | 4 |
| [`../runs/cache_gptoss20b.jsonl`](../runs/cache_gptoss20b.jsonl) | gpt-oss-20b (Groq) | 50 | 3 |

One JSON object per task:

| Field | Meaning |
|---|---|
| `notes` | Researcher outputs, one per researcher call, with retrieved `docs` and token counts |
| `levels` | Solver samples keyed by how many researcher calls came before (`"0"` = no evidence) |
| `text`, `conf` | Solver answer and its self-reported confidence |
| `tin`, `tout`, `sec` | Input tokens, output tokens, seconds |
| `ok_raw`, `ok_final` | Graded correct before and after the summarizer |
| `v_pass`, `v_tin`, `v_tout`, `v_sec` | Verifier verdict and its cost |

Every policy replays the same recorded samples, so differences in score come from the policy and not from sampling luck.
The tasks and graders are defined in [`../llm/tasks.py`](../llm/tasks.py), and the traces are rebuilt with
`python3 -m llm.build_cache <backend> <model> <output.jsonl>`.
