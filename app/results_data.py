SIM = [("Fixed pipeline (always use every helper)", 0.0034, False),
       ("Single agent", 0.1528, False),
       ("Best hand-written rule (sees difficulty and confidence)", 0.1732, False),
       ("DQN (sees difficulty and confidence)", 0.189, True)]
SIM_BLIND = [("Best hand-written rule (blind)", 0.1528, False), ("DQN (blind)", 0.1768, True)]

SHIFT = [("Training world", 0.153, 0.166, 0.187), ("Tight budget", 0.153, 0.172, 0.167), ("Loose budget", 0.153, 0.166, 0.181),
         ("Research-heavy tasks", 0.067, 0.029, 0.077), ("Harder tasks", 0.088, 0.085, 0.103), ("Easier tasks", 0.213, 0.246, 0.263)]

ABLATE = [("No cost penalty during training", -0.3123), ("No memory of what happened so far", -0.0205),
          ("No difficulty estimate", -0.0104), ("No confidence score", -0.0075), ("No difficulty and no confidence", -0.0122)]

REAL = {"qwen2.5:3b (local)": (0.462, 0.336, 0.569, 0.580), "gpt-oss-20b (Groq)": (0.563, 0.282, 0.670, 0.770)}

SOURCES = [("docs/logs/sim_baselines_test.log", s) for s in ("+0.0034", "+0.1528", "+0.1732")] + [
    ("docs/logs/plan_a_tune_dqn.log", "+0.189"), ("docs/logs/ablate_blind.log", "+0.1768"), ("docs/logs/ablate_blind.log", "+0.1528"),
    ("docs/logs/shift.log", "+0.263"), ("docs/logs/shift.log", "+0.029"), ("docs/logs/shift.log", "+0.077"),
    ("docs/logs/ablate.log", "-0.3123"), ("docs/logs/ablate.log", "-0.0205"), ("docs/logs/ablate.log", "-0.0104"),
    ("docs/logs/ablate.log", "-0.0075"),
    ("docs/logs/real_eval_qwen3b.log", "+0.462"), ("docs/logs/real_eval_qwen3b.log", "+0.336"),
    ("docs/logs/real_eval_qwen3b.log", "+0.569"), ("docs/logs/real_eval_qwen3b.log", "+0.580"),
    ("docs/logs/real_eval_gptoss20b.log", "+0.563"), ("docs/logs/real_eval_gptoss20b.log", "+0.282"),
    ("docs/logs/real_eval_gptoss20b.log", "+0.670"), ("docs/logs/real_eval_gptoss20b.log", "+0.770"),
    ("docs/logs/typed_gptoss20b.log", "+0.819"), ("docs/logs/typed_qwen3b.log", "+0.576")]
