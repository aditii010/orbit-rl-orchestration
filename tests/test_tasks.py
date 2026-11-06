import pytest

from llm.tasks import DOCS, all_tasks, grade, retrieve

TASKS = all_tasks()


def test_counts():
    kinds = [t["type"] for t in TASKS]
    assert (kinds.count("coding"), kinds.count("analytical"), kinds.count("research")) == (20, 15, 15)


@pytest.mark.parametrize("t", [t for t in TASKS if t["type"] == "coding"], ids=lambda t: t["id"])
def test_reference_solution_passes_all_tests(t):
    assert grade(t, t["ref"])


@pytest.mark.parametrize("t", [t for t in TASKS if t["type"] == "coding"], ids=lambda t: t["id"])
def test_wrong_code_fails(t):
    assert not grade(t, f"def {t['fn']}(*a, **k):\n    return 0")
    assert not grade(t, "this is not python at all (")


def test_infinite_loop_is_cut_off():
    t = next(t for t in TASKS if t["type"] == "coding")
    assert not grade(t, f"def {t['fn']}(*a):\n    while True: pass")


def test_hidden_tests_are_stricter_than_visible():
    from llm.tasks import run_asserts
    caught = 0
    for t in TASKS:
        if t["type"] != "coding":
            continue
        cheat = f"def {t['fn']}(*args):\n    return None"
        ok, n = run_asserts(cheat, t["visible"] + t["hidden"])
        caught += ok < n
    assert caught == 20


def test_analytical_grading():
    t = next(t for t in TASKS if t["id"] == "ana00")
    assert grade(t, "The change is 16 rupees.") and not grade(t, "21")
    t = next(t for t in TASKS if t["id"] == "ana01")
    assert grade(t, "1890") and grade(t, "Final price: 1,890.00")


def test_research_grading_and_forbid():
    t = next(t for t in TASKS if t["id"] == "res11")
    assert grade(t, "Ardent") and not grade(t, "Ardent, not the Brannock") and not grade(t, "Brannock")
    t = next(t for t in TASKS if t["id"] == "res02")
    assert grade(t, "212,000") and not grade(t, "about 200,000")


def test_every_answer_is_in_the_documents():
    for t in TASKS:
        if t["type"] != "research":
            continue
        text = " ".join(DOCS[i] for i in t["gold"]).lower().replace(",", "")
        assert all(a in text for a in t["accept"]), t["id"]


def test_first_retrieval_finds_gold_for_single_hop_questions():
    for t in TASKS:
        if t["type"] == "research" and len(t["gold"]) == 1:
            assert t["gold"][0] in retrieve(t["question"]), t["id"]


def test_extra_research_calls_reach_more_gold_docs():
    one, two = 0, 0
    multi = [t for t in TASKS if t["type"] == "research" and len(t["gold"]) > 1]
    for t in multi:
        seen = retrieve(t["question"])
        one += set(t["gold"]) <= set(seen)
        q2 = t["question"] + " " + " ".join(DOCS[i] for i in seen)
        seen2 = seen + retrieve(q2, exclude=seen)
        two += set(t["gold"]) <= set(seen2)
    assert two >= one and two >= len(multi) // 2
