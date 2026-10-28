"""The four specialist agents on top of a real LLM. Each returns a dict with the output and the call's usage."""
import re

from llm.tasks import DOCS, extract_code, retrieve, run_asserts

CONF_LINE = "\nAfter your answer, write one last line: CONFIDENCE: <0-100> (how sure you are it is correct)."


def _usage(r):
    return {"tin": r["tin"], "tout": r["tout"], "seconds": r["seconds"]}


def parse_confidence(text):
    m = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
    return min(100, int(m.group(1))) / 100 if m else None


def strip_confidence(text):
    return re.sub(r"\n?\s*CONFIDENCE:.*$", "", text, flags=re.S).strip()


def final_answer(task, raw):
    """Summarizer: deterministic formatter. Coding -> code only; analytical -> last number; research -> text."""
    raw = strip_confidence(raw)
    if task["type"] == "coding":
        return "```python\n" + extract_code(raw) + "\n```"
    if task["type"] == "analytical":
        m = re.findall(r"-?\d[\d,]*\.?\d*", raw.replace(",", ""))
        return m[-1] if m else raw
    return raw


def researcher(llm, task, seen_docs=()):
    """Research tasks: retrieve new passages (query grows with what was found), no LLM call.
    Others: ask for a short plan and edge cases. Returns (evidence_text, doc_ids, usage)."""
    if task["type"] == "research":
        # Pure retrieval, no LLM call: the evidence is the passages themselves. (An earlier version let the model
        # summarise them; on irrelevant passages it wrote "none of these help", which poisoned the solver.)
        query = task["question"] + " " + " ".join(DOCS[i] for i in seen_docs)
        new = retrieve(query, exclude=seen_docs)
        return "\n".join(f"[{i}] {DOCS[i]}" for i in new), list(new), {"tin": 0, "tout": 0, "seconds": 0.0}
    what = task["spec"] if task["type"] == "coding" else task["question"]
    r = llm.chat(f"Task: {what}\nGive a short plan (3 bullet points) and list 2 tricky edge cases. Do not solve it.",
                 max_tokens=150, temperature=0.0)
    return r["text"].strip(), [], _usage(r)


def solver(llm, task, evidence="", feedback="", temperature=0.7):
    if task["type"] == "coding":
        shown = "\n".join(f"assert {a}" for a in task["visible"])
        p = (f"Write a Python function `{task['fn']}`. {task['spec']}\nExample tests:\n{shown}\n"
             "Reply with only the code in a ```python block." + CONF_LINE)
        mt = 400
    elif task["type"] == "analytical":
        p = f"{task['question']}\nThink step by step briefly, then write 'Answer: <number>'." + CONF_LINE
        mt = 300
    else:
        p = (f"Use ONLY these passages:\n{evidence or '(none)'}\n\nQuestion: {task['question']}\n"
             "If the passages are not enough, answer 'unknown'. Reply with a short answer." + CONF_LINE)
        mt = 120
    if evidence and task["type"] != "research":
        p += f"\nHelpful notes:\n{evidence}"
    if feedback:
        p += f"\nYour previous attempt was rejected: {feedback}\nTry again."
    r = llm.chat(p, max_tokens=mt, temperature=temperature)
    return {"text": r["text"], "confidence": parse_confidence(r["text"]), **_usage(r)}


def verifier(llm, task, candidate_text, evidence=""):
    """Coding: run the VISIBLE tests (deterministic). Others: an LLM check. Returns (passed, feedback, usage)."""
    cand = strip_confidence(candidate_text)
    if task["type"] == "coding":
        ok, n = run_asserts(extract_code(cand), task["visible"])
        return ok == n, f"{n - ok} of {n} example tests failed", {"tin": 0, "tout": 0, "seconds": 0.0}
    q = task["question"]
    ev = f"\nPassages:\n{evidence}" if task["type"] == "research" and evidence else ""
    r = llm.chat(f"Problem: {q}{ev}\nProposed answer: {cand}\n"
                 "Is the proposed answer correct? Check it carefully. Reply 'VERDICT: PASS' or 'VERDICT: FAIL', "
                 "then one short sentence.", max_tokens=150, temperature=0.0)
    t = r["text"].upper()
    passed = "VERDICT: PASS" in t or ("PASS" in t and "FAIL" not in t)
    return passed, r["text"].strip()[:200], _usage(r)
