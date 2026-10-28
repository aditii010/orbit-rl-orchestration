"""One client for a local Ollama model and Groq. Logs tokens, latency and request count for every call.
Keys come from the environment only (GROQ_API_KEY); they are never logged or printed."""
import json
import os
import time
import urllib.error
import urllib.request


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    def __init__(self, backend="ollama", model="qwen2.5:3b", max_requests=None, log_path=None):
        self.backend, self.model = backend, model
        self.max_requests, self.n_requests = max_requests, 0
        self.log_path = log_path
        if backend == "groq" and not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set (put it in .env and load it)")

    def chat(self, user, system=None, max_tokens=400, temperature=0.7):
        if self.max_requests is not None and self.n_requests >= self.max_requests:
            raise BudgetExceeded(f"request cap {self.max_requests} reached")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        t0 = time.time()
        if self.backend == "ollama":
            body = {"model": self.model, "messages": msgs, "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}}
            d = self._post("http://localhost:11434/api/chat", body, {})
            text, tin, tout = d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)
        else:
            body = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
            hdr = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
            d = self._post("https://api.groq.com/openai/v1/chat/completions", body, hdr)
            text = d["choices"][0]["message"]["content"] or ""
            tin, tout = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
        self.n_requests += 1
        rec = {"text": text, "tin": tin, "tout": tout, "seconds": time.time() - t0}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"model": self.model, "tin": tin, "tout": tout,
                                    "seconds": round(rec["seconds"], 2)}) + "\n")     # metadata only, no text
        return rec

    def _post(self, url, body, extra_headers, retries=5):
        for attempt in range(retries):
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "User-Agent": "orbit-rl/0.1",
                                                  **extra_headers})
            try:
                return json.load(urllib.request.urlopen(req, timeout=600))
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:       # rate limited: wait and retry
                    time.sleep(float(e.headers.get("retry-after", 5 * (attempt + 1))))
                    continue
                raise
