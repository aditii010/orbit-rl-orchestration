#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
MODEL="qwen2.5:3b"
PORT=8501

say() { printf '\n== %s\n' "$1"; }

say "1/4 Is Ollama running?"
if ! curl -s -o /dev/null http://localhost:11434/api/tags; then
  echo "Ollama is not running. Starting the Ollama app..."
  open -a Ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
  for i in $(seq 1 30); do
    curl -s -o /dev/null http://localhost:11434/api/tags && break
    sleep 1
  done
fi
if curl -s -o /dev/null http://localhost:11434/api/tags; then echo "Ollama is up."; else
  echo "WARNING: could not start Ollama. The demo still works in Replay mode (no model needed)."; fi

say "2/4 Is the model $MODEL downloaded?"
if curl -s http://localhost:11434/api/tags | grep -q "\"$MODEL\""; then echo "Yes."; else
  echo "WARNING: $MODEL not found. Run:  ollama pull $MODEL   (Replay mode works without it)."; fi

say "3/4 Warming the model up (first call is slow, so we do it now)"
curl -s -m 120 http://localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"keep_alive\":\"30m\",\"options\":{\"num_predict\":3}}" >/dev/null \
  && echo "Warm." || echo "Skipped (model not available)."

say "4/4 Starting the dashboard on http://localhost:$PORT"
pkill -f "streamlit run" 2>/dev/null; sleep 1
mkdir -p docs/logs
nohup python3 -m streamlit run app/dashboard.py --server.port "$PORT" --server.headless true > docs/logs/streamlit.log 2>&1 &
for i in $(seq 1 20); do curl -s -o /dev/null "http://localhost:$PORT" && break; sleep 1; done
if curl -s -o /dev/null "http://localhost:$PORT"; then
  echo "Dashboard is up: http://localhost:$PORT"
  [ -z "$NO_OPEN" ] && open "http://localhost:$PORT" 2>/dev/null
else
  echo "Dashboard did not start. See docs/logs/streamlit.log"; exit 1
fi
echo; echo "Stop it later with:  pkill -f 'streamlit run'"
