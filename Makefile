.PHONY: install test bench bench-gemma4 bench-gemma3-server bench-hybrid bench-qwen lint clean

install:
	pip install -e .[dev]

test:
	pytest -q

# === Recommended production-tier benchmark (all-Gemma default) ============
bench-gemma4:
	agentic-bench run \
	  --backend ollama \
	  --model gemma4:e4b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/gemma4_e4b.json

# Server-tier (Gemma 3 27B via structured-output ReAct adapter).
bench-gemma3-server:
	agentic-bench run \
	  --backend ollama-react \
	  --model gemma3:27b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/gemma3_27b_react.json

# Hybrid planner+seeker (production setup mirroring agent-mcp/hybrid_agent.py).
bench-hybrid:
	agentic-bench run \
	  --backend hybrid \
	  --planner gemma4:e4b \
	  --seeker gemma4:e4b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/hybrid_combined.json

# === Historical baselines (kept for transparency; not the default) ========
bench-gemma3:
	agentic-bench run \
	  --backend ollama \
	  --model gemma3:4b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/gemma3_4b.json

bench-qwen:
	agentic-bench run \
	  --backend ollama \
	  --model qwen2.5:7b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/qwen2_5_7b.json

bench: bench-gemma4 bench-hybrid

lint:
	ruff check agentic_bench tests

clean:
	rm -rf build dist *.egg-info .pytest_cache
