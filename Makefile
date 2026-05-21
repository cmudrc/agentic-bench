.PHONY: install test bench bench-gemma4 bench-qwen lint clean

install:
	pip install -e .[dev]

test:
	pytest -q

bench-gemma4:
	agentic-bench run \
	  --backend ollama \
	  --model gemma4:e4b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/gemma4_e4b.json

bench-qwen:
	agentic-bench run \
	  --backend ollama \
	  --model qwen2.5:7b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/qwen2_5_7b.json

bench-gemma3:
	agentic-bench run \
	  --backend ollama \
	  --model gemma3:4b \
	  --suite agentic_bench/tasks/aircraft_design.yaml \
	  --report reports/gemma3_4b.json

bench: bench-gemma4 bench-qwen

lint:
	ruff check agentic_bench tests

clean:
	rm -rf build dist *.egg-info .pytest_cache
