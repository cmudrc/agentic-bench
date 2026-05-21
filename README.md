# agentic-bench

A small, opinionated benchmark for **local agentic LLMs** in engineering
domains. Pluggable adapters, deterministic scoring, a single aggregate
loss in `[0, 1]`. Works out of the box with Ollama; ~50 LOC to add a new
provider.

> Status: alpha (v0.1.0). Released because we needed a way to compare
> Gemma 4, Qwen 2.5, Gemma 3, and friends on an aerospace-flavoured
> agent workload — and decided to make it reusable for anyone wiring
> LLMs to engineering tools.

---

## Why another benchmark?

Most public LLM benchmarks (MMLU, AIME, LiveCodeBench, …) measure raw
text generation. They do not measure the four things that *actually*
break when you stick an LLM in front of an engineering toolchain:

1. **Numerical sanity.** Does the model know that a transonic
   narrowbody cruises at `CL ≈ 0.55`, or does it make up `CL = 7.2`?
2. **Tool routing.** Given "give me a STEP file of the geometry", does
   the model call the geometry-export tool, or the CFD tool?
3. **Argument extraction.** When the user says "Mach 0.78, AoA 2.5°,
   FL350, workstation preset", does the model put 0.78 in `mach`, 2.5
   in `aoa_deg`, 35 000 in `altitude_ft`, and `workstation` in
   `preset`? Or does it scramble them?
4. **Multi-step planning.** Given "take the geometry through aero,
   propulsion, and mission to get block fuel", does it produce a
   coherent ordered plan, or a soup of out-of-order tool calls?
5. **Multimodal verdicts** (optional). Given a CFD post-processing
   image, can the model judge whether the mesh looks under-resolved?

Every task type has a deterministic scorer in `[0, 1]`, and the suite
collapses into **one weighted aggregate loss** so you can rank models
on a single number.

## Install

```bash
pip install -e .
# or
pip install -e .[dev]   # adds pytest + ruff
```

You also need an LLM backend. The reference adapter is
[Ollama](https://ollama.com):

```bash
brew install ollama
brew services start ollama
ollama pull gemma4:e4b      # the default model in the default suite
ollama pull qwen2.5:7b      # comparison baseline
```

## Quickstart

```bash
agentic-bench run \
  --backend ollama \
  --model gemma4:e4b \
  --suite agentic_bench/tasks/aircraft_design.yaml \
  --report reports/gemma4_e4b.json
```

Output:

```
=== AGENTIC-BENCH REPORT ===
  adapter  : ollama:gemma4:e4b
  suite    : aircraft_design_v1
  items    : 19
  wall (s) : 509.91
  loss     : 0.2537
  per-category scores:
    args        0.739
    numerical   0.854
    planning    0.525
    routing     0.800
```

The full per-item report is written to `reports/gemma4_e4b.json`.

## What's in a suite

A suite is a single YAML file containing:

- A **`tools`** list of OpenAI-shaped function specs. Adapters that
  support function calling get these natively; adapters that don't see
  only the names in planning-style prompts.
- A list of **`items`**. Each item has a `kind`:
  - `numerical` — single-number QA, scored by relative-error decay.
  - `routing` — first-tool-call must match `expected_tool`.
  - `args` — first-tool-call arguments are compared to `expected_args`.
  - `planning` — model emits a `{"plan": [...]}` JSON of tool names;
    scored by normalised Levenshtein distance.
  - `multimodal` — model is given an image and a prompt; verdict is
    scored by substring match against `expected_label`.
- An optional **`weights`** block overriding category weights in the
  aggregate loss. Default weights are
  `{numerical: 0.25, routing: 0.25, args: 0.20, planning: 0.20, multimodal: 0.10}`,
  renormalised over the categories that actually appear in the suite.

See [`agentic_bench/tasks/aircraft_design.yaml`](agentic_bench/tasks/aircraft_design.yaml)
for the reference suite (19 items, aerospace-flavoured).

## Scoring details

| Kind         | Scorer                                                                      |
| ------------ | --------------------------------------------------------------------------- |
| `numerical`  | `1.0` if first emitted number is within `tolerance_pct`; exp-decay outside. |
| `routing`    | `1.0` if first tool call name matches; `0.5` if it appears later; else `0`. |
| `args`       | Per-key score, mean over keys. Numbers use exp-decay; strings/enums exact.  |
| `planning`   | `1 - levenshtein(expected_plan, got_plan) / max(|exp|, |got|)`.             |
| `multimodal` | `1.0` if `expected_label.lower()` is a substring of the reply; else `0.0`.  |

The aggregate loss is

```
L = sum_c w_c * (1 - mean_score_c)   over categories c that appear
```

with weights renormalised over present categories.

## Adding a new adapter

Drop a file in `agentic_bench/adapters/` that implements `LLMAdapter`
from `agentic_bench.adapters.base`. The protocol is two methods:

```python
class LLMAdapter(Protocol):
    def name(self) -> str: ...
    def chat(self, messages, tools=None, temperature=0.0) -> ChatResult: ...
    def chat_with_image(self, messages, image_path, temperature=0.0) -> ChatResult: ...
```

Then register it in `agentic_bench/adapters/__init__.py` `REGISTRY`.
~50 lines for a typical provider — see
[`ollama_adapter.py`](agentic_bench/adapters/ollama_adapter.py).

## Adding tasks

Tasks are plain YAML, no Python required. The simplest is a numerical
item:

```yaml
- id: my_question
  kind: numerical
  prompt: "What is the typical Reynolds number for a 737 wing at cruise?"
  expected: 2.0e7
  tolerance_pct: 30
```

For routing / args / planning items, see the reference suite.

## Reproducibility

- Every adapter calls the underlying model at `temperature=0.0`.
- The Ollama adapter sets `keep_alive: "10m"` so the model stays loaded
  across items — otherwise short-task suites measure model-load latency
  instead of model quality.
- Reports include the adapter name, the model tag, per-item raw outputs
  (truncated), and the aggregate loss — sufficient to diff two runs.

## Known limitations

- Aerospace-specific suite. We expect users to write domain-specific
  suites for their own engineering toolchains.
- No reasoning-trace scoring. Open question whether to add it.
- No stochastic re-runs / confidence intervals yet. On the roadmap.
- Multimodal images are passed verbatim; some Ollama models silently
  ignore them. The adapter has no way to detect this.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citation

If you use agentic-bench in published work, please cite the
[aircraft-analysis](https://github.com/cmudrc/aircraft-analysis)
project as the originating context:

```bibtex
@misc{dixit2026agentic_bench,
  author = {Dixit, Mayank and McComb, Christopher},
  title  = {agentic-bench: a small benchmark for local agentic LLMs in engineering domains},
  year   = {2026},
  howpublished = {\url{https://github.com/cmudrc/agentic-bench}},
}
```
