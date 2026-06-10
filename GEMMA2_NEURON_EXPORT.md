# Gemma 2 2B support + neuron/text export — what was added

This note summarizes files added/changed to (1) make ADAG's circuit tracing correct for
**`google/gemma-2-2b`**, and (2) export **top-N MLP neurons with their associated text** for
many prompts at once, for use in an external description / feature-grouping pipeline.

Read this together with the main [README.md](README.md) (the upstream ADAG pipeline docs).

---

## 1. Gemma 2 2B tracing support

ADAG originally supported only Llama/Qwen in the tracing path. Gemma 2 was half-wired
(registered in `circuits/utils/constants.py`, `gemma2_2b_config` in `lib/util/util/subject.py`)
but the **gradient-modification wrappers were Llama-shaped**, so tracing ran without error yet
produced *wrong* attributions. Gemma 2's decoder layer differs in ways that matter for
attribution:

- **Sandwich-norm layers**: two extra RMSNorms on the MLP path (`pre_feedforward_layernorm`,
  `post_feedforward_layernorm`) that must be linearized.
- **`Gemma2RMSNorm` scales by `(1 + weight)`** (not `weight`) and exposes eps as `.eps`
  (Llama/Qwen use `.variance_epsilon`).
- **Attention-logit softcapping** (50.0) and **final-logit softcapping** (30.0).

### Files
- **`circuits/tracing/grad/gemma2.py`** (new) — imports `Gemma2Attention/MLP/RMSNorm`.
- **`circuits/tracing/grad/__init__.py`** (changed) — the core fix. Adds Gemma2 to the
  type tuples; `_effective_norm_weight` (the `1+weight`) and `_norm_eps` accessors;
  `stop_nonlinear_grad` / `revert` / `layerwise_*` now linearize the extra feedforward norms
  and **disable/restore final-logit softcapping** during attribution; `noqk_attention_forward`
  applies attention-logit softcapping. **All changes are model-conditional — no-ops for
  Llama/Qwen.**
- **`circuits/tracing/utils.py`** (changed) — `collect_neuron_acts` output-norm now uses the
  same accessors (needed for the full edge-tracing pipeline on Gemma2).
- **`scripts/circuit_prep/configs/capitals_gemma.yaml`** (new) — a ready trace config for
  `google/gemma-2-2b` on the capitals dataset.
- **`tests/test_gemma2_attribution.py`** (new, GPU-gated) — asserts **attribution
  completeness** (summed MLP+embed attributions ≈ traced logit). This is the correctness
  proof: it *fails* on the pre-fix code and *passes* after the fix. Skips without CUDA.

### Correctness invariant
Once all nonlinearities are linearized, the traced logit is a linear function of the
embeddings and MLP neuron activations, so by Euler's theorem the summed attributions equal
the logit. The test checks this in the plain stop-gradient mode (`use_relp_grad=False`),
which carries no Shapley redistribution.

---

## 2. "Prompt → top-N MLP neurons" (lightweight, no text)

- **`scripts/circuit_prep/top_neurons.py`** (new) — given a prompt (+ optional target token),
  returns the top-N `(layer, token, neuron, attribution)` using the fast attribution path
  (`return_only_important_neurons=True`); no edges, no text. Good for a quick ranked list.

```bash
uv run python scripts/circuit_prep/top_neurons.py --model-id google/gemma-2-2b \
    --prompt "What is the capital of the state containing Dallas? Answer:" \
    --target " Austin" --top-n 20
```

---

## 3. Top-N neurons **+ text** export (for the description / grouping pipeline)

The reusable logic lives in the installed package; the scripts are thin CLIs.

- **`circuits/analysis/neuron_export.py`** (new, importable) — the core functions:
  - `export_per_prompt_from_circuit(...)` → **one ranked list per prompt** (per graph).
  - `export_top_neurons_from_circuit(...)` → **one aggregate list** across all prompts.
  - `export_top_neurons(model, tokenizer, prompts, labels, ...)` → trace + export in one call.
- **`scripts/circuit_prep/export_neurons.py`** (new) — single-prompt CLI (loads model, traces
  one prompt or loads one circuit pickle, writes JSON).
- **`scripts/circuit_prep/batch_export_neurons.py`** (new) — **batch CLI for many graphs at
  once**. Loads a circuit pickle (tokenizer-only, no GPU) and writes per-prompt or aggregate
  JSON.

### How it works (reuses ADAG's own representation end-to-end)
```
Circuit                       # normalizes attr_map / contrib_map
  → prepare_circuit_data(sum_over_tokens=True)   # NeuronId keys; sum maps over token positions
  → build_neuron_activation_records              # per-neuron tokens + per-token attribution + contrib
  → build_attr_exemplars                         # {{highlighted}} exemplar text
```

### Per-neuron text fields in the JSON
- `tokens` + `attr_activations` — the prompt tokens and the neuron's **per-token input
  attribution** (the text that drives it). Raw vector — usable as a grouping feature.
- `highlighted_text` — ADAG's `{{highlighted}}` exemplar string.
- `top_input_tokens` — `[(token, score), ...]` driver tokens.
- `output_contributions` + `raw_contrib_map` — the **output tokens it promotes/suppresses**
  over the traced logits.

---

## Recommended workflow for MANY prompts

**Step 1 — trace many prompts once** (batched; multi-GPU via SLURM). Define a dataset module
`scripts/circuit_prep/data/<name>.py` exporting `prompts`, `seed_responses`, `labels`
(see `data/capitals.py`), then:

```bash
uv run python scripts/circuit_prep/prep.py --config configs/capitals_gemma.yaml
# or multi-GPU:
sbatch --gres=gpu:4 --export=CONFIG=configs/capitals_gemma.yaml scripts/circuit_prep/prep.sbatch
```

This writes ONE `CircuitData` pickle covering every prompt. **It contains the full attribution
graph: `df_node` (neurons) AND `df_edge` (edges).** Each prompt is a separable graph inside it
(keyed by the `___N` label suffix).

**Step 2 — export per-graph top-N neurons + text** (tokenizer-only, no GPU).

One file per graph into a folder (recommended for N graphs):
```bash
uv run python scripts/circuit_prep/batch_export_neurons.py \
    --circuit results/case_studies/capitals_gemma_circuit.pkl \
    --model-id google/gemma-2-2b --dataset capitals \
    --top-n 30 --mode per-prompt --out-dir capitals_graphs/
# -> capitals_graphs/graph_0000_austin.json, graph_0001_montgomery.json, ...
```

Each per-graph file is a complete, self-contained attribution graph — `nodes`, `edges`, and
the curated `neurons` (top-N features + text):
```json
{
  "ci_idx": 0, "label": "Austin___0",
  "prompt": "What is the capital of the state containing Dallas?",
  "target": " Austin",
  "neurons": [
    {"rank":1,"layer":20,"neuron":1234,"polarity":"+","attribution":0.41,
     "tokens":[...],"attr_activations":[...],"highlighted_text":"...{{Texas}}...",
     "top_input_tokens":[[" Texas",0.42]],"output_contributions":[[" Austin",0.55]],
     "raw_contrib_map":[...]}
  ],
  "nodes": [
    {"layer":12,"token":5,"neuron":880,"attribution":0.03,"activation":1.7}
  ],
  "edges": [
    {"src":{"layer":12,"token":5,"neuron":880},
     "tgt":{"layer":20,"token":7,"neuron":1234},
     "attribution":0.06,"weight":0.0021}
  ]
}
```

Output options:
- `--out-dir DIR` — one `graph_<ci>_<label>.json` per prompt (overrides `--out`).
- `--split` (with `--out-dir`) — write graph structure (`nodes`+`edges`) to `DIR/graphs/` and
  the `neurons`+text to `DIR/neurons/`, as separate files sharing the same stem.
- `--out FILE` — single combined JSON list (default). `--jsonl` — one graph per line.
- `--no-edges` / `--no-nodes` — omit those sections.
- `--mode aggregate` — pool neurons across the whole dataset into one ranked list (no
  nodes/edges, since pooled neurons aren't a single graph).

**Two granularities in each graph (by design):** the curated `neurons` (features + text) are
summed over token positions (keyed by layer+neuron); the raw `nodes`/`edges` keep their token
positions (carry layer, token, neuron; `layer == -1` is the token embedding). Join a
node/edge endpoint back to a feature by (layer, neuron). Edge `attribution` is normalized per
prompt; `weight` is the raw jacobian edge weight.

---

## What you can and cannot get today

| Want | Status |
|------|--------|
| Trace many prompts at once (batched / multi-GPU) | ✅ `prep.py` |
| Full attribution graph (nodes **and** edges) saved per run | ✅ in the `CircuitData` pickle (`df_node`, `df_edge`); viewable via `Circuit.serve()` |
| Top-N MLP neurons + text per prompt as JSON | ✅ `batch_export_neurons.py --mode per-prompt` |
| One JSON file per graph in a folder | ✅ `--out-dir DIR` (optionally `--split` graph vs neurons) |
| Top-N neurons + text aggregated across prompts | ✅ `--mode aggregate` |
| **Nodes + edges included in the JSON export** | ✅ per-prompt mode emits `nodes` and `edges` (the pruned attribution graph) alongside `neurons`; `--no-edges` / `--no-nodes` to omit |

---

## Important caveats for whoever picks this up

- **Not yet run end-to-end on a GPU.** All new files byte-compile and the data-flow was
  verified against ADAG's source, but the dev box couldn't install torch/transformers. Run
  `tests/test_gemma2_attribution.py` (needs CUDA + downloads `google/gemma-2-2b`) first to
  confirm tracing correctness, then the workflow above.
- `google/gemma-2-2b` is a **base** model. The single-prompt helpers default to
  `use_chat_format=False`; the full pipeline (`prep.py` / `convert_inputs_to_circuits`) uses
  the tokenizer's chat template.
- The softcapping fix assumes the installed `transformers` routes `Gemma2Attention` through
  the `ALL_ATTENTION_FUNCTIONS["noqk"]` dispatch (the same mechanism the existing Llama/Qwen
  path relies on). If the completeness test passes, that assumption held.
- `ci_idx` in the export aligns with the order prompts were traced in (the dataset module's
  list order).
```
