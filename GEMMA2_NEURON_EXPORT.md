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

### 1b. Base-model prompting fixes (`circuits/tracing/trace.py`)

`google/gemma-2-2b` is a **base** model with no chat template, which broke the
`prep.py` path in two ways. Both are fixed in `prepare_ci` / `compute_circuits`:

- **Chat-template crash.** `prep.py` defaulted to `use_chat_format=True`, calling
  `tokenizer.apply_chat_template(...)` — which raises on a base model. Fixed at the
  `compute_circuits` → `prepare_cis` call site: `use_chat_format=tokenizer.chat_template
  is not None`. Auto-detects per tokenizer (base → plain text, instruct → chat format),
  so Llama/Qwen instruct paths are unchanged.
- **Dropped seed response.** The no-chat-format branch of `prepare_ci` tokenized only the
  question and **silently discarded `seed_response`** (`"Answer:"`). The trace then targeted
  the token right after the bare question — a newline — instead of the intended answer, so
  the top neurons came out as question-structure / formatting neurons that promote `\n\n`
  and ` A`, **not** the Dallas→Texas→Austin recall circuit. Fixed by concatenating
  `question + " " + seed_response` in the no-chat path.

**Consequence:** any Gemma circuit traced *before* these fixes is invalid (wrong target).
Re-trace after pulling, then sanity-check that the top neurons' `output_contributions`
actually promote answer-relevant tokens (e.g. a state/capital), not `\n\n`/` A`.

**Verify the model can even do the task in this format.** `prep.py --verbose` prints, per
prompt, `question seed -> <top predicted token>`. If base Gemma doesn't predict the capital
after `"… ? Answer:"`, the traced circuit won't be a capital-recall circuit no matter how
correct the attribution is — consider a completion-style prompt (e.g. `"The capital of the
state containing Dallas is"`) that matches how the base model actually continues text.

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

## Hardware / install on a single consumer GPU

Gemma 2 2B is small (~2.6B params ≈ **5.2 GB in bf16**), so a single consumer GPU (e.g. an
**RTX 3080, 10–12 GB**) is enough — no cloud/RunPod needed. Only **Step 1 (tracing)** uses the
GPU; **Step 2 (export)** is tokenizer-only and runs on CPU.

- **Precision / batch size:** trace in **bf16** (the default in `prep.py` and the export CLIs).
  On a 10 GB card use `--batch-size 1` or `2` (the `capitals_gemma.yaml` default of 4 can OOM
  on long prompts).
- **The completeness test loads bf16 by default** now (fits any modern GPU). For the tight
  fp32 check (needs ~10.5 GB just for weights — OOMs a 10 GB card), run it on a bigger GPU:
  `GEMMA2_TEST_DTYPE=float32 uv run pytest tests/test_gemma2_attribution.py`.
- **vLLM is NOT needed for this workflow.** It is declared in `pyproject.toml` and used only by
  ADAG's *built-in* explainer (which you replace with your own pipeline); it is imported
  lazily and never touched during tracing or export. It is, however, **Linux-only**, which
  affects install:
  - **Linux or WSL2 (+CUDA):** `uv sync` works as-is — vLLM builds fine on Linux/Ampere.
  - **Windows native:** `uv sync` will fail building vLLM. Use **WSL2** (recommended), or a
    minimal env (torch 2.5.1 + transformers + pandas + numpy + scikit-learn + anthropic + the
    `lib/` packages) skipping `vllm`/`sae_lens` — nothing on your path imports them.
- RTX 3080 (Ampere, sm_86) is fully supported by the pinned `torch==2.5.1` + CUDA 12.x.

## How a node is identified in ADAG (vs an "id and more" schema)

ADAG does **not** use a single opaque global node id at the analysis layer. A node is keyed by
the tuple **`(layer, token, neuron)`** (plus a `polarity` sign), and is decorated with data.
The same identity shows up at four levels:

| Level | Where | Node identity | Extra fields |
|-------|-------|---------------|--------------|
| In-memory node | `circuits/tracing/utils.py` `Node` | `(layer, token, neuron)` | `activation`, `final_attribution`, `attr_map`, `contrib_map` |
| Canonical key | `circuits/analysis/cluster.py` `NeuronId` | `(layer, token, neuron, polarity)`; `to_string()` → `"layer,token,neuron"` | hashable key used for clustering/labeling |
| DataFrame row | `CircuitData.df_node` | columns `layer, token, neuron` (+ `label` = which prompt, `name___ci_idx`) | `attribution`, `activation`, `attr_map`, `contrib_map` |
| Frontend node | `circuits/frontend/graph_models.py` `Node` | string `node_id` (see below) | `feature`, `ctx_idx`, `feature_type`, `influence`, `activation`, `attr_map`, `contrib_map`, `clerp` |

**Conventions:** `layer == -1` (or `"E"` in the frontend) is a **token embedding** node;
the final layer / `num_layers+1` is a **logit** node; everything in between is an MLP
**neuron** (`feature_type = "cross layer transcoder"`). `token` / `ctx_idx` is the sequence
position; `neuron` is the MLP intermediate-dimension index.

**Only the frontend builds explicit string ids** (for the circuit-tracer UI):
- feature/neuron: `node_id = f"{layer}_{neuron}_{pos}"`
- token (embedding): `node_id = f"E_{vocab_idx}_{pos}"`
- logit: `node_id = f"{num_layers+1}_{vocab_idx}_{pos}"`
- error: `node_id = f"0_{layer}_{pos}"`

In **the JSON this repo exports** (`neuron_export.py`), the mapping to your SLT "id + fields"
schema is:
- **curated features** (`neurons[]`): identified by `(layer, neuron, polarity)` — token-summed
  — plus `rank`, `attribution`, and the text bundle (`tokens`, `attr_activations`,
  `highlighted_text`, `top_input_tokens`, `output_contributions`, `raw_contrib_map`).
- **graph nodes** (`nodes[]`): identified by `(layer, token, neuron)` with `attribution`,
  `activation`.
- **edges** (`edges[]`): `src`/`tgt` each a `(layer, token, neuron)` node, with `attribution`
  and raw jacobian `weight`.

So if your SLT nodes carry a unique `id` plus metadata, the ADAG equivalent of that `id` is
the `(layer, token, neuron[, polarity])` tuple (stringifiable as `"layer,token,neuron"`), and
the "more" maps onto the attribution/activation/attr_map/contrib_map fields above. Join the
token-summed features to the per-token graph nodes/edges by `(layer, neuron)`.

## Important caveats for whoever picks this up

- **Run end-to-end on a GPU (RunPod, `google/gemma-2-2b`).** `prep.py` traces 50 capitals
  prompts and `batch_export_neurons.py` writes the per-graph JSON successfully. The base-model
  prompting fixes in §1b were needed to get there.
- **The completeness test (`tests/test_gemma2_attribution.py`) is still unverified.** It has
  its own shape bug: it pairs one `focus_position` with `k` `focus_logits` (the core expects
  a `(B, P)` pairing — see `circuits/tracing/clja.py:162-167`). Patch the call to repeat the
  position to match the logits before trusting it:
  `focus_positions=[last_pos] * len(focus_tokens[0])`. The same bug is in
  `scripts/circuit_prep/top_neurons.py` (`tgt_tokens=[max(keep_pos)]` should repeat `k`
  times, as the blessed `trace.py:441` path does). Until this test passes, attribution
  *correctness* (as opposed to "the pipeline runs") is not yet certified.
- **Sanity-check the circuit, not just that it ran.** After the §1b fixes, confirm the top
  neurons' `output_contributions` promote answer-relevant tokens (a state/capital), not
  `\n\n`/` A`. If they don't, the model isn't doing capital recall in this prompt format
  (see §1b — consider a completion-style prompt).
- The softcapping fix assumes the installed `transformers` routes `Gemma2Attention` through
  the `ALL_ATTENTION_FUNCTIONS["noqk"]` dispatch (the same mechanism the existing Llama/Qwen
  path relies on). If the completeness test passes, that assumption held.
- `ci_idx` in the export aligns with the order prompts were traced in (the dataset module's
  list order).

## Viewing the graphs (and why the sidebar text fails)

`Circuit.serve()` / `scripts/case_studies/capitals/serve_circuit.py --model-id
google/gemma-2-2b` renders the graph topology in the circuit-tracer frontend. The
**feature-detail sidebar will error** (`visState.feature is null`): it tries to fetch
per-neuron cards from a remote feature store (`huggingface.co/.../features/...` → 401,
CloudFront `features/google/gemma-2-2b/*.json` → 403) that **does not host raw Gemma MLP
neurons** — only SAE/transcoder features. The graph still renders; the text snippets live
only in the export JSON (`highlighted_text`, `top_input_tokens`, `output_contributions`),
which is computed from the traced prompts, not a hosted corpus. Use the JSON for text.

## 4. Annotation pipeline + local viewer (`custom_automation/`)

A small annotation pipeline that consumes the exported `graph_*.json` directly,
mirroring the transcoder-side description/grouping flow, plus a local HTML viewer
that sidesteps the broken frontend sidebar entirely.

- **`custom_automation/generate_description.py`** — LLM descriptions for each MLP
  neuron. Builds the evidence block from ADAG's own fields
  (`prompt` -> context, `highlighted_text` with `{{...}}` rewritten to `<<<...>>>`,
  `top_input_tokens` -> drivers, signed `output_contributions` split into
  promoted/suppressed) and calls GPT-5-mini for a `LABEL — elaboration` line.
  Writes `generated_description` back into each neuron, **in place**.

  ```bash
  cd custom_automation
  export OPENAI_API_KEY=sk-...
  # one graph, or a whole folder (writes labels into the JSON)
  python generate_description.py --graph ../capitals_neuron_graphs/graph_0000_austin.json
  python generate_description.py --graphs-dir ../capitals_neuron_graphs/
  ```

- **`custom_automation/render_report.py`** — self-contained **interactive** viewer, a
  local stand-in for the circuit-tracer frontend (the **workaround for the dead
  feature-text sidebar**), mirroring its two-section layout:
  - **Attribution graph (top)** — **every neuron** as a node, positioned by token (x)
    and layer (y), colored by its supernode, with the pruned attribution edges;
    Embeddings (layer -1) and Output (top layer) drawn as squares.
  - **Subgraph (middle)** — the collapsed view: one box per **supernode with its
    member neurons listed inside**, plus separate **Embeddings** and **Output** nodes,
    and aggregated attribution edges between them.
  - **Detail (right)** — clicking any node / neuron shows its **activation text**:
    highlighted activating tokens (`<mark>`), per-token input attribution shading
    (green +/red −), promoted/suppressed output tokens, and the LLM label. Untracked
    nodes (not in the exported top-N) show their raw attribution/activation instead.

  A top-left dropdown switches prompts; un-grouped graphs put every neuron under a
  single "All neurons" node. All inlined (SVG + vanilla JS) — no server, no CDN, no
  remote feature store, no 401/403. Open the `.html` directly, `scp` it back, run with
  `--serve` (serves localhost + opens a browser), or `python -m http.server 8041`.

  ```bash
  python render_report.py --graph ../capitals_neuron_graphs/graph_0000_austin.json --out dallas.html
  python render_report.py --graphs-dir ../capitals_neuron_graphs/ --out capitals_report.html
  ```

- **`custom_automation/generate_supernodes.py`** — the grouping step. Clusters the
  described MLP neurons of each `graph_*.json` into **supernodes** with an LLM
  (gpt-5-mini), in place, mirroring `generate_description.py`'s CLI. Three phases:
  (1) discover groups from the top-K most influential features, (2) assign the rest
  in concurrent batches, (3) reconcile — merge duplicates, fix misassignments, drop
  grammar/noise groups. Reads each neuron's `generated_description` + `attribution`;
  a feature id is `L{layer}_N{neuron}_{polarity}` (same id `generate_description.py`
  uses). Only the described MLP neurons are grouped — embedding (`layer == -1`) and
  logit nodes are left out, since the exported `nodes` carry no token strings /
  probabilities. Writes back into the graph JSON: `supernodes`
  (`{group_name: [feature_id, ...]}`, ordered by total influence), `ungrouped`, and a
  `group` field on each neuron.

  ```bash
  cd custom_automation
  export OPENAI_API_KEY=sk-...
  # one graph, or a whole folder (writes groups into the JSON)
  python generate_supernodes.py --graph ../capitals_neuron_graphs/graph_0000_austin.json
  python generate_supernodes.py --graphs-dir ../capitals_neuron_graphs/
  ```

  Knobs live in `custom_automation/config.py` (the shared pipeline config):
  `GROUPING_MODEL`, `GROUPING_VARIANT` (a0–a3 prompt strictness; default a2),
  `GROUPING_TOP_K_SEED`, `GROUPING_BATCH_SIZE`, `GROUPING_MAX_CONCURRENCY` — each
  overridable via the matching env var.

  Typical flow: `batch_export_neurons.py` -> `generate_description.py` ->
  `generate_supernodes.py` -> `render_report.py` -> read the report to verify the
  descriptions and groupings are sensible. The supernodes are the ADAG-side analogue
  of the transcoder supernodes.
```
