# Spec — corpus raw-activation exemplars for traced Gemma MLP neurons

**Status:** design only. No harvest code until this is agreed.
**Goal:** give the bounded set of MLP neurons that appear in our attribution graphs
**corpus-wide, raw-activation, multi-window** exemplars — the same evidence regime the
SLT/transcoder (Neuronpedia) side uses — so the descriptions the matcher in
`cross_graph_analysis.py` consumes are methodologically comparable instead of confounded.

## Why
Today the MLP description evidence is **per-prompt input-attribution** (one window, the
traced prompt); the SLT side is **corpus raw-activation** (many windows over a big
dataset). Those are two independent differences (corpus-vs-per-prompt AND
activation-vs-attribution) and they land exactly on `generated_description`. See the
caveat banner now emitted by `cross_graph_analysis.py`. This spec removes the confound.

For ≤~10-token Neuronpedia prompts the per-prompt window is also a near-useless
neuron-identity signal (the text is nearly identical across neurons), which is the
strongest practical argument that this is required, not optional, for the comparison.

---

## Resolved decisions (pinned to existing code — not open)

### R1. The harvest reuses ADAG's own capture, not a hand-rolled hook
`circuits/tracing/utils.py:56` `collect_neuron_acts` already hooks
`layer_module.mlp.down_proj` (forward hook, `utils.py:83-88`) and returns `neurons_LBTI`
= the **down_proj input** (post-activation MLP hidden) per (layer, batch, token,
intermediate). It is already Gemma-2-aware (sandwich norms via
`_effective_norm_weight` / `_norm_eps`).

→ **The harvest calls `collect_neuron_acts`.** This makes "corpus activation = the exact
quantity ADAG attributes" true *by construction*, and inherits tokenization/BOS handling
for free. This collapses the former highest-risk item (custom hook) and the tokenization
item to zero.

### R2. Polarity = sign of the activation
`circuits/analysis/cluster.py:140`:
`df_node["polarity"] = df_node["activation"].apply(lambda x: "+" if x >= 0 else "-")`.
Polarity is the sign of the down_proj input, **not** an attribution sign. So:
- `+` polarity → keep **max**-activating windows.
- `-` polarity → keep **most-negative** windows.
The same physical neuron can legitimately be harvested **twice** (once per polarity) —
GeGLU activations are genuinely signed. The store key is `(layer, neuron, polarity)`.

---

## Pipeline

### 1. Scope = the traced-neuron union (what makes this tractable)
Collect every `(layer, neuron, polarity)` appearing in **any** `graph_*.json` across the
prompt set. Dedup. This is hundreds–low-thousands of neurons, not all ~240k. One bounded
batch job, not a full Gemma dashboard.

### 2. Harvest
- One forward pass of `google/gemma-2-2b` over a fixed corpus sample, via
  `collect_neuron_acts` (R1), capturing only the traced `(layer, neuron)` subset.
- Maintain running **top-k windows by activation** per `(layer, neuron, polarity)` (R2).
- Window = **±N tokens around the peak token**; choose N to match the gemmascope/
  Neuronpedia context width (see OPEN-1).

### 3. Quantile bands
Match the Neuronpedia lens: top band plus sampled lower-activation bands, so descriptions
see the distribution, not just the max. (Banding config in OPEN-1.)

### 4. Output schema (drop-in, zero transform)
Store keyed by `(layer, neuron, polarity)`:
```
{ "act_min", "act_max",
  "examples_quantiles": [
     {"quantile_name", "examples": [{"tokens", "tokens_acts_list", "train_token_ind"}]}
  ] }
```
This is exactly the frontend / Neuronpedia card schema, so it feeds **both**
`serve_with_cards.py` and `generate_description.py` with no transform.

### 5. Integration
- **Descriptions:** corpus exemplars become **primary** evidence; the per-prompt
  attribution window stays as a labeled **secondary "role in this circuit"** overlay.
- **Viewer:** serve corpus exemplars from the store; per-prompt card secondary.
- **Promote/demote:** stays locally computed on both sides (the Modal endpoint zeroes
  logits), so it is model-independent and unaffected by this change.

---

## Additions (the real gaps beyond the original sketch)

### A1. Validation gate — before trusting the store
Take 3–4 neurons we already understand from the graphs (e.g. a "basketball" neuron from
`michael-jordan`, a "season/summer" one from `saison`) and eyeball that their corpus
max-activating windows are semantically coherent. A systematic hook/normalization error
is invisible in aggregate but obvious here. Cheap; catches the disaster case.

### A2. The one residual correctness assert
During the validation pass, assert that the harvested value for a given
`(layer, token, neuron)` equals `df_node["activation"]` for the same node from a trace
(raw down_proj input, pre-output-norm). This closes the only leftover piece of the old
"hook point" risk.

### A3. Dead / rare-neuron fallback + coverage reporting
Some traced neurons are prompt-domain-specific and barely fire on generic web text →
sparse/empty exemplars. Rule: **if corpus exemplars are empty/weak for a neuron, the
per-prompt snippet stays primary for that neuron.** Report per-neuron hit counts so we
know what fraction of the traced union actually got real exemplars.

### A4. Polysemanticity asymmetry is a finding, not a bug
Even after the regimes match, transcoder latents are sparse/near-monosemantic by
construction while raw MLP neurons are polysemantic → MLP exemplars will be messier
across bands and MLP descriptions legitimately vaguer. Surface this as a real
representational difference; the matcher/report must not read it as "the method failed."
(Already noted in the `cross_graph_analysis.py` caveat.)

### A5. Incremental harvest / caching
The traced-neuron union grows as prompts are added. Key the store by
`(model, corpus_hash, window_radius, quantile_config)` and only harvest neurons not
already cached.

### A6. Confirm the SLT context width, don't default
Matching the SLT lens is the entire point. Look up the actual gemmascope/Neuronpedia
context window and quantile banding rather than guessing ±32 (see OPEN-1).

---

## Open decisions (must resolve before code)
- **OPEN-1: window width N + quantile banding** — look up the gemmascope/Neuronpedia
  values and match them. This is the one number that makes the lens comparable.
- **OPEN-2: corpus + token budget** — ideally the same dataset family Neuronpedia used
  for gemmascope; otherwise a defensible diverse-web sample (~10–50M tokens). Diversity
  is the requirement; exact match is a nice-to-have.
- **OPEN-3: keep the per-prompt attribution overlay, or drop it** once corpus exemplars
  exist (A3 keeps it as fallback regardless).

## Cost / scale
~1–3k neurons × one corpus pass (~10–50M tokens) on a single GPU; sparse capture means
only the traced subset is tracked, plus small top-k buffers. Tens of minutes to a couple
hours. Tractable precisely because of scope (§1).

## Out of scope
The top-K-threshold sweep (100/200/300/400/500) — deferred per current direction.
