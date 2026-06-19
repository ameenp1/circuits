# Spec — Gemma MLP description evidence: corpus exemplars + logit-weights

**Status:** design only. No code until this is agreed.

There are **two distinct fixes** to the MLP-side evidence, with **different faithfulness
stories**. Do not conflate them:

- **Fix #1 — corpus exemplars (activating text).** Genuinely *causally faithful*: it
  captures the real neuron activation over a corpus. This is the substantive upgrade and
  the H100 job. **Primary fix.**
- **Fix #2 — promote/demote via logit-weights (direct path).** A *standard heuristic for
  comparability*, **not** causally faithful. It answers "what does this neuron write
  toward in vocab space, ignoring everything downstream," not "what does it causally
  promote." Build it **only** to match Neuronpedia's transcoder `top_logits/bottom_logits`
  lens — and label it honestly as logit-weights, not causal promote/demote.

**Goal of Fix #1:** give the bounded set of MLP neurons that appear in our attribution
graphs **corpus-wide, raw-activation, multi-window** exemplars — the same evidence regime
the SLT/transcoder (Neuronpedia) side uses — so the descriptions the matcher in
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

### 2. Harvest — match SAEDashboard's lens exactly (verified config)
From `SAEDashboard` `NeuronpediaRunnerConfig` (the tool Neuronpedia runs):
`n_tokens_in_prompt=128`, `n_quantiles=5`, `top_acts_group_size=20`, `quantile_group_size=5`.

- One forward pass of `google/gemma-2-2b` over `monology/pile-uncopyrighted`, via
  `collect_neuron_acts` (R1), capturing only the traced `(layer, neuron)` subset.
- **An example is the FULL 128-token prompt** (the corpus chunk), with the peak token
  highlighted — **NOT a ±N re-centered window.** SAEDashboard shows the whole 128-token
  prompt; matching that is the lens. (Correcting an earlier ±N assumption.)
- Maintain per `(layer, neuron, polarity)` (R2): a **top-20** heap by activation
  (`+`→max, `−`→most-negative), plus the quantile bands below.

### 3. Quantile bands — 5 bands × 5 examples (verified)
Reproduce SAEDashboard: **20 top-activating examples + 5 quantile bands × 5 examples each**
(45 examples/feature). Bands are intervals over the activation range, sampled. **Confirm
SAEDashboard's exact interval definition** (how it bins the 5 quantiles) and match it.

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

### A3. Dead / rare-neuron handling — FAIL LOUD, no fallback
Some traced neurons barely fire on generic web text → sparse/empty exemplars.
**Rule (per user): do NOT substitute the per-prompt snippet or any other guess.** If a
neuron has no/weak corpus exemplars, simply **omit it from the store** — the existing
serve_with_cards endpoint then falls through to the Transluce/Llama default, which is
*obviously wrong* and makes the coverage gap visible at a glance. A plausible-looking
fallback would mask the problem; the Llama garbage is the intended signal that *this neuron
got no real data*. Still **report per-neuron hit counts** so the dead rate is quantified,
not just visible. (This makes the corpus token budget matter more — see the budget note.)

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

## Corpus decision (Fix #1) — match the gemma-2-2b SLT side EXACTLY
**Faithful comparison = identical corpus + budget. Match it; do not deviate.** Transluce's
Llama recipe is the wrong target (Llama tokenizer, chat prefix) — irrelevant here.

VERIFIED for **gemma-2-2b** gemmascope dashboards (Neuronpedia + SAEDashboard
`NeuronpediaRunnerConfig`):
- **Corpus:** `monology/pile-uncopyrighted` — **must be the identical dataset.**
- **Budget:** **36,864 prompts × 128 tokens = 4,718,592 ≈ 4.7M tokens.** Match this exact
  budget. (NOT 10M — the SLT side scanned 4.7M; scanning more breaks parity.)
- **Context:** 128 tokens per prompt.
- **Tokenization / BOS:** VERIFIED from the SAE cfg metadata
  (`SAE.from_pretrained("gemma-scope-2b-pt-res-canonical",...).cfg.metadata`):
  `prepend_bos = True`, `dataset_path = monology/pile-uncopyrighted`. Each 128-token context
  starts with `<bos>`. `shuffle_tokens=True`. Base gemma-2-2b, no chat template.
- **Window = 128 (NOT 1024):** the SAE metadata's `context_size = 1024` is the SAE's
  **training** context — NOT the exemplar window. The exemplar/display window is
  SAEDashboard's `n_tokens_in_prompt = 128` (verified from the runner config AND
  Neuronpedia's published 36,864×128 figure). Harvest at **128**.
- **Banding:** 20 top + 5 quantile bands × 5 (verified).

## Remaining open / to-verify
- **OPEN-1: exact quantile *interval definition*.** Count (5×5) verified; *how* SAEDashboard
  bins the 5 intervals over the activation range — match its actual scheme.
- **OPEN-2 — BOS final check:** confirm the gemma-scope SAE cfg's `prepend_bos` value
  directly (the runner defers to it; SAE Lens convention is `True`).
- **RESOLVED — polarity = MLP-specific.** Keep `−` polarity as **most-negative** windows.
  Per user: this does NOT ruin the comparison as long as the corpus is identical — it's a
  legitimate MLP-specific signal that the (non-negative) SLT side simply lacks. Harvest it;
  label `−` neurons as MLP-only / no-SLT-counterpart, don't drop them.
- **RESOLVED — budget = 4.7M, match exactly** (not 10M).
- **RESOLVED — dead/rare = fail loud** (A3). MVP. No per-prompt fallback, no substitution.
- **CORRECTION — the SLT transcoders are SINGLE-LAYER, not cross-layer.** The data is
  labeled "cross layer transcoder" by a default misnomer; they are **single-layer (per-layer)
  transcoders**. (Doesn't change the harvest — same corpus/window — but fix the terminology
  anywhere it says "cross-layer".)

---

## Fix #2 — promote/demote via logit-weights (comparability only — NOT causally faithful)

**Framing (do not overstate this):** this is the standard "logit-weights" heuristic, a
*direct-path* approximation. It is **not** the causal contribution. Build it to match
Neuronpedia's transcoder `top_logits/bottom_logits` lens, and label it **"logit weights
(direct path)"**, not "promote/demote."

**Pre-req to confirm:** that Neuronpedia's transcoder promotes/demotes (what
`fetch_all_activation_text.py` pulls) are themselves logit-weights. They almost certainly
are — but confirm, so you are matching the *same* lens on both sides.

**What we already have is MORE causally faithful, just narrower.** `output_contributions`
uses the real attributed `contrib_map` over `output_logits = target_logits[ci_idx]`
(`label.py:69`) — the genuine causal contribution, but only to the handful of traced target
tokens. **Keep `contrib_map` as a separate, genuinely-causal-but-local signal.** The
logit-lens is the *global-but-shallow* complement; they answer different questions.

**Gemma-2 correctness requirements (the formula `W_U @ final_norm(down_proj[:,n])` is
WRONG for Gemma-2):**
1. **Route through `post_feedforward_layernorm`.** Gemma-2 applies an extra sandwich norm
   to the MLP output before it reaches the residual
   (`grad/__init__.py:23` `_GEMMA2_EXTRA_NORM_ATTRS`). The neuron's true write direction is
   `post_feedforward_layernorm(down_proj_output)` — skipping it is not even the correct
   *direct* path.
2. **Use the existing norm accessors, never hardcode Llama-shaped norms.** Gemma-2 RMSNorm
   is `(1 + weight)` with `.eps`; reuse `_effective_norm_weight` / `_norm_eps`
   (`grad/__init__.py:26,33`) for both `post_feedforward_layernorm` and the final
   `model.model.norm`. This is exactly the Llama-shaped trap that produced wrong
   attributions before the Gemma-2 fixes (GEMMA2_NEURON_EXPORT.md §1).
3. **Tied embeddings:** the unembedding is the tied input embedding.
4. **Final logit softcapping is monotonic** → it does not change top-k token *ranking*, so
   it is safe to skip for the promote/demote *list* (note it if you ever report magnitudes).
5. **Downstream caveat (inherent, not fixable):** the direct path ignores every layer
   between the neuron and the unembedding. For mid/early-layer neurons this is a small,
   often misleading slice of the true effect. This is the core reason it's "logit-weights,"
   not "causal." State it in the report.

**Cost:** weights-only, no corpus, no forward passes — seconds on CPU for all traced
neurons. (This is the "cheap, no corpus" fix; the H100s are for Fix #1 only.)

---

## Correctness gate (before trusting EITHER fix)
Spot-check 3–4 neurons you already understand from the graphs — a "basketball" neuron from
`michael-jordan`, a "season/summer" one from `saison`. Corpus exemplars (#1) should show
coherent contexts; logit-weights (#2) top tokens should be sane. A systematic Gemma-2 norm
bug is invisible in aggregate but obvious here. This is the single cheapest guard against a
Llama-shaped norm error in #2 or a hook/normalization error in #1.

## Quantile binning — VERIFIED (SAEDashboard `sequence_data_generator.py`)
`quantiles = torch.linspace(0, feat_acts.max(), n_quantiles + 1)` → **equal-width bands over
[0, max_activation]** (NOT distribution quantiles). TOP group = deterministic top-k; each
quantile band = randomly sample `quantile_group_size` examples whose activation is in
`[lower, upper]`. `harvest_corpus_exemplars.py` reproduces this exactly (linspace bins; the
only approximation is sampling each band from a per-neuron reservoir of firing examples
rather than from all — documented in the script).

## Cost / scale
Traced union (hundreds–low-thousands of neurons) × one pass over 4.7M tokens (36,864×128) on
an H100 = **minutes**. Sparse capture (only traced columns) + a top-20 heap + a ~400-example
reservoir per `(layer,neuron,polarity)` ≈ a few hundred MB host RAM.

---

## Files & run order (everything needed)
All under `circuits/`. Build the MLP side, harvest exemplars, then compare.

| step | file | what it does |
|---|---|---|
| trace | `scripts/circuit_prep/prep.py --config configs/neuronpedia_gemma.yaml` | ADAG trace → `neuronpedia_circuit.pkl` |
| export | `scripts/circuit_prep/batch_export_neurons.py` | per-prompt `neuronpedia_neuron_graphs/graph_*.json` |
| describe | `custom_automation/generate_description.py` | LLM descriptions into the graphs (needs `OPENAI_API_KEY`) |
| supernodes | `custom_automation/generate_supernodes.py` | LLM groups into the graphs |
| **harvest** | **`custom_automation/harvest_corpus_exemplars.py`** | **the H100 job → `np_data/mlp_exemplars.json`** |
| SLT fetch | `custom_automation/fetch_neuronpedia_artifacts.py` | downloads the SLT side → `np_data/` |
| compare | `custom_automation/cross_graph_analysis.py` | MLP↔SLT report (emits the evidence-regime caveat) |
| view | `custom_automation/serve_with_cards.py` | real frontend; (wire `mlp_exemplars.json` post-validation) |

**Run commands (box, after `cp .env.template .env` + `export HF_TOKEN=… OPENAI_API_KEY=…`):**
```bash
# (MLP side already built per the existing runbook: prep → export → describe → supernodes)

# Fix #1 — corpus exemplars (the H100 job)
uv run python custom_automation/harvest_corpus_exemplars.py \
    --graphs-dir neuronpedia_neuron_graphs/ \
    --out custom_automation/np_data/mlp_exemplars.json

# eyeball the validation gate it prints (basketball / season neurons coherent?)
# then (post-validation) wire mlp_exemplars.json into serve_with_cards + generate_description.
```

## Ruthless verification checklist (for an independent agent — re-verify, do not trust)
Every one of these is a claim that, if wrong, silently corrupts the comparison.

1. **Capture == the traced quantity.** Assert the harvest's `down_proj`-input value for a
   `(layer,token,neuron)` equals `df_node['activation']` for the same node from a real trace.
   (A2.) The hook is `utils.py:84-88`; confirm it's the *input* to `down_proj`, not output.
2. **BOS/tokenization.** Confirm `prepend_bos=True` and `dataset_path=monology/pile-uncopyrighted`
   from the SLT SAE cfg (`SAE.from_pretrained(...).cfg.metadata`). Confirm the harvest prepends
   `<bos>` as token 0 of every 128-context.
3. **Window = 128, not 1024.** The SAE cfg `context_size=1024` is *training* context. The
   exemplar window is SAEDashboard `n_tokens_in_prompt=128`. Confirm the SLT exemplars the
   descriptions used are 128 tokens (count tokens in a raw hosted feature example).
4. **Budget = 4.7M, matched.** 36,864 × 128. Not 10M. Confirm the harvest scans exactly this.
5. **Banding == SAEDashboard.** 20 TOP (top-k) + 5 bands × 5, bins `linspace(0, max)`.
   Confirm the harvest's band edges and per-band counts match.
6. **Polarity.** `+` = max(act), `−` = max(−act). Confirm `−` neurons are kept + labeled
   MLP-only (no SLT analogue, since JumpReLU SAEs are ≥0), NOT dropped, NOT compared as if
   the SLT side had them.
7. **Dead/rare = fail loud.** Confirm dead neurons are OMITTED from the store (→ Llama
   default), with NO per-prompt fallback. Confirm coverage is reported.
8. **SLT side is single-layer transcoders** (the "cross layer transcoder" label is a default
   misnomer). Confirm the SLT exemplar config (corpus/window) matches gemmascope.
9. **Fix #2 (logit-weights) is labeled, not "faithful," and is Gemma-2-correct** (routes
   through `post_feedforward_layernorm` + final norm via `_effective_norm_weight`/`_norm_eps`;
   tied embeddings; softcap monotonic→skippable). Not yet implemented — flag if it gets built
   Llama-shaped.
10. **Validation gate ran.** The basketball (`michael-jordan`) + season (`saison`) neurons
    show coherent corpus contexts. If they look random, STOP — likely a norm/hook bug.

## Out of scope
- Fix #2 (logit-weights promote/demote) — speced, not yet built. Cheap, weights-only.
- Wiring `mlp_exemplars.json` into serve_with_cards / generate_description — do AFTER the
  validation gate passes (don't wire unvalidated data into the viewer).
- The top-K-threshold sweep (100/200/…/500) — deferred.
