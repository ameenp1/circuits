# tracing/

Circuit tracing via Cross-Layer Jacobian Attribution (CLJA). Identifies important MLP neurons and computes edge weights between them to build a circuit graph.

## Files

- **`trace.py`** — High-level tracing pipeline: prepares tokenized inputs, runs CLJA, and returns a `CircuitData` artifact containing node/edge DataFrames, tokenized inputs, target logits, and config metadata.
- **`clja.py`** — Core CLJA algorithm (`get_all_pairs_cl_ja_effects_with_attributions`). Orchestrates node selection, attribution/contribution computation, and edge tracing via jacobians.
- **`attribution.py`** — Gradient-based attribution helpers for scoring neuron importance (used by `clja.py` to filter neurons before edge computation).
- **`grad.py`** — Custom backward-pass wrappers (straight-through layernorm, stop-grad on attention/MLP gate, Shapley gradient approximations).
- **`utils.py`** — Data classes (`NeuronIdx`, `Node`, `Edge`) and activation collection utilities.
