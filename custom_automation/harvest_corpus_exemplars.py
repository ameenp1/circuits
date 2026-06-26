"""
harvest_corpus_exemplars.py — corpus raw-activation exemplars for traced Gemma-2-2b MLP neurons.

Fix #1 of CORPUS_EXEMPLARS_SPEC.md. Produces the SLT-matched, causally-faithful exemplar
store so MLP descriptions become comparable to the transcoder side (replaces the weak
per-prompt attribution snippet).

Lens matched to gemmascope (ALL verified — see the spec):
  corpus    monology/pile-uncopyrighted  (== SAE cfg.metadata.dataset_path)
  tokens    prepend_bos=True, 128-token contexts, 36,864 prompts (~4.7M tokens)
            (1024 in the SAE cfg is the *training* context, NOT the exemplar window)
  banding   20 TOP (deterministic top-k) + 5 quantile bands × 5, bins = linspace(0, max)
            equal-width over the activation range (SAEDashboard sequence_data_generator)
  capture   the down_proj-input forward hook — byte-identical to collect_neuron_acts
            (utils.py:84-88): the raw post-GeGLU neuron value, == df_node['activation'].
            (No unembed machinery needed; that part of collect_neuron_acts is for logits.)
  polarity  MLP-specific. '+' uses act, '−' uses -act (most-negative). Both reduced to a
            non-negative "max-activating" quantity, so act_min=0 and the frontend orange
            scale just works. '−' neurons have NO SLT analogue (JumpReLU SAEs are ≥0) — that
            is expected and fine; they are labeled MLP-only, not dropped.
  dead      FAIL LOUD. Neurons that never fire on the corpus are OMITTED (no fallback) so the
            viewer falls through to the obviously-wrong Llama default — the visible signal
            that this neuron got no real data. Coverage is reported.

Output: JSON keyed "L{layer}_N{neuron}_{pol}" in the frontend card schema — a drop-in for
serve_with_cards.py and generate_description.py.

Run on a GPU box (H100). Needs: HF_TOKEN (gated gemma) + `.env` (cp .env.template .env).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Traced-neuron union
# ---------------------------------------------------------------------------

def build_union(graphs_dir: Path) -> dict[int, list[int]]:
    """layer -> sorted unique neuron indices that appear in ANY graph (both polarities)."""
    per_layer: dict[int, set[int]] = {}
    for fp in sorted(graphs_dir.glob("graph_*.json")):
        g = json.loads(fp.read_text(encoding="utf-8"))
        for n in g.get("neurons", []):
            per_layer.setdefault(int(n["layer"]), set()).add(int(n["neuron"]))
    out = {layer: sorted(neur) for layer, neur in per_layer.items()}
    total = sum(len(v) for v in out.values())
    print(f"Traced-neuron union: {total} (layer,neuron) pairs across {len(out)} layers.")
    return out


# ---------------------------------------------------------------------------
# 2. Corpus -> BOS-prefixed 128-token contexts
# ---------------------------------------------------------------------------

def build_contexts(dataset: str, tokenizer, context: int, n_prompts: int) -> np.ndarray:
    """[n_prompts, context] int32 of token ids; each context starts with <bos> (prepend_bos)."""
    from datasets import load_dataset

    bos = tokenizer.bos_token_id
    ds = load_dataset(dataset, split="train", streaming=True)
    contexts = np.empty((n_prompts, context), dtype=np.int32)
    buf: list[int] = []
    filled = 0
    for ex in ds:
        ids = tokenizer(ex.get("text", ""), add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        while len(buf) >= context - 1 and filled < n_prompts:
            contexts[filled, 0] = bos
            contexts[filled, 1:] = buf[: context - 1]
            buf = buf[context - 1 :]
            filled += 1
        if filled >= n_prompts:
            break
    if filled < n_prompts:
        raise RuntimeError(f"corpus exhausted at {filled}/{n_prompts} contexts — pick a larger split")
    print(f"Built {filled} contexts × {context} tokens (BOS-prefixed) from {dataset}.")
    return contexts


# ---------------------------------------------------------------------------
# 3. Per-neuron streaming state: top-k heap + reservoir
# ---------------------------------------------------------------------------

class NeuronState:
    """Keeps top-k examples by activation + a reservoir of all firing examples (for quantiles)."""

    __slots__ = ("top", "res", "res_seen", "reservoir_size", "top_k")

    def __init__(self, top_k: int, reservoir_size: int):
        self.top: list[tuple[float, int, int, np.ndarray]] = []  # (ctx_max, ctx_idx, pos, acts_fp16)
        self.res: list[tuple[float, int, int, np.ndarray]] = []
        self.res_seen = 0
        self.reservoir_size = reservoir_size
        self.top_k = top_k

    def add(self, ctx_max: float, ctx_idx: int, pos: int, acts: np.ndarray) -> None:
        import heapq

        ex = (ctx_max, ctx_idx, pos, acts)
        if len(self.top) < self.top_k:
            heapq.heappush(self.top, ex)
        elif ctx_max > self.top[0][0]:
            heapq.heapreplace(self.top, ex)
        # reservoir sampling over ALL firing examples
        self.res_seen += 1
        if len(self.res) < self.reservoir_size:
            self.res.append(ex)
        else:
            j = random.randint(0, self.res_seen - 1)
            if j < self.reservoir_size:
                self.res[j] = ex


# ---------------------------------------------------------------------------
# 4. Card assembly (frontend schema)
# ---------------------------------------------------------------------------

def _example(tokenizer, contexts: np.ndarray, ex, buffer: int) -> dict:
    """Crop to peak ± buffer to MATCH the SLT exported exemplar window.

    The SLT descriptions were built from cropped windows (~35 tokens, measured from
    feature_descriptions_v2.json), NOT the 128-token forward prompt. We forward 128 for
    correct activations but export only peak ± buffer so both sides describe from the same
    amount of context.
    """
    _ctx_max, ctx_idx, pos, acts = ex
    ids = contexts[ctx_idx]
    lo = max(0, pos - buffer)
    hi = min(len(ids), pos + buffer + 1)
    win_ids = ids[lo:hi].tolist()
    win_acts = acts[lo:hi]
    return {
        "tokens": [tokenizer.decode([i]) for i in win_ids],
        "tokens_acts_list": [round(float(a), 4) for a in win_acts.astype(np.float32)],
        "train_token_ind": int(pos - lo),
    }


def build_card(tokenizer, contexts, st: NeuronState, n_quantiles: int, q_group: int, buffer: int) -> dict | None:
    if not st.top:
        return None  # dead -> omit (fail loud)
    max_act = max(e[0] for e in st.top)
    top_sorted = sorted(st.top, key=lambda e: -e[0])[: st.top_k]
    bands = [{"quantile_name": "TOP", "examples": [_example(tokenizer, contexts, e, buffer) for e in top_sorted]}]
    edges = np.linspace(0.0, max_act, n_quantiles + 1)
    for b in range(n_quantiles):
        lo, hi = float(edges[b]), float(edges[b + 1])
        in_band = [e for e in st.res if lo <= e[0] < hi] if b < n_quantiles - 1 \
            else [e for e in st.res if lo <= e[0] <= hi]
        if not in_band:
            continue
        sample = random.sample(in_band, min(q_group, len(in_band)))
        sample.sort(key=lambda e: -e[0])
        bands.append({
            "quantile_name": f"INTERVAL {lo:.3f}-{hi:.3f}",
            "examples": [_example(tokenizer, contexts, e, buffer) for e in sample],
        })
    return {"act_min": 0, "act_max": round(float(max_act), 4), "examples_quantiles": bands}


# ---------------------------------------------------------------------------
# 5. Harvest
# ---------------------------------------------------------------------------

def harvest(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    union = build_union(args.graphs_dir)
    collect_layers = sorted(union)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    contexts = build_contexts(args.dataset, tok, args.context, args.n_prompts)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # eager attention: Gemma-2 SDPA can diverge from the standard/dashboard forward because of
    # attention softcapping — match the dashboard by forcing eager.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, attn_implementation="eager"
    ).to(args.device).eval()

    # state[(layer, neuron, pol)] -> NeuronState
    state: dict[tuple[int, int, str], NeuronState] = {}
    for layer, neurons in union.items():
        for nidx in neurons:
            for pol in ("+", "-"):
                state[(layer, nidx, pol)] = NeuronState(args.top_k, args.reservoir)

    # per-layer column index of the traced neurons (gather only those columns)
    cols = {layer: torch.tensor(neurons, device=args.device) for layer, neurons in union.items()}

    # register the down_proj hooks ONCE (identical capture to collect_neuron_acts utils.py:84-88)
    cache: dict[int, torch.Tensor] = {}
    handles = []
    for layer in collect_layers:
        def hook(m, inp, out, _l=layer):  # input to down_proj == the neuron value
            cache[_l] = inp[0].detach()
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_hook(hook))

    n = len(contexts)
    try:
        for start in range(0, n, args.batch_size):
            batch_idx = list(range(start, min(start + args.batch_size, n)))
            ids = torch.tensor(contexts[batch_idx], dtype=torch.long, device=args.device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids))

            for layer in collect_layers:
                acts = cache[layer].index_select(2, cols[layer]).float()  # [B, T, n_traced]
                for pol, sign in (("+", 1.0), ("-", -1.0)):
                    pacts = acts * sign                                   # polarity activation (>0 = fires)
                    peak_src = pacts.clone()
                    peak_src[:, 0, :] = float("-inf")                     # mask BOS (attention-sink) from the peak
                    ctx_max, ctx_pos = peak_src.max(dim=1)               # [B, n_traced]
                    cm = ctx_max.cpu().numpy()
                    cp = ctx_pos.cpu().numpy()
                    pa = pacts.cpu().numpy().astype(np.float16)          # [B, T, n_traced]
                    for j, nidx in enumerate(union[layer]):
                        st = state[(layer, nidx, pol)]
                        for bi, ctx_i in enumerate(batch_idx):
                            v = float(cm[bi, j])
                            if v <= 0.0:                                 # didn't fire for this polarity
                                continue
                            # .copy() — pa[bi,:,j] is a VIEW that would pin the whole batch array
                            st.add(v, ctx_i, int(cp[bi, j]), pa[bi, :, j].copy())
            if (start // args.batch_size) % 20 == 0:
                print(f"  {min(start + args.batch_size, n)}/{n} contexts")
    finally:
        for h in handles:
            h.remove()

    # 6. assemble + coverage
    store: dict[str, dict] = {}
    cov = {"+": [0, 0], "-": [0, 0]}  # [with_exemplars, total]
    for (layer, nidx, pol), st in state.items():
        cov[pol][1] += 1
        card = build_card(tok, contexts, st, args.n_quantiles, args.quantile_group_size, args.buffer)
        if card is not None:
            store[f"L{layer}_N{nidx}_{pol}"] = card
            cov[pol][0] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(store), encoding="utf-8")
    print("\n=== coverage ===")
    for pol in ("+", "-"):
        got, tot = cov[pol]
        print(f"  {pol} polarity: {got}/{tot} neurons got exemplars ({tot - got} dead -> fail loud)")
    print(f"Wrote {len(store)} cards -> {args.out}")

    # 7. validation-gate hint: print a few neurons' top example for eyeballing
    print("\n=== validation gate — eyeball these (expect coherent contexts) ===")
    for key in list(store)[:4]:
        ex = store[key]["examples_quantiles"][0]["examples"][0]
        peak = ex["tokens"][ex["train_token_ind"]]
        ctx = "".join(ex["tokens"]).replace("\n", " ")
        print(f"  {key}  peak={peak!r}  act_max={store[key]['act_max']}\n    …{ctx[max(0, 0):220]}…")


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest corpus raw-activation exemplars for traced Gemma MLP neurons.")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="ADAG graphs (the traced-neuron union).")
    ap.add_argument("--out", type=Path, default=Path("custom_automation/np_data/mlp_exemplars.json"))
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--dataset", default="monology/pile-uncopyrighted")  # verified == SLT
    ap.add_argument("--n-prompts", type=int, default=36864)              # verified == SLT (×128 = 4.7M)
    ap.add_argument("--context", type=int, default=128)                  # forward prompt length (acts need full context)
    ap.add_argument("--buffer", type=int, default=16,                     # EXPORTED window = peak ± buffer (~33 tokens)
                    help="tokens each side of the peak in the EXPORTED exemplar. Match the SLT "
                         "exemplar length: measure tokens-per-context in feature_descriptions_v2.json "
                         "and set buffer≈(median-1)/2. ~16 ≈ the measured ~35-token SLT windows.")
    ap.add_argument("--top-k", type=int, default=20)                     # verified == SAEDashboard
    ap.add_argument("--n-quantiles", type=int, default=5)                # verified == SAEDashboard
    ap.add_argument("--quantile-group-size", type=int, default=5)        # verified == SAEDashboard
    ap.add_argument("--reservoir", type=int, default=400, help="per-neuron firing-example buffer for quantile sampling.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set — gemma-2-2b is gated and the load will 401.")
    harvest(args)


if __name__ == "__main__":
    main()
