"""
Thin "prompt -> top-N MLP neurons" helper for ADAG circuit tracing.

Given a prompt (and an optional target token), this returns the MLP neuron activations with
the largest attribution to the model's next-token prediction, using the fast attribution
path (`return_only_important_neurons=True`) — i.e. *without* edge tracing or clustering.

This is the lightweight "prompt -> top N neurons" step. It is model-agnostic and works for
any architecture the grad wrappers support (Llama, Qwen3, Gemma2).

Example:
    uv run python scripts/circuit_prep/top_neurons.py \
        --model-id google/gemma-2-2b \
        --prompt "What is the capital of the state containing Dallas? Answer:" \
        --target " Austin" --top-n 20
"""

import argparse
from dataclasses import dataclass

import torch
from circuits.tracing.clja import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions
from circuits.tracing.trace import prepare_cis
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class TopNeuron:
    """A single MLP neuron's attribution to the traced prediction."""

    layer: int
    token: int
    neuron: int
    attribution: float


def prompt_to_top_neurons(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    target: str | None = None,
    top_n: int = 20,
    k: int = 1,
    device: str | None = None,
    use_chat_format: bool = False,
    verbose: bool = False,
) -> list[TopNeuron]:
    """
    Return the `top_n` MLP neurons (by absolute attribution) for `prompt`.

    Args:
        model, tokenizer: a loaded HF causal LM and its tokenizer.
        prompt: the input text.
        target: optional target token string (e.g. " Austin"); its first token is the logit
            traced to. If None, the model's own top-1 prediction is used.
        top_n: number of neurons to return, sorted by |attribution| descending.
        k: number of top logits to trace when `target` is None.
        device: device string; defaults to the model's device.
        use_chat_format: wrap the prompt with the tokenizer's chat template (default False,
            appropriate for base models such as google/gemma-2-2b).

    Returns:
        A list of `TopNeuron(layer, token, neuron, attribution)`, longest-attribution first.
    """
    device = device or str(next(model.parameters()).device)
    true_answers = [target] if target is not None else None

    cis, attention_masks, focus_tokens, _focus_probs, keep_pos, _starts = prepare_cis(
        model,
        tokenizer,
        [prompt],
        seed_responses=[""],
        k=1 if target is not None else k,
        true_answers=true_answers,
        use_chat_format=use_chat_format,
        verbose=verbose,
    )

    config = ADAGConfig(
        device=device,
        use_relp_grad=True,
        return_only_important_neurons=True,
        verbose=verbose,
    )
    mlp_attr, _embed_attr, _mlp_acts, _embed_acts = get_all_pairs_cl_ja_effects_with_attributions(
        model=model,
        tokenizer=tokenizer,
        cis=cis,
        config=config,
        attention_masks=attention_masks,
        focus_logits=focus_tokens,
        src_tokens=keep_pos,
        tgt_tokens=[max(keep_pos)],
    )

    # mlp_attr: (L, B, T, D_ff, 1); single prompt (B=1), focus-logit dim already summed.
    attr_LTI = mlp_attr[:, 0, :, :, 0]  # (L, T, D_ff)
    L, T, D_ff = attr_LTI.shape

    # Restrict to kept (non-padding) token positions before ranking.
    keep_mask = torch.zeros(T, dtype=torch.bool, device=attr_LTI.device)
    keep_mask[torch.tensor(keep_pos, device=attr_LTI.device)] = True
    attr_LTI = torch.where(keep_mask[None, :, None], attr_LTI, torch.zeros_like(attr_LTI))

    flat = attr_LTI.reshape(-1)
    n = min(top_n, int((flat != 0).sum().item()))
    if n == 0:
        return []
    _, top_idx = torch.topk(flat.abs(), n)

    results: list[TopNeuron] = []
    for idx in top_idx.tolist():
        layer = idx // (T * D_ff)
        token = (idx % (T * D_ff)) // D_ff
        neuron = idx % D_ff
        results.append(
            TopNeuron(layer=layer, token=token, neuron=neuron, attribution=flat[idx].item())
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prompt -> top-N MLP neurons (fast attribution).")
    parser.add_argument("--model-id", default="google/gemma-2-2b", help="HuggingFace model ID.")
    parser.add_argument("--prompt", required=True, help="Input prompt text.")
    parser.add_argument(
        "--target",
        default=None,
        help="Target token string to trace to (e.g. ' Austin'). Default: model's top-1 token.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of neurons to return.")
    parser.add_argument("--k", type=int, default=1, help="Top-k logits to trace when no target.")
    parser.add_argument("--device", default=None, help="Device (default: model's device).")
    parser.add_argument(
        "--use-chat-format",
        action="store_true",
        help="Wrap the prompt with the tokenizer's chat template (for instruct models).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_id} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    neurons = prompt_to_top_neurons(
        model,
        tokenizer,
        prompt=args.prompt,
        target=args.target,
        top_n=args.top_n,
        k=args.k,
        device=device,
        use_chat_format=args.use_chat_format,
        verbose=args.verbose,
    )

    print(f"\nTop {len(neurons)} MLP neurons for prompt:\n  {args.prompt!r}")
    if args.target is not None:
        print(f"  (traced to target {args.target!r})")
    print(f"\n{'rank':>4}  {'layer':>5}  {'token':>5}  {'neuron':>6}  {'attribution':>12}")
    for rank, nrn in enumerate(neurons, 1):
        print(f"{rank:>4}  {nrn.layer:>5}  {nrn.token:>5}  {nrn.neuron:>6}  {nrn.attribution:>12.6f}")


if __name__ == "__main__":
    _main()
