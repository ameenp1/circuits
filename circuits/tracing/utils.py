"""
Utility functions/classes for indicating circuit components and for collecting neuron activations
in prep for circuit tracing.
"""

from typing import NamedTuple

import torch
from circuits.tracing.grad import _effective_norm_weight, _norm_eps
from transformers import PreTrainedTokenizer
from util.gpu import gpu_mem_str
from util.parallel import TensorDict


class NeuronIdx(NamedTuple):
    layer: int
    token: int
    neuron: int


class Node(NamedTuple):
    layer: int
    token: int
    neuron: int
    activation: torch.Tensor | None = None
    final_attribution: torch.Tensor | None = None
    attr_map: torch.Tensor | None = None
    contrib_map: torch.Tensor | None = None


class Edge(NamedTuple):
    src: NeuronIdx
    tgt: NeuronIdx
    weight: torch.Tensor | None = None
    final_attribution: torch.Tensor | None = None


def _llama3_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension
    of estimator_X1D, then applying the same normalization to constant to X2D for all X1.
    """
    device = x_X1X2D.device
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


def collect_neuron_acts(
    model,
    tokenizer: PreTrainedTokenizer,
    cis: list[list[int]],
    attention_masks: list[list[int]] | torch.Tensor,
    collect_layers: list[int],
    keep_tokens: list[int] | None = None,
    device: str = "cuda",
    verbose: bool = False,
) -> tuple[TensorDict,]:
    """
    Collect MLP neuron activations using plain forward hooks.
    """
    input_ids = torch.tensor(cis, device=device)
    if isinstance(attention_masks, torch.Tensor):
        attn_mask = attention_masks.to(device)
    else:
        attn_mask = torch.tensor(attention_masks, device=device)
    # Storage for hook captures
    neurons_cache: dict[int, torch.Tensor] = {}
    resids_cache: dict[int, torch.Tensor] = {}

    # Register hooks
    handles = []
    for layer_idx in collect_layers:
        layer_module = model.model.layers[layer_idx]

        # neurons = input to down_proj (post-activation MLP hidden states)
        def _neuron_hook(module, input, output, _idx=layer_idx):
            # input is a tuple; first element is the tensor
            neurons_cache[_idx] = input[0].detach()

        handles.append(layer_module.mlp.down_proj.register_forward_hook(_neuron_hook))

        # resid = layer output (may be a tuple or plain tensor depending on transformers version)
        def _resid_hook(module, input, output, _idx=layer_idx):
            resids_cache[_idx] = (output[0] if isinstance(output, tuple) else output).detach()

        handles.append(layer_module.register_forward_hook(_resid_hook))

    # Forward pass
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attn_mask)

    # Remove hooks
    for h in handles:
        h.remove()

    # Build TensorDicts
    resids_LBTD = TensorDict({layer: resids_cache[layer].to(device) for layer in collect_layers})
    neurons_LBTI = TensorDict({layer: neurons_cache[layer].to(device) for layer in collect_layers})

    del neurons_cache, resids_cache

    # Key constants
    L = model.config.num_hidden_layers
    D = model.config.hidden_size
    B = resids_LBTD[0].size(0)
    Tf = resids_LBTD[0].size(1)

    # Derive output norm constants. Use the architecture-aware accessors so Gemma2's
    # (1 + weight) RMSNorm scale and `eps` attribute name are handled (no-op for Llama/Qwen).
    # `model.model.norm` is the raw HF norm here (the replacement model is reverted before
    # edge tracing collects activations).
    unembed_norm_weight_D = _effective_norm_weight(model.model.norm).detach().to(device)
    unembed_norm_variance_epsilon = _norm_eps(model.model.norm)
    output_norm_const_BTf11D = _llama3_layernorm_fn(
        resids_LBTD[L - 1].new_ones((B * Tf, 1, 1), device=device),
        resids_LBTD[L - 1][:, :].view(B * Tf, D),
        unembed_norm_weight_D,
        unembed_norm_variance_epsilon,
    ).view(B, Tf, 1, 1, D)

    if verbose:
        print(f"After collecting acts: {gpu_mem_str()}")
        print(
            "neurons_LBTI",
            neurons_LBTI[0].shape,
            neurons_LBTI[0].dtype,
            neurons_LBTI[0].device,
        )

    return (neurons_LBTI, resids_LBTD, cis, output_norm_const_BTf11D)
