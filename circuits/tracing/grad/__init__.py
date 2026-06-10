"""
Code for constructing the replacement model with various modifications to the original backward pass
in order to improve gradient-based attribution techniques.

Supports multiple model architectures (Llama, Qwen3, Gemma2) with a shared interface.
"""

import torch
from circuits.tracing.grad.gemma2 import Gemma2Attention, Gemma2MLP, Gemma2RMSNorm
from circuits.tracing.grad.llama import LlamaAttention, LlamaMLP, LlamaRMSNorm, repeat_kv
from circuits.tracing.grad.qwen3 import Qwen3Attention, Qwen3MLP, Qwen3RMSNorm
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Map from HF module types to their "kind" for dispatching
_NORM_TYPES: tuple[type[nn.Module], ...] = (LlamaRMSNorm, Qwen3RMSNorm, Gemma2RMSNorm)
_ATTN_TYPES: tuple[type[nn.Module], ...] = (LlamaAttention, Qwen3Attention, Gemma2Attention)
_MLP_TYPES: tuple[type[nn.Module], ...] = (LlamaMLP, Qwen3MLP, Gemma2MLP)

# Gemma2 RMSNorm scales by (1 + weight) rather than weight, and normalizes in float32.
# The extra feedforward layernorms below sit directly on the MLP path in Gemma2's
# sandwich-norm decoder layer and must also be linearized for correct attribution.
_GEMMA2_EXTRA_NORM_ATTRS: tuple[str, ...] = ("pre_feedforward_layernorm", "post_feedforward_layernorm")


def _effective_norm_weight(norm: nn.Module) -> torch.Tensor:
    """Return the effective RMSNorm scale: (1 + weight) for Gemma2, else weight."""
    if isinstance(norm, Gemma2RMSNorm):
        return 1.0 + norm.weight
    return norm.weight


def _norm_eps(norm: nn.Module) -> float:
    """RMSNorm epsilon — Llama/Qwen expose `variance_epsilon`, Gemma2 exposes `eps`."""
    if hasattr(norm, "variance_epsilon"):
        return norm.variance_epsilon
    return norm.eps


def _rms_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of
    estimator_X1D, then applying the same normalization to constant to X2D for all X1.

    We cast to float32 for numeric stability.
    """
    device = x_X1X2D.device
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


def remove_forward_hooks(main_module: nn.Module):
    """Remove all forward and pre-forward hooks from a module and its sub-modules."""
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys())
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(submodule._forward_pre_hooks.keys())
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)


class StopGradientModule(nn.Module):
    _stop_gradient = True


class StraightThroughRMSNorm(StopGradientModule):
    """
    Wrap an existing RMSNorm so that

      forward  = real RMSNorm value
      backward = identity wrt input  (dout/dx = I)
      weight   is frozen (requires_grad = False)

    Works with any RMSNorm module that has .weight and .variance_epsilon attributes.
    Gemma2RMSNorm scales by (1 + weight) instead of weight; `_effective_norm_weight`
    handles that so the linearized coefficient matches the real forward value.
    """

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm
        self.norm.weight.requires_grad_(False)
        self.weight = self.norm.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        coeff = _rms_layernorm_fn(
            x.new_ones(B * L, 1, 1),
            x.view(B * L, D),
            _effective_norm_weight(self.norm),
            _norm_eps(self.norm),
        ).detach()
        return x * coeff.permute(1, 0, 2).view(B, L, D)


def noqk_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention forward that detaches attention weights so gradient only flows through OV."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # Gemma2 applies tanh softcapping to the attention logits (attn_logit_softcapping=50.0);
    # apply it here so the (detached) attention map matches the real forward pass. No-op for
    # Llama/Qwen where the config attribute is absent or None.
    softcap = getattr(getattr(module, "config", None), "attn_logit_softcapping", None)
    if softcap is not None:
        attn_scores = attn_scores / softcap
        attn_scores = torch.tanh(attn_scores)
        attn_scores = attn_scores * softcap
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_scores = attn_scores + causal_mask

    attn_weights = (
        nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype).detach()
    )

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS["noqk"] = noqk_attention_forward


class NoQKGradAttention(StopGradientModule):
    """
    Wraps an existing attention module so that the soft-maxed attention
    map gets no gradient. Gradient only flows through the OV path.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.attn.config._attn_implementation = "noqk"

    def forward(self, *args, **kwargs):
        attn_output, attn_weights = self.attn(*args, **kwargs)
        return attn_output, attn_weights


class StopGradGateMLP(StopGradientModule):
    """
    Wrap an existing gated MLP so the activation-gate side
      act_fn( gate_proj(x) )
    is detached from the autograd graph.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.gate_proj.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj
        self.act_fn = self.mlp.act_fn
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_act = self.mlp.act_fn(self.mlp.gate_proj(x)).detach()
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(gate_act * up_branch)


class StopGradMLP(StopGradientModule):
    """
    Wrap an existing MLP and stop *all* gradient through it.
    """

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.down_proj = self.mlp.down_proj
        self.act_fn = self.mlp.act_fn
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.mlp(x)
        return out.detach()


class ShapleyElementwiseMult(torch.autograd.Function):
    """
    Shapley gradient for elementwise multiplication. This distributes the attribution equally to
    both branches, avoiding double-counting (which normal gradient would do).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor, use_half_rule: bool = True):
        ctx.save_for_backward(x, y)
        ctx.use_half_rule = use_half_rule
        return x * y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, y = ctx.saved_tensors
        return (
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * y,
            (0.5 if ctx.use_half_rule else 1.0) * grad_output * x,
            None,
        )


class RelPGradMLP(StopGradientModule):
    """
    RelP gradient for a gated MLP. Linearizes the activation gate by detaching it as a constant,
    then uses Shapley halving for the gate*up elementwise multiplication.
    """

    def __init__(self, mlp: nn.Module, use_half_rule: bool = True):
        super().__init__()
        self.mlp = mlp
        self.down_proj = self.mlp.down_proj
        self.gate_proj = self.mlp.gate_proj
        self.up_proj = self.mlp.up_proj
        self.use_half_rule = use_half_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_proj = self.mlp.gate_proj(x)
        # Linearize: treat act_fn(z)/z as a detached constant so gradient flows only
        # through the linear gate_proj. For SiLU this equals sigmoid(z).
        coeff = (self.mlp.act_fn(gate_proj) / (gate_proj + 1e-10)).detach()
        gate_act = gate_proj * coeff
        up_branch = self.mlp.up_proj(x)
        return self.mlp.down_proj(
            ShapleyElementwiseMult.apply(gate_act, up_branch, self.use_half_rule)
        )


def _wrap_extra_ff_norms(layer: nn.Module) -> None:
    """Linearize Gemma2's extra sandwich-norm feedforward LayerNorms, if present."""
    for attr in _GEMMA2_EXTRA_NORM_ATTRS:
        norm = getattr(layer, attr, None)
        if norm is not None and not isinstance(norm, StraightThroughRMSNorm):
            setattr(layer, attr, StraightThroughRMSNorm(norm))


def _unwrap_extra_ff_norms(layer: nn.Module) -> None:
    """Restore Gemma2's extra feedforward LayerNorms wrapped by `_wrap_extra_ff_norms`."""
    for attr in _GEMMA2_EXTRA_NORM_ATTRS:
        norm = getattr(layer, attr, None)
        if isinstance(norm, StraightThroughRMSNorm):
            setattr(layer, attr, norm.norm)


def _disable_final_softcap(model) -> None:
    """
    Disable Gemma2's final-logit softcapping during attribution so the logits are a linear
    function of the residual stream (required for attribution completeness). The original
    value is stashed on the model so `_restore_final_softcap` can put it back.
    """
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        model._adag_saved_final_softcap = softcap
        model.config.final_logit_softcapping = None


def _restore_final_softcap(model) -> None:
    """Restore final-logit softcapping disabled by `_disable_final_softcap`."""
    if getattr(model, "_adag_saved_final_softcap", None) is not None:
        model.config.final_logit_softcapping = model._adag_saved_final_softcap
        model._adag_saved_final_softcap = None


def stop_nonlinear_grad(
    model,
    use_relp_grad: bool = False,
    use_half_rule: bool = True,
):
    """
    Stop gradient for all non-linear layers in the model.

    - LayerNorms: linearized via StraightThroughRMSNorm (detached coefficients). For Gemma2's
      sandwich-norm layers, the extra pre/post feedforward norms are linearized too.
    - Attention: QK path detached, gradient flows only through OV
    - MLP: if use_relp_grad, activation gate is detached and Shapley halving applied;
           otherwise, entire gate branch is detached
    - Output: Gemma2 final-logit softcapping is disabled so logits stay linear in the residual.
    """
    _disable_final_softcap(model)
    model.model.norm = StraightThroughRMSNorm(model.model.norm)
    for layer in range(len(model.model.layers)):
        model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        _wrap_extra_ff_norms(model.model.layers[layer])
        model.model.layers[layer].self_attn = NoQKGradAttention(model.model.layers[layer].self_attn)
        if use_relp_grad:
            model.model.layers[layer].mlp = RelPGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        else:
            model.model.layers[layer].mlp = StopGradGateMLP(model.model.layers[layer].mlp)
    return model


def revert_stop_nonlinear_grad(model):
    """
    Revert stop gradient for all non-linear layers in the model.
    """
    _restore_final_softcap(model)
    model.model.norm = model.model.norm.norm
    for layer in range(len(model.model.layers)):
        model.model.layers[layer].input_layernorm = model.model.layers[layer].input_layernorm.norm
        model.model.layers[layer].post_attention_layernorm = model.model.layers[
            layer
        ].post_attention_layernorm.norm
        _unwrap_extra_ff_norms(model.model.layers[layer])
        model.model.layers[layer].self_attn = model.model.layers[layer].self_attn.attn
        model.model.layers[layer].mlp = model.model.layers[layer].mlp.mlp
    return model


def layerwise_stop_nonlinear_grad(
    model,
    start_layer: int,
    end_layer: int,
    use_relp_grad: bool = False,
    use_stop_grad_on_mlps: bool = True,
    use_half_rule: bool = True,
):
    _disable_final_softcap(model)
    model.model.norm = StraightThroughRMSNorm(model.model.norm)
    # for the start and the end layer, we don't do stop grad on mlp
    for layer in [start_layer, end_layer]:
        if layer < 0 or layer >= len(model.model.layers):
            continue
        model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        _wrap_extra_ff_norms(model.model.layers[layer])
        model.model.layers[layer].self_attn = NoQKGradAttention(model.model.layers[layer].self_attn)
        if use_relp_grad:
            model.model.layers[layer].mlp = RelPGradMLP(
                model.model.layers[layer].mlp, use_half_rule
            )
        else:
            model.model.layers[layer].mlp = StopGradGateMLP(model.model.layers[layer].mlp)

    # for layers in between, we do stop grad on mlp
    for layer in range(start_layer + 1, end_layer):
        if layer < 0 or layer >= len(model.model.layers):
            continue
        model.model.layers[layer].input_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].input_layernorm
        )
        model.model.layers[layer].post_attention_layernorm = StraightThroughRMSNorm(
            model.model.layers[layer].post_attention_layernorm
        )
        _wrap_extra_ff_norms(model.model.layers[layer])
        model.model.layers[layer].self_attn = NoQKGradAttention(model.model.layers[layer].self_attn)
        model.model.layers[layer].mlp = StopGradMLP(model.model.layers[layer].mlp)

    return model


def layerwise_revert_stop_nonlinear_grad(
    model,
    start_layer: int,
    end_layer: int,
):
    _restore_final_softcap(model)
    model.model.norm = model.model.norm.norm
    for layer in range(start_layer, end_layer + 1):
        if layer < 0 or layer >= len(model.model.layers):
            continue
        model.model.layers[layer].input_layernorm = model.model.layers[layer].input_layernorm.norm
        model.model.layers[layer].post_attention_layernorm = model.model.layers[
            layer
        ].post_attention_layernorm.norm
        _unwrap_extra_ff_norms(model.model.layers[layer])
        model.model.layers[layer].self_attn = model.model.layers[layer].self_attn.attn
        model.model.layers[layer].mlp = model.model.layers[layer].mlp.mlp
    return model


# Backward-compatible aliases
StraightThroughLlamaRMSNorm = StraightThroughRMSNorm
stop_nonlinear_grad_for_llama = stop_nonlinear_grad
revert_stop_nonlinear_grad_for_llama = revert_stop_nonlinear_grad
layerwise_stop_nonlinear_grad_for_llama = layerwise_stop_nonlinear_grad
layerwise_revert_stop_nonlinear_grad_for_llama = layerwise_revert_stop_nonlinear_grad
