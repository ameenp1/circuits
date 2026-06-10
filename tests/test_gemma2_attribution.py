"""
Correctness test for Gemma 2 2B MLP-neuron attribution ("prompt -> top N neurons").

The key invariant is *attribution completeness*: once every nonlinearity on the path is
linearized (RMSNorms via straight-through coefficients, attention through OV only, the MLP
gate detached, and Gemma2's final-logit softcapping disabled), the traced logit is a linear
function of the embeddings and the MLP neuron activations. By Euler's theorem for a
degree-1-homogeneous function, the summed attributions must equal the target logit:

    sum(MLP attributions) + sum(embed attributions) ~= goal logit

This holds for the plain stop-gradient mode (`use_relp_grad=False`), which carries no
Shapley redistribution. It is exactly the property that *breaks* on the pre-fix code,
because Gemma2's sandwich-norm `pre_feedforward_layernorm` / `post_feedforward_layernorm`
sit on the MLP path and were never linearized.

These tests require a GPU and download `google/gemma-2-2b`, so they skip without CUDA.
"""

import pytest
import torch

MODEL_ID = "google/gemma-2-2b"
PROMPT = "What is the capital of the state containing Dallas? Answer:"
TARGET = " Austin"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Gemma2 attribution tests require a CUDA GPU."
)


@pytest.fixture(scope="module")
def gemma_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # float32 for a numerically clean completeness check (bf16 is too lossy to assert on).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": "cuda:0"}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def test_attribution_completeness(gemma_model_and_tokenizer):
    """Summed MLP + embed attributions must reconstruct the traced logit (within tol)."""
    from circuits.tracing.attribution import _get_grad_attributions_from_logits
    from circuits.tracing.grad import revert_stop_nonlinear_grad, stop_nonlinear_grad
    from circuits.tracing.trace import prepare_cis

    model, tokenizer = gemma_model_and_tokenizer
    device = "cuda:0"

    cis, attention_masks, focus_tokens, _probs, keep_pos, _starts = prepare_cis(
        model,
        tokenizer,
        [PROMPT],
        seed_responses=[""],
        k=1,
        true_answers=[TARGET],
        use_chat_format=False,
    )
    input_ids = torch.tensor(cis, device=device)
    attn_mask = torch.tensor(attention_masks, device=device)
    last_pos = max(keep_pos)

    # Linearize the model (plain stop-grad mode → clean conservation), then attribute.
    model = stop_nonlinear_grad(model, use_relp_grad=False)
    try:
        layer_grad_attr, embed_grad_attr, goal_value, _acts, _embeds = (
            _get_grad_attributions_from_logits(
                model,
                input_ids,
                keep_tokens=keep_pos,
                focus_positions=[last_pos],
                focus_logits=[focus_tokens[0]],  # shape (B=1, P=1)
                attention_masks=attn_mask,
                ablation_mode="zero",
                disable_stop_grad=False,
            )
        )
    finally:
        model = revert_stop_nonlinear_grad(model)

    total_attr = layer_grad_attr.float().sum() + embed_grad_attr.float().sum()
    goal = goal_value.float().sum()

    rel_err = (total_attr - goal).abs() / goal.abs().clamp_min(1e-6)
    assert rel_err < 1e-2, (
        f"Attribution completeness violated: sum(attr)={total_attr.item():.4f} vs "
        f"goal={goal.item():.4f} (rel err {rel_err.item():.3%}). The MLP-path nonlinearities "
        "are not fully linearized for Gemma2."
    )

    # Final-logit softcapping must have been restored after revert.
    assert model.config.final_logit_softcapping is not None


def test_top_neurons_helper(gemma_model_and_tokenizer):
    """The thin helper returns a sorted, in-range top-N neuron list for Gemma 2 2B."""
    from scripts.circuit_prep.top_neurons import prompt_to_top_neurons

    model, tokenizer = gemma_model_and_tokenizer

    neurons = prompt_to_top_neurons(
        model, tokenizer, prompt=PROMPT, target=TARGET, top_n=20, use_chat_format=False
    )

    assert len(neurons) == 20
    n_layers = model.config.num_hidden_layers
    intermediate = model.config.intermediate_size
    for nrn in neurons:
        assert 0 <= nrn.layer < n_layers
        assert 0 <= nrn.neuron < intermediate
    # Sorted by descending absolute attribution.
    abs_attrs = [abs(n.attribution) for n in neurons]
    assert abs_attrs == sorted(abs_attrs, reverse=True)
