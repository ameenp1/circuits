"""Gemma2-specific model imports for circuit tracing."""

from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2MLP,
    Gemma2RMSNorm,
)

# Note: Gemma2's GQA `repeat_kv` is intentionally not imported — the noqk attention path
# reuses the Llama `repeat_kv` (identical implementation) and importing it here would add a
# version-fragile dependency for no benefit.
__all__ = ["Gemma2Attention", "Gemma2MLP", "Gemma2RMSNorm"]
