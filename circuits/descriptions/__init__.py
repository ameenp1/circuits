"""Description generation and scoring v2 — no observatory dependency."""

from circuits.descriptions.api_backend import (
    AnthropicAttrExplainer,
    AnthropicAttrScorer,
    AnthropicContribExplainer,
    AnthropicContribScorer,
)
from circuits.descriptions.label import build_neuron_activation_records, label_clusters
from circuits.descriptions.scoring import compute_correlation_and_rsquared
from circuits.descriptions.types import (
    ActivationRecord,
    ActivationRecordWithContrib,
    ActSign,
    ExemplarDict,
    ExemplarResults,
    ExplanationResults,
    ScoredExplanation,
    SignedExplanations,
)
from circuits.descriptions.vllm_backend import FinetunedSimulator, VLLMExplainer

__all__ = [
    "ActivationRecord",
    "ActivationRecordWithContrib",
    "ActSign",
    "AnthropicAttrExplainer",
    "AnthropicAttrScorer",
    "AnthropicContribExplainer",
    "AnthropicContribScorer",
    "ExemplarDict",
    "ExemplarResults",
    "ExplanationResults",
    "FinetunedSimulator",
    "ScoredExplanation",
    "SignedExplanations",
    "VLLMExplainer",
    "build_neuron_activation_records",
    "compute_correlation_and_rsquared",
    "label_clusters",
]
