"""Data models and type aliases for description generation and scoring."""

from typing import Any, Literal

from circuits.analysis.cluster import NeuronId
from pydantic import BaseModel


class ActivationRecord(BaseModel):
    """A sequence of tokens and their corresponding activations for a single neuron."""

    tokens: list[str]
    activations: list[float]
    token_ids: list[int] | None = None


class ScoredExplanation(BaseModel):
    """An explanation with its simulation score and predicted activations."""

    explanation: str
    score: float | None  # Correlation score between simulated and true activations
    rsquared: float | None  # R-squared: variance explained after linear calibration
    # Per-exemplar predictions: list of {"tokens": [...], "true": [...], "predicted": [...]}
    predictions: list[dict[str, Any]] | None = None
    # Last-token-only scores (for contrib exemplars where only the appended logit matters)
    score_last_token: float | None = None
    rsquared_last_token: float | None = None


class ActivationRecordWithContrib(ActivationRecord):
    """Extended activation record that includes contribution data for output logits."""

    contrib_map: list[float] | None = None
    output_logits: list[int] | None = None


# Activation sign: "pos" or "neg"
ActSign = Literal["pos", "neg"]

# Type alias for explanation results with pos/neg subkeys
# Each neuron maps to {"pos": [...], "neg": [...]} or {"combined": [...]}
SignedExplanations = dict[str, list[str] | list[ScoredExplanation]]
ExplanationResults = dict[NeuronId, SignedExplanations]
# Exemplar dict contains: text (highlighted), tokens (list), activations (list of floats)
ExemplarDict = dict[str, Any]
ExemplarResults = dict[NeuronId, dict[str, list[ExemplarDict]]]
