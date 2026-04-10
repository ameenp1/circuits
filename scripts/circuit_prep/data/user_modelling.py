"""Wikipedia user modelling dataset: predict user attributes from Wikipedia browsing prompts.

Uses get_dataset(subject) pattern since data loading requires the Subject (tokenizer).
Configure the split via the SPLIT environment variable (default: "country").
Supported splits: country, gender, occupation, religion.
"""

from __future__ import annotations

import os

from user_modeling.datasets.wikipedia import (
    WIKIPEDIA_PROMPT_RESPONSE,
    get_wikipedia_dataset_by_split,
)
from util.subject import Subject

NUM_EXAMPLES = 128


def get_dataset(model, tokenizer) -> tuple[list[str], list[str], list[str]]:
    """Load Wikipedia user modelling dataset for the configured split."""
    from circuits.utils.constants import SUBJECT_CONFIG_MAPPING

    model_id = getattr(model.config, "_name_or_path", "") or getattr(model, "name_or_path", "")
    lm_config = SUBJECT_CONFIG_MAPPING.get(model_id)
    if lm_config is None:
        raise ValueError(
            f"No Subject config for model '{model_id}'. Add it to SUBJECT_CONFIG_MAPPING."
        )
    subject = Subject(lm_config, preloaded_model=model, preloaded_tokenizer=tokenizer)

    split = os.environ.get("SPLIT", "country")
    seed = int(os.environ.get("SEED", "0"))

    datasets = get_wikipedia_dataset_by_split(
        subject,
        question_types=[split],
        only_good=True,
        seed=seed,
    )
    combined = (
        datasets["train"].datapoints + datasets["valid"].datapoints + datasets["test"].datapoints
    )
    if len(combined) < NUM_EXAMPLES:
        raise ValueError(
            f"Not enough datapoints for split '{split}' (needed {NUM_EXAMPLES}, "
            f"got {len(combined)})."
        )

    selected = combined[:NUM_EXAMPLES]

    prompts = [dp.conversation[0]["content"] for dp in selected]
    labels = [dp.latent_attributes[split][0] for dp in selected]
    seed_responses = [f"{WIKIPEDIA_PROMPT_RESPONSE} {split} ="] * NUM_EXAMPLES
    return prompts, seed_responses, labels
