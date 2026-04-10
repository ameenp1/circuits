"""Sensitivity analysis dataset: jailbreak prompt variants with systematic part substitutions."""

import json
import os
from pathlib import Path

DATA_PATH = Path(os.environ.get("CIRCUITS_RESULTS_DIR", "results")) / "case_studies/sensitivity_analysis/root.json"

with open(DATA_PATH) as f:
    root = json.load(f)

# Base prompt
base = root["base_prompt"]
base_asr = base["success_count"] / base["num_responses"]
prompts = [base["prompt"]]
labels = [f"base_asr{base_asr:.3f}"]

# Alternatives from each part
for part_idx, part in enumerate(root["parts"]):
    for alt_idx, alt in enumerate(part["alternatives"]):
        sub = alt["substituted_prompt"]
        asr = sub["success_count"] / sub["num_responses"]
        prompts.append(sub["prompt"])
        labels.append(f"part{part_idx}_alt{alt_idx}_asr{asr:.3f}")

seed_responses = ["[EMPTY]"] * len(prompts)
