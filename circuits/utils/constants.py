import os
from pathlib import Path

import torch as t
from circuits.evals.enap import ENAP
from circuits.evals.nap import NAP
from util.subject import gemma2_2b_config, llama31_8B_config, llama31_8B_instruct_config

RESULTS_DIR = Path(os.environ.get("CIRCUITS_RESULTS_DIR", "results"))

UNIT_1M = 1_000_000

N_LAYERS_MAPPING = {
    "google/gemma-2-2b": 26,
    "google/gemma-2-9b": 42,
    "meta-llama/Llama-3.1-8B": 32,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen3-32B": 64,
}

PARALLEL_ATTN_MAPPING = {
    "google/gemma-2-2b": False,
    "google/gemma-2-9b": False,
    "meta-llama/Llama-3.1-8B": False,
    "meta-llama/Llama-3.1-8B-Instruct": False,
    "Qwen/Qwen3-32B": False,
}

INCLUDE_EMBED_MAPPING = {
    "google/gemma-2-2b": False,
    "google/gemma-2-9b": False,
    "meta-llama/Llama-3.1-8B": False,
    "meta-llama/Llama-3.1-8B-Instruct": False,
    "Qwen/Qwen3-32B": False,
}

DTYPE_MAPPING = {
    "google/gemma-2-2b": t.bfloat16,
    "google/gemma-2-9b": t.bfloat16,
    "meta-llama/Llama-3.1-8B": t.bfloat16,
    "meta-llama/Llama-3.1-8B-Instruct": t.bfloat16,
    "Qwen/Qwen3-32B": t.bfloat16,
}

METHOD_MAPPING = {
    "nap": NAP,
    "enap": ENAP,
}

START_LAYER_MAPPING = {
    "google/gemma-2-2b": 0,
    "google/gemma-2-9b": 0,
    "meta-llama/Llama-3.1-8B": 0,
    "meta-llama/Llama-3.1-8B-Instruct": 0,
    "Qwen/Qwen3-32B": 0,
}

THRESHOLD_RANGE_MAPPING = {
    "google/gemma-2-2b": [0] + t.logspace(-4, 0, 15).tolist(),
    "google/gemma-2-9b": [0] + t.logspace(-4, 0, 15).tolist(),
    "meta-llama/Llama-3.1-8B": [0] + t.logspace(-6, 0, 30).tolist(),
    "meta-llama/Llama-3.1-8B-Instruct": [0] + t.logspace(-6, 0, 30).tolist(),
    "Qwen/Qwen3-32B": [0] + t.logspace(-6, 0, 30).tolist(),
}

DATASET_TASK_MAPPING = {
    "simple": "sva",
    "nounpp": "sva",
    "rc": "sva",
    "within_rc": "sva",
    "causalgym_pp": "sva",
    "causalgym_npi": "sva",
    "gender": "user_modeling",
    "gender_nopair": "user_modeling_nopair",
    "country": "user_modeling",
    "country_nopair": "user_modeling_nopair",
    "occupation": "user_modeling",
    "occupation_nopair": "user_modeling_nopair",
    "religion": "user_modeling",
    "religion_nopair": "user_modeling_nopair",
    "obituary": "user_modeling",
    "obituary_nopair": "user_modeling_nopair",
}

SUBJECT_CONFIG_MAPPING = {
    "meta-llama/Llama-3.1-8B": llama31_8B_config,
    "meta-llama/Llama-3.1-8B-Instruct": llama31_8B_instruct_config,
    "google/gemma-2-2b": gemma2_2b_config,
}
