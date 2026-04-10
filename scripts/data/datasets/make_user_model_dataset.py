#!/usr/bin/env python
"""
Utility script for generating paired user modeling datasets in the
feature-circuit (SVA) format.

The resulting files mirror the JSONL structure consumed by the NAP/SVA
evaluation scripts: each line contains a tuple of `clean_*` and `patch_*`
fields representing counterfactual pairs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from circuits.utils.data_loading_utils import make_user_model_examples
from util.subject import get_subject_config, make_subject

SUPPORTED_DATASETS = {"obituary", "country", "gender", "occupation", "religion"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paired user modeling datasets in SVA format."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=sorted(SUPPORTED_DATASETS),
        help="Datasets to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/user_modeling",
        help="Directory where generated JSONL files will be written.",
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=500,
        help="Number of examples to sample for each split (train/valid/test).",
    )
    parser.add_argument(
        "--subject-hf-model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id used to instantiate the Subject wrapper.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling.",
    )
    parser.add_argument(
        "--dispatch",
        action="store_true",
        help="If set, dispatch the subject to the default device (mirrors make_subject).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    invalid = [ds for ds in args.datasets if ds not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(
            f"Unsupported dataset(s): {', '.join(invalid)}. "
            f"Supported values are: {', '.join(sorted(SUPPORTED_DATASETS))}."
        )

    random.seed(args.seed)

    subject_config = get_subject_config(args.subject_hf_model_id)
    subject = make_subject(subject_config, dispatch=args.dispatch)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        target_dir = args.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Generating paired dataset '{dataset}' "
            f"-> {target_dir} (per-split examples={args.examples_per_split})"
        )

        make_user_model_examples(
            subject=subject,
            split=dataset,
            mode="pair",
            train_examples=args.examples_per_split,
            test_examples=args.examples_per_split,
            output_path=str(target_dir),
        )


if __name__ == "__main__":
    main()
