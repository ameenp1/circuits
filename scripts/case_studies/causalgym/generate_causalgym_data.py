"""Generate CausalGym SVA dataset in the circuits JSONL format.

Uses the agr_sv_num_pp task from CausalGym (subject-verb agreement with PP intervener).
Template: "The {subject} {prep} the {object}" -> "is"/"are"/"was"/etc.

Paired examples differ only in the subject number (singular vs plural),
identical to the nounpp task from the feature_circuits data.

Usage:
    python generate_causalgym_data.py --output /path/to/causalgym_pp_train.json --n 1000
    python generate_causalgym_data.py --output /path/to/causalgym_pp_train.json --n 1000 --task agr_sv_num_subj-relc
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# CausalGym agr_sv_num_pp task definition
# ---------------------------------------------------------------------------

TASKS = {
    "agr_sv_num_pp": {
        "templates": ["The {subject} {prep} the {object}"],
        "label": "subject",
        "labels": {
            "singular": ["is", "was", "has"],
            "plural": ["are", "were", "have"],
        },
        "variables": {
            "subject": {
                "singular": [
                    "guard",
                    "doctor",
                    "farmer",
                    "author",
                    "officer",
                    "secretary",
                    "athlete",
                    "manager",
                    "senator",
                    "consultant",
                    "teacher",
                    "customer",
                    "pilot",
                    "executive",
                    "minister",
                    "actor",
                    "architect",
                    "clerk",
                ],
                "plural": [
                    "customers",
                    "authors",
                    "actors",
                    "senators",
                    "athletes",
                    "officers",
                    "teachers",
                    "ministers",
                    "managers",
                    "farmers",
                    "guards",
                    "pilots",
                    "secretaries",
                    "architects",
                    "doctors",
                    "consultants",
                    "executives",
                    "clerks",
                ],
            },
            "prep": ["in front of", "near", "to the side of", "next to", "across from", "behind"],
            "object": [
                "manager",
                "senator",
                "actor",
                "teacher",
                "doctor",
                "customer",
                "executive",
                "pilot",
                "farmer",
                "guard",
                "author",
                "officer",
                "architect",
                "secretary",
                "minister",
                "clerk",
                "athlete",
            ],
        },
    },
    "agr_sv_num_subj-relc": {
        "templates": ["The {subject} that {verbed} the {object}"],
        "label": "subject",
        "labels": {
            "singular": ["is", "was", "has"],
            "plural": ["are", "were", "have"],
        },
        "variables": {
            "subject": {
                "singular": [
                    "guard",
                    "doctor",
                    "farmer",
                    "author",
                    "officer",
                    "secretary",
                    "athlete",
                    "manager",
                    "senator",
                    "consultant",
                    "teacher",
                    "customer",
                    "pilot",
                    "executive",
                    "minister",
                    "actor",
                    "architect",
                    "clerk",
                ],
                "plural": [
                    "customers",
                    "authors",
                    "actors",
                    "senators",
                    "athletes",
                    "officers",
                    "teachers",
                    "ministers",
                    "managers",
                    "farmers",
                    "guards",
                    "pilots",
                    "secretaries",
                    "architects",
                    "doctors",
                    "consultants",
                    "executives",
                    "clerks",
                ],
            },
            "verbed": [
                "hated",
                "injured",
                "disguised",
                "ignored",
                "embarrassed",
                "admired",
                "liked",
                "hurt",
            ],
            "object": [
                "manager",
                "senator",
                "actor",
                "teacher",
                "doctor",
                "customer",
                "executive",
                "pilot",
                "farmer",
                "guard",
                "author",
                "officer",
                "architect",
                "secretary",
                "minister",
                "clerk",
                "athlete",
            ],
        },
    },
    "npi_any_subj-relc": {
        "templates": ["{det1} {np} that {rc_verb} {det2} {rc_obj} has {matrix_v}"],
        "label": "det1",
        "labels": {
            "any": ["any"],
            "some": ["some"],
        },
        "variables": {
            "det1": {
                "any": ["No"],
                "some": ["The"],
            },
            "np": [
                "farmer",
                "executive",
                "architect",
                "customer",
                "athlete",
                "officer",
                "pilot",
                "minister",
                "secretary",
                "author",
                "teacher",
                "manager",
                "journalist",
                "senator",
                "actor",
                "clerk",
                "consultant",
                "guard",
                "doctor",
            ],
            "det2": ["the", "no"],
            "rc_obj": [
                "chef",
                "teachers",
                "senator",
                "senators",
                "managers",
                "athlete",
                "officer",
                "assistant",
                "consultants",
                "architects",
                "farmers",
                "clerks",
                "customers",
                "guard",
                "farmer",
                "secretaries",
                "executives",
                "authors",
                "customer",
                "doctors",
                "journalists",
                "architect",
                "officers",
                "pilot",
                "pilots",
                "surgeon",
                "ministers",
                "dancer",
                "consultant",
                "executive",
                "guards",
                "minister",
                "teacher",
                "manager",
            ],
            "rc_verb": [
                "knew",
                "loved",
                "impressed",
                "helped",
                "praised",
                "hated",
                "respected",
                "contacted",
                "discussed",
                "admired",
                "liked",
            ],
            "matrix_v": [
                "completed",
                "landed",
                "crashed",
                "caught",
                "had",
                "failed",
                "refused",
                "passed",
                "burned",
                "missed",
                "read",
                "shown",
                "planted",
                "arrested",
                "known",
                "seen",
                "spent",
                "fired",
            ],
        },
    },
}


def generate_examples(
    task_name: str,
    n: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate paired examples from a CausalGym SVA task.

    Each example pairs a singular subject with a plural subject, keeping
    all other variables identical. The clean/patch distinction is which
    number the subject takes.
    """
    rng = random.Random(seed)
    task = TASKS[task_name]
    template = task["templates"][0]
    label_var = task["label"]
    labels = task["labels"]
    variables = task["variables"]

    # Get the two contrastive types
    types = list(labels.keys())  # ["singular", "plural"]
    assert len(types) == 2

    # Build shared variable lists (non-label variables)
    shared_vars = {k: v for k, v in variables.items() if k != label_var}

    # Build paired subject lists: zip singular[i] with plural[i]
    subj_lists = variables[label_var]  # dict: {"singular": [...], "plural": [...]}
    n_pairs = min(len(subj_lists[t]) for t in types)
    subject_pairs = list(zip(subj_lists[types[0]][:n_pairs], subj_lists[types[1]][:n_pairs]))

    examples = []
    for _ in range(n):
        # Pick a random subject pair
        subj_a, subj_b = rng.choice(subject_pairs)

        # Pick random values for shared variables
        shared_values = {k: rng.choice(v) for k, v in shared_vars.items()}

        # Pick a random answer pair (e.g., "is"/"are", "was"/"were", "has"/"have")
        answer_idx = rng.randrange(len(labels[types[0]]))
        answer_a = labels[types[0]][answer_idx]
        answer_b = labels[types[1]][answer_idx]

        # Fill templates
        fill_a = {label_var: subj_a, **shared_values}
        fill_b = {label_var: subj_b, **shared_values}
        prefix_a = template.format(**fill_a)
        prefix_b = template.format(**fill_b)

        examples.append(
            {
                "clean_prefix": prefix_a,
                "clean_answer": " " + answer_a,
                "patch_prefix": prefix_b,
                "patch_answer": " " + answer_b,
            }
        )

    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate CausalGym SVA data")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--n", type=int, default=1000, help="Number of examples")
    parser.add_argument(
        "--task",
        type=str,
        default="agr_sv_num_pp",
        choices=list(TASKS.keys()),
        help="CausalGym task name",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    examples = generate_examples(args.task, args.n, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")
    print(f"Example: {json.dumps(examples[0], indent=2)}")


if __name__ == "__main__":
    main()
