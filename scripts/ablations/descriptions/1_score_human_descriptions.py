#!/usr/bin/env python3
"""Score hardcoded human descriptions of manual circuit clusters using the v2 system.

Attr scoring: VLLM finetuned simulator (default) or Anthropic API
Contrib scoring: Haiku API (claude-haiku-4-5-20251001)

Usage:
    python scripts/ablations/descriptions/1_score_human_descriptions.py
    python scripts/ablations/descriptions/1_score_human_descriptions.py --attr-scorer api
"""

import argparse
import json
import logging
from datetime import datetime

from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId
from circuits.descriptions.api_backend import AnthropicAttrScorer, AnthropicContribScorer
from circuits.descriptions.exemplars import build_contrib_minibatch
from circuits.descriptions.label import build_neuron_activation_records
from circuits.descriptions.types import ActivationRecord, ScoredExplanation
from circuits.descriptions.vllm_backend import FinetunedSimulator, score_attr_explanations
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/capitals_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals/human_descriptions"

# Manual clusters
manual_clusters_raw: dict[str, list[tuple[int, int, str]]] = {
    "capital": [
        (3, 14335, "-"),
        (4, 13489, "-"),
        (19, 2520, "-"),
        (20, 3520, "+"),
        (16, 13326, "-"),
        (13, 4038, "+"),
    ],
    "state": [
        (0, 9296, "-"),
        (2, 5246, "+"),
        (4, 604, "-"),
        (19, 4478, "+"),
        (21, 5790, "-"),
        (21, 12118, "-"),
    ],
    "Dallas": [(0, 12136, "-"), (5, 8659, "+")],
    "Texas": [(6, 10965, "-"), (21, 3093, "+")],
    "say a capital": [
        (23, 8079, "-"),
        (21, 4924, "-"),
        (23, 2709, "-"),
        (17, 3663, "+"),
    ],
    "say Austin": [(30, 8371, "+"), (31, 4876, "+"), (31, 6705, "+")],
    "location": [
        (5, 7404, "-"),
        (23, 9355, "+"),
        (19, 4982, "-"),
        (17, 1276, "+"),
        (17, 9922, "-"),
    ],
    "say a location": [
        (27, 4319, "+"),
        (29, 1846, "+"),
        (29, 8838, "-"),
        (23, 2825, "+"),
        (22, 10506, "-"),
        (28, 2580, "-"),
        (28, 8928, "+"),
        (30, 13458, "+"),
        (30, 1644, "-"),
        (30, 3283, "+"),
        (29, 5785, "-"),
        (25, 13461, "+"),
        (24, 8483, "-"),
    ],
}

# Human descriptions from annotation UI session (2026-03-06)
# Each cluster has an attr description and a contrib description.
HUMAN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Texas": {
        "attr": ("primarily {{Dallas}} and some other terms only in Texas-specific context"),
        "contrib": ("promotes Texas and cities in Texas, does nothing in other cases"),
    },
    "say Austin": {
        "attr": (
            "{{capital}} of the {{state}} containing city "
            "(e.g. {{Charleston}}, {{Memphis}}, {{Portland}}), "
            "firing most strongly on the name of the city, "
            "as well as on the token {{Answer}} when responding"
        ),
        "contrib": (
            "strongly promotes full state name, and more weakly promotes full city names. "
            "does not promote abbreviations of states, partial city names, "
            "and weakly demotes function words like {{The}}. "
            "an exception: strongly demotes {{Austin}}"
        ),
    },
    "Dallas": {
        "attr": "only fires on {{Dallas}} and {{Idaho}}",
        "contrib": (
            "promotes city and state names only in the context of Dallas and Idaho Falls, "
            "does nothing on any other inputs"
        ),
    },
    "location": {
        "attr": (
            "extremely strongly fires on {{capital}}, "
            "as well as a little bit on {{state}} and some city names"
        ),
        "contrib": (
            "strongly promotes correct capital city names as well as partial prefixes of "
            "correct names (e.g. both {{Sacramento}} and {{Sac}}), "
            "and weakly promotes 2-letter state abbreviations and state names, "
            "otherwise doing nothing."
        ),
    },
    "capital": {
        "attr": "{{capital}}",
        "contrib": (
            "strongly promotes correct capital city responses to the question, "
            "including partial responses or forms with different capitalisation, "
            "and nothing in other cases"
        ),
    },
    "say a capital": {
        "attr": (
            "extremely strong firing on {{capital}} and also a little bit of firing "
            "on {{Answer}}, but nothing on other words"
        ),
        "contrib": (
            "promotes correct capital city responses to the questions and also "
            "partial responses which are correct, otherwise does nothing"
        ),
    },
    "state": {
        "attr": "{{state}}",
        "contrib": (
            "promotes US states the strongest and US city locations pretty strongly too "
            "as well as partial responses to those, some grammatical terms like {{The}}, "
            "but not other locations or terms"
        ),
    },
    "say a location": {
        "attr": (
            "fires most strongly on US city names "
            "(e.g. {{Burlington}}, {{Savannah}}), "
            'and some firing on "{{capital}} of the {{state}} containing", '
            "as well as the token {{Answer}}"
        ),
        "contrib": (
            "strongly promotes US state names and two-letter state abbreviations "
            "and generic prefixes for such names like {{New}} and {{North}}, "
            "and somewhat promotes correct US capital city names which are full words, "
            "and does not promote partial responses or grammatical terms"
        ),
    },
}


def build_manual_clusters_map() -> dict[NeuronId, str]:
    mapping: dict[NeuronId, str] = {}
    for cluster_name, neurons in manual_clusters_raw.items():
        for layer, neuron, polarity in neurons:
            mapping[NeuronId(layer=layer, token=-1, neuron=neuron, polarity=polarity)] = (
                cluster_name
            )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attr-scorer",
        choices=["vllm", "api"],
        default="vllm",
        help="Which attr scorer to use: vllm (finetuned simulator) or api (Anthropic API)",
    )
    parser.add_argument(
        "--attr-api-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for API attr scorer",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Load circuit and cluster ---
    logger.info("Loading circuit from %s", CIRCUIT_PICKLE)
    circuit = Circuit.load_from_pickle(str(CIRCUIT_PICKLE))

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    logger.info("Clustering with manual clusters...")
    manual_map = build_manual_clusters_map()
    circuit.cluster(
        n_clusters=0,
        manual_clusters=manual_map,
        include_attr_contrib=True,
        get_desc=False,
        verbose=True,
    )

    # Prepare clustered df
    df_clustered, cluster_to_neurons = circuit._prepare_clustered_df_for_labelling()
    cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})
    name_to_cluster_id = {v: k for k, v in cluster_id_to_name.items()}

    # Build activation records from the clustered df
    neuron_data, neuron_ci_mapping = build_neuron_activation_records(
        df_clustered,
        circuit.cis,
        tokenizer,
        circuit.target_logits,
        num_layers=None,  # cluster IDs are used as layer, no filtering needed
    )

    # Map cluster names -> NeuronId keys in neuron_data
    cluster_neuron_ids: dict[str, NeuronId] = {}
    for nid in neuron_data:
        cid = nid.layer
        name = cluster_id_to_name.get(cid, "")
        if name in HUMAN_DESCRIPTIONS:
            cluster_neuron_ids[name] = nid

    logger.info(
        "Matched %d/%d human descriptions to cluster neuron IDs",
        len(cluster_neuron_ids),
        len(HUMAN_DESCRIPTIONS),
    )
    for name, nid in sorted(cluster_neuron_ids.items()):
        logger.info("  %s -> %s (%d records)", name, nid, len(neuron_data[nid]))

    # --- PHASE 1: Score attr descriptions ---
    attr_scores: dict[str, list[ScoredExplanation]] = {}

    if args.attr_scorer == "api":
        logger.info("PHASE 1: Scoring attr descriptions via API (%s)", args.attr_api_model)
        api_attr_scorer = AnthropicAttrScorer(model_name=args.attr_api_model)

        for cluster_name, nid in sorted(cluster_neuron_ids.items()):
            desc = HUMAN_DESCRIPTIONS[cluster_name]["attr"]
            records = neuron_data[nid]

            raw_records = [
                ActivationRecord(
                    tokens=rec.tokens, token_ids=rec.token_ids, activations=rec.activations
                )
                for rec in records
            ]

            try:
                scored = api_attr_scorer.score_explanations(
                    [desc],
                    raw_records,
                    "pos",
                    keep_only_top_predictions=False,
                )
                attr_scores[cluster_name] = scored
                for se in scored:
                    logger.info(
                        "  [attr] %s: score=%.3f r2=%.3f | %s",
                        cluster_name,
                        se.score if se.score is not None else float("nan"),
                        se.rsquared if se.rsquared is not None else float("nan"),
                        se.explanation,
                    )
            except Exception as e:
                logger.error("Error scoring attr for %s: %s", cluster_name, e, exc_info=True)
                attr_scores[cluster_name] = [
                    ScoredExplanation(explanation=desc, score=None, rsquared=None)
                ]
    else:
        import gc

        import torch

        logger.info("PHASE 1: Scoring attr descriptions via VLLM simulator")
        simulator = FinetunedSimulator(model_name="Transluce/llama_8b_simulator", gpu_idx=0)

        for cluster_name, nid in sorted(cluster_neuron_ids.items()):
            desc = HUMAN_DESCRIPTIONS[cluster_name]["attr"]
            records = neuron_data[nid]

            raw_records = [
                ActivationRecord(
                    tokens=rec.tokens, token_ids=rec.token_ids, activations=rec.activations
                )
                for rec in records
            ]

            # Score as "pos" sign (human descriptions describe what activates the cluster)
            try:
                scored = score_attr_explanations(
                    simulator,
                    [desc],
                    raw_records,
                    "pos",
                    use_raw_activations=True,
                    keep_only_top_predictions=False,
                )
                attr_scores[cluster_name] = scored
                for se in scored:
                    logger.info(
                        "  [attr] %s: score=%.3f r2=%.3f | %s",
                        cluster_name,
                        se.score if se.score is not None else float("nan"),
                        se.rsquared if se.rsquared is not None else float("nan"),
                        se.explanation,
                    )
            except Exception as e:
                logger.error("Error scoring attr for %s: %s", cluster_name, e, exc_info=True)
                attr_scores[cluster_name] = [
                    ScoredExplanation(explanation=desc, score=None, rsquared=None)
                ]

        # Unload simulator
        simulator.cleanup()
        del simulator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- PHASE 2: Score contrib descriptions via Haiku API ---
    logger.info("PHASE 2: Scoring contrib descriptions via Haiku API")
    contrib_scorer = AnthropicContribScorer(model_name="claude-haiku-4-5-20251001")

    contrib_scores: dict[str, list[ScoredExplanation]] = {}
    for cluster_name, nid in sorted(cluster_neuron_ids.items()):
        desc = HUMAN_DESCRIPTIONS[cluster_name]["contrib"]
        records = neuron_data[nid]

        has_contrib = any(
            rec.contrib_map is not None and rec.output_logits is not None for rec in records
        )
        if not has_contrib:
            logger.warning("No contrib data for %s, skipping", cluster_name)
            contrib_scores[cluster_name] = [
                ScoredExplanation(explanation=desc, score=None, rsquared=None)
            ]
            continue

        try:
            minibatch_data = build_contrib_minibatch(records, tokenizer, max_prompts=20)
            if not minibatch_data:
                logger.warning("Empty minibatch for %s, skipping", cluster_name)
                contrib_scores[cluster_name] = [
                    ScoredExplanation(explanation=desc, score=None, rsquared=None)
                ]
                continue

            scored = contrib_scorer.score_explanations(
                [desc],
                minibatch_data,
                keep_only_top_predictions=False,
            )
            contrib_scores[cluster_name] = scored
            for se in scored:
                logger.info(
                    "  [contrib] %s: score=%.3f r2=%.3f | %s",
                    cluster_name,
                    se.score if se.score is not None else float("nan"),
                    se.rsquared if se.rsquared is not None else float("nan"),
                    se.explanation,
                )
        except Exception as e:
            logger.error("Error scoring contrib for %s: %s", cluster_name, e, exc_info=True)
            contrib_scores[cluster_name] = [
                ScoredExplanation(explanation=desc, score=None, rsquared=None)
            ]

    # --- Save results ---
    scorer_suffix = f"_{args.attr_scorer}" if args.attr_scorer != "vllm" else ""
    output_file = OUTPUT_DIR / f"human_descriptions{scorer_suffix}_{timestamp}.json"
    attr_scorer_name = (
        args.attr_api_model if args.attr_scorer == "api" else "Transluce/llama_8b_simulator"
    )
    output_data: dict = {
        "metadata": {
            "circuit_pickle": str(CIRCUIT_PICKLE),
            "timestamp": timestamp,
            "backend": "descriptions",
            "attr_scorer": args.attr_scorer,
            "attr_simulator": attr_scorer_name,
            "contrib_scorer": "claude-haiku-4-5-20251001",
            "source": "human annotation UI (2026-03-06)",
        },
        "results": {},
    }

    for cluster_name in sorted(HUMAN_DESCRIPTIONS.keys()):
        attr = attr_scores.get(cluster_name, [])
        contrib = contrib_scores.get(cluster_name, [])
        output_data["results"][cluster_name] = {
            "attr_description": HUMAN_DESCRIPTIONS[cluster_name]["attr"],
            "contrib_description": HUMAN_DESCRIPTIONS[cluster_name]["contrib"],
            "attr_score": attr[0].score if attr else None,
            "attr_rsquared": attr[0].rsquared if attr else None,
            "contrib_score": contrib[0].score if contrib else None,
            "contrib_rsquared": contrib[0].rsquared if contrib else None,
            "attr_predictions": (
                [p for p in attr[0].predictions] if attr and attr[0].predictions else None
            ),
            "contrib_predictions": (
                [p for p in contrib[0].predictions] if contrib and contrib[0].predictions else None
            ),
        }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info("Results saved to %s", output_file)

    # Print summary
    print("\n" + "=" * 80)
    print("HUMAN DESCRIPTION SCORES")
    print("=" * 80)
    print(
        f"{'Cluster':<20} {'Attr Score':>12} {'Attr R²':>10} {'Contrib Score':>14} {'Contrib R²':>12}"
    )
    print("-" * 70)
    for cluster_name in sorted(HUMAN_DESCRIPTIONS.keys()):
        r = output_data["results"][cluster_name]
        a_s = f"{r['attr_score']:.3f}" if r["attr_score"] is not None else "N/A"
        a_r = f"{r['attr_rsquared']:.3f}" if r["attr_rsquared"] is not None else "N/A"
        c_s = f"{r['contrib_score']:.3f}" if r["contrib_score"] is not None else "N/A"
        c_r = f"{r['contrib_rsquared']:.3f}" if r["contrib_rsquared"] is not None else "N/A"
        print(f"{cluster_name:<20} {a_s:>12} {a_r:>10} {c_s:>14} {c_r:>12}")


if __name__ == "__main__":
    main()
