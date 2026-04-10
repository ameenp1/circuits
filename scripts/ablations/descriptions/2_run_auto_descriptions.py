"""Generate + score automated v2 descriptions for manual capitals clusters.

Uses quantile thresholding, min_highlights=1, 20 explanation samples.
Saves results in the same JSON format as run_explanations_v2.py so that
3_table_human_vs_auto.py can load them via load_auto().

Runs two configurations:
1. VLLM attr backend (default, requires GPU)
2. API attr backend (uses Anthropic API, no GPU needed for attr)

Usage:
    python scripts/ablations/descriptions/2_run_auto_descriptions.py
    python scripts/ablations/descriptions/2_run_auto_descriptions.py --backend api
    python scripts/ablations/descriptions/2_run_auto_descriptions.py --backend both
    sbatch scripts/ablations/descriptions/2_run_auto_descriptions.sbatch
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Any

from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import NeuronId
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/capitals_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals/human_vs_auto"

# Same manual clusters used in score_human_descriptions.py
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


def build_manual_clusters_map() -> dict[NeuronId, str]:
    mapping: dict[NeuronId, str] = {}
    for cluster_name, neurons in manual_clusters_raw.items():
        for layer, neuron, polarity in neurons:
            mapping[NeuronId(layer=layer, token=-1, neuron=neuron, polarity=polarity)] = (
                cluster_name
            )
    return mapping


def run_descriptions(
    circuit: Circuit,
    attr_backend: str = "vllm",
    attr_api_model_name: str = "claude-haiku-4-5-20251001",
) -> tuple[str, dict[str, Any]]:
    """Run description generation + scoring with the given attr backend.

    Returns (suffix, output_data) where suffix identifies the backend.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "api" if attr_backend == "api" else "vllm"

    logger.info(
        "Generating + scoring v2 descriptions (quantile, mh=1, n=20, attr_backend=%s)...",
        attr_backend,
    )
    result = circuit.label_clusters_simulator_v2(
        score_explanations=True,
        num_expl_samples=20,
        min_highlights=1,
        threshold_mode="quantile",
        contrib_model_name="claude-haiku-4-5-20251001",
        verbose=True,
        attr_backend=attr_backend,
        attr_api_model_name=attr_api_model_name,
    )
    attr_results, contrib_results, attr_exemplars, contrib_exemplars, cluster_to_neurons = result

    # Read cluster_id_to_name AFTER label_clusters_simulator_v2, which populates it
    cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})

    # Print summary
    print("\n" + "=" * 80)
    print(f"AUTO V2 DESCRIPTION SCORES (quantile, mh=1, n=20, attr_backend={attr_backend})")
    print("=" * 80)

    for neuron_key, sign_data in sorted(attr_results.items(), key=lambda x: str(x[0])):
        cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
        name = cluster_id_to_name.get(cid, f"C{cid}")
        print(f"\n--- {name} (C{cid}) ---")
        for sign in ["pos", "neg"]:
            for expl in sign_data.get(sign, []):
                if hasattr(expl, "explanation"):
                    score = f"{expl.score:.3f}" if expl.score is not None else "N/A"
                    print(f"  attr/{sign}: [{score}] {expl.explanation}")

    for neuron_key, sign_data in sorted(contrib_results.items(), key=lambda x: str(x[0])):
        cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
        name = cluster_id_to_name.get(cid, f"C{cid}")
        for sign in ["combined"]:
            for expl in sign_data.get(sign, []):
                if hasattr(expl, "explanation"):
                    score = f"{expl.score:.3f}" if expl.score is not None else "N/A"
                    print(f"  contrib/{sign}: [{score}] {expl.explanation}")

    def neuron_key_to_name(neuron_key: object) -> str:
        cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
        return cluster_id_to_name.get(cid, f"C{cid}")

    output_data: dict[str, Any] = {
        "metadata": {
            "circuit_pickle": str(CIRCUIT_PICKLE),
            "timestamp": timestamp,
            "scored": True,
            "max_clusters": None,
            "threshold_mode": "quantile",
            "enforce_top_exemplars": 0,
            "num_expl_samples": 20,
            "min_highlights": 1,
            "backend": "descriptions",
            "attr_backend": attr_backend,
        },
        "cluster_id_to_name": {str(k): v for k, v in cluster_id_to_name.items()},
        "attr": {},
        "contrib": {},
        "attr_exemplars": {},
        "contrib_exemplars": {},
    }

    for results, key in [(attr_results, "attr"), (contrib_results, "contrib")]:
        for neuron_key, sign_data in results.items():
            name = neuron_key_to_name(neuron_key)
            output_data[key][name] = {}
            for sign in ["pos", "neg", "combined"]:
                output_data[key][name][sign] = [
                    (
                        {"explanation": expl.explanation, "score": expl.score}
                        if hasattr(expl, "explanation")
                        else {"explanation": str(expl), "score": None}
                    )
                    for expl in sign_data.get(sign, [])
                ]

    for exemplars, key in [
        (attr_exemplars, "attr_exemplars"),
        (contrib_exemplars, "contrib_exemplars"),
    ]:
        for neuron_key, sign_exemplars in exemplars.items():
            name = neuron_key_to_name(neuron_key)
            output_data[key][name] = dict(sign_exemplars)

    return suffix, output_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["vllm", "api", "both"],
        default="both",
        help="Which attr backend(s) to run: vllm, api, or both (default: both)",
    )
    parser.add_argument(
        "--attr-api-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for API attr backend",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    backends = []
    if args.backend in ("vllm", "both"):
        backends.append("vllm")
    if args.backend in ("api", "both"):
        backends.append("api")

    for backend in backends:
        suffix, output_data = run_descriptions(
            circuit,
            attr_backend=backend,
            attr_api_model_name=args.attr_api_model,
        )
        timestamp = output_data["metadata"]["timestamp"]
        output_file = OUTPUT_DIR / f"auto_descriptions_{suffix}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
