"""Cluster the math circuit and save cluster state as lightweight JSON.

Usage:
    python scripts/case_studies/math/cluster_math.py --n-clusters 256
    python scripts/case_studies/math/cluster_math.py --n-clusters 128 --combine mean
"""

import argparse
import logging

from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/math_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/math"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--combine", type=str, default="harmonic")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    logger.info("Clustering with multiview (k=%d, combine=%s)...", args.n_clusters, args.combine)
    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        combine=args.combine,
        get_desc=False,
        verbose=True,
    )

    out_path = OUTPUT_DIR / f"cluster_state_k{args.n_clusters}_{args.combine}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    circuit.save_cluster_state(out_path)


if __name__ == "__main__":
    main()
