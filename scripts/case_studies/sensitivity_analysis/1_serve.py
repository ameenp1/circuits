"""Serve the sensitivity analysis circuit."""

from circuits.analysis.circuit_ops import Circuit
from transformers import AutoTokenizer

CLUSTER_STATE = "results/case_studies/sensitivity_analysis/cluster_state_20260323_131824_mv_k20_harmonic.json"

c = Circuit.load_from_pickle("results/case_studies/sensitivity_analysis_circuit.pkl")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
c.set_tokenizer(tokenizer, num_layers=32)
c.load_cluster_state(CLUSTER_STATE)
c.serve(port=8032, slug="sensitivity_analysis")

input("Press Enter to stop the server...")
