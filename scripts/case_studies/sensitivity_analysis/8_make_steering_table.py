"""Generate a LaTeX table of cluster steering results with ASR and correlations.

Usage:
    python scripts/case_studies/sensitivity_analysis/make_steering_table.py
    python scripts/case_studies/sensitivity_analysis/make_steering_table.py --output table.tex
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
from scipy import stats

CIRCUIT_PICKLE = "results/case_studies/sensitivity_analysis_circuit.pkl"
CLUSTER_STATE = "results/case_studies/sensitivity_analysis/cluster_state_20260323_131824_mv_k20_harmonic.json"
SWEEP_RESULTS = "results/case_studies/sensitivity_analysis/sweep_results.json"


def parse_asr_from_label(label: str) -> float:
    match = re.search(r"asr([\d.]+)", label)
    return float(match.group(1)) if match else 0.0


def compute_cluster_correlations(
    circuit_pickle: str, cluster_state: str
) -> dict[str, dict[str, float]]:
    """Compute per-cluster correlation with ASR from circuit data."""
    from circuits.analysis.circuit_ops import Circuit

    c = Circuit.load_from_pickle(circuit_pickle)
    c.load_cluster_state(cluster_state)

    df = c.df_node.copy()
    label_to_asr = {}
    for label in df["label"].unique():
        label_to_asr[label] = parse_asr_from_label(label)
    df["asr"] = df["label"].map(label_to_asr)

    num_layers = int(df["layer"].max())
    df = df[(df["layer"] >= 0) & (df["layer"] < num_layers)]

    grouped = (
        df.groupby(["layer", "neuron", "label"])
        .agg(total_attr=("attribution", "sum"), asr=("asr", "first"))
        .reset_index()
    )
    all_labels = sorted(df["label"].unique())
    asr_vec = np.array([label_to_asr[l] for l in all_labels])

    # Build cluster map
    cluster_map = {}
    for nid, cl in c._cluster_map.items():
        cluster_map[(int(nid.layer), int(nid.neuron))] = cl

    # Sum attribution per cluster per CI
    cluster_attr: dict[str, np.ndarray] = {}
    for (layer, neuron), sub in grouped.groupby(["layer", "neuron"]):
        cl = cluster_map.get((int(layer), int(neuron)), "?")
        attr_by_label = dict(zip(sub["label"], sub["total_attr"]))
        attr_vec = np.array([attr_by_label.get(l, 0.0) for l in all_labels])
        if cl not in cluster_attr:
            cluster_attr[cl] = np.zeros(len(all_labels))
        cluster_attr[cl] += attr_vec

    # Count unique neurons per cluster (hidden layers only)
    cluster_n_neurons: dict[str, int] = defaultdict(int)
    seen = set()
    for nid, cl in c._cluster_map.items():
        layer = int(nid.layer)
        neuron = int(nid.neuron)
        if 0 <= layer < num_layers and (layer, neuron) not in seen:
            seen.add((layer, neuron))
            cluster_n_neurons[cl] += 1

    result = {}
    for cl, attr_vec in cluster_attr.items():
        r, p = stats.pearsonr(attr_vec, asr_vec)
        result[cl] = {
            "correlation": r,
            "p_value": p,
            "n_neurons": cluster_n_neurons.get(cl, 0),
        }
    return result


def binomial_sd(n_success: int, n_total: int) -> float:
    """Standard deviation of a binomial proportion."""
    if n_total == 0:
        return 0.0
    p = n_success / n_total
    return np.sqrt(p * (1 - p) / n_total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default=CIRCUIT_PICKLE)
    parser.add_argument("--cluster-state", default=CLUSTER_STATE)
    parser.add_argument("--sweep-results", default=SWEEP_RESULTS)
    parser.add_argument("--output", type=str, default=None, help="Save LaTeX to file")
    args = parser.parse_args()

    # Load sweep results
    with open(args.sweep_results) as f:
        sweep = json.load(f)

    # Load cluster state for labels
    with open(args.cluster_state) as f:
        cluster_state = json.load(f)
    summary_labels = cluster_state.get("cluster_summary_labels", {})

    # Compute correlations
    corr_data = compute_cluster_correlations(args.circuit, args.cluster_state)

    # Sort by effect size (0x vs 1x, or 2x vs 1x)
    clusters = sorted(sweep.keys(), key=lambda x: int(re.search(r"\d+", x).group()))

    def effect_size(cl):
        r = sweep[cl]
        a0 = r.get("0.0", {}).get("asr", 0)
        a1 = r.get("1.0", {}).get("asr", 0)
        a2 = r.get("2.0", {}).get("asr", 0)
        if a0 < 0 or a1 < 0 or a2 < 0:
            return -1
        return max(abs(a0 - a1), abs(a2 - a1))

    clusters_sorted = sorted(clusters, key=effect_size, reverse=True)

    # Get unsteered (1x) data from any cluster (they're all the same)
    first_cl = clusters_sorted[0]
    unsteered = sweep[first_cl].get("1.0", {})
    n_total = (
        unsteered.get("n_yes", 0) + unsteered.get("n_no", 0) + unsteered.get("n_incoherent", 0)
    )

    def color_cell(asr: float, sd: float) -> str:
        if asr < 0:
            return "---"
        pct = int(asr * 100)
        sd_pct = int(sd * 100)
        if asr == 0:
            return f"0$^{{\\pm{sd_pct}}}$\\%"
        red_int = int(asr * 100)
        val = f"{pct}$^{{\\pm{sd_pct}}}$\\%"
        if asr >= 0.8:
            return f"\\cellcolor{{red!{red_int}}} \\textbf{{{val}}}"
        return f"\\cellcolor{{red!{red_int}}} {val}"

    def incoh_cell(d: dict) -> str:
        if d.get("asr", 0) < 0:
            return "---"
        n_incoh = d.get("n_incoherent", 0)
        n_tot = d.get("n_yes", 0) + d.get("n_no", 0) + d.get("n_incoherent", 0)
        if n_tot == 0:
            return "---"
        pct = int(n_incoh / n_tot * 100)
        sd_pct = int(binomial_sd(n_incoh, n_tot) * 100)
        return f"{pct}$^{{\\pm{sd_pct}}}$\\%"

    def escape_label(label):
        return (
            label.replace("_", r"\_")
            .replace("&", r"\&")
            .replace("[", "{[}")
            .replace("]", "{]}")
            .replace('"', "``")
        )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cluster steering sweep on the base prompt ($n{=}50$ generations per condition). "
        r"Clusters sorted by max effect size. $r$ is Pearson correlation of cluster attribution "
        r"with ASR across all 151 prompt variants. Superscripts show $\pm 1$ binomial SD.}"
    )
    lines.append(r"\label{tab:cluster-steering}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{rlrr cc cc}")
    lines.append(r"\toprule")
    lines.append(
        r"& & & & \multicolumn{2}{c}{\textbf{ASR}} & \multicolumn{2}{c}{\textbf{\% Incoherent}} \\"
    )
    lines.append(r"\cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    lines.append(
        r"\textbf{Cluster} & \textbf{Label} & \textbf{\#N} & \textbf{$r$} "
        r"& \textbf{0$\times$} & \textbf{2$\times$} "
        r"& \textbf{0$\times$} & \textbf{2$\times$} \\"
    )
    lines.append(r"\midrule")

    # Unsteered baseline row
    unsteered_asr = unsteered.get("asr", 0)
    unsteered_n_yes = unsteered.get("n_yes", 0)
    unsteered_n_no = unsteered.get("n_no", 0)
    unsteered_n_incoh = unsteered.get("n_incoherent", 0)
    unsteered_n_tot = unsteered_n_yes + unsteered_n_no + unsteered_n_incoh
    unsteered_asr_sd = binomial_sd(unsteered_n_yes, unsteered_n_tot)
    unsteered_incoh_sd = binomial_sd(unsteered_n_incoh, unsteered_n_tot)
    unsteered_incoh_pct = int(unsteered_n_incoh / unsteered_n_tot * 100) if unsteered_n_tot else 0
    unsteered_incoh_sd_pct = int(unsteered_incoh_sd * 100)
    unsteered_asr_cell = color_cell(unsteered_asr, unsteered_asr_sd)
    unsteered_incoh_cell = f"{unsteered_incoh_pct}$^{{\\pm{unsteered_incoh_sd_pct}}}$\\%"

    lines.append(
        f"--- & \\textit{{unsteered}} & & "
        f"& \\multicolumn{{2}}{{c}}{{{unsteered_asr_cell}}} "
        f"& \\multicolumn{{2}}{{c}}{{{unsteered_incoh_cell}}} \\\\"
    )
    lines.append(r"\midrule")

    for cl in clusters_sorted:
        r = sweep[cl]
        label = escape_label(summary_labels.get(cl, r.get("label", cl)))
        cd = corr_data.get(cl, {})
        nn = cd.get("n_neurons", "?")
        corr = cd.get("correlation", 0)
        corr_str = f"{corr:+.2f}"

        r0 = r.get("0.0", {})
        r2 = r.get("2.0", {})

        # ASR with SD
        n0_tot = r0.get("n_yes", 0) + r0.get("n_no", 0) + r0.get("n_incoherent", 0)
        n2_tot = r2.get("n_yes", 0) + r2.get("n_no", 0) + r2.get("n_incoherent", 0)
        asr0_sd = binomial_sd(r0.get("n_yes", 0), n0_tot)
        asr2_sd = binomial_sd(r2.get("n_yes", 0), n2_tot)

        lines.append(
            f"{cl} & {label} & {nn} & {corr_str} & "
            f"{color_cell(r0.get('asr', 0), asr0_sd)} & {color_cell(r2.get('asr', 0), asr2_sd)} & "
            f"{incoh_cell(r0)} & {incoh_cell(r2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table = "\n".join(lines)
    print(table)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table)
        print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
