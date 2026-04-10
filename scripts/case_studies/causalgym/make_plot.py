"""Plot faithfulness & completeness for CausalGym: MLP neurons vs SAE MLP outputs.

Usage:
    python scripts/case_studies/causalgym/make_plot.py
    python scripts/case_studies/causalgym/make_plot.py --results-dir results/causalgym/pair
"""

import argparse
import glob
import logging
import warnings
from pathlib import Path

import pandas as pd
import plotnine as p9
from circuits.utils.constants import RESULTS_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

PLOT_SUFFIX = ".pdf"

p9.theme_set(
    p9.theme_bw(base_size=10, base_family="Palatino")
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=1, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)


def load_fc_results(
    results_dir: str, model: str = "Llama-3.1-8B", agg: str = "mean"
) -> pd.DataFrame:
    """Load faithfulness/completeness JSON results from the CausalGym experiment directories."""
    base = Path(results_dir)
    dfs = []

    # MLP neurons (activations), IG
    neurons_pattern = (
        f"{base}/{model}_causalgym_pp_N*_AGG{agg}_Mnap_USE_NEURONS_USE_MLPACTS_DISABLE_STOP_GRAD"
        f"_EDGE_THRESHOLD0.02_TOPK_NEURONS100/faithfulness_and_completeness/*.json"
    )
    for f in glob.glob(neurons_pattern):
        df = pd.read_json(f)
        if not df.empty:
            df["method"] = "MLP neurons (acts.)"
            dfs.append(df)
            logger.info("Loaded neurons results: %s (%d rows)", f, len(df))

    # SAE MLP outputs, IG
    sae_pattern = (
        f"{base}/{model}_causalgym_pp_N*_AGG{agg}_Mnap"
        f"_DISABLE_STOP_GRAD_EDGE_THRESHOLD0.02_TOPK_NEURONS100/faithfulness_and_completeness/*.json"
    )
    for f in glob.glob(sae_pattern):
        df = pd.read_json(f)
        if not df.empty:
            df["method"] = "SAE MLP outputs"
            dfs.append(df)
            logger.info("Loaded SAE results: %s (%d rows)", f, len(df))

    if not dfs:
        logger.warning("No results found in %s", results_dir)
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def plot_fc(df: pd.DataFrame, output_dir: str, use_log: bool = False) -> None:
    """Plot faithfulness and completeness curves for the two methods."""
    for c in ["n_nodes", "faithfulness", "completeness"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["n_nodes", "faithfulness", "completeness"])

    if df.empty:
        logger.warning("No valid data to plot.")
        return

    # Melt to long form
    df_long = df.melt(
        id_vars=["method", "n_nodes"],
        value_vars=["faithfulness", "completeness"],
        var_name="metric",
        value_name="value",
    )
    df_long["metric"] = df_long["metric"].str.capitalize()
    df_long["metric"] = pd.Categorical(
        df_long["metric"], categories=["Faithfulness", "Completeness"], ordered=True
    )

    plot = (
        p9.ggplot(df_long, p9.aes(x="n_nodes", y="value", color="method"))
        + p9.geom_line(size=1, alpha=0.8)
        + p9.scale_color_brewer(type="qual", palette="Set1")
        + p9.facet_wrap("~ metric", nrow=1)
        + p9.theme(figure_size=(5, 2.5), legend_position="right")
        + p9.labs(x="Circuit size (# neurons)", y="", color="Method")
    )

    if use_log:
        plot = plot + p9.scale_x_log10(breaks=[10**i for i in range(10)])
        suffix = "log"
    else:
        plot = plot + p9.scale_x_continuous()
        suffix = "linear"

    out_path = Path(output_dir) / f"causalgym_fc_{suffix}{PLOT_SUFFIX}"
    plot.save(str(out_path), dpi=300)
    logger.info("Saved plot to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Plot CausalGym faithfulness/completeness")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR / "causalgym" / "pair"),
        help="Directory containing CausalGym experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "causalgym"),
        help="Directory to save plots",
    )
    parser.add_argument("--model", type=str, default="Llama-3.1-8B")
    parser.add_argument("--agg", type=str, default="mean")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df = load_fc_results(args.results_dir, model=args.model, agg=args.agg)
    if df.empty:
        logger.error("No data found. Check results directory: %s", args.results_dir)
        return

    for use_log in [True, False]:
        plot_fc(df.copy(), args.output_dir, use_log=use_log)


if __name__ == "__main__":
    main()
