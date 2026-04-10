"""Plot benchmark results: batch size vs throughput and peak GPU memory.

Usage:
    python scripts/case_studies/benchmark/plot_bench.py \
        --results results_1gpu.json results_2gpu.json \
        --output-dir outputs/
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import plotnine as p9

p9.theme_set(
    p9.theme_bw(base_size=10, base_family="Palatino")
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
    )
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True, help="JSON result files")
    parser.add_argument("--output-dir", type=str, default="outputs/")
    args = parser.parse_args()

    rows = []
    for path in args.results:
        with open(path) as f:
            data = json.load(f)
        gpu_label = f"{data['num_gpus']} GPU"
        for run in data["runs"]:
            rows.append(
                {
                    "batch_size": run["batch_size"],
                    "per_prompt_s": run["per_prompt_s"],
                    "peak_gpu_gb": run["peak_gpu_gb"] if run["peak_gpu_gb"] >= 0 else None,
                    "gpus": gpu_label,
                }
            )

    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot 1: throughput
    p1 = (
        p9.ggplot(df, p9.aes(x="batch_size", y="per_prompt_s", color="gpus"))
        + p9.geom_line(size=1)
        + p9.geom_point(size=3)
        + p9.scale_x_continuous(breaks=[1, 2, 4, 8])
        + p9.labs(x="Batch Size", y="Seconds / Prompt", color="")
        + p9.scale_color_brewer(type="qual", palette="Set1")
    )
    p1_path = out / "bench_throughput.pdf"
    p9.ggsave(p1, filename=str(p1_path), width=4, height=3, dpi=300)
    print(f"Saved {p1_path}")

    # Plot 2: peak GPU memory (only rows with data)
    df_mem = df.dropna(subset=["peak_gpu_gb"])
    if len(df_mem) > 0:
        p2 = (
            p9.ggplot(df_mem, p9.aes(x="batch_size", y="peak_gpu_gb", color="gpus"))
            + p9.geom_line(size=1)
            + p9.geom_point(size=3)
            + p9.scale_x_continuous(breaks=[1, 2, 4, 8])
            + p9.labs(x="Batch Size", y="Peak GPU Memory (GB)", color="")
            + p9.scale_color_brewer(type="qual", palette="Set1")
        )
        p2_path = out / "bench_memory.pdf"
        p9.ggsave(p2, filename=str(p2_path), width=4, height=3, dpi=300)
        print(f"Saved {p2_path}")


if __name__ == "__main__":
    main()
