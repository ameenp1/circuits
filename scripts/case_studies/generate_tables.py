"""
Generate LaTeX tables and heatmap visualizations for circuit analysis.

Usage:
    python generate_tables.py

Output:
    - LaTeX tables in subdirectories (math/, multilingual/, user_modeling/)
    - PNG heatmaps for math case study
"""

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

RESULTS_DIR = Path(".")
DATA_URLS = {
    "sum_mod_n": "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/scores_sum_mod_n.json",
    "sum_mod_10": "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/scores_sum_mod_10.json",
    "sum_tens": "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/scores_sum_tens.json",
}

p9.theme_set(
    p9.theme_bw(base_size=10, base_family="Palatino")
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_blank(),
        panel_grid_minor=p9.element_blank(),
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)


def fetch_json(url: str) -> list[dict] | dict:
    """Fetch JSON data from URL."""
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def make_heatmap_df(attribution: list[float], normalize: bool = True) -> pd.DataFrame:
    """Convert flat 10000-element attribution list to DataFrame for heatmap."""
    arr = np.array(attribution).reshape(100, 100)
    if normalize:
        max_abs = np.abs(arr).max()
        if max_abs > 0:
            arr = arr / max_abs
    rows = []
    for i in range(100):
        for j in range(100):
            rows.append({"x": j, "y": i, "value": arr[i, j]})
    return pd.DataFrame(rows)


def plot_single_heatmap(entry: dict, output_path: Path) -> None:
    """Create a single heatmap for one neuron entry."""
    df = make_heatmap_df(entry["attribution"])
    description = entry["description"]
    layer = entry["layer"]
    neuron = entry["neuron"]
    polarity = entry["polarity"]

    plot = (
        p9.ggplot(df, p9.aes(x="x", y="y", fill="value"))
        + p9.geom_tile()
        + p9.scale_fill_gradient2(
            low="#b2182b", mid="white", high="#2166ac", midpoint=0, limits=(-1, 1)
        )
        + p9.coord_equal()
        + p9.scale_x_continuous(breaks=range(0, 100, 10), expand=(0, 0))
        + p9.scale_y_continuous(breaks=range(0, 100, 10), expand=(0, 0))
        + p9.labs(
            title=f"{description} (L{layer}/N{neuron}{polarity})",
            x="Addend 1",
            y="Addend 2",
            fill="Norm. Attr.",
        )
        + p9.theme(figure_size=(5, 4.5))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output_path)
    print(f"Saved: {output_path}")


def plot_all_heatmaps(
    entries: list[dict],
    output_path: Path,
    ncol: int = 3,
    figure_size: tuple[float, float] = (8, 4),
) -> None:
    """Create a faceted plot with all heatmaps."""
    all_rows = []
    for entry in entries:
        df = make_heatmap_df(entry["attribution"])
        df["description"] = entry["description"]
        df["label"] = (
            f"{entry['description']}\n(L{entry['layer']}/N{entry['neuron']}{entry['polarity']})"
        )
        all_rows.append(df)

    combined_df = pd.concat(all_rows, ignore_index=True)

    plot = (
        p9.ggplot(combined_df, p9.aes(x="x", y="y", fill="value"))
        + p9.geom_tile()
        + p9.facet_wrap("~label", ncol=ncol)
        + p9.scale_fill_gradient2(
            low="#b2182b", mid="white", high="#2166ac", midpoint=0, limits=(-1, 1)
        )
        + p9.coord_equal()
        + p9.scale_x_continuous(breaks=range(0, 100, 20), expand=(0, 0))
        + p9.scale_y_continuous(breaks=range(0, 100, 20), expand=(0, 0))
        + p9.labs(
            x="Addend 1",
            y="Addend 2",
            fill="Norm. Attr.",
        )
        + p9.theme(
            figure_size=figure_size,
            strip_text=p9.element_text(size=8),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot.save(output_path, dpi=300)
    print(f"Saved: {output_path}")


# Colors matching the heatmap scheme (flipped: blue=high, red=low)
COLOR_HIGH = "2166ac"  # blue
COLOR_LOW = "b2182b"  # red


def score_to_color(score: float) -> str:
    """Convert AUROC score to LaTeX xcolor specification."""
    # Using blue-red diverging scheme: blue for high scores, red for low scores
    if score >= 0.5:
        # Blue tint for high scores
        intensity = int((score - 0.5) * 2 * 40)  # 0-40 range
        return f"highcolor!{intensity}!white"
    else:
        # Red tint for low scores
        intensity = int((0.5 - score) * 2 * 40)
        return f"lowcolor!{intensity}!white"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters and non-ASCII Unicode."""
    import unicodedata

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Handle unicode superscripts
    text = text.replace("⁺", "$^+$").replace("⁻", "$^-$")

    # Replace non-ASCII characters with their Unicode name or codepoint
    result = []
    for char in text:
        if ord(char) > 127:
            try:
                name = unicodedata.name(char, None)
                if name:
                    # Use abbreviated name for common scripts
                    if "ARABIC" in name:
                        result.append(f"[AR:{name.split()[-1]}]")
                    elif "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name:
                        result.append("[CJK]")
                    elif "CYRILLIC" in name:
                        result.append(f"[CYR:{name.split()[-1]}]")
                    elif "HEBREW" in name:
                        result.append(f"[HEB:{name.split()[-1]}]")
                    else:
                        result.append(f"[U+{ord(char):04X}]")
                else:
                    result.append(f"[U+{ord(char):04X}]")
            except ValueError:
                result.append(f"[U+{ord(char):04X}]")
        else:
            result.append(char)
    return "".join(result)


def generate_latex_table_sva(
    url: str,
    output_path: Path,
    caption: str = "Feature scores.",
    label: str = "tab:features",
) -> None:
    """Generate a LaTeX table from SVA-style JSON (nodes as lists)."""
    data = fetch_json(url)
    nodes = data["nodes"]

    # Group by token
    grouped: dict[int, list] = {}
    for node in nodes:
        token = node[2]
        if token not in grouped:
            grouped[token] = []
        grouped[token].append(node)

    # Sort each group by abs score (descending)
    for token in grouped:
        grouped[token] = sorted(
            grouped[token], key=lambda n: abs(n[5][0]) if n[5] else 0, reverse=True
        )

    lines = [
        r"% Requires: \usepackage[table]{xcolor}, \usepackage{longtable}, \usepackage{booktabs}",
        rf"\definecolor{{highcolor}}{{HTML}}{{{COLOR_HIGH}}}",
        rf"\definecolor{{lowcolor}}{{HTML}}{{{COLOR_LOW}}}",
        r"\begingroup\small",
        r"\begin{longtable}{llr}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}} \\",
        r"\toprule",
        r"Feature & Description & Score \\",
        r"\endhead",
    ]

    for token in sorted(grouped.keys()):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{Token: ${token}$}}}} \\")

        for node in grouped[token]:
            # node = [id, layer, token, neuron, description, scores, abs_scores, cluster, edges]
            layer = node[1]
            neuron = node[3]
            description_raw = node[4]
            scores = node[5]
            score = scores[0] if scores else 0.0

            # Parse description into positive and negative parts
            # Format: "⁺positive desc | ⁻negative desc"
            desc_parts = description_raw.split(" | ")
            pos_desc = ""
            neg_desc = ""
            for part in desc_parts:
                if part.startswith("⁺"):
                    pos_desc = part[1:]  # Remove ⁺ prefix
                elif part.startswith("⁻"):
                    neg_desc = part[1:]  # Remove ⁻ prefix

            desc_limit = 45

            # Color based on score sign and magnitude
            abs_score = abs(score)
            intensity = min(int(abs_score * 40), 40) if abs_score > 0 else 0
            if score >= 0:
                color = f"highcolor!{intensity}!white" if intensity > 0 else "white"
            else:
                color = f"lowcolor!{intensity}!white" if intensity > 0 else "white"

            # Positive row (with score)
            feature_pos = rf"\neuron{{{layer}}}{{{neuron}}}{{+}}"
            pos_text = escape_latex(pos_desc)
            pos_text = pos_text[:desc_limit] + "..." if len(pos_text) > desc_limit else pos_text
            row_pos = rf"\rowcolor{{{color}}} " rf"{feature_pos} & {pos_text} & ${score:.3f}$ \\"
            lines.append(row_pos)

            # Negative row (no score)
            feature_neg = rf"\neuron{{{layer}}}{{{neuron}}}{{-}}"
            neg_text = escape_latex(neg_desc)
            neg_text = neg_text[:desc_limit] + "..." if len(neg_text) > desc_limit else neg_text
            row_neg = rf"\rowcolor{{{color}}} " rf"{feature_neg} & {neg_text} & \\"
            lines.append(row_neg)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved LaTeX table: {output_path}")


def generate_latex_table(
    url: str,
    output_path: Path,
    caption: str = "Feature scores.",
    label: str = "tab:features",
    top_k: int | None = None,
) -> None:
    """Generate a LaTeX table from JSON data with row coloring based on AUROC score."""
    entries = fetch_json(url)

    # Group entries by target_variable
    grouped: dict[int, list[dict]] = {}
    for entry in entries:
        target = entry["target_variable"]
        if "math" in url:
            target = int(target)
        if target not in grouped:
            grouped[target] = []
        grouped[target].append(entry)

    lines = [
        r"% Requires: \usepackage[table]{xcolor}, \usepackage{longtable}, \usepackage{booktabs}",
        rf"\definecolor{{highcolor}}{{HTML}}{{{COLOR_HIGH}}}",
        rf"\definecolor{{lowcolor}}{{HTML}}{{{COLOR_LOW}}}",
        r"\begingroup\small",
        r"\begin{longtable}{llrrr}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}} \\",
        r"\toprule",
        r"Feature & Description & AUROC & In-class & Out-of-class \\",
        r"\endhead",
    ]

    for target in sorted(grouped.keys()):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{Target: {target}}}}} \\")

        sorted_entries = sorted(
            grouped[target],
            key=lambda e: abs(float(e["avg_attribution_in_class"])),
            reverse=True,
        )
        if top_k is not None:
            sorted_entries = sorted_entries[:top_k]
        for entry in sorted_entries:
            layer = entry["layer"]
            neuron = entry["neuron"]
            polarity = entry["polarity"]
            feature = rf"\neuron{{{layer}}}{{{neuron}}}{{{polarity}}}"
            desc_limit = 45
            desc_text = escape_latex(entry["description"])
            description = (
                desc_text[:desc_limit] + "..." if len(desc_text) > desc_limit else desc_text
            )
            score = float(entry["roc_auc_score"])
            in_class = float(entry["avg_attribution_in_class"]) * 100
            out_class = float(entry["avg_attribution_out_of_class"]) * 100
            color = score_to_color(score)

            row = (
                rf"\rowcolor{{{color}}} "
                rf"{feature} & {description} & ${score:.3f}$ & ${in_class:.2f}\%$ & ${out_class:.2f}\%$ \\"
            )
            lines.append(row)

    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved LaTeX table: {output_path}")


def main() -> None:
    # Settings for each dataset: (ncol, figure_size)
    plot_settings = {
        "sum_mod_n": (3, (8, 4)),
        "sum_mod_10": (5, (12, 5)),
        "sum_tens": (5, (12, 5)),
    }

    for name, url in DATA_URLS.items():
        print(f"Fetching data from {url}...")
        entries = fetch_json(url)
        print(f"Loaded {len(entries)} entries for {name}")

        ncol, figure_size = plot_settings.get(name, (3, (8, 4)))
        plot_all_heatmaps(
            entries,
            RESULTS_DIR / f"{name}_heatmaps.png",
            ncol=ncol,
            figure_size=figure_size,
        )

    # Generate LaTeX tables
    # Math tables
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/math_sum_mod_10.json",
        RESULTS_DIR / "math" / "sum_mod_10_table.tex",
        caption="Feature scores for sum mod 10 task.",
        label="tab:sum_mod_10",
    )
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/math_sum_tens_digit.json",
        RESULTS_DIR / "math" / "sum_tens_digit_table.tex",
        caption="Feature scores for sum tens digit task.",
        label="tab:sum_tens_digit",
    )

    # Multilingual tables
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/multilingual/language.json",
        RESULTS_DIR / "multilingual" / "language_table.tex",
        caption="Feature scores for language task.",
        label="tab:language",
        top_k=20,
    )
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/multilingual/word.json",
        RESULTS_DIR / "multilingual" / "word_table.tex",
        caption="Feature scores for word task.",
        label="tab:word",
    )
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/multilingual/axis.json",
        RESULTS_DIR / "multilingual" / "axis_table.tex",
        caption="Feature scores for axis task.",
        label="tab:axis",
    )

    # User modeling tables
    generate_latex_table(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/user_modeling/gender.json",
        RESULTS_DIR / "user_modeling" / "gender_table.tex",
        caption="Feature scores for gender task.",
        label="tab:gender",
    )

    # SVA tables (different format)
    generate_latex_table_sva(
        "https://transluce-public.s3.us-east-1.amazonaws.com/circuits/sva_nounpp_mlpacts_ig.json",
        RESULTS_DIR / "sva" / "nounpp_mlpacts_ig_table.tex",
        caption="Feature scores for SVA nounpp task.",
        label="tab:sva_nounpp",
    )


if __name__ == "__main__":
    main()
