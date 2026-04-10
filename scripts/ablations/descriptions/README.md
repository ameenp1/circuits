# Human vs. Auto Description Ablation

Compares human-written cluster descriptions against automated descriptions
on the capitals circuit (8 manual clusters, Llama 3.1 8B Instruct).

## Scripts

| Script | Purpose |
|---|---|
| `0_annotate_clusters.py` | Interactive HTML annotation tool — view attr/contrib highlights for shuffled clusters and write descriptions, then reveal true labels |
| `1_score_human_descriptions.py` | Score hardcoded human descriptions (attr via VLLM simulator, contrib via Haiku API) |
| `2_run_auto_descriptions.py` | Generate + score auto descriptions (quantile, mh=1, 20 samples) |
| `3_table_human_vs_auto.py` | LaTeX table comparing human vs auto scores |

## Running

### 0. Annotate clusters (optional, for writing new human descriptions)

Run locally (needs circuit pickle + tokenizer access):

```bash
uv run python scripts/ablations/descriptions/0_annotate_clusters.py
```

Opens a local web UI at `http://localhost:8765` showing shuffled clusters with
attr_map/contrib_map highlights. Write attr + contrib descriptions for each
cluster, then reveal the true labels. Annotations are saved to
`$CIRCUITS_RESULTS_DIR/case_studies/capitals/annotations/`.

### 1. Score human descriptions

```bash
uv run python scripts/ablations/descriptions/1_score_human_descriptions.py
```

Results go to `$CIRCUITS_RESULTS_DIR/case_studies/capitals/human_descriptions/`.

### 2. Generate + score auto descriptions

```bash
uv run python scripts/ablations/descriptions/2_run_auto_descriptions.py
```

Results go to `$CIRCUITS_RESULTS_DIR/case_studies/capitals/human_vs_auto/`.

### 3. Generate comparison table

After both jobs finish:

```bash
uv run python scripts/ablations/descriptions/3_table_human_vs_auto.py
```

This picks the latest JSON from each results folder by default. To specify
files explicitly:

```bash
uv run python scripts/ablations/descriptions/3_table_human_vs_auto.py \
    --human /path/to/human_descriptions.json \
    --auto V2=/path/to/auto.json MoreSamples=/path/to/another.json
```

## Output folders

All under `$CIRCUITS_RESULTS_DIR` (default `results`):

- `case_studies/capitals/annotations/` — raw annotations from the annotation UI
- `case_studies/capitals/human_descriptions/` — human description scores
- `case_studies/capitals/human_vs_auto/` — auto description scores
- `case_studies/capitals_circuit.pkl` — shared circuit pickle
