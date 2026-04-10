# Clustering Ablations

Compares clustering algorithms, multi-view vs baseline clustering, and
visualizes cluster structure on the capitals circuit.

## Scripts

| Script | Purpose |
|---|---|
| `0_sweep_clustering.py` | Sweep clustering algorithms (KMeans, spectral, agglomerative, etc.) at varying k, report balance metrics |
| `1_sweep_multiview_clustering.py` | Sweep multi-view similarity clustering (spectral on per-CI similarity, Leiden) and report diagnostics |
| `2_analyse_multiview_clusters.py` | Run multi-view spectral clustering at a chosen k, fetch neuron descriptions, print per-cluster summaries |
| `3_dump_cluster_descriptions.py` | Dump cluster-to-neuron description mappings for a given clustering |
| `4_heatmap_multiview_sim.py` | Interactive Plotly heatmap of multi-view similarity matrix (neurons ordered by cluster) |
| `5_plot_sweep_min_highlights.py` | Plot min_highlights x threshold_mode sweep results from existing explanation JSONs |

## Running

### 0. Baseline clustering sweep

```bash
python scripts/ablations/clustering/0_sweep_clustering.py \
    --circuit-pickle results/case_studies/capitals_circuit.pkl \
    --output results/case_studies/clustering_sweep.json
```

### 1. Multi-view clustering sweep

```bash
python scripts/ablations/clustering/1_sweep_multiview_clustering.py \
    --circuit-pickle results/case_studies/capitals_circuit.pkl \
    --output-dir results/case_studies/multiview_clustering
```

### 2-4. Analysis and visualization (run after sweeps)

```bash
python scripts/ablations/clustering/2_analyse_multiview_clusters.py
python scripts/ablations/clustering/3_dump_cluster_descriptions.py
python scripts/ablations/clustering/4_heatmap_multiview_sim.py
```

### 5. Plot description sweep results

```bash
python scripts/ablations/clustering/5_plot_sweep_min_highlights.py
```

## Output folders

All under `$CIRCUITS_RESULTS_DIR` (default `results`):

- `case_studies/clustering_sweep.json` — baseline clustering sweep results
- `case_studies/multiview_clustering/` — multi-view clustering sweep results
- `case_studies/capitals_circuit.pkl` — shared circuit pickle
