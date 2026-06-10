<div align="center">
  <h1 align="center">ADAG</h1>
  <a href="https://arxiv.org/abs/2604.07615"><strong>Read our paper »</strong></a>
</div>
<br/>

**ADAG** (Automatically Describing Attribution Graphs) is Transluce's circuit tracing library, developed and maintained by [Aryaman Arora](https://aryaman.io/) and [Zhengxuan Wu](https://nlp.stanford.edu/~wuzhengx/index.html) (and, of course, [Claude Code](https://code.claude.com/docs/en/overview)). This library includes circuit tracing code for MLP neurons in Llama and Qwen-family Transformer LLMs, along with a whole analysis pipeline which automates circuit interpretation.

The pipeline goes like this:

1. **Prepare a circuit** -- trace neuron attributions across a dataset
2. **Cluster neurons** -- group neurons by their activation/contribution patterns
3. **Generate descriptions** -- explain what each cluster does (input attribution + output contribution)
4. **Summarize** -- produce short labels for each cluster
5. **Analyze** -- steering, correlation, visualization

We also adapt the interactive frontend from [Decode Research's excellent `circuit-tracer` library](https://github.com/decoderesearch/circuit-tracer) to enable viewing of all this information. We explain how to do each step below, along with downstream analysis like steering.

If you use this code, please cite:

```bibtex
@article{arora2026sparse,
    title={Language Model Circuits Are Sparse in the Neuron Basis},
    author={Aryaman Arora and Zhengxuan Wu and Jacob Steinhardt and Sarah Schwettmann},
    year={2026},
    journal={arXiv:2601.22594},
    url={https://arxiv.org/abs/2601.22594},
}

@article{arora2026adag,
    title={ADAG: Automatically Describing Attribution Graphs},
    author={Aryaman Arora and Zhengxuan Wu and Jacob Steinhardt and Sarah Schwettmann},
    year={2026},
    journal={arXiv:2604.07615},
    url={https://arxiv.org/abs/2604.07615},
}
```

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and configure environment
cp .env.template .env
# Fill in API keys in .env

# Trace a circuit from a config
uv run python scripts/circuit_prep/prep.py --config configs/capitals.yaml

# Or submit as a SLURM job
sbatch --export=CONFIG=configs/capitals.yaml scripts/circuit_prep/prep.sbatch
```

## Pipeline Overview

1. **Prepare a circuit** -- trace neuron attributions across a dataset
2. **Cluster neurons** -- group neurons by their activation/contribution patterns
3. **Generate descriptions** -- explain what each cluster does (input attribution + output contribution)
4. **Summarize** -- produce short labels for each cluster
5. **Analyze** -- steering, correlation, visualization

## 1. Preparing a Circuit

### Dataset format

Create a Python module in `scripts/circuit_prep/data/` that exports:

```python
# scripts/circuit_prep/data/my_dataset.py
prompts = ["What is the capital of Texas?", ...]
seed_responses = ["Answer:"] * len(prompts)  # prefix for model response
labels = [" Austin", ...]                    # target tokens to trace
```

For dynamic datasets that need model access, export a function instead:

```python
def get_dataset(model, tokenizer) -> tuple[list[str], list[str], list[str]]:
    # ... generate prompts dynamically ...
    return prompts, seed_responses, labels
```

### Config file

```yaml
# scripts/circuit_prep/configs/my_dataset.yaml
dataset: my_dataset
output_path: results/case_studies/output_circuit.pkl
model-id: meta-llama/Llama-3.1-8B-Instruct
percentage_threshold: 0.005  # attribution pruning cutoff
batch_size: 4
k: 5                        # top-k logits to trace
apply-blacklist: true
```

### Tracing

```bash
# Single GPU
uv run python scripts/circuit_prep/prep.py --config configs/my_dataset.yaml

# Multi-GPU data parallelism (e.g. 4 GPUs, 1 per worker = 4 workers)
sbatch --gres=gpu:4 --export=CONFIG=configs/my_dataset.yaml scripts/circuit_prep/prep.sbatch

# Large model across 2 GPUs per copy (e.g. 8 GPUs = 4 workers)
uv run python scripts/circuit_prep/prep.py --config configs/my_dataset.yaml --gpus-per-model 2
```

This produces a `CircuitData` pickle containing per-neuron attributions, activation maps, and edge
weights.

### Tracing API

```python
from circuits.tracing.trace import convert_inputs_to_circuits, CircuitData
from circuits.tracing.clja import ADAGConfig

config = ADAGConfig(
    device="cuda:0",
    percentage_threshold=0.005,
    use_relp_grad=True,
    apply_blacklist=True,
    # ... see ADAGConfig for all options
)

data = convert_inputs_to_circuits(
    model, tokenizer, prompts,
    config=config,
    seed_responses=seed_responses,
    labels=labels,
    batch_size=4,
    k=5,
)
data.save_to_pickle("circuit.pkl")
```

## 2. Clustering

Load the circuit and cluster neurons into interpretable groups:

```python
from circuits.analysis.circuit_ops import Circuit
from transformers import AutoTokenizer

c = Circuit.load_from_pickle("circuit.pkl")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
c.set_tokenizer(tokenizer, num_layers=32)

# Multi-view spectral clustering
c.cluster_multiview(n_clusters=20, get_desc=True, combine="harmonic")

# Save cluster assignments
c.save_cluster_state("cluster_state.json")
```

To reload cluster assignments later:

```python
c = Circuit.load_from_pickle("circuit.pkl")
c.set_tokenizer(tokenizer, num_layers=32)
c.load_cluster_state("cluster_state.json")
```

## 3. Generating Descriptions

Generate natural-language explanations for each cluster. There are two types:

- **Attribution (attr)**: what input tokens cause this cluster to activate
- **Contribution (contrib)**: what output tokens this cluster promotes/suppresses

```python
result = c.label_clusters_simulator_v2(
    score_explanations=True,
    num_expl_samples=5,
    attr_backend="vllm",                       # local finetuned model (no refusals)
    contrib_model_name="claude-haiku-4-5-20251001",
    verbose=True,
)
attr_results, contrib_results, attr_exemplars, contrib_exemplars, cluster_to_neurons = result
```

**Important**: Use `attr_backend="vllm"` (not `"api"`) for attribution descriptions. The finetuned
local model won't refuse on adversarial or safety-related content, whereas the Claude API may refuse
to analyze jailbreak-style excerpts.

This requires 2 GPUs: one for the vllm explainer/simulator, one for scoring.

## 4. Summarizing Clusters

Generate short (1--3 word) labels for each cluster:

```python
# Best results: attr + contrib descriptions only, no neuron descriptions
labels = c.summarize_clusters(mode="rich", attr_only=True)

# Full rich mode: attr + contrib + individual neuron descriptions
c.fetch_descriptions()  # fetch per-neuron descriptions from database
labels = c.summarize_clusters(mode="rich")

# Include top dataset examples where cluster is most active
labels = c.summarize_clusters(mode="batch", top_k_examples=10)

# Save (labels are stored in cluster state)
c.save_cluster_state("cluster_state.json")
```

`summarize_clusters()` uses `claude-opus-4-6` with adaptive thinking by default.

| Mode | Description | Best for |
|------|-------------|----------|
| `mode="rich", attr_only=True` | Per-cluster API call with attr + contrib only | Task-specific circuits |
| `mode="rich"` | Per-cluster with attr + contrib + neuron descriptions | General circuits |
| `mode="rich", neurons_only=True` | Per-cluster with neuron descriptions only | When attr/contrib are missing |
| `mode="batch"` | All clusters in one prompt | Quick/cheap labeling |

There's also a standalone script:

```bash
uv run python scripts/case_studies/sensitivity_analysis/3b_summarize_clusters.py \
    --mode rich --attr-only \
    --circuit circuit.pkl --cluster-state cluster_state.json
```

## 5. Visualization

### Circuit Tracer frontend

The built-in web UI shows the full circuit interactively -- clusters, neurons, edges, token
attributions, and descriptions. Uses the circuit-tracer frontend.

```python
# Serve directly (launches local HTTP server)
c.serve(port=8032, slug="my_circuit")
# Open http://localhost:8032/index.html?slug=my_circuit_0

# Or export JSON files for hosting elsewhere
c.export_to_circuit_tracer("output_dir/", slug="my_circuit")
```

### Circuit graph (Graphviz)

For a static cluster-level graph of a single circuit instance:

```bash
uv run python scripts/case_studies/capitals/plot_circuit_graph.py circuit.pkl \
    --label 0 --cluster-state cluster_state.json --dot-only
dot -Tpdf circuit_graph.dot -o circuit_graph.pdf
```

## 6. Downstream Analysis

### Steering

Intervene on cluster activations and measure effects on model output:

```python
# Ablate a cluster (0x multiplier)
diversity_stats, cluster_to_cluster, cluster_to_output = c.steer(
    model, multiplier=0.0, store_results=True
)

# Amplify a cluster (2x multiplier)
c.steer(model, multiplier=2.0, store_results=True)
```

For more controlled steering with ASR (attack success rate) measurement and LLM judging, see the
sensitivity analysis scripts.

### Correlation analysis

Correlate cluster attribution with a metric (e.g., ASR) across circuit instances:

```python
from scipy import stats
import numpy as np

# Per-cluster: sum attribution across neurons, correlate with metric
for cl, attr_vec in cluster_attr.items():
    r, p = stats.pearsonr(attr_vec, metric_vec)
```

See `scripts/case_studies/sensitivity_analysis/3_describe_and_correlate.py` for a full example.

## Case Studies

| Directory | Description |
|-----------|-------------|
| `sensitivity_analysis/` | Medical safety jailbreak analysis on Llama-3.1-8B-Instruct |
| `capitals/` | State capitals factual recall |
| `capitals_qwen3/` | Capitals with Qwen3 |
| `math/` | Arithmetic circuits |
| `user_modelling/` | User behavior modeling |
| `causalgym/` | CausalGym SVA evaluation |
| `benchmark/` | Tracing throughput benchmarks |

### Sensitivity analysis pipeline (numbered scripts)

```
1_serve.py                     Browse circuit interactively
2_correlate_asr.py             Per-neuron attribution vs ASR correlation
3_describe_and_correlate.py    Cluster, describe, correlate
3b_summarize_clusters.py       Generate summary labels (standalone)
4_steer_and_judge.py           Steer individual neurons, judge ASR
5_steer_cluster_and_judge.py   Steer clusters, judge ASR
6_sweep_clusters.py            Sweep all clusters (0x/2x), report ASR
7_plot_asr_vs_attribution.py   Scatter plots
8_make_steering_table.py       LaTeX table for paper
```

## Neuronpedia graph prompts (Gemma 2 2B, end-to-end)

A ready-made set of **base-model completion prompts** taken from public Neuronpedia
attribution graphs for `google/gemma-2-2b`, plus the full pipeline that traces them,
exports per-prompt MLP neurons + text, describes each neuron, and groups the neurons
into supernodes. See [GEMMA2_NEURON_EXPORT.md](GEMMA2_NEURON_EXPORT.md) for the
Gemma 2 tracing fixes and the export/description/grouping scripts.

The graph list lives in `custom_automation/prompts/neuronpedia_graphs.csv` (`slug,
share_url, notes`). `fetch_neuronpedia_prompts.py` fetches each slug's **canonical
prompt** from the Neuronpedia graph API
(`https://www.neuronpedia.org/api/graph/gemma-2-2b/<slug>`) — more reliable than
parsing the note — strips the leading `<bos>`, and regenerates the dataset module
`scripts/circuit_prep/data/neuronpedia.py` (already checked in, so you can skip
step 0 unless the CSV changes).

### Prompts

| # | slug | prompt | answer |
|---|------|--------|--------|
| 0 | gemma-G | `The International Advanced Security Group (IAS` | G |
| 1 | gemma-addition | `3 + 5 = ` | 8 |
| 2 | gemma-addition2 | `2 + 1 = ` | 3 |
| 3 | gemma-basket | `Fait: Michael Jordan joue au` | basket |
| 4 | gemma-dollar | `Mexico:peso :: US:` | dollar |
| 5 | gemma-english | `Mexico:Spanish :: US:` | English |
| 6 | gemma-euro | `Mexico:peso :: Europe:` | euro |
| 7 | gemma-girl-is | `The girl that the teacher sees` | is |
| 8 | gemma-girls-are | `The girls that the teacher sees` | are |
| 9 | gemma-gp-nps | `The guitarist knew the song` | . / is |
| 10 | gemma-keys-cabinet | `The keys on the cabinet` | are |
| 11 | gemma-michael-jordan | `Fact: Michael Jordan plays the sport of` | basketball |
| 12 | gemma-michael-jordan-es | `Hecho: Michael Jordan juega al` | baloncesto |
| 13 | gemma-saison | `La saison après le printemps s'apelle l'` | été |
| 14 | gemma-verano | `La estación después de la primavera se llama el` | verano |

Prompts are stored with `<bos>` stripped (the tokenizer re-adds it) and trailing
spaces preserved, so tracing reproduces Neuronpedia's target token. gemma-2-2b is a
base model, so `seed_responses` are empty and the trace targets the model's own
next-token prediction; `labels` are the slugs, so each exported graph file maps back
to its source Neuronpedia graph. The `answer` column is documentation only.

### Run the pipeline

```bash
# 0. (optional) refresh prompts from Neuronpedia — needs network, regenerates
#    scripts/circuit_prep/data/neuronpedia.py + prompts/neuronpedia_prompts.json
python custom_automation/fetch_neuronpedia_prompts.py

# 1. Trace all 15 prompts at once → one CircuitData pickle (GPU; ~5 GB in bf16).
uv run python scripts/circuit_prep/prep.py --config configs/neuronpedia_gemma.yaml
#    sanity-check (with verbose) that each "prompt seed -> <top token>" matches the answer.

# 2. Export per-graph MLP neurons + text (tokenizer-only, no GPU). --top-n 0 exports
#    ALL neurons that survived pruning so every one is eligible for grouping; use a
#    number (e.g. 30) to cap to the top-N by |attribution|.
uv run python scripts/circuit_prep/batch_export_neurons.py \
    --circuit results/case_studies/neuronpedia_circuit.pkl \
    --model-id google/gemma-2-2b --dataset neuronpedia \
    --top-n 0 --mode per-prompt --out-dir neuronpedia_neuron_graphs/
#    -> neuronpedia_neuron_graphs/graph_0000_gemma_g.json, graph_0001_gemma_addition.json, ...

# 3-4. Describe each neuron, then group neurons into supernodes. --graphs-dir loops
#      over every prompt, so this covers all 15 in one call each (in place).
cd custom_automation
export OPENAI_API_KEY=sk-...
python generate_description.py  --graphs-dir ../neuronpedia_neuron_graphs/
python generate_supernodes.py   --graphs-dir ../neuronpedia_neuron_graphs/   # --top-k-seed N to retune Phase 1

# 5. Local interactive HTML viewer (circuit-tracer style). Top: the full attribution
#    graph — every neuron as a node (token x layer), colored by supernode, with edges.
#    Middle "Subgraph": one box per supernode with its member neurons listed inside,
#    plus separate Embeddings/Output nodes. Right: detail panel — click any node to see
#    that neuron's activation text. The top-left dropdown switches prompts.
python render_report.py --graphs-dir ../neuronpedia_neuron_graphs/ --out neuronpedia_report.html
```

View the report (it is a self-contained file — no Neuronpedia push, no metadata
update, fully local):

```powershell
# one step — render AND serve on localhost, opens the browser for you
cd custom_automation
python render_report.py --graphs-dir ../neuronpedia_neuron_graphs/ --out neuronpedia_report.html --serve --port 8041
# -> http://localhost:8041/neuronpedia_report.html

# the report is self-contained (data inlined, nothing fetched), so you can also just
# open the file directly — no server needed:
start neuronpedia_report.html
# or serve a folder manually (e.g. for remote/RunPod via SSH port-forward):
python -m http.server 8041   # then http://localhost:8041/neuronpedia_report.html
```

`render_report.py` only reads the `graph_*.json` files. Click a box in the subgraph
to jump to that supernode's neurons. (Don't confuse this with
`scripts/case_studies/.../serve_circuit.py`, which launches the circuit-tracer
frontend — that renders topology but its feature-text sidebar 401/403s for Gemma
neurons; the HTML report is the workaround that shows the text.)

Each `graph_*.json` ends up self-contained: the pruned attribution graph (`nodes`,
`edges`), the curated top-N `neurons` (with `generated_description`), and the
`supernodes` grouping. The Neuronpedia ground-truth (transcoder) supernodes are
encoded in the `share_url` of each CSV row and mirrored in
`prompts/ground_truth_neuronpedia.csv` for comparison.

## Performance

Benchmarked on H100 80GB with Llama-3.1-8B-Instruct (capitals, 50 prompts, k=5):

| Batch Size | 1 GPU (s/prompt) | 2 GPU (s/prompt) | Peak GPU (1 GPU) |
|------------|-----------------|-----------------|------------------|
| 1 | 22.3 | 11.2 | 16.7 GB |
| 2 | 13.7 | 7.1 | 17.7 GB |
| **4** | **10.0** | **5.5** | **22.1 GB** |
| 8 | 15.1 | 8.0 | 40.9 GB |

Batch size 4 is optimal. Batch size 8 is slower due to memory pressure.

Multi-GPU uses data parallelism: each GPU gets its own model copy and processes a shard of the
dataset. For models that don't fit on a single GPU (e.g. Qwen3-32B), use `--gpus-per-model 2`.

## Project Structure

```
circuits/
  tracing/              Core tracing algorithm
    trace.py            High-level API: convert_inputs_to_circuits()
    clja.py             ADAGConfig, core jacobian attribution
    attribution.py      Gradient attribution computation
    grad/               Model-specific gradient wrappers (RelP, stop-grad)
  analysis/
    circuit_ops.py      Circuit class: clustering, labeling, steering, serving
    steer.py            Activation steering implementation
    label.py            Batch summarization
  descriptions/         Description generation
    label.py            Orchestration: generate + score explanations
    prompts.py          All LLM prompts (attr, contrib, summary)
    exemplars.py        Token highlighting and exemplar formatting
    vllm_backend.py     Local finetuned explainer/simulator
    api_backend.py      Anthropic API-based explainer
  frontend/             Web UI for interactive circuit exploration
scripts/
  circuit_prep/         Dataset loading, config, tracing entry point
  case_studies/         Per-task analysis scripts
```
