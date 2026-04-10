#!/usr/bin/env python3
"""Interactive HTML annotation tool for circuit clusters.

Serves a local web page showing attr_map and contrib_map highlights for manually-defined
clusters. The user writes descriptions for each cluster, then reveals the true labels.

Usage:
    python scripts/ablations/descriptions/0_annotate_clusters.py
"""

import http.server
import json
import logging
import random
import socketserver
import string
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
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
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals/annotations"

# Manual clusters from capitals_analysis.py
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


def prepare_cluster_data(
    circuit: Circuit,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Extract per-cluster, per-example data for the annotation UI.

    Returns:
        (cluster_data, id_to_true_name) where cluster_data maps shuffled IDs like "A", "B", ...
        to dicts with examples containing prompt_tokens, attr_map, contrib_map, target_logit_tokens.
    """
    df_clustered, _ = circuit._prepare_clustered_df_for_labelling()
    cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})
    tokenizer = circuit.tokenizer

    # Group by cluster_id (stored in "layer" column)
    raw_clusters: dict[int, list[dict]] = {}
    for _, row in df_clustered.iterrows():
        cid = int(row["layer"])
        label = row["label"]

        # Labels in df have "___N" suffix encoding the ci index
        if "___" in label:
            example_idx = int(label.rsplit("___", 1)[1])
            display_label = label.rsplit("___", 1)[0]
        else:
            example_idx = circuit.labels.index(label)
            display_label = label

        # Decode prompt tokens (strip padding via attention_mask)
        input_ids = circuit.cis[example_idx]
        attn_mask = circuit.attention_masks[example_idx]
        unpadded_ids = [tid for tid, m in zip(input_ids, attn_mask) if m == 1]
        prompt_tokens = [tokenizer.decode(tid) for tid in unpadded_ids]

        # attr_map may be shorter than prompt tokens (e.g. BOS removed), align to suffix
        attr_map_raw = np.asarray(row["attr_map"], dtype=float)
        n_attr = len(attr_map_raw)
        n_toks = len(unpadded_ids)
        attr_map = ([0.0] * max(0, n_toks - n_attr)) + attr_map_raw[:n_toks].tolist()

        # contrib_map: per target logit contribution
        contrib_map_raw = np.asarray(row["contrib_map"], dtype=float)
        target_logit_ids = circuit.target_logits[example_idx]
        target_logit_tokens = [tokenizer.decode(tid) for tid in target_logit_ids]
        contrib_map = contrib_map_raw[: len(target_logit_ids)].tolist()

        if cid not in raw_clusters:
            raw_clusters[cid] = []
        raw_clusters[cid].append(
            {
                "label": display_label,
                "prompt_tokens": prompt_tokens,
                "attr_map": attr_map,
                "contrib_map": contrib_map,
                "target_logit_tokens": target_logit_tokens,
            }
        )

    # Shuffle cluster order and assign letter IDs
    cluster_ids = list(raw_clusters.keys())
    random.shuffle(cluster_ids)
    letters = list(string.ascii_uppercase)

    cluster_data: dict[str, dict] = {}
    id_to_true_name: dict[str, str] = {}
    for i, cid in enumerate(cluster_ids):
        letter = letters[i] if i < len(letters) else f"Z{i}"
        cluster_data[letter] = {
            "name": f"Cluster {letter}",
            "examples": raw_clusters[cid],
        }
        id_to_true_name[letter] = cluster_id_to_name.get(cid, f"C{cid}")

    return cluster_data, id_to_true_name


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Circuit Cluster Annotation</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
       display: flex; height: 100vh; background: #1a1a2e; color: #e0e0e0; }

/* Sidebar */
#sidebar { width: 220px; background: #16213e; padding: 16px; overflow-y: auto;
           border-right: 1px solid #0f3460; flex-shrink: 0; }
#sidebar h2 { font-size: 14px; color: #8888cc; margin-bottom: 12px; text-transform: uppercase;
              letter-spacing: 1px; }
.cluster-btn { display: flex; align-items: center; gap: 8px; width: 100%; padding: 10px 12px;
               margin-bottom: 4px; background: transparent; border: 1px solid transparent;
               color: #c0c0c0; cursor: pointer; border-radius: 6px; font-size: 13px;
               text-align: left; transition: all 0.15s; }
.cluster-btn:hover { background: #1a1a3e; border-color: #333366; }
.cluster-btn.active { background: #0f3460; border-color: #4444aa; color: #fff; }
.cluster-btn .dot { width: 8px; height: 8px; border-radius: 50%; background: #444; flex-shrink: 0; }
.cluster-btn.done .dot { background: #44cc44; }
.cluster-btn.revealed .dot { background: #cc8844; }

/* Main */
#main { flex: 1; overflow-y: auto; padding: 24px 32px; }
#main h1 { font-size: 18px; margin-bottom: 16px; color: #aaaaee; }
.section-header { font-size: 13px; color: #8888bb; text-transform: uppercase;
                  letter-spacing: 1px; margin: 20px 0 10px; border-bottom: 1px solid #333;
                  padding-bottom: 4px; }

/* Token display */
.example-row { margin-bottom: 14px; }
.example-label { font-size: 11px; color: #666; margin-bottom: 2px; }
.token-row { display: flex; flex-wrap: wrap; gap: 1px; line-height: 1.6; }
.token { padding: 1px 3px; border-radius: 3px; font-size: 13px; white-space: pre; }

/* Inputs */
.desc-section { margin-top: 24px; padding: 16px; background: #16213e; border-radius: 8px; }
.desc-section label { display: block; font-size: 12px; color: #8888bb; margin-bottom: 4px;
                      text-transform: uppercase; letter-spacing: 0.5px; }
.desc-section textarea { width: 100%; padding: 8px; background: #0f0f23; border: 1px solid #333366;
                         color: #e0e0e0; border-radius: 4px; font-size: 13px; resize: vertical;
                         min-height: 48px; margin-bottom: 12px; font-family: inherit; }
.desc-section textarea:focus { outline: none; border-color: #6666cc; }
.btn { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;
       font-weight: 600; transition: all 0.15s; }
.btn-save { background: #2244aa; color: #fff; }
.btn-save:hover { background: #3355cc; }
.btn-reveal { background: #884400; color: #fff; margin-left: 8px; }
.btn-reveal:hover { background: #aa5500; }
.btn-reveal:disabled { opacity: 0.4; cursor: not-allowed; }

/* Reveal */
.true-label { margin-top: 16px; padding: 12px 16px; background: #2a1a00; border: 1px solid #884400;
              border-radius: 8px; font-size: 15px; }
.true-label strong { color: #ffaa44; }

/* Legend */
.legend { display: flex; gap: 16px; margin-bottom: 16px; font-size: 12px; color: #888; }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-swatch { width: 32px; height: 14px; border-radius: 3px; }

#placeholder { color: #555; margin-top: 40px; text-align: center; font-size: 14px; }
</style>
</head>
<body>

<div id="sidebar">
  <h2>Clusters</h2>
  <div id="cluster-list"></div>
  <div style="margin-top: 20px; padding-top: 12px; border-top: 1px solid #333;">
    <button class="btn btn-reveal" id="reveal-all-btn" onclick="revealAll()">Reveal All</button>
  </div>
</div>

<div id="main">
  <div id="placeholder">Select a cluster from the sidebar to begin annotating.</div>
  <div id="content" style="display:none;">
    <h1 id="cluster-title"></h1>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-swatch" style="background: linear-gradient(90deg, rgba(100,140,255,0.8), rgba(100,140,255,0.1))"></div>
        <span>Positive</span>
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background: linear-gradient(90deg, rgba(255,80,80,0.8), rgba(255,80,80,0.1))"></div>
        <span>Negative</span>
      </div>
    </div>

    <div class="section-header">Attribution &mdash; what input tokens activate this cluster?</div>
    <div id="attr-examples"></div>

    <div class="section-header">Contribution &mdash; what output tokens does this cluster promote?</div>
    <div id="contrib-examples"></div>

    <div class="desc-section">
      <label for="attr-desc">Your attr description (what inputs activate this?)</label>
      <textarea id="attr-desc" placeholder="e.g. tokens related to US states..."></textarea>
      <label for="contrib-desc">Your contrib description (what outputs does this promote?)</label>
      <textarea id="contrib-desc" placeholder="e.g. promotes capital city names..."></textarea>
      <button class="btn btn-save" onclick="saveDescription()">Save</button>
      <button class="btn btn-reveal" onclick="revealCurrent()">Reveal Label</button>
    </div>

    <div id="true-label-box" class="true-label" style="display:none;"></div>
  </div>
</div>

<script>
const CLUSTER_DATA = __CLUSTER_DATA__;
const clusterIds = Object.keys(CLUSTER_DATA).sort();
const descriptions = {};  // {clusterId: {attr: "", contrib: ""}}
const revealed = new Set();
let currentCluster = null;

function init() {
  const list = document.getElementById('cluster-list');
  clusterIds.forEach(id => {
    const btn = document.createElement('button');
    btn.className = 'cluster-btn';
    btn.id = 'btn-' + id;
    btn.innerHTML = `<span class="dot"></span>${CLUSTER_DATA[id].name}`;
    btn.onclick = () => selectCluster(id);
    list.appendChild(btn);
  });
}

function selectCluster(id) {
  currentCluster = id;
  document.querySelectorAll('.cluster-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + id).classList.add('active');
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('content').style.display = 'block';
  document.getElementById('cluster-title').textContent = CLUSTER_DATA[id].name;

  // Restore saved descriptions
  const saved = descriptions[id] || {};
  document.getElementById('attr-desc').value = saved.attr || '';
  document.getElementById('contrib-desc').value = saved.contrib || '';

  // Render attr examples
  const attrDiv = document.getElementById('attr-examples');
  attrDiv.innerHTML = '';
  CLUSTER_DATA[id].examples.forEach(ex => {
    attrDiv.appendChild(renderTokenRow(ex.label, ex.prompt_tokens, ex.attr_map));
  });

  // Render contrib examples
  const contribDiv = document.getElementById('contrib-examples');
  contribDiv.innerHTML = '';
  CLUSTER_DATA[id].examples.forEach(ex => {
    contribDiv.appendChild(renderTokenRow(ex.label, ex.target_logit_tokens, ex.contrib_map));
  });

  // Show/hide reveal box
  const labelBox = document.getElementById('true-label-box');
  if (revealed.has(id)) {
    labelBox.style.display = 'block';
    // Will be populated by reveal response
  } else {
    labelBox.style.display = 'none';
  }
}

function renderTokenRow(label, tokens, values) {
  const row = document.createElement('div');
  row.className = 'example-row';

  const labelEl = document.createElement('div');
  labelEl.className = 'example-label';
  labelEl.textContent = label;
  row.appendChild(labelEl);

  const tokenRow = document.createElement('div');
  tokenRow.className = 'token-row';

  // Normalize: divide by max abs value for color scaling
  const maxAbs = Math.max(...values.map(Math.abs), 1e-8);

  tokens.forEach((tok, i) => {
    const span = document.createElement('span');
    span.className = 'token';
    span.textContent = tok;
    const v = (values[i] || 0) / maxAbs;
    if (v > 0.01) {
      span.style.background = `rgba(100, 140, 255, ${Math.min(Math.abs(v), 1) * 0.8})`;
    } else if (v < -0.01) {
      span.style.background = `rgba(255, 80, 80, ${Math.min(Math.abs(v), 1) * 0.8})`;
    }
    tokenRow.appendChild(span);
  });

  row.appendChild(tokenRow);
  return row;
}

function saveDescription() {
  if (!currentCluster) return;
  descriptions[currentCluster] = {
    attr: document.getElementById('attr-desc').value,
    contrib: document.getElementById('contrib-desc').value,
  };
  document.getElementById('btn-' + currentCluster).classList.add('done');
}

function revealCurrent() {
  if (!currentCluster) return;
  // Save first
  saveDescription();
  // POST to server
  fetch('/reveal', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cluster_id: currentCluster, descriptions: descriptions}),
  })
  .then(r => r.json())
  .then(data => {
    revealed.add(currentCluster);
    document.getElementById('btn-' + currentCluster).classList.add('revealed');
    const box = document.getElementById('true-label-box');
    box.style.display = 'block';
    box.innerHTML = `<strong>True label:</strong> ${data.true_name}`;
  });
}

function revealAll() {
  // Save current first
  if (currentCluster) saveDescription();
  fetch('/reveal_all', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({descriptions: descriptions}),
  })
  .then(r => r.json())
  .then(data => {
    // Mark all as revealed
    clusterIds.forEach(id => {
      revealed.add(id);
      document.getElementById('btn-' + id).classList.add('revealed');
    });
    // Show current cluster's label if viewing one
    if (currentCluster && data.labels[currentCluster]) {
      const box = document.getElementById('true-label-box');
      box.style.display = 'block';
      box.innerHTML = `<strong>True label:</strong> ${data.labels[currentCluster]}`;
    }
    alert('All labels revealed and annotations saved! Check the results directory.');
  });
}

init();
</script>
</body>
</html>
"""


def make_handler(
    cluster_data: dict[str, dict],
    id_to_true_name: dict[str, str],
    output_dir: Path,
) -> type:
    class AnnotationHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path == "/index.html":
                # Inject cluster data into HTML
                data_json = json.dumps(cluster_data, default=str)
                html = HTML_TEMPLATE.replace("__CLUSTER_DATA__", data_json)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}

            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/reveal":
                cid = body.get("cluster_id", "")
                true_name = id_to_true_name.get(cid, "???")
                # Save descriptions
                self._save_annotations(body.get("descriptions", {}))
                self._respond_json({"true_name": true_name})

            elif parsed.path == "/reveal_all":
                self._save_annotations(body.get("descriptions", {}))
                self._respond_json({"labels": id_to_true_name})

            else:
                self.send_response(404)
                self.end_headers()

        def _save_annotations(self, descriptions: dict) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = output_dir / f"annotations_{timestamp}.json"
            result = {
                "timestamp": timestamp,
                "annotations": {},
            }
            for cid, descs in descriptions.items():
                result["annotations"][cid] = {
                    "display_name": cluster_data[cid]["name"],
                    "true_name": id_to_true_name.get(cid, "???"),
                    "attr_description": descs.get("attr", ""),
                    "contrib_description": descs.get("contrib", ""),
                }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Saved annotations to %s", out_path)

        def _respond_json(self, data: dict) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            # Suppress default request logging
            pass

    return AnnotationHandler


def main() -> None:
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

    logger.info("Preparing cluster data for annotation UI...")
    cluster_data, id_to_true_name = prepare_cluster_data(circuit)
    logger.info("Prepared %d clusters", len(cluster_data))

    port = 8765
    handler_cls = make_handler(cluster_data, id_to_true_name, OUTPUT_DIR)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler_cls) as httpd:
        logger.info("Annotation tool running at http://localhost:%d", port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down.")


if __name__ == "__main__":
    main()
