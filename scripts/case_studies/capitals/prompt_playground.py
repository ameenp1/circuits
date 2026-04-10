"""Interactive prompt playground for cluster summary labels.

Serves a web UI to experiment with the summary prompt template
and see results for all clusters in real time.

Usage:
    python scripts/case_studies/capitals/prompt_playground.py \
        --descriptions-json /path/to/explanations.json
"""

import argparse
import asyncio
import http.server
import json
import logging
import socketserver
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cluster Summary Prompt Playground</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
       background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { font-size: 16px; color: #aaaaee; margin-bottom: 16px; }
.layout { display: flex; gap: 20px; height: calc(100vh - 60px); }
.left { flex: 1; display: flex; flex-direction: column; gap: 12px; }
.right { flex: 1; overflow-y: auto; }

textarea { width: 100%; background: #0f0f23; border: 1px solid #333366;
           color: #e0e0e0; border-radius: 6px; font-size: 12px; padding: 10px;
           font-family: monospace; resize: vertical; }
textarea:focus { outline: none; border-color: #6666cc; }
#prompt-box { flex: 1; min-height: 200px; }

.btn { padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer;
       font-size: 13px; font-weight: 600; }
.btn-run { background: #2244aa; color: #fff; }
.btn-run:hover { background: #3355cc; }
.btn-run:disabled { opacity: 0.5; cursor: not-allowed; }

.controls { display: flex; gap: 12px; align-items: center; }
.controls label { font-size: 12px; color: #888; }
.controls select, .controls input { background: #0f0f23; border: 1px solid #333;
    color: #e0e0e0; padding: 4px 8px; border-radius: 4px; font-size: 12px; }

.status { font-size: 12px; color: #888; margin-top: 4px; }

table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #333; }
th { color: #8888cc; font-size: 11px; text-transform: uppercase; position: sticky; top: 0;
     background: #1a1a2e; }
td.label { color: #ffcc66; font-weight: 600; }
td.cluster { color: #888; font-family: monospace; }
</style>
</head>
<body>
<h1>Cluster Summary Prompt Playground</h1>
<div class="layout">
  <div class="left">
    <div class="controls">
      <label>Model:</label>
      <select id="model">
        <option value="claude-haiku-4-5-20251001">haiku</option>
        <option value="claude-sonnet-4-20250514">sonnet</option>
      </select>
      <button class="btn btn-run" id="run-btn" onclick="runPrompt()">Run</button>
      <span class="status" id="status"></span>
    </div>
    <textarea id="prompt-box">__DEFAULT_PROMPT__</textarea>
    <div style="font-size:11px; color:#666;">
      Variables: {n_clusters}, {cluster_block}. Output format must have "CLUSTER_ID: label" lines.
    </div>
  </div>
  <div class="right">
    <table>
      <thead><tr><th>Cluster</th><th>Label</th></tr></thead>
      <tbody id="results"></tbody>
    </table>
  </div>
</div>

<script>
const CLUSTER_DATA = __CLUSTER_DATA__;

async function runPrompt() {
  const btn = document.getElementById('run-btn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Running...';

  const prompt = document.getElementById('prompt-box').value;
  const model = document.getElementById('model').value;

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt, model}),
    });
    const data = await resp.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
    } else {
      status.textContent = `Done in ${data.elapsed_ms}ms`;
      const tbody = document.getElementById('results');
      tbody.innerHTML = '';
      // Sort by cluster name
      const entries = Object.entries(data.labels).sort((a, b) => {
        const na = parseInt(a[0].replace('C', ''));
        const nb = parseInt(b[0].replace('C', ''));
        return na - nb;
      });
      for (const [cid, label] of entries) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="cluster">${cid}</td><td class="label">${label}</td>`;
        tbody.appendChild(tr);
      }
    }
  } catch (e) {
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""


def load_descriptions(path: str) -> dict[str, dict[str, str]]:
    """Load best attr + contrib descriptions per cluster from explanations JSON."""
    with open(path) as f:
        desc_data = json.load(f)

    result: dict[str, dict[str, str]] = {}
    for cluster_name in desc_data.get("attr", {}):
        attr_expls = desc_data["attr"][cluster_name]
        contrib_expls = desc_data.get("contrib", {}).get(cluster_name, {})

        best_attr, best_attr_score = "", -1.0
        for sign in ("pos", "neg"):
            for e in attr_expls.get(sign, []):
                s = e.get("score")
                if s is not None and s > best_attr_score:
                    best_attr_score = s
                    best_attr = e["explanation"]

        best_contrib, best_contrib_score = "", -1.0
        for sign in ("pos", "neg", "combined"):
            for e in contrib_expls.get(sign, []):
                s = e.get("score")
                if s is not None and s > best_contrib_score:
                    best_contrib_score = s
                    best_contrib = e["explanation"]

        result[cluster_name] = {"attr": best_attr, "contrib": best_contrib}
    return result


def build_cluster_block(cluster_descs: dict[str, dict[str, str]]) -> str:
    lines = []
    for name in sorted(cluster_descs.keys()):
        descs = cluster_descs[name]
        block = f"### {name}\n"
        if descs.get("attr"):
            block += f"- Attribution: {descs['attr']}\n"
        if descs.get("contrib"):
            block += f"- Contribution: {descs['contrib']}\n"
        lines.append(block)
    return "\n".join(lines)


def make_handler(
    cluster_descs: dict[str, dict[str, str]],
    cluster_block: str,
    default_prompt: str,
) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path == "/index.html":
                data_json = json.dumps(cluster_descs, default=str)
                html = HTML_TEMPLATE.replace("__CLUSTER_DATA__", data_json)
                html = html.replace(
                    "__DEFAULT_PROMPT__",
                    default_prompt.replace("\\", "\\\\").replace("`", "\\`"),
                )
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

            if urllib.parse.urlparse(self.path).path == "/run":
                import time

                from anthropic import AsyncAnthropic

                prompt_template = body.get("prompt", default_prompt)
                model = body.get("model", "claude-haiku-4-5-20251001")

                prompt = prompt_template.replace("{n_clusters}", str(len(cluster_descs))).replace(
                    "{cluster_block}", cluster_block
                )

                start = time.time()
                try:
                    client = AsyncAnthropic()
                    response = asyncio.run(
                        client.messages.create(
                            model=model,
                            max_tokens=2048,
                            messages=[{"role": "user", "content": prompt}],
                        )
                    )
                    content = response.content[0].text if response.content else ""
                    elapsed_ms = int((time.time() - start) * 1000)

                    labels = {}
                    for line in content.strip().splitlines():
                        line = line.strip()
                        if ":" in line:
                            cid, lbl = line.split(":", 1)
                            cid = cid.strip()
                            lbl = lbl.strip()
                            if cid in cluster_descs:
                                labels[cid] = lbl

                    self._respond_json({"labels": labels, "raw": content, "elapsed_ms": elapsed_ms})
                except Exception as e:
                    self._respond_json({"error": str(e)})
            else:
                self.send_response(404)
                self.end_headers()

        def _respond_json(self, data: dict) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    from circuits.descriptions.prompts import CLUSTER_SUMMARY_BATCH_PROMPT

    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions-json", type=str, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    cluster_descs = load_descriptions(args.descriptions_json)
    cluster_block = build_cluster_block(cluster_descs)
    logger.info("Loaded %d clusters", len(cluster_descs))

    handler_cls = make_handler(cluster_descs, cluster_block, CLUSTER_SUMMARY_BATCH_PROMPT)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), handler_cls) as httpd:
        logger.info("Playground at http://localhost:%d", args.port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down.")


if __name__ == "__main__":
    main()
