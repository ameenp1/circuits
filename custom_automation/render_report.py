"""
render_report.py — self-contained HTML viewer for ADAG neuron graphs.

Works around the broken circuit-tracer feature sidebar (which 401/403s trying to
fetch per-neuron cards from a remote store that doesn't host raw MLP neurons).
Everything here is local: it renders, per neuron, the highlighted activating
text, the driver tokens, the promoted/suppressed output tokens, and — if
generate_description.py has run — the LLM label.

No server needed. Open the .html in any browser (or scp it to your Mac).

Usage:
    python render_report.py --graph ../capitals_neuron_graphs/graph_0000_austin.json
    python render_report.py --graphs-dir ../capitals_neuron_graphs/ --out report.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#666;margin:0 0 20px}
.prompt{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 16px;margin-bottom:20px}
.prompt b{color:#444}
.neuron{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:14px 16px;margin-bottom:12px}
.nid{font-weight:600;color:#2b5dd0} .rank{color:#999;font-weight:400;margin-left:6px}
.attr{float:right;color:#888;font-variant-numeric:tabular-nums}
.label{margin:8px 0;padding:8px 12px;background:#eef4ff;border-left:3px solid #2b5dd0;border-radius:4px}
.label.none{background:#f4f4f4;border-left-color:#bbb;color:#999}
.txt{margin:8px 0;padding:8px 12px;background:#fafbfc;border-radius:4px;white-space:pre-wrap}
mark{background:#ffe89e;padding:0 2px;border-radius:2px;font-weight:600}
.row{display:flex;gap:24px;margin-top:8px;flex-wrap:wrap}
.col{flex:1;min-width:240px}
.col h4{margin:0 0 4px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#888}
.tok{display:inline-block;background:#f0f2f5;border-radius:4px;padding:1px 7px;margin:2px 3px 2px 0;font-size:13px}
.tok.promote{background:#e3f6e9} .tok.suppress{background:#fde8e8}
.score{color:#999;font-size:11px;margin-left:3px}
nav{margin-bottom:16px} nav a{margin-right:10px;color:#2b5dd0;text-decoration:none}
"""


def _esc(s: str) -> str:
    return html.escape(str(s))


def _highlight(text: str) -> str:
    """ADAG {{token}} -> <mark>token</mark> (escaped)."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("{{", i):
            j = text.find("}}", i + 2)
            if j != -1:
                out.append(f"<mark>{_esc(text[i + 2:j])}</mark>")
                i = j + 2
                continue
        out.append(_esc(text[i]))
        i += 1
    return "".join(out)


def _toks(items, cls: str) -> str:
    spans = []
    for it in (items or [])[:8]:
        try:
            tok, score = it[0], float(it[1])
        except (TypeError, ValueError, IndexError):
            continue
        spans.append(f'<span class="tok {cls}">{_esc(tok)}<span class="score">{score:.2g}</span></span>')
    return "".join(spans) or '<span class="score">none</span>'


def _split_contribs(contribs):
    promote, suppress = [], []
    for it in contribs or []:
        try:
            tok, score = it[0], float(it[1])
        except (TypeError, ValueError, IndexError):
            continue
        (promote if score >= 0 else suppress).append([tok, score])
    return promote, suppress


def render_graph(graph: dict) -> str:
    prompt = _esc(graph.get("prompt", ""))
    target = _esc(graph.get("target", ""))
    parts = [
        f'<div class="prompt"><b>Prompt:</b> {prompt} &nbsp; <b>Target:</b> {target}</div>'
    ]
    for n in graph.get("neurons", []):
        nid = f"L{n['layer']} N{n['neuron']} {n.get('polarity','')}"
        rank = n.get("rank", "")
        attr = n.get("attribution", 0.0)
        desc = n.get("generated_description")
        label_html = (
            f'<div class="label">{_esc(desc)}</div>' if desc
            else '<div class="label none">(no description — run generate_description.py)</div>'
        )
        promote, suppress = _split_contribs(n.get("output_contributions"))
        parts.append(f"""
        <div class="neuron">
          <span class="nid">{_esc(nid)}</span><span class="rank">rank {_esc(rank)}</span>
          <span class="attr">attr {attr:.3f}</span>
          {label_html}
          <div class="txt">{_highlight(n.get('highlighted_text',''))}</div>
          <div class="row">
            <div class="col"><h4>Driver tokens (input)</h4>{_toks(n.get('top_input_tokens'), 'tok')}</div>
            <div class="col"><h4>Promotes (output)</h4>{_toks(promote, 'promote')}</div>
            <div class="col"><h4>Suppresses (output)</h4>{_toks(suppress, 'suppress')}</div>
          </div>
        </div>""")
    return "".join(parts)


def build_html(graphs: list[tuple[str, dict]]) -> str:
    nav = ""
    if len(graphs) > 1:
        nav = '<nav>' + "".join(
            f'<a href="#g{i}">{_esc(g.get("target", name))}</a>' for i, (name, g) in enumerate(graphs)
        ) + '</nav>'
    body = []
    for i, (name, g) in enumerate(graphs):
        body.append(f'<h1 id="g{i}">{_esc(name)}</h1>')
        body.append(f'<p class="sub">{len(g.get("neurons", []))} neurons</p>')
        body.append(render_graph(g))
    return f"<!doctype html><meta charset='utf-8'><title>ADAG neuron report</title><style>{_CSS}</style><div class='wrap'>{nav}{''.join(body)}</div>"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render ADAG neuron graphs to a local HTML report.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph", type=Path)
    g.add_argument("--graphs-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    args = ap.parse_args()

    if args.graph:
        paths = [args.graph]
    else:
        paths = sorted(args.graphs_dir.glob("graph_*.json"))

    graphs = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    args.out.write_text(build_html(graphs), encoding="utf-8")
    print(f"Wrote {args.out}  ({len(graphs)} graph(s))  — open it in a browser.")


if __name__ == "__main__":
    main()
