"""
render_report.py — Neuronpedia-style local viewer for ADAG neuron graphs.

A simple, self-contained HTML page (no server needed, no remote feature store,
none of the circuit-tracer frontend's 401/403 on raw MLP neurons). One clean
card per neuron, laid out like a Neuronpedia feature page:

  - description (from generate_description.py, if present)
  - ACTIVATIONS: the prompt text with each token shaded by its input attribution
    (orange = activates), the activating token{{s}} marked
  - LOGITS: Negative (suppressed / demotes) | Positive (promoted) token columns

A dropdown switches between prompts when given a folder. Open the .html directly.

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
*{box-sizing:border-box}
body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1d2025}
.wrap{max-width:880px;margin:0 auto;padding:20px}
.topbar{display:flex;gap:12px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
.topbar h1{font-size:15px;font-weight:600;color:#888;margin:0}
select{font-size:14px;padding:6px 10px;border:1px solid #cbd2dd;border-radius:8px;background:#fff;min-width:260px}
.prompt{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:10px 14px;margin-bottom:16px}
.prompt b{color:#444}
.card{background:#fff;border:1px solid #e4e7ec;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.chead{display:flex;align-items:baseline;gap:8px;margin-bottom:6px}
.nid{font-weight:700;color:#1a56db;font-size:15px}
.rank{color:#9aa1ac;font-size:12px}
.attr{margin-left:auto;color:#9aa1ac;font-size:12px;font-variant-numeric:tabular-nums}
.desc{margin:6px 0 12px;padding:8px 12px;background:#eef3ff;border-left:3px solid #1a56db;border-radius:5px;font-size:13.5px}
.desc.none{background:#f5f6f8;border-left-color:#cdd2da;color:#9aa1ac;font-style:italic}
.sect{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#9aa1ac;margin:12px 0 5px}
.acts{padding:9px 11px;background:#fbfbfc;border:1px solid #eef0f3;border-radius:7px;line-height:2.1}
.atok{padding:1px 0;border-radius:3px}
.atok.mark{outline:2px solid #f59e0b;outline-offset:-1px}
.logits{display:flex;gap:14px}
.lcol{flex:1;min-width:0}
.lcol .h{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}
.lcol.pos .h{color:#15803d}.lcol.neg .h{color:#b91c1c}
.lrow{display:flex;justify-content:space-between;align-items:center;padding:3px 8px;border-radius:5px;margin-bottom:3px;font-size:13px}
.lcol.pos .lrow{background:#ecfdf3}.lcol.neg .lrow{background:#fef2f2}
.lrow .v{color:#9aa1ac;font-size:11.5px;font-variant-numeric:tabular-nums}
.empty{color:#bcc2cb;font-size:12px;padding:3px 8px}
"""

_JS = r"""
var cur=0;
function show(i){cur=i;var g=GRAPHS[i];
  document.getElementById('pbar').innerHTML='<b>Prompt:</b> '+esc(g.prompt)+' &nbsp;&nbsp; <b>Target:</b> '+esc(g.target);
  document.getElementById('cards').innerHTML=g.cards;}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
window.addEventListener('DOMContentLoaded',function(){
  var s=document.getElementById('sel');
  if(s)s.addEventListener('change',function(){show(parseInt(s.value));});
  show(0);});
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _split_contribs(contribs):
    pos, neg = [], []
    for it in contribs or []:
        try:
            tok, score = it[0], float(it[1])
        except (TypeError, ValueError, IndexError):
            continue
        (pos if score >= 0 else neg).append((tok, score))
    pos.sort(key=lambda x: -x[1])
    neg.sort(key=lambda x: x[1])
    return pos, neg


def _activation_line(neuron: dict) -> str:
    """Prompt tokens shaded by per-token input attribution; activating tokens outlined."""
    toks = neuron.get("tokens") or []
    acts = neuron.get("attr_activations") or []
    # which tokens are inside {{...}} in highlighted_text → mark them
    marked = set()
    ht = neuron.get("highlighted_text", "")
    k = 0
    while True:
        a = ht.find("{{", k)
        if a == -1:
            break
        b = ht.find("}}", a + 2)
        if b == -1:
            break
        marked.add(ht[a + 2:b].strip())
        k = b + 2

    if not toks:
        return f'<div class="acts">{_esc(ht) or "(no activation text)"}</div>'

    mx = max((abs(float(a)) for a in acts), default=0.0) or 1.0
    spans = []
    for t, a in zip(toks, acts):
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = 0.0
        inten = min(1.0, abs(a) / mx)
        if a > 0:
            bg = f"rgba(245,158,11,{0.10 + 0.6 * inten:.2f})"   # orange = activates
        elif a < 0:
            bg = f"rgba(120,140,170,{0.08 + 0.35 * inten:.2f})"
        else:
            bg = "transparent"
        cls = "atok mark" if t.strip() in marked and t.strip() else "atok"
        spans.append(f'<span class="{cls}" style="background:{bg}">{_esc(t)}</span>')
    return '<div class="acts">' + "".join(spans) + "</div>"


def _logit_col(items, cls: str, header: str) -> str:
    rows = "".join(
        f'<div class="lrow"><span>{_esc(tok)}</span><span class="v">{score:+.2f}</span></div>'
        for tok, score in items[:8]
    ) or '<div class="empty">none</div>'
    return f'<div class="lcol {cls}"><div class="h">{header}</div>{rows}</div>'


def _card(neuron: dict) -> str:
    nid = f"L{neuron['layer']} · N{neuron['neuron']} {neuron.get('polarity','')}"
    desc = neuron.get("generated_description")
    desc_html = (
        f'<div class="desc">{_esc(desc)}</div>' if desc
        else '<div class="desc none">no description — run generate_description.py</div>'
    )
    try:
        attr_s = f"attr {float(neuron.get('attribution', 0.0)):.3f}"
    except (TypeError, ValueError):
        attr_s = ""
    pos, neg = _split_contribs(neuron.get("output_contributions"))
    return (
        '<div class="card">'
        f'<div class="chead"><span class="nid">{_esc(nid)}</span>'
        f'<span class="rank">rank {_esc(neuron.get("rank",""))}</span>'
        f'<span class="attr">{attr_s}</span></div>'
        f'{desc_html}'
        '<div class="sect">Activations</div>'
        f'{_activation_line(neuron)}'
        '<div class="sect">Logits</div>'
        '<div class="logits">'
        f'{_logit_col(neg, "neg", "Negative")}'
        f'{_logit_col(pos, "pos", "Positive")}'
        '</div>'
        '</div>'
    )


def _cards_for(graph: dict) -> str:
    return "".join(_card(n) for n in graph.get("neurons", []))


def build_html(graphs: list[tuple[str, dict]]) -> str:
    payload = []
    for name, g in graphs:
        payload.append({
            "prompt": g.get("prompt", ""),
            "target": g.get("target", ""),
            "cards": _cards_for(g),
        })
    sel = ""
    if len(graphs) > 1:
        opts = "".join(
            f'<option value="{i}">{_esc(g.get("target", name) or name)}</option>'
            for i, (name, g) in enumerate(graphs)
        )
        sel = f'<span style="color:#888">prompt:</span><select id="sel">{opts}</select>'
    data = json.dumps(payload, ensure_ascii=False)
    return (
        "<!doctype html><meta charset='utf-8'><title>ADAG neuron cards</title>"
        f"<style>{_CSS}</style><div class='wrap'>"
        f"<div class='topbar'><h1>ADAG MLP-neuron cards</h1>{sel}</div>"
        "<div class='prompt' id='pbar'></div><div id='cards'></div></div>"
        f"<script>var GRAPHS={data};{_JS}</script>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render ADAG neuron graphs to Neuronpedia-style HTML cards.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph", type=Path)
    g.add_argument("--graphs-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    args = ap.parse_args()

    paths = [args.graph] if args.graph else sorted(args.graphs_dir.glob("graph_*.json"))
    graphs = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    args.out.write_text(build_html(graphs), encoding="utf-8")
    print(f"Wrote {args.out}  ({len(graphs)} graph(s)) — open it in a browser.")


if __name__ == "__main__":
    main()
