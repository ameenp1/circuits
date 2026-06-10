"""
render_report.py — self-contained interactive viewer for ADAG neuron graphs.

A local stand-in for the circuit-tracer frontend (whose feature-text sidebar
401/403s for raw Gemma MLP neurons). Mirrors that UI's two-section layout:

  TOP  "Attribution graph" — ALL neurons as nodes, positioned by token (x) and
        layer (y), colored by their supernode, with the pruned attribution edges.
        Embeddings (layer -1, squares at the bottom) and logits/Output (top row,
        squares) are shown too.
  MID  "Subgraph" — the collapsed view: one box per SUPERNODE with its member
        neurons listed inside, plus separate Embeddings and Output nodes, and
        aggregated attribution edges between them.
  RIGHT "Detail" — click any node / neuron to see its ACTIVATION TEXT: highlighted
        activating tokens, per-token input attribution (green +/red -), the
        promoted/suppressed output tokens, and the LLM description.

A dropdown (top-left) switches between prompts. Everything is inlined (SVG + vanilla
JS) — no server, no CDN, no remote feature store. Open the .html directly, or:
    python render_report.py ... --serve        # serves localhost + opens a browser

Usage:
    python render_report.py --graph ../neuronpedia_neuron_graphs/graph_0001_gemma_addition.json
    python render_report.py --graphs-dir ../neuronpedia_neuron_graphs/ --serve
"""
from __future__ import annotations

import argparse
import html
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

EMB, OUT = "EMB", "OUT"

# Distinct, readable palette for supernodes (cycled if there are more groups).
_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2", "#edc948",
    "#ff9da7", "#9c755f", "#79706e", "#86bcb6", "#d37295", "#a0cbe8", "#8cd17d",
    "#b6992d", "#fabfd2",
]
_UNGROUPED_COLOR = "#c9c9cf"
_EMB_COLOR, _OUT_COLOR = "#59a14f", "#e15759"

_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
.wrap{max-width:1500px;margin:0 auto;padding:18px}
.topbar{display:flex;gap:14px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.topbar h1{font-size:16px;margin:0;color:#999;font-weight:500}
.sellbl{font-size:13px;color:#666}
select{font-size:14px;padding:6px 10px;border:1px solid #cbd2dd;border-radius:8px;background:#fff;min-width:240px}
.prompt{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:9px 14px;margin-bottom:12px}
.prompt b{color:#444}
.viewer{display:flex;gap:16px;align-items:flex-start}
.graphcol{flex:0 0 auto;max-width:60%}
.box{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:6px;margin-bottom:14px;overflow:auto}
.box>h4{margin:6px 8px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#888}
.panel{flex:1;min-width:300px;position:sticky;top:12px;max-height:94vh;overflow:auto;background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 14px}
.panel h3{font-size:16px;margin:2px 0 10px;padding-bottom:4px;border-bottom:2px solid #2b5dd0}
.panel h3 .cnt{color:#999;font-weight:400;font-size:13px}.panel h3 .dep{float:right;color:#aaa;font-weight:400;font-size:12px}
.hint{color:#999}
.neuron{border:1px solid #e8ebef;border-radius:8px;padding:10px 12px;margin-bottom:10px;background:#fcfdfe}
.nid{font-weight:600;color:#2b5dd0}.rank{color:#999;font-weight:400;margin-left:6px}
.attr{float:right;color:#888;font-variant-numeric:tabular-nums}
.label{margin:8px 0;padding:7px 11px;background:#eef4ff;border-left:3px solid #2b5dd0;border-radius:4px}
.label.none{background:#f4f4f4;border-left-color:#bbb;color:#999}
.txt{margin:8px 0;padding:7px 11px;background:#fafbfc;border-radius:4px;white-space:pre-wrap}
mark{background:#ffe89e;padding:0 2px;border-radius:2px;font-weight:600}
.attrtoks{margin:6px 0;padding:7px 11px;background:#fafbfc;border-radius:4px;line-height:2}
.atok{padding:1px 1px;border-radius:2px}
.row{display:flex;gap:18px;margin-top:8px;flex-wrap:wrap}.col{flex:1;min-width:200px}
.col h5,.neuron h5{margin:8px 0 4px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:#888}
.tok{display:inline-block;background:#f0f2f5;border-radius:4px;padding:1px 7px;margin:2px 3px 2px 0;font-size:13px}
.tok.promote{background:#e3f6e9}.tok.suppress{background:#fde8e8}
.score{color:#999;font-size:11px;margin-left:3px}
.kv{color:#555;font-size:13px;margin:2px 0}
svg [data-id]{cursor:pointer}
svg .sel{stroke:#111 !important;stroke-width:2.2 !important}
svg .edge{stroke:#9bb0d6;fill:none}
svg text{font:11px -apple-system,Segoe UI,Roboto,sans-serif}
svg .mrow:hover{fill:#2b5dd0}
"""

_JS = r"""
var cur=0;
function renderGraph(i){
  cur=i; var g=GRAPHS[i];
  document.getElementById('promptbar').innerHTML='<b>Prompt:</b> '+esc(g.prompt)+' &nbsp; <b>Target:</b> '+esc(g.target);
  document.getElementById('graph').innerHTML=g.scatter;
  document.getElementById('subgraph').innerHTML=g.sub;
  document.querySelectorAll('#graph [data-id],#subgraph [data-id]').forEach(function(el){
    el.addEventListener('click',function(ev){ev.stopPropagation();showDetail(el.getAttribute('data-id'));});});
  showDetail(g.def);
}
function showDetail(id){var g=GRAPHS[cur];
  document.getElementById('panel').innerHTML=(g.detail[id]||'<p class="hint">Untracked node — not in the exported top-N (no text).</p>');
  document.querySelectorAll('[data-id]').forEach(function(el){el.classList.toggle('sel', el.getAttribute('data-id')===id);});
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
window.addEventListener('DOMContentLoaded',function(){
  var sel=document.getElementById('sel');
  if(sel) sel.addEventListener('change',function(){renderGraph(parseInt(sel.value));});
  renderGraph(0);
});
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _a(s) -> str:
    """Escape for an SVG/HTML attribute value."""
    return html.escape(str(s), quote=True)


def _highlight(text: str) -> str:
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


def _attr_tokens(n: dict) -> str:
    toks = n.get("tokens") or []
    acts = n.get("attr_activations") or []
    if not toks:
        return ""
    mx = max((abs(float(a)) for a in acts), default=0.0) or 1.0
    spans = []
    for t, a in zip(toks, acts):
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = 0.0
        inten = min(1.0, abs(a) / mx)
        if a > 0:
            bg = f"rgba(60,170,90,{0.12 + 0.55 * inten:.2f})"
        elif a < 0:
            bg = f"rgba(212,72,72,{0.12 + 0.55 * inten:.2f})"
        else:
            bg = "transparent"
        spans.append(f'<span class="atok" style="background:{bg}">{_esc(t)}</span>')
    return '<div class="attrtoks">' + "".join(spans) + "</div>"


def _fid(n: dict) -> str:
    pol = n.get("polarity", "")
    return f"L{n['layer']}_N{n['neuron']}{('_' + pol) if pol else ''}"


def _neuron_card(n: dict) -> str:
    nid = f"L{n['layer']} N{n['neuron']} {n.get('polarity','')}"
    desc = n.get("generated_description")
    label_html = (
        f'<div class="label">{_esc(desc)}</div>' if desc
        else '<div class="label none">(no description — run generate_description.py)</div>'
    )
    promote, suppress = _split_contribs(n.get("output_contributions"))
    try:
        attr_s = f"{float(n.get('attribution', 0.0)):.3f}"
    except (TypeError, ValueError):
        attr_s = str(n.get("attribution", ""))
    return f"""
    <div class="neuron">
      <span class="nid">{_esc(nid)}</span><span class="rank">rank {_esc(n.get('rank',''))}</span>
      <span class="attr">attr {attr_s}</span>
      {label_html}
      <div class="txt">{_highlight(n.get('highlighted_text',''))}</div>
      <h5>Input attribution (per token)</h5>{_attr_tokens(n)}
      <div class="row">
        <div class="col"><h5>Driver tokens</h5>{_toks(n.get('top_input_tokens'), 'tok')}</div>
        <div class="col"><h5>Promotes</h5>{_toks(promote, 'promote')}</div>
        <div class="col"><h5>Suppresses</h5>{_toks(suppress, 'suppress')}</div>
      </div>
    </div>"""


def _raw_card(layer: int, token: int, neuron: int, attribution, activation, tok_str: str) -> str:
    def num(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)
    where = f' at token {token} (<code>{_esc(tok_str)}</code>)' if tok_str else f" at token {token}"
    return (
        f'<h3>L{layer} · N{neuron}</h3>'
        '<p class="hint">Graph node not in the exported top-N features, so it has no '
        'description / activation text. Increase <code>--top-n</code> at export time to '
        'curate more neurons.</p>'
        f'<div class="kv">layer <b>{layer}</b>{where}</div>'
        f'<div class="kv">attribution <b>{num(attribution)}</b> · activation <b>{num(activation)}</b></div>'
    )


def _group_members(graph: dict) -> "OrderedDict[str, list[dict]]":
    members: "OrderedDict[str, list[dict]]" = OrderedDict()
    neurons = graph.get("neurons", [])
    grouped = bool(graph.get("supernodes")) or any(n.get("group") for n in neurons)
    for n in neurons:
        g = (n.get("group") or "Ungrouped") if grouped else "All neurons"
        members.setdefault(g, []).append(n)
    if "Ungrouped" in members:
        members.move_to_end("Ungrouped")
    return members


# ---------------------------------------------------------------------------
# Top graph: scatter of every neuron (token x layer)
# ---------------------------------------------------------------------------

def _svg_scatter(graph, feat_by_ln, gcolor, max_layer, prompt_tokens):
    nodes = graph.get("nodes") or []
    if not nodes:
        return ('<p class="hint" style="padding:10px">No node-level graph in this file '
                '(exported with --no-nodes). The Subgraph below still works.</p>'), {}

    buckets = defaultdict(list)
    for nd in nodes:
        buckets[(nd["layer"], nd["token"])].append(nd)
    tokens = sorted({nd["token"] for nd in nodes})
    layers = sorted({nd["layer"] for nd in nodes})
    ti = {t: i for i, t in enumerate(tokens)}
    li = {l: i for i, l in enumerate(layers)}

    PADL, PADT, PADB, XGAP, YGAP = 52, 16, 66, 48, 25
    nrows, ncols = len(layers), len(tokens)
    width = PADL + (ncols - 1) * XGAP + 64
    height = PADT + (nrows - 1) * YGAP + PADB

    def X(t):
        return PADL + ti[t] * XGAP

    def Y(l):
        return PADT + (nrows - 1 - li[l]) * YGAP

    axis = []
    for l in layers:
        lab = "Emb" if l == -1 else ("Out" if l >= max_layer else f"L{l}")
        axis.append(f'<text x="6" y="{Y(l) + 3:.1f}" fill="#aaa">{_esc(lab)}</text>')
    for t in tokens:
        tok = prompt_tokens[t] if 0 <= t < len(prompt_tokens) else "?"
        tx, tyy = X(t), height - PADB + 14
        axis.append(
            f'<text x="{tx:.1f}" y="{tyy:.1f}" fill="#999" text-anchor="end" '
            f'transform="rotate(-45 {tx:.1f} {tyy:.1f})">{_esc(str(tok)[:14])}</text>'
        )

    pos = {}
    node_svg = []
    for (layer, token), nds in buckets.items():
        k = len(nds)
        bx, by = X(token), Y(layer)
        for i, nd in enumerate(nds):
            cx = bx + (i - (k - 1) / 2) * 9
            cy = by
            nid = f"N:{layer}:{token}:{nd['neuron']}"
            pos[nid] = (cx, cy)
            feat = feat_by_ln.get((layer, nd["neuron"]))
            if layer == -1:
                fill, did, shape = _EMB_COLOR, nid, "rect"
            elif layer >= max_layer:
                fill, did, shape = _OUT_COLOR, nid, "rect"
            elif feat:
                fill, did, shape = gcolor.get(feat[1], _UNGROUPED_COLOR), "F:" + feat[0], "circle"
            else:
                fill, did, shape = _UNGROUPED_COLOR, nid, "circle"
            if shape == "rect":
                node_svg.append(
                    f'<rect data-id="{_a(did)}" x="{cx - 4.5:.1f}" y="{cy - 4.5:.1f}" width="9" '
                    f'height="9" rx="2" fill="{fill}" stroke="#7a7a7a" stroke-width="0.5"/>'
                )
            else:
                node_svg.append(
                    f'<circle data-id="{_a(did)}" cx="{cx:.1f}" cy="{cy:.1f}" r="4.6" '
                    f'fill="{fill}" stroke="#7a7a7a" stroke-width="0.5"/>'
                )

    es = graph.get("edges") or []
    ws = [abs(float(e.get("attribution", e.get("weight", 0)) or 0)) for e in es]
    mw = max(ws) if ws else 1.0
    edge_svg = []
    for e in es:
        s = f"N:{e['src']['layer']}:{e['src']['token']}:{e['src']['neuron']}"
        t = f"N:{e['tgt']['layer']}:{e['tgt']['token']}:{e['tgt']['neuron']}"
        if s in pos and t in pos:
            (x1, y1), (x2, y2) = pos[s], pos[t]
            w = abs(float(e.get("attribution", e.get("weight", 0)) or 0))
            edge_svg.append(
                f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke-width="{0.5 + 1.5 * (w / mw):.2f}" stroke-opacity="{0.07 + 0.5 * (w / mw):.2f}"/>'
            )

    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{"".join(edge_svg)}{"".join(node_svg)}{"".join(axis)}</svg>'
    )
    return svg, pos


# ---------------------------------------------------------------------------
# Subgraph: supernode boxes (members inside) + Embeddings + Output
# ---------------------------------------------------------------------------

def _svg_subgraph(members, depths, gcolor, agg, used):
    HEADER, ROWH, BPAD, COLW, BOXW, GAPY, PAD = 22, 15, 7, 210, 178, 16, 16

    nodelist = []  # (id, kind, depth, group_name)
    for g in members:
        nodelist.append(("SN:" + g, "ung" if g == "Ungrouped" else "sn", depths[g], g))
    base = list(depths.values()) or [0.0]
    if EMB in used:
        nodelist.append((EMB, "emb", min(base) - 1, None))
    if OUT in used:
        nodelist.append((OUT, "out", max(base) + 1, None))
    if not nodelist:
        return ""

    MAXCOL = 7
    ds = [d for _, _, d, _ in nodelist]
    lo, hi = min(ds), max(ds)
    span = (hi - lo) or 1.0
    cols = defaultdict(list)
    for nd in nodelist:
        cols[round((nd[2] - lo) / span * (MAXCOL - 1))].append(nd)
    for c in cols:
        cols[c].sort(key=lambda nd: nd[2])
    used_cols = sorted(cols)

    def box_h(nid, gname):
        if gname is None:
            return 34
        return HEADER + max(1, len(members[gname])) * ROWH + BPAD

    pos = {}
    col_h = {}
    for c in used_cols:
        y = PAD
        for nid, kind, dep, gname in cols[c]:
            h = box_h(nid, gname)
            pos[nid] = (PAD + used_cols.index(c) * COLW, y, h, kind, gname)
            y += h + GAPY
        col_h[c] = y
    height = max(col_h.values()) if col_h else 80
    width = PAD * 2 + (len(used_cols) - 1) * COLW + BOXW

    # Edges between box centers (drawn first).
    mw = max(agg.values()) if agg else 1.0
    edge_svg = []
    for (a, b), w in sorted(agg.items(), key=lambda kv: kv[1]):
        if a not in pos or b not in pos:
            continue
        ax, ay, ah, _, _ = pos[a]
        bx, by, bh, _, _ = pos[b]
        sx, sy = ax + BOXW, ay + ah / 2
        tx, ty = bx, by + bh / 2
        dx = max(24.0, abs(tx - sx) / 2)
        edge_svg.append(
            f'<path class="edge" d="M{sx:.1f},{sy:.1f} C{sx + dx:.1f},{sy:.1f} {tx - dx:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" '
            f'stroke-width="{1.0 + 5.0 * (w / mw):.2f}" stroke-opacity="{0.25 + 0.5 * (w / mw):.2f}" marker-end="url(#sarr)"/>'
        )

    box_svg = []
    for nid, (x, y, h, kind, gname) in pos.items():
        if kind == "emb":
            stroke, fill, title = _EMB_COLOR, "#f1faf4", "Embeddings"
        elif kind == "out":
            stroke, fill, title = _OUT_COLOR, "#fdf3f3", "Output"
        else:
            stroke, fill, title = gcolor.get(gname, _UNGROUPED_COLOR), "#ffffff", gname
        box_svg.append(
            f'<g><rect data-id="{_a(nid)}" x="{x}" y="{y:.1f}" width="{BOXW}" height="{h:.1f}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text data-id="{_a(nid)}" x="{x + 9}" y="{y + 15:.1f}" font-weight="700" fill="{stroke}">'
            f'{_esc(title[:24])}'
            + (f'  ({len(members[gname])})' if gname is not None else "")
            + "</text>"
        )
        if gname is not None:
            for j, n in enumerate(members[gname]):
                ry = y + HEADER + 4 + j * ROWH
                lab = f"L{n['layer']} N{n['neuron']} {n.get('polarity','')}".strip()
                box_svg.append(
                    f'<text class="mrow" data-id="F:{_a(_fid(n))}" x="{x + 12}" y="{ry:.1f}" fill="#444">'
                    f'{_esc(lab)}</text>'
                )
        box_svg.append("</g>")

    return (
        f'<svg width="{width}" height="{height:.0f}" viewBox="0 0 {width} {height:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<defs><marker id="sarr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 z" fill="#9bb0d6"/></marker></defs>'
        f'{"".join(edge_svg)}{"".join(box_svg)}</svg>'
    )


def _aggregate_edges(graph, ln_to_group, max_layer):
    """Collapse (layer,token,neuron) edges into SN/EMB/OUT node-id sums."""
    edges = graph.get("edges") or []

    def endpoint(ep):
        layer, neuron = ep["layer"], ep["neuron"]
        if layer == -1:
            return EMB
        g = ln_to_group.get((layer, neuron))
        if g is not None:
            return "SN:" + g
        if layer >= max_layer:
            return OUT
        return None

    agg = defaultdict(float)
    used = set()
    for e in edges:
        a, b = endpoint(e["src"]), endpoint(e["tgt"])
        if a is None or b is None or a == b:
            continue
        try:
            w = abs(float(e.get("attribution", e.get("weight", 0.0))))
        except (TypeError, ValueError):
            continue
        agg[(a, b)] += w
        used.update((a, b))
    return agg, used


def build_view(graph: dict) -> dict:
    members = _group_members(graph)
    depths = {g: (sum(n["layer"] for n in ns) / len(ns) if ns else 0.0) for g, ns in members.items()}
    prompt_tokens: list[str] = []
    for ns in members.values():
        if ns and ns[0].get("tokens"):
            prompt_tokens = ns[0]["tokens"]
            break

    # (layer, neuron) -> (feature_id, group, neuron dict); and a color per group.
    feat_by_ln: dict[tuple, tuple] = {}
    ln_to_group: dict[tuple, str] = {}
    for g, ns in members.items():
        for n in ns:
            feat_by_ln[(n["layer"], n["neuron"])] = (_fid(n), g, n)
            ln_to_group[(n["layer"], n["neuron"])] = g
    gcolor = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(g for g in members if g != "Ungrouped")}
    gcolor["Ungrouped"] = _UNGROUPED_COLOR
    gcolor["All neurons"] = _PALETTE[0]

    node_layers = [nd["layer"] for nd in graph.get("nodes", [])]
    edge_layers = [e["src"]["layer"] for e in graph.get("edges", [])] + \
                  [e["tgt"]["layer"] for e in graph.get("edges", [])]
    max_layer = max(node_layers + edge_layers + [int(max(depths.values(), default=0))], default=0)

    scatter, _pos = _svg_scatter(graph, feat_by_ln, gcolor, max_layer, prompt_tokens)
    agg, used = _aggregate_edges(graph, ln_to_group, max_layer)
    if any(nd["layer"] == -1 for nd in graph.get("nodes", [])):
        used.add(EMB)
    if node_layers and max(node_layers) >= max_layer and max_layer > 0:
        used.add(OUT)
    sub = _svg_subgraph(members, depths, gcolor, agg, used)

    # Detail panels: features, raw nodes, supernodes, EMB, OUT.
    detail: dict[str, str] = {}
    for g, ns in members.items():
        dep = "" if not ns else f'<span class="dep">~L{depths[g]:.0f}</span>'
        detail["SN:" + g] = (
            f'<h3>{_esc(g)} <span class="cnt">({len(ns)})</span>{dep}</h3>'
            + "".join(_neuron_card(n) for n in ns)
        )
        for n in ns:
            detail["F:" + _fid(n)] = f"<h3>L{n['layer']} N{n['neuron']} {_esc(n.get('polarity',''))}</h3>" + _neuron_card(n)
    for nd in graph.get("nodes", []):
        layer, token, neuron = nd["layer"], nd["token"], nd["neuron"]
        if (layer, neuron) in feat_by_ln or layer == -1 or layer >= max_layer:
            continue
        nid = f"N:{layer}:{token}:{neuron}"
        tok = prompt_tokens[token] if 0 <= token < len(prompt_tokens) else ""
        detail[nid] = _raw_card(layer, token, neuron, nd.get("attribution"), nd.get("activation"), tok)
    if EMB in used:
        chips = "".join(f'<span class="tok">{_esc(t)}</span>' for t in prompt_tokens) or "—"
        detail[EMB] = ('<h3>Embeddings</h3><p class="hint">Input token embeddings (layer -1), '
                       'kept separate from the supernodes.</p>'
                       f'<div class="attrtoks">{chips}</div>')
    if OUT in used:
        promoted: dict[str, float] = defaultdict(float)
        for n in graph.get("neurons", [])[:12]:
            for it in n.get("output_contributions") or []:
                try:
                    tok, sc = it[0], float(it[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if sc > 0:
                    promoted[tok] += sc
        top = sorted(promoted, key=lambda t: promoted[t], reverse=True)[:10]
        chips = "".join(f'<span class="tok promote">{_esc(t)}</span>' for t in top) or "—"
        detail[OUT] = (f'<h3>Output</h3><p class="hint">Logit / output node. Traced target: '
                       f'<b>{_esc(graph.get("target",""))}</b>.</p>'
                       f'<h5>Most promoted output tokens across top features</h5>'
                       f'<div class="attrtoks">{chips}</div>')

    real = [g for g in members if g != "Ungrouped"] or list(members)
    default = ("SN:" + max(real, key=lambda g: sum(abs(float(n.get("attribution", 0.0))) for n in members[g]))) \
        if real else (EMB if EMB in used else "")

    return {
        "name": "", "prompt": graph.get("prompt", "") or "", "target": graph.get("target", "") or "",
        "scatter": scatter, "sub": sub, "detail": detail, "def": default,
    }


def build_html(graphs: list[tuple[str, dict]]) -> str:
    views = []
    for name, g in graphs:
        v = build_view(g)
        v["name"] = name
        views.append(v)

    options = "".join(
        f'<option value="{i}">{_esc(v["target"] or v["name"])}</option>' for i, v in enumerate(views)
    )
    selector = f'<label class="sellbl">Prompt&nbsp;<select id="sel">{options}</select></label>'
    data = json.dumps(views, ensure_ascii=False).replace("</", "<\\/")

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>ADAG neuron viewer</title>"
        f"<style>{_CSS}</style></head><body><div class='wrap'>"
        f"<div class='topbar'>{selector}<h1>ADAG supernode viewer</h1></div>"
        "<div class='prompt' id='promptbar'></div>"
        "<div class='viewer'><div class='graphcol'>"
        "<div class='box'><h4>Attribution graph — all neurons (token × layer)</h4><div id='graph'></div></div>"
        "<div class='box'><h4>Subgraph — supernodes (members inside) + embeddings + output</h4><div id='subgraph'></div></div>"
        "</div><div class='panel' id='panel'></div></div>"
        "</div>"
        f"<script>var GRAPHS={data};</script><script>{_JS}</script>"
        "</body></html>"
    )


def _serve(out: Path, port: int) -> None:
    import functools
    import http.server
    import socketserver
    import webbrowser

    out = out.resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out.parent))
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://localhost:{port}/{out.name}"
    try:
        httpd = socketserver.TCPServer(("", port), handler)
    except OSError as exc:
        print(f"Could not bind port {port} ({exc}). Try a different --port.")
        return
    with httpd:
        print(f"Serving at {url}   (Ctrl+C to stop)")
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render ADAG neuron graphs to a local interactive HTML report.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph", type=Path)
    g.add_argument("--graphs-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("report.html"))
    ap.add_argument("--serve", action="store_true",
                    help="After writing, serve the report on localhost and open it in a browser.")
    ap.add_argument("--port", type=int, default=8041, help="Port for --serve (default: 8041).")
    args = ap.parse_args()

    if args.graph:
        paths = [args.graph]
    else:
        paths = sorted(args.graphs_dir.glob("graph_*.json"))

    graphs = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    args.out.write_text(build_html(graphs), encoding="utf-8")
    print(f"Wrote {args.out}  ({len(graphs)} graph(s)).")

    if args.serve:
        _serve(args.out, args.port)
    else:
        print(f"Open it directly, or serve it:  python render_report.py ... --serve  "
              f"(or: python -m http.server {args.port})")


if __name__ == "__main__":
    main()
