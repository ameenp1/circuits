"""
cross_graph_analysis.py — compare the MLP (ADAG) attribution graph against the
SLT (cross-layer transcoder) graph for the same neuronpedia prompt.

This is the "Cross-Graph Analysis" step. For a given slug it answers three questions:

  (A) STRUCTURAL DIFFERENCES — how do the two graphs differ in node counts, node
      types, layer distribution, edge density, supernode count, and % grouped? Are
      the differences meaningful (e.g. the transcoder graph carries an MLP-error
      basis the MLP graph cannot, the MLP graph is sparser, etc.)?

  (B) SUPERNODE COMPOSITION (both vs one type) — build one UNIFIED set of supernode
      concepts by matching MLP supernodes to transcoder supernodes via description
      similarity, then classify each concept as:
        * BOTH            — found by both methods (transcoder features AND MLP neurons)
        * MLP-only        — only the MLP graph surfaces it
        * transcoder-only — only the transcoder graph surfaces it

  (C) AGREEMENT vs DIVERGENCE — characterize, via description matching, where the two
      bases agree and where they diverge: per-matched-concept similarity, member-level
      nearest-neighbour similarity, shared promoted-token overlap, layer-band overlap,
      and feature-level coverage (what fraction of each side has a close analogue on
      the other).

Both sides expose the SAME evidence fields, which makes the comparison clean:
  - a per-feature `generated_description` (the matching key)
  - a supernode `group` name (the same a2 LLM-grouping scheme on both sides)
  - promoted / suppressed output tokens

DATA SCHEMAS
------------
MLP / ADAG graph_*.json (produced by this repo: batch_export_neurons.py ->
generate_description.py -> generate_supernodes.py):
    {
      "label": "gemma-G___0", "prompt": "...", "target": " G",
      "neurons": [{"layer":20,"neuron":1234,"polarity":"+","attribution":0.41,
                   "generated_description":"...", "group":"say capital",
                   "output_contributions":[[" Austin",0.55],[" foo",-0.1]]}, ...],
      "supernodes": {"say capital": ["L20_N1234_+", ...], ...},
      "ungrouped": ["L3_N7_+", ...],
      "nodes": [...], "edges": [...]          # token-level graph (optional)
    }
  feature id = L{layer}_N{neuron}_{polarity}

Transcoder / SLT side (HF circuit-tracer-automation/pipeline_automation, pulled by
fetch_neuronpedia_artifacts.py):
    test_graphs/gemma-<slug>.json            : {metadata, nodes[], links[]}
        node = {node_id "L_feat_ctx", feature_type, layer, ctx_idx, influence, ...}
        feature_type in {cross layer transcoder, mlp reconstruction error,
                         embedding, logit}
    artifacts/<slug>__feature_groups_v2_a2.json    : {node_id -> group_name}
    artifacts/<slug>__feature_descriptions_v2.json : [{id, layer, ctx_idx,
        influence_score, generated_description, promotes:[tok], demotes:[tok]}, ...]

Usage:
    # one slug (transcoder side auto-resolved from --np-dir)
    python cross_graph_analysis.py --slug gemma-dollar \
        --mlp-graph ../neuronpedia_mlp_graphs/graph_0004_gemma-dollar.json \
        --np-dir np_data --out-dir cross_graph_out

    # a whole folder of ADAG graphs (one per slug), batched + a summary
    python cross_graph_analysis.py --mlp-dir ../neuronpedia_mlp_graphs \
        --np-dir np_data --out-dir cross_graph_out

    # no MLP graphs yet? synthesize a stand-in from the transcoder side to see the
    # full report shape end-to-end (clearly labelled synthetic):
    python cross_graph_analysis.py --slug gemma-dollar --np-dir np_data \
        --demo-mlp --out-dir cross_graph_out

Embeddings: uses OpenAI text-embedding-3-small when OPENAI_API_KEY is set (the same
key the rest of the pipeline uses); otherwise falls back to a deterministic offline
TF-IDF matcher (sklearn). Override with --embed {auto,openai,tfidf}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Tolerant stdout for non-ASCII tokens on Windows terminals.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ===========================================================================
# Evidence-regime caveat — emitted at the top of every report (analysis (c))
# ===========================================================================
# The two sides' descriptions are built from DIFFERENT evidence, so a divergence
# in (B)/(C) can be a method artifact rather than a representational difference.
# This banner makes that explicit so results aren't over-read before corpus
# raw-activation exemplars exist (see CORPUS_EXEMPLARS_SPEC.md).
EVIDENCE_REGIME_CAVEAT = (
    "> ⚠️ **Evidence-regime caveat — read before interpreting divergences.**\n"
    "> The two sides' `generated_description` text — the key this analysis matches on —\n"
    "> is built from **different evidence**, so a divergence below may be a *method\n"
    "> artifact*, not a real representational difference:\n"
    ">\n"
    "> - **MLP (ADAG):** per-prompt, **input-attribution** window — the tokens that drove\n"
    ">   this neuron on *this* prompt. One example, no corpus.\n"
    "> - **Transcoder (SLT):** corpus-wide, **raw-activation** max-activating exemplars —\n"
    ">   the feature's global selectivity. Many examples.\n"
    ">\n"
    "> Two axes differ at once (corpus-vs-per-prompt AND activation-vs-attribution).\n"
    "> Separately, transcoder latents are sparse/near-monosemantic by construction while\n"
    "> raw MLP neurons are polysemantic, so MLP descriptions are legitimately vaguer —\n"
    "> surface that as a finding, don't read it as failure. Until corpus raw-activation\n"
    "> exemplars exist for the traced neurons (CORPUS_EXEMPLARS_SPEC.md), treat the\n"
    "> BOTH / MLP-only / transcoder-only splits and agree/diverge numbers as\n"
    "> **hypotheses**, not conclusions."
)


# ===========================================================================
# Normalized representation
# ===========================================================================

@dataclass
class Feature:
    fid: str            # id within its own graph
    layer: int
    ctx: int            # token position
    score: float        # |attribution| (MLP) or influence (transcoder)
    desc: str           # generated_description (the matching key)
    group: str          # supernode name, or "Ungrouped"
    promotes: list[str] = field(default_factory=list)
    demotes: list[str] = field(default_factory=list)


@dataclass
class Graph:
    side: str           # "mlp" | "transcoder"
    slug: str
    prompt: str
    target: str
    features: list[Feature]
    node_type_counts: dict      # node-type -> count (structural, all nodes)
    n_token_nodes: int          # token-level node count
    n_edges: int
    n_tokens: int               # prompt length in tokens
    note: str = ""              # e.g. "SYNTHETIC" for demo graphs

    def concept_supernodes(self) -> dict[str, list[Feature]]:
        """name -> member features, concept groups only (drops Ungrouped / Emb / Output)."""
        out: dict[str, list[Feature]] = {}
        for f in self.features:
            if is_concept_group(f.group):
                out.setdefault(f.group, []).append(f)
        return out

    def grouped_fraction(self) -> float:
        if not self.features:
            return 0.0
        grouped = sum(1 for f in self.features if is_concept_group(f.group))
        return grouped / len(self.features)


def is_concept_group(name: str | None) -> bool:
    if not name:
        return False
    if name == "Ungrouped":
        return False
    if name.startswith(("Emb:", "Output:")):
        return False
    return True


# ===========================================================================
# Loaders
# ===========================================================================

def _split_signed(contribs) -> tuple[list[str], list[str]]:
    """ADAG output_contributions [[tok, signed]] -> (promoted, suppressed) token lists."""
    pos, neg = [], []
    for item in contribs or []:
        try:
            tok, score = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        (pos if score >= 0 else neg).append(tok)
    return pos, neg


def load_mlp_graph(path: Path) -> Graph:
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    slug = (g.get("label") or g.get("slug") or path.stem).split("___")[0]
    feats: list[Feature] = []
    for n in g.get("neurons", []):
        desc = (n.get("generated_description") or "").strip()
        if not desc or desc == "Error generating description":
            continue
        pol = n.get("polarity", "")
        fid = f"L{n['layer']}_N{n['neuron']}{('_' + pol) if pol else ''}"
        promotes, demotes = _split_signed(n.get("output_contributions"))
        feats.append(Feature(
            fid=fid, layer=int(n["layer"]), ctx=int(n.get("token", 0) or 0),
            score=abs(float(n.get("attribution", 0.0))), desc=desc,
            group=n.get("group", "Ungrouped"), promotes=promotes, demotes=demotes,
        ))
    nodes = g.get("nodes", []) or []
    return Graph(
        side="mlp", slug=slug, prompt=g.get("prompt", ""), target=g.get("target", ""),
        features=feats,
        node_type_counts={"mlp neuron": len(nodes)} if nodes else {"mlp neuron (feature)": len(feats)},
        n_token_nodes=len(nodes), n_edges=len(g.get("edges", []) or []),
        n_tokens=len(g.get("prompt_tokens") or []) or 0,
    )


def load_transcoder_graph(tg_path: Path, groups_path: Path, desc_path: Path) -> Graph:
    tg = json.loads(Path(tg_path).read_text(encoding="utf-8"))
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8"))
    descs = json.loads(Path(desc_path).read_text(encoding="utf-8"))

    meta = tg.get("metadata", {})
    slug = meta.get("slug", Path(tg_path).stem)
    prompt = meta.get("prompt", "")
    n_tokens = len(meta.get("prompt_tokens") or [])

    # Structural node-type counts + edges from the test graph.
    from collections import Counter
    nodes = tg.get("nodes", [])
    type_counts = dict(Counter(n.get("feature_type", "?") for n in nodes))
    n_edges = len(tg.get("links", []))

    # target token from the logit / target node
    target = ""
    for n in nodes:
        if n.get("is_target_logit") or n.get("feature_type") == "logit":
            target = (n.get("clerp") or n.get("token") or "").strip() or target
            if n.get("is_target_logit"):
                break

    feats: list[Feature] = []
    for e in descs:
        desc = (e.get("generated_description") or "").strip()
        if not desc:
            continue
        fid = e["id"]
        feats.append(Feature(
            fid=fid, layer=int(e.get("layer", 0)), ctx=int(e.get("ctx_idx", 0)),
            score=float(e.get("influence_score", 0.0)), desc=desc,
            group=groups.get(fid, "Ungrouped"),
            promotes=list(e.get("promotes") or []), demotes=list(e.get("demotes") or []),
        ))
    return Graph(
        side="transcoder", slug=slug, prompt=prompt, target=target, features=feats,
        node_type_counts=type_counts, n_token_nodes=len(nodes), n_edges=n_edges,
        n_tokens=n_tokens,
    )


def make_demo_mlp_from_transcoder(tc: Graph, keep_frac: float = 0.45, seed: int = 0) -> Graph:
    """Synthesize a *stand-in* MLP graph from a transcoder graph so the full report
    path can be exercised before the real Runpod MLP graphs are available.

    It keeps a random subset of the transcoder concept features (simulating the MLP
    graph's sparser, partly-overlapping coverage), re-keys them as L{layer}_N{n}_+,
    drops a couple of concept groups entirely (to create transcoder-only concepts),
    and lightly perturbs descriptions. CLEARLY SYNTHETIC — not a real attribution.
    """
    rng = np.random.default_rng(seed)
    concept = [f for f in tc.features if is_concept_group(f.group)]
    # drop ~1/4 of concept groups so some end up transcoder-only
    gnames = sorted({f.group for f in concept})
    drop = set(rng.choice(gnames, size=max(1, len(gnames) // 4), replace=False)) if gnames else set()
    feats: list[Feature] = []
    for i, f in enumerate(concept):
        if f.group in drop:
            continue
        if rng.random() > keep_frac:
            continue
        feats.append(Feature(
            fid=f"L{f.layer}_N{1000 + i}_+", layer=f.layer, ctx=f.ctx,
            score=float(rng.random()), desc=f.desc, group=f.group,
            promotes=f.promotes[:5], demotes=f.demotes[:5],
        ))
    return Graph(
        side="mlp", slug=tc.slug, prompt=tc.prompt, target=tc.target, features=feats,
        node_type_counts={"mlp neuron (feature)": len(feats)},
        n_token_nodes=len(feats), n_edges=int(len(feats) * 2.5), n_tokens=tc.n_tokens,
        note="SYNTHETIC demo MLP graph (NOT a real attribution) — for report-shape validation only",
    )


# ===========================================================================
# Embedding backends
# ===========================================================================

class Embedder:
    """Maps descriptions/concepts to L2-normalized vectors. OpenAI or offline TF-IDF."""

    def __init__(self, mode: str, corpus: list[str]):
        self.mode = mode
        self._cache: dict[str, np.ndarray] = {}
        uniq = sorted(set(t for t in corpus if t))
        if mode == "openai":
            self._embed_openai(uniq)
            self.default_match_threshold = 0.55
            self.default_feature_threshold = 0.55
        else:  # tfidf
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            if uniq:
                mat = self._vec.fit_transform(uniq)
                for t, row in zip(uniq, mat):
                    v = np.asarray(row.todense()).ravel()
                    nrm = np.linalg.norm(v)
                    self._cache[t] = v / nrm if nrm else v
            self.default_match_threshold = 0.30
            self.default_feature_threshold = 0.30

    def _embed_openai(self, texts: list[str]) -> None:
        from openai import OpenAI
        client = OpenAI()
        for i in range(0, len(texts), 256):
            chunk = texts[i:i + 256]
            resp = client.embeddings.create(model="text-embedding-3-small", input=chunk)
            for t, d in zip(chunk, resp.data):
                v = np.asarray(d.embedding, dtype=np.float32)
                nrm = np.linalg.norm(v)
                self._cache[t] = v / nrm if nrm else v

    def vec(self, text: str) -> np.ndarray | None:
        if not text:
            return None
        if text in self._cache:
            return self._cache[text]
        if self.mode == "tfidf" and getattr(self, "_vec", None) is not None:
            try:
                row = self._vec.transform([text])
                v = np.asarray(row.todense()).ravel()
                nrm = np.linalg.norm(v)
                v = v / nrm if nrm else v
                self._cache[text] = v
                return v
            except Exception:  # noqa: BLE001
                return None
        return None

    def mean_vec(self, texts: list[str]) -> np.ndarray | None:
        vs = [v for v in (self.vec(t) for t in texts) if v is not None]
        if not vs:
            return None
        m = np.mean(np.stack(vs), axis=0)
        nrm = np.linalg.norm(m)
        return m / nrm if nrm else m


def build_embedder(mode: str, *graphs: Graph) -> Embedder:
    if mode == "auto":
        mode = "openai" if os.environ.get("OPENAI_API_KEY") else "tfidf"
    corpus: list[str] = []
    for g in graphs:
        corpus.extend(f.desc for f in g.features)
        corpus.extend(set(f.group for f in g.features if is_concept_group(f.group)))
    return Embedder(mode, corpus)


def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# ===========================================================================
# (A) Structural comparison
# ===========================================================================

def _layer_stats(feats: list[Feature]) -> dict:
    if not feats:
        return {"min": None, "median": None, "max": None}
    layers = sorted(f.layer for f in feats)
    return {
        "min": layers[0], "median": int(np.median(layers)), "max": layers[-1],
        "hist": dict(sorted({l: layers.count(l) for l in set(layers)}.items())),
    }


def structural_comparison(mlp: Graph, tc: Graph) -> dict:
    mlp_sn = mlp.concept_supernodes()
    tc_sn = tc.concept_supernodes()
    return {
        "mlp": {
            "described_features": len(mlp.features),
            "token_level_nodes": mlp.n_token_nodes,
            "edges": mlp.n_edges,
            "node_types": mlp.node_type_counts,
            "concept_supernodes": len(mlp_sn),
            "grouped_fraction": round(mlp.grouped_fraction(), 3),
            "layers": _layer_stats(mlp.features),
            "avg_supernode_size": round(np.mean([len(v) for v in mlp_sn.values()]), 2) if mlp_sn else 0,
        },
        "transcoder": {
            "described_features": len(tc.features),
            "token_level_nodes": tc.n_token_nodes,
            "edges": tc.n_edges,
            "node_types": tc.node_type_counts,
            "concept_supernodes": len(tc_sn),
            "grouped_fraction": round(tc.grouped_fraction(), 3),
            "layers": _layer_stats(tc.features),
            "avg_supernode_size": round(np.mean([len(v) for v in tc_sn.values()]), 2) if tc_sn else 0,
        },
    }


# ===========================================================================
# (B) Unified supernodes — both vs one type  (+ similarity for (C))
# ===========================================================================

def _norm_name(name: str) -> str:
    """Normalize a supernode name for exact-match comparison: lowercase, collapse
    whitespace, drop surrounding punctuation/quotes."""
    import re
    s = name.lower().strip().strip("\"'")
    s = re.sub(r"[\s_]+", " ", s)
    s = re.sub(r"[^\w /]+", "", s)
    return s.strip()


def match_supernodes(mlp: Graph, tc: Graph, emb: Embedder, threshold: float) -> dict:
    """Greedy 1-1 matching of MLP concept-supernodes to transcoder concept-supernodes.

    Both pipelines use the same a2 grouping scheme, so group NAMES carry real signal
    (often identical). The match score blends name similarity with mean-member-
    description similarity, and an exact normalized-name match is forced (sim 1.0)."""
    mlp_sn = mlp.concept_supernodes()
    tc_sn = tc.concept_supernodes()
    mlp_names = list(mlp_sn)
    tc_names = list(tc_sn)

    mlp_mvec = {n: emb.mean_vec([f.desc for f in mlp_sn[n]]) for n in mlp_names}
    tc_mvec = {n: emb.mean_vec([f.desc for f in tc_sn[n]]) for n in tc_names}
    mlp_nvec = {n: emb.vec(n) for n in mlp_names}
    tc_nvec = {n: emb.vec(n) for n in tc_names}
    mlp_norm = {n: _norm_name(n) for n in mlp_names}
    tc_norm = {n: _norm_name(n) for n in tc_names}

    # combined score: exact normalized-name -> 1.0; else 0.5*name + 0.5*members
    pairs = []
    for mn in mlp_names:
        for tn in tc_names:
            if mlp_norm[mn] == tc_norm[tn]:
                name_sim, member_sim, combined = 1.0, _cos(mlp_mvec[mn], tc_mvec[tn]), 1.0
            else:
                name_sim = _cos(mlp_nvec[mn], tc_nvec[tn])
                member_sim = _cos(mlp_mvec[mn], tc_mvec[tn])
                combined = 0.5 * name_sim + 0.5 * member_sim
            pairs.append((combined, name_sim, member_sim, mn, tn))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched, used_m, used_t = [], set(), set()
    for combined, name_sim, member_sim, mn, tn in pairs:
        if combined < threshold or mn in used_m or tn in used_t:
            continue
        used_m.add(mn)
        used_t.add(tn)
        matched.append({
            "mlp_group": mn, "transcoder_group": tn,
            "concept_similarity": round(combined, 3),
            "name_similarity": round(name_sim, 3),
            "member_similarity": round(member_sim, 3),
            "exact_name": mlp_norm[mn] == tc_norm[tn],
            "mlp_size": len(mlp_sn[mn]), "transcoder_size": len(tc_sn[tn]),
        })

    mlp_only = [n for n in mlp_names if n not in used_m]
    tc_only = [n for n in tc_names if n not in used_t]
    return {
        "matched_both": matched,
        "mlp_only": [{"group": n, "size": len(mlp_sn[n])} for n in mlp_only],
        "transcoder_only": [{"group": n, "size": len(tc_sn[n])} for n in tc_only],
        "_mlp_sn": mlp_sn, "_tc_sn": tc_sn,  # internal, stripped before JSON dump
    }


# ===========================================================================
# (C) Agreement vs divergence
# ===========================================================================

def _best_member_sim(srcs: list[Feature], tgts: list[Feature], emb: Embedder) -> float:
    """Mean over src features of max cosine to any tgt feature (description-level)."""
    if not srcs or not tgts:
        return 0.0
    tvecs = [emb.vec(f.desc) for f in tgts]
    sims = []
    for s in srcs:
        sv = emb.vec(s.desc)
        sims.append(max((_cos(sv, tv) for tv in tvecs), default=0.0))
    return float(np.mean(sims))


def _token_jaccard(a: list[Feature], b: list[Feature]) -> float:
    sa = {t for f in a for t in f.promotes}
    sb = {t for f in b for t in f.promotes}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _layer_overlap(a: list[Feature], b: list[Feature]) -> float:
    """Histogram-intersection overlap of the two layer distributions (0..1)."""
    if not a or not b:
        return 0.0
    la = [f.layer for f in a]
    lb = [f.layer for f in b]
    layers = set(la) | set(lb)
    da = {l: la.count(l) / len(la) for l in layers}
    db = {l: lb.count(l) / len(lb) for l in layers}
    return float(sum(min(da.get(l, 0), db.get(l, 0)) for l in layers))


def agreement_analysis(match: dict, emb: Embedder, feature_threshold: float,
                       mlp: Graph, tc: Graph) -> dict:
    mlp_sn, tc_sn = match["_mlp_sn"], match["_tc_sn"]

    per_concept = []
    for m in match["matched_both"]:
        a = mlp_sn[m["mlp_group"]]
        b = tc_sn[m["transcoder_group"]]
        per_concept.append({
            "mlp_group": m["mlp_group"], "transcoder_group": m["transcoder_group"],
            "concept_similarity": m["concept_similarity"],
            "mlp_to_tc_member_sim": round(_best_member_sim(a, b, emb), 3),
            "tc_to_mlp_member_sim": round(_best_member_sim(b, a, emb), 3),
            "promoted_token_jaccard": round(_token_jaccard(a, b), 3),
            "layer_overlap": round(_layer_overlap(a, b), 3),
        })
    # strong agreement = high concept sim AND decent member sim; nominal-but-diverging otherwise
    for c in per_concept:
        member = (c["mlp_to_tc_member_sim"] + c["tc_to_mlp_member_sim"]) / 2
        c["verdict"] = "agree" if member >= feature_threshold else "nominal (diverging members)"

    # feature-level coverage in both directions
    cov = feature_coverage(mlp, tc, emb, feature_threshold)
    return {"per_concept": per_concept, "feature_coverage": cov}


def feature_coverage(mlp: Graph, tc: Graph, emb: Embedder, threshold: float) -> dict:
    mvecs = [(f, emb.vec(f.desc)) for f in mlp.features]
    tvecs = [(f, emb.vec(f.desc)) for f in tc.features]
    t_only = [v for _, v in tvecs if v is not None]
    m_only = [v for _, v in mvecs if v is not None]

    def covered(srcs, pool):
        if not srcs or not pool:
            return 0.0, []
        P = np.stack(pool)
        best = []
        for _, v in srcs:
            if v is None:
                best.append(0.0)
                continue
            best.append(float(np.max(P @ v)))
        best = np.array(best)
        return float(np.mean(best >= threshold)), best

    mlp_cov, mlp_best = covered(mvecs, t_only)
    tc_cov, tc_best = covered(tvecs, m_only)
    return {
        "threshold": threshold,
        "mlp_features_with_transcoder_analogue": round(mlp_cov, 3),
        "transcoder_features_with_mlp_analogue": round(tc_cov, 3),
        "mlp_nn_sim_mean": round(float(np.mean(mlp_best)), 3) if len(mlp_best) else 0.0,
        "transcoder_nn_sim_mean": round(float(np.mean(tc_best)), 3) if len(tc_best) else 0.0,
    }


# ===========================================================================
# Orchestration + reporting
# ===========================================================================

def analyze_pair(mlp: Graph, tc: Graph, emb: Embedder,
                 match_threshold: float, feature_threshold: float) -> dict:
    structural = structural_comparison(mlp, tc)
    match = match_supernodes(mlp, tc, emb, match_threshold)
    agreement = agreement_analysis(match, emb, feature_threshold, mlp, tc)

    n_both = len(match["matched_both"])
    n_m = len(match["mlp_only"])
    n_t = len(match["transcoder_only"])
    match_public = {k: v for k, v in match.items() if not k.startswith("_")}

    report = {
        "slug": mlp.slug,
        "prompt": mlp.prompt or tc.prompt,
        "target": {"mlp": mlp.target, "transcoder": tc.target},
        "embedding_backend": emb.mode,
        "thresholds": {"supernode_match": match_threshold, "feature": feature_threshold},
        "notes": {k: v.note for k, v in (("mlp", mlp), ("transcoder", tc)) if v.note},
        "A_structural": structural,
        "B_supernode_composition": {
            "counts": {"both": n_both, "mlp_only": n_m, "transcoder_only": n_t},
            **match_public,
        },
        "C_agreement_divergence": agreement,
    }
    return report


def _fmt_layers(d: dict) -> str:
    return f"{d.get('min')}–{d.get('max')} (median {d.get('median')})"


def render_markdown(rep: dict) -> str:
    A = rep["A_structural"]
    m, t = A["mlp"], A["transcoder"]
    B = rep["B_supernode_composition"]
    C = rep["C_agreement_divergence"]
    cov = C["feature_coverage"]
    L: list[str] = []
    P = L.append
    P(f"# Cross-graph analysis — `{rep['slug']}`")
    P("")
    P(f"**Prompt:** `{rep['prompt']}`  ")
    P(f"**Target:** MLP `{rep['target']['mlp']}` · transcoder `{rep['target']['transcoder']}`  ")
    P(f"**Embedding backend:** {rep['embedding_backend']} · "
      f"supernode-match≥{rep['thresholds']['supernode_match']}, "
      f"feature≥{rep['thresholds']['feature']}")
    if rep.get("notes"):
        for k, v in rep["notes"].items():
            P(f"> ⚠️ {k}: {v}")
    P("")
    P(EVIDENCE_REGIME_CAVEAT)
    P("")

    P("## A. Structural differences")
    P("")
    P("| metric | MLP (ADAG) | transcoder (SLT) |")
    P("|---|---:|---:|")
    P(f"| described features | {m['described_features']} | {t['described_features']} |")
    P(f"| token-level nodes | {m['token_level_nodes']} | {t['token_level_nodes']} |")
    P(f"| edges / links | {m['edges']} | {t['edges']} |")
    P(f"| concept supernodes | {m['concept_supernodes']} | {t['concept_supernodes']} |")
    P(f"| grouped fraction | {m['grouped_fraction']} | {t['grouped_fraction']} |")
    P(f"| avg supernode size | {m['avg_supernode_size']} | {t['avg_supernode_size']} |")
    P(f"| layer range | {_fmt_layers(m['layers'])} | {_fmt_layers(t['layers'])} |")
    P("")
    P(f"- **MLP node types:** {m['node_types']}")
    P(f"- **Transcoder node types:** {t['node_types']}")
    P("")

    P("## B. Supernode composition — both vs one type")
    P("")
    c = B["counts"]
    P(f"**{c['both']}** concepts found by BOTH · **{c['mlp_only']}** MLP-only · "
      f"**{c['transcoder_only']}** transcoder-only")
    P("")
    if B["matched_both"]:
        P("### Found by both (transcoder features AND MLP neurons)")
        P("| MLP supernode | transcoder supernode | concept | name | member | exact | MLP n | tc n |")
        P("|---|---|---:|---:|---:|:--:|---:|---:|")
        for x in B["matched_both"]:
            P(f"| {x['mlp_group']} | {x['transcoder_group']} | {x['concept_similarity']} "
              f"| {x.get('name_similarity','')} | {x.get('member_similarity','')} "
              f"| {'✓' if x.get('exact_name') else ''} | {x['mlp_size']} | {x['transcoder_size']} |")
        P("")
    if B["mlp_only"]:
        P("### MLP-only concepts")
        P(", ".join(f"{x['group']} ({x['size']})" for x in B["mlp_only"]))
        P("")
    if B["transcoder_only"]:
        P("### Transcoder-only concepts")
        P(", ".join(f"{x['group']} ({x['size']})" for x in B["transcoder_only"]))
        P("")

    P("## C. Agreement vs divergence")
    P("")
    P(f"- MLP features with a close transcoder analogue: **{cov['mlp_features_with_transcoder_analogue']:.0%}** "
      f"(mean NN sim {cov['mlp_nn_sim_mean']})")
    P(f"- Transcoder features with a close MLP analogue: **{cov['transcoder_features_with_mlp_analogue']:.0%}** "
      f"(mean NN sim {cov['transcoder_nn_sim_mean']})")
    P("")
    if C["per_concept"]:
        P("### Per-matched-concept agreement")
        P("| MLP group | transcoder group | concept | member→ | member← | promo-token J | layer overlap | verdict |")
        P("|---|---|---:|---:|---:|---:|---:|---|")
        for x in C["per_concept"]:
            P(f"| {x['mlp_group']} | {x['transcoder_group']} | {x['concept_similarity']} "
              f"| {x['mlp_to_tc_member_sim']} | {x['tc_to_mlp_member_sim']} "
              f"| {x['promoted_token_jaccard']} | {x['layer_overlap']} | {x['verdict']} |")
        P("")
    return "\n".join(L)


# ===========================================================================
# Slug -> transcoder file resolution + CLI
# ===========================================================================

def resolve_transcoder_paths(np_dir: Path, slug: str, groups_variant: str) -> tuple[Path, Path, Path]:
    tg = np_dir / "test_graphs" / f"{slug}.json"
    groups = np_dir / "artifacts" / f"{slug}__{groups_variant}.json"
    descs = np_dir / "artifacts" / f"{slug}__feature_descriptions_v2.json"
    return tg, groups, descs


def slug_from_mlp_path(path: Path) -> str:
    g = json.loads(path.read_text(encoding="utf-8"))
    return (g.get("label") or g.get("slug") or path.stem).split("___")[0]


def run_one(slug: str, mlp_graph: Graph | None, np_dir: Path, groups_variant: str,
            embed_mode: str, match_threshold: float | None, feature_threshold: float | None,
            out_dir: Path, demo: bool) -> dict | None:
    tg_p, gr_p, de_p = resolve_transcoder_paths(np_dir, slug, groups_variant)
    for p in (tg_p, gr_p, de_p):
        if not p.exists():
            print(f"  [skip {slug}] missing transcoder file {p.name} "
                  f"(run fetch_neuronpedia_artifacts.py)")
            return None
    tc = load_transcoder_graph(tg_p, gr_p, de_p)

    if mlp_graph is None:
        if not demo:
            print(f"  [skip {slug}] no MLP graph provided (use --mlp-graph/--mlp-dir or --demo-mlp)")
            return None
        mlp_graph = make_demo_mlp_from_transcoder(tc)

    emb = build_embedder(embed_mode, mlp_graph, tc)
    mt = match_threshold if match_threshold is not None else emb.default_match_threshold
    ft = feature_threshold if feature_threshold is not None else emb.default_feature_threshold

    rep = analyze_pair(mlp_graph, tc, emb, mt, ft)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}__cross_graph.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{slug}__cross_graph.md").write_text(render_markdown(rep), encoding="utf-8")
    b = rep["B_supernode_composition"]["counts"]
    print(f"  [ok {slug}] both={b['both']} mlp_only={b['mlp_only']} "
          f"tc_only={b['transcoder_only']} backend={emb.mode}")
    return rep


def write_summary(reports: list[dict], out_dir: Path) -> None:
    if not reports:
        return
    L = ["# Cross-graph analysis — summary", "",
         EVIDENCE_REGIME_CAVEAT, "",
         "| slug | MLP feats | tc feats | both | MLP-only | tc-only | MLP→tc cov | tc→MLP cov |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        A = r["A_structural"]
        b = r["B_supernode_composition"]["counts"]
        cov = r["C_agreement_divergence"]["feature_coverage"]
        L.append(f"| {r['slug']} | {A['mlp']['described_features']} | "
                 f"{A['transcoder']['described_features']} | {b['both']} | {b['mlp_only']} "
                 f"| {b['transcoder_only']} | {cov['mlp_features_with_transcoder_analogue']:.0%} "
                 f"| {cov['transcoder_features_with_mlp_analogue']:.0%} |")
    (out_dir / "cross_graph_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nSummary -> {out_dir / 'cross_graph_summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare MLP (ADAG) vs SLT (transcoder) graphs.")
    ap.add_argument("--np-dir", type=Path, default=Path(__file__).resolve().parent / "np_data",
                    help="Dir with test_graphs/ and artifacts/ (from fetch_neuronpedia_artifacts.py).")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--mlp-graph", type=Path, help="A single ADAG graph_*.json.")
    src.add_argument("--mlp-dir", type=Path, help="A folder of ADAG graph_*.json (one per slug).")
    ap.add_argument("--slug", help="Slug to analyze (required with --mlp-graph or --demo-mlp).")
    ap.add_argument("--demo-mlp", action="store_true",
                    help="Synthesize a stand-in MLP graph from the transcoder side (no real MLP data).")
    ap.add_argument("--groups-variant", default="feature_groups_v2_a2")
    ap.add_argument("--embed", choices=["auto", "openai", "tfidf"], default="auto")
    ap.add_argument("--match-threshold", type=float, default=None)
    ap.add_argument("--feature-threshold", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "cross_graph_out")
    args = ap.parse_args()

    reports: list[dict] = []

    if args.mlp_dir:
        graphs = sorted(args.mlp_dir.glob("graph_*.json")) or sorted(args.mlp_dir.glob("*.json"))
        if not graphs:
            print(f"No graph_*.json in {args.mlp_dir}")
            sys.exit(1)
        for gp in graphs:
            slug = slug_from_mlp_path(gp)
            rep = run_one(slug, load_mlp_graph(gp), args.np_dir, args.groups_variant,
                          args.embed, args.match_threshold, args.feature_threshold,
                          args.out_dir, demo=False)
            if rep:
                reports.append(rep)
    elif args.demo_mlp and not args.slug and not args.mlp_graph:
        # demo over every slug present under np_dir/test_graphs (single process -> real summary)
        slugs = sorted(p.stem for p in (args.np_dir / "test_graphs").glob("*.json")
                       if not p.stem.startswith("graph-"))
        if not slugs:
            print(f"No transcoder graphs in {args.np_dir / 'test_graphs'}")
            sys.exit(1)
        for slug in slugs:
            rep = run_one(slug, None, args.np_dir, args.groups_variant, args.embed,
                          args.match_threshold, args.feature_threshold, args.out_dir, demo=True)
            if rep:
                reports.append(rep)
    else:
        # single slug (real MLP graph, or demo)
        if args.mlp_graph:
            slug = args.slug or slug_from_mlp_path(args.mlp_graph)
            mlp_graph = load_mlp_graph(args.mlp_graph)
        else:
            if not args.slug:
                ap.error("--slug is required with --demo-mlp (or pass --mlp-graph / --mlp-dir).")
            slug = args.slug
            mlp_graph = None
        rep = run_one(slug, mlp_graph, args.np_dir, args.groups_variant, args.embed,
                      args.match_threshold, args.feature_threshold, args.out_dir,
                      demo=args.demo_mlp)
        if rep:
            reports.append(rep)

    write_summary(reports, args.out_dir)


if __name__ == "__main__":
    main()