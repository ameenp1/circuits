"""Exemplar construction for explanation generation (no observatory dependency)."""

import random
from typing import Any

import numpy as np
from circuits.descriptions.types import ActivationRecord, ActivationRecordWithContrib, ActSign

# Same quantile keys as observatory's exemplars_wrapper.QUANTILE_KEYS
QUANTILE_KEYS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    1 - 1e-2,
    1 - 1e-3,
    1 - 1e-4,
    1 - 1e-5,
    1 - 1e-6,
    1 - 1e-7,
    1 - 1e-8,
)

EPSILON_THRESHOLD = 1e-5


def _add_brackets(token_str: str, left: str, right: str) -> str:
    """Add brackets around token, pushing spaces outside."""
    stripped = token_str.strip(" ")
    if not stripped:
        return f"{left}{token_str}{right}"
    l_idx = 0
    while l_idx < len(token_str) and token_str[l_idx] == " ":
        l_idx += 1
    r_idx = len(token_str) - 1
    while r_idx >= 0 and token_str[r_idx] == " ":
        r_idx -= 1
    return token_str[:l_idx] + left + token_str[l_idx : r_idx + 1] + right + token_str[r_idx + 1 :]


def format_tokens_with_highlights(
    tokens: list[str],
    activations: list[float],
    threshold: float,
) -> str:
    """Format tokens as a string with {{highlighted}} tokens above threshold.

    Consecutive highlighted tokens are merged into a single pair of brackets.
    """
    # Build (token, highlighted?) pairs
    elements: list[tuple[str, bool]] = []
    for tok, act in zip(tokens, activations):
        elements.append((tok, act >= threshold and act > EPSILON_THRESHOLD))

    # Merge consecutive highlighted tokens
    merged: list[tuple[str, bool]] = [("", False)]
    for tok, highlighted in elements:
        prev_tok, prev_h = merged[-1]
        if highlighted and prev_h:
            merged[-1] = (prev_tok + tok, True)
        else:
            merged.append((tok, highlighted))

    # Format
    result = ""
    for tok, highlighted in merged:
        if highlighted:
            result += _add_brackets(tok, "{{", "}}")
        else:
            result += tok
    return result


def compute_percentiles(
    records: list[ActivationRecord],
) -> dict[float, float]:
    """Compute activation percentiles from all non-zero activations across all records.

    Matches v1: percentiles are computed on the FULL pool (not just top records).
    """
    all_acts = [a for rec in records for a in rec.activations if a != 0.0]
    if not all_acts:
        return {}
    arr = np.array(all_acts)
    return {q: float(np.percentile(arr, max(0.0, min(1.0, q)) * 100)) for q in QUANTILE_KEYS}


def compute_highlight_threshold_quantile(
    records: list[ActivationRecord],
    percentiles: dict[float, float],
    min_highlights: int = 1,
) -> float:
    """Find the highlight threshold using quantile walk.

    - Uses globally-computed percentiles (from full pool)
    - Walks quantiles from most extreme downward
    - Counts unique highlighted TOKEN STRINGS (merged consecutive), not positions
    - Returns threshold when enough unique highlighted strings are found

    Args:
        records: The subset of records to check highlights for.
        percentiles: Pre-computed percentiles from the full pool.
        min_highlights: Minimum number of unique highlighted token strings required.
    """
    if not percentiles:
        return 0.0

    q_idx = len(QUANTILE_KEYS) - 1
    while q_idx >= 0 and QUANTILE_KEYS[q_idx] > 0.5:
        q = QUANTILE_KEYS[q_idx]
        thresh = percentiles.get(q, float("inf"))

        # Count unique highlighted token strings (matching v1's activating_tokens set)
        activating_tokens: set[str] = set()
        for rec in records:
            current_group = ""
            for tok, act in zip(rec.tokens, rec.activations):
                if act >= thresh and act > EPSILON_THRESHOLD:
                    current_group += tok
                else:
                    if current_group:
                        activating_tokens.add(current_group.strip())
                        current_group = ""
            if current_group:
                activating_tokens.add(current_group.strip())

        if len(activating_tokens) >= min_highlights:
            return thresh
        q_idx -= 1

    # Fallback
    all_acts = sorted(a for rec in records for a in rec.activations if a > 0)
    if all_acts:
        low_idx = max(0, int(len(all_acts) * 0.1))
        return all_acts[low_idx]
    return 0.0


def compute_highlight_threshold_topk(
    records: list[ActivationRecord],
    min_highlights: int = 2,
) -> float:
    """Find the highlight threshold by sorting activations and picking the k-th highest.

    Directly selects the threshold so that approximately min_highlights tokens are
    highlighted. Simpler and more predictable than the quantile walk approach.

    Args:
        records: The records to compute threshold for.
        min_highlights: Number of top activations to highlight.
    """
    all_activations = sorted(a for rec in records for a in rec.activations)
    if not all_activations:
        return 0.0
    idx = max(0, len(all_activations) - min_highlights - 1)
    thresh = all_activations[idx]
    # Ensure we don't highlight near-zero activations
    return max(thresh, EPSILON_THRESHOLD)


def compute_highlight_threshold(
    records: list[ActivationRecord],
    percentiles: dict[float, float],
    min_highlights: int = 1,
    mode: str = "topk",
) -> float:
    """Compute highlight threshold using the specified mode.

    Args:
        records: The records to compute threshold for.
        percentiles: Pre-computed percentiles from full pool (used by "quantile" mode).
        min_highlights: Minimum number of highlights.
        mode: "topk" (sort activations, pick k-th highest) or "quantile" (walk quantiles).
    """
    if mode == "topk":
        return compute_highlight_threshold_topk(records, min_highlights)
    elif mode == "quantile":
        return compute_highlight_threshold_quantile(records, percentiles, min_highlights)
    else:
        raise ValueError(f"Unknown threshold mode: {mode!r}. Use 'topk' or 'quantile'.")


def _filter_by_sign(
    records: list[ActivationRecord] | list[ActivationRecordWithContrib],
    act_sign: ActSign,
) -> list[ActivationRecord]:
    """Filter activations by sign: set wrong-sign values to 0, take abs for neg."""
    filtered = []
    for rec in records:
        if act_sign == "pos":
            new_acts = [max(0.0, act) for act in rec.activations]
        else:
            new_acts = [abs(min(0.0, act)) for act in rec.activations]
        filtered.append(
            ActivationRecord(
                tokens=rec.tokens,
                token_ids=rec.token_ids,
                activations=new_acts,
            )
        )
    return filtered


def build_attr_exemplar_pool(
    records: list[ActivationRecord] | list[ActivationRecordWithContrib],
    act_sign: ActSign,
    max_exemplars: int = 20,
    min_highlights: int = 1,
    max_records: int = 80,
    random_pool_records: list[ActivationRecord] | None = None,
    threshold_mode: str = "quantile",
) -> tuple[list[ActivationRecord], dict[float, float], int, list[dict[str, Any]]]:
    """Build a pool of filtered records and compute percentiles on full pool.

    Matches v1 behavior:
    - Percentiles computed on ALL records in pool (not just top records)
    - Per-prompt threshold is computed later using these percentiles
    - Full pool returned for per-prompt subset sampling

    Returns (pool_records, percentiles, min_highlights, exemplar_dicts_for_top_records).
    """
    # Filter by sign
    filtered = _filter_by_sign(records, act_sign)

    # Keep top records by max activation
    if len(filtered) > max_records:
        filtered.sort(key=lambda r: max(r.activations) if r.activations else 0, reverse=True)
        filtered = filtered[:max_records]

    # Append random pool, deduplicating by token_ids
    if random_pool_records:
        existing_ids = {tuple(r.token_ids) for r in filtered if r.token_ids}
        for rec in random_pool_records:
            key = tuple(rec.token_ids) if rec.token_ids else None
            if key is None or key not in existing_ids:
                filtered.append(rec)
                if key is not None:
                    existing_ids.add(key)

    if not any(any(a != 0 for a in rec.activations) for rec in filtered):
        return [], {}, min_highlights, []

    # Compute percentiles on FULL pool (matching v1)
    percentiles = compute_percentiles(filtered)

    # Sort by max activation for exemplar dicts
    sorted_records = sorted(
        filtered,
        key=lambda r: max(r.activations) if r.activations else 0,
        reverse=True,
    )
    top_records = sorted_records[:max_exemplars]

    # Compute threshold for the top records (for exemplar dict storage)
    threshold = compute_highlight_threshold(
        top_records, percentiles, min_highlights, mode=threshold_mode
    )

    # Build exemplar dicts for the top records (for storage/visualization)
    exemplar_dicts: list[dict[str, Any]] = []
    for rec in top_records:
        text = format_tokens_with_highlights(rec.tokens, rec.activations, threshold)
        exemplar_dicts.append(
            {
                "text": text,
                "tokens": rec.tokens,
                "activations": rec.activations,
                "highlight_threshold": threshold,
            }
        )

    return filtered, percentiles, min_highlights, exemplar_dicts


def sample_and_format_exemplars(
    pool: list[ActivationRecord],
    percentiles: dict[float, float],
    min_highlights: int,
    rng: random.Random,
    num_exemplars_range: tuple[int, int] = (10, 20),
    threshold_mode: str = "quantile",
    enforce_top_exemplars: int = 0,
) -> list[str]:
    """Sample a subset from the pool and format with highlights.

    - Sample a random count from num_exemplars_range
    - If enforce_top_exemplars > 0, always include the top records by max activation,
      then fill remaining slots randomly from the rest of the pool
    - Otherwise, draw all indices uniformly at random
    - Shuffle order
    - Compute per-subset threshold
    - Format each with highlights
    """
    num_exemplars = rng.randint(num_exemplars_range[0], num_exemplars_range[1])
    num_exemplars = min(num_exemplars, len(pool))

    if enforce_top_exemplars > 0 and num_exemplars > 0:
        # Rank pool records by max activation
        ranked = sorted(
            range(len(pool)),
            key=lambda i: max(pool[i].activations) if pool[i].activations else 0,
            reverse=True,
        )
        n_forced = min(enforce_top_exemplars, num_exemplars, len(pool))
        forced_indices = set(ranked[:n_forced])
        remaining_pool = [i for i in range(len(pool)) if i not in forced_indices]
        n_random = min(num_exemplars - n_forced, len(remaining_pool))
        random_indices = rng.sample(remaining_pool, k=n_random) if n_random > 0 else []
        indices = sorted(list(forced_indices) + random_indices)
    else:
        indices = rng.sample(range(len(pool)), k=num_exemplars)
        indices = sorted(indices)

    rng.shuffle(indices)
    subset = [pool[idx] for idx in indices]

    # Compute threshold adapted to this specific subset
    threshold = compute_highlight_threshold(
        subset, percentiles, min_highlights, mode=threshold_mode
    )

    formatted: list[str] = []
    for rec in subset:
        text = format_tokens_with_highlights(rec.tokens, rec.activations, threshold)
        formatted.append(text)
    return formatted


def build_attr_exemplars(
    records: list[ActivationRecord] | list[ActivationRecordWithContrib],
    act_sign: ActSign,
    rng: random.Random | None = None,
    max_exemplars: int = 20,
    min_highlights: int = 1,
    max_records: int = 80,
    random_pool_records: list[ActivationRecord] | None = None,
    threshold_mode: str = "quantile",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build formatted attr exemplars for explanation generation.

    Returns (formatted_strings, exemplar_dicts_with_scores).
    """
    pool, _percentiles, _mh, exemplar_dicts = build_attr_exemplar_pool(
        records,
        act_sign,
        max_exemplars,
        min_highlights,
        max_records,
        random_pool_records,
        threshold_mode=threshold_mode,
    )
    if not pool:
        return [], []

    # Format all top records (for backward compat / single-prompt usage)
    formatted: list[str] = []
    for d in exemplar_dicts:
        formatted.append(d["text"])

    return formatted, exemplar_dicts


def build_contrib_minibatch(
    records: list[ActivationRecordWithContrib],
    tokenizer: Any,
    max_prompts: int = 20,
) -> list[dict[str, Any]]:
    """Build minibatch data structure from contrib records.

    Groups by prompt, normalizes scores to -10..10 scale.
    Returns list of {"tokens": [...], "continuations": [{"token": str, "score": float,
    "normalized_score": int}, ...]}.
    """
    grouped: dict[int, dict[str, Any]] = {}
    for i, rec in enumerate(records):
        if rec.contrib_map is None or rec.output_logits is None:
            continue
        conts: list[dict[str, Any]] = []
        for score, logit_id in zip(rec.contrib_map, rec.output_logits):
            token_str = tokenizer.decode([logit_id])
            conts.append({"token": token_str, "score": score})
        if conts:
            grouped[i] = {"tokens": rec.tokens, "continuations": conts}

    if not grouped:
        return []

    # Normalize scores to -10..10
    all_scores = [c["score"] for data in grouped.values() for c in data["continuations"]]
    max_abs = max(abs(s) for s in all_scores) if all_scores else 1.0
    if max_abs == 0:
        max_abs = 1.0

    result: list[dict[str, Any]] = []
    for data in grouped.values():
        for c in data["continuations"]:
            c["normalized_score"] = int(round(c["score"] / max_abs * 10))
        result.append(data)

    # Sort by max absolute contribution
    result.sort(
        key=lambda d: max(abs(c["score"]) for c in d["continuations"]),
        reverse=True,
    )

    return result[:max_prompts]


def build_contrib_exemplar_dicts(
    minibatch_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert minibatch data to exemplar dicts for storage."""
    return [
        {
            "tokens": data["tokens"],
            "continuations": [
                {
                    "token": c["token"],
                    "score": c["score"],
                    "normalized_score": c["normalized_score"],
                }
                for c in data["continuations"]
            ],
        }
        for data in minibatch_data
    ]
