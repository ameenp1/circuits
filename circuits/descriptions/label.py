"""Main orchestration: explanation generation and scoring for circuit neurons."""

import asyncio
import gc
import logging
import random
from collections import defaultdict
from typing import Any

import pandas as pd
from circuits.analysis.cluster import NeuronId
from circuits.descriptions.api_backend import (
    AnthropicAttrExplainer,
    AnthropicAttrScorer,
    AnthropicContribExplainer,
    AnthropicContribScorer,
)
from circuits.descriptions.exemplars import (
    build_attr_exemplar_pool,
    build_contrib_exemplar_dicts,
    build_contrib_minibatch,
)
from circuits.descriptions.types import (
    ActivationRecord,
    ActivationRecordWithContrib,
    ActSign,
    ExemplarResults,
    ExplanationResults,
    ScoredExplanation,
)
from circuits.descriptions.vllm_backend import (
    FinetunedSimulator,
    VLLMExplainer,
    generate_attr_explanations,
    score_attr_explanations,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _build_activation_records(
    neuron_id: NeuronId,
    neuron_examples: list[tuple[int, list[float], list[float] | None]],
    cis: list[Any],
    tokenizer: Any,
    target_logits: list[list[int]] | None = None,
) -> list[ActivationRecordWithContrib]:
    """Build activation records for a neuron from its examples."""
    activation_records: list[ActivationRecordWithContrib] = []
    for ci_idx, attr_map, contrib_map in neuron_examples:
        input_ids = cis[ci_idx] if isinstance(cis[ci_idx], list) else cis[ci_idx].input_ids
        tokens = [tokenizer.decode([tid]) for tid in input_ids]
        activations = list(attr_map)

        # Handle length mismatch (BOS token)
        if len(input_ids) != len(activations):
            if len(input_ids) > len(activations):
                num_missing = len(input_ids) - len(activations)
                activations = [0.0] * num_missing + activations
            else:
                activations = activations[: len(input_ids)]

        output_logits = target_logits[ci_idx] if target_logits is not None else None

        activation_records.append(
            ActivationRecordWithContrib(
                tokens=tokens,
                token_ids=list(input_ids),
                activations=activations,
                contrib_map=list(contrib_map) if contrib_map is not None else None,
                output_logits=output_logits,
            )
        )

    return activation_records


def build_neuron_activation_records(
    df_node: pd.DataFrame,
    cis: list[Any],
    tokenizer: Any,
    target_logits: list[list[int]] | None,
    num_layers: int | None,
) -> tuple[
    dict[NeuronId, list[ActivationRecordWithContrib]],
    dict[NeuronId, dict[int, ActivationRecordWithContrib]],
]:
    """Build activation records for all neurons from df_node.

    Skips embedding layer (-1) and unembedding layer (num_layers if specified).
    """
    examples: dict[NeuronId, list[tuple[int, list[float], list[float] | None]]] = defaultdict(list)
    for _, neuron in df_node.iterrows():
        ci_idx = int(neuron.label.split("___")[1])
        contrib_map = (
            neuron.contrib_map
            if "contrib_map" in neuron and neuron.contrib_map is not None
            else None
        )
        examples[neuron.input_variable].append((ci_idx, neuron.attr_map, contrib_map))

    neuron_data: dict[NeuronId, list[ActivationRecordWithContrib]] = {}
    neuron_ci_mapping: dict[NeuronId, dict[int, ActivationRecordWithContrib]] = {}
    skipped_count = 0
    for neuron_id, neuron_examples in examples.items():
        layer = neuron_id[0]
        if layer == -1:
            skipped_count += 1
            continue
        if num_layers is not None and layer == num_layers:
            skipped_count += 1
            continue

        activation_records = _build_activation_records(
            neuron_id, neuron_examples, cis, tokenizer, target_logits
        )

        if not activation_records:
            skipped_count += 1
            continue

        neuron_data[neuron_id] = activation_records

        ci_mapping: dict[int, ActivationRecordWithContrib] = {}
        for (ci_idx, _, _), record in zip(neuron_examples, activation_records):
            ci_mapping[ci_idx] = record
        neuron_ci_mapping[neuron_id] = ci_mapping

    logger.info("Prepared %d neurons (%d skipped)", len(neuron_data), skipped_count)
    return neuron_data, neuron_ci_mapping


def label_clusters(
    df_node: pd.DataFrame,
    cis: list[Any],
    tokenizer: Any,
    target_logits: list[list[int]] | None = None,
    attr_explainer_name: str = "Transluce/llama_8b_explainer",
    attr_simulator_name: str = "Transluce/llama_8b_simulator",
    contrib_model_name: str = "claude-haiku-4-5-20251001",
    num_expl_samples: int = 5,
    score_explanations: bool = False,
    gpu_idx: int = 0,
    num_layers: int | None = None,
    max_neurons: int | None = None,
    min_highlights: int = 1,
    threshold_mode: str = "quantile",
    enforce_top_exemplars: int = 0,
    verbose: bool = False,
    return_neuron_data: bool = False,
    target_neuron_ids: list[NeuronId] | None = None,
    max_exemplars_per_batch: int = 25,
    skip_attr: bool = False,
    skip_contrib: bool = False,
    attr_backend: str = "vllm",
    attr_api_model_name: str = "claude-haiku-4-5-20251001",
) -> (
    tuple[ExplanationResults, ExplanationResults, ExemplarResults, ExemplarResults]
    | tuple[
        ExplanationResults,
        ExplanationResults,
        ExemplarResults,
        ExemplarResults,
        dict[NeuronId, list[ActivationRecordWithContrib]],
    ]
):
    """Generate explanations for each neuron cluster, optionally scoring them.

    Simplified pipeline:
    - Phase 1: VLLM or API explainer for attr (pos/neg) — skipped if skip_attr=True
    - Phase 2: Anthropic API for contrib (minibatch, combined pos/neg)
    - Phase 3: Finetuned simulator or API scorer for attr scoring — skipped if skip_attr=True
    - Phase 4: API scorer for contrib scoring

    Args:
        attr_backend: "vllm" (default, uses GPU models) or "api" (uses Anthropic API).
        attr_api_model_name: Anthropic model for API attr backend (default haiku).

    Returns:
        (attr_results, contrib_results, attr_exemplars, contrib_exemplars)
        or 5-tuple with neuron_data if return_neuron_data=True.
    """
    import torch

    # Build activation records for all neurons
    neuron_data, neuron_ci_mapping = build_neuron_activation_records(
        df_node, cis, tokenizer, target_logits, num_layers
    )

    # Select neurons to process
    all_neurons = list(neuron_data.items())
    if target_neuron_ids is not None:
        target_set = set(target_neuron_ids)
        sampled_neurons = [(nid, recs) for nid, recs in all_neurons if nid in target_set]
    elif max_neurons is not None and max_neurons < len(all_neurons):
        sampled_neurons = random.sample(all_neurons, max_neurons)
    else:
        sampled_neurons = all_neurons

    # Build random pools helper
    def build_attr_random_pool(
        neuron_id: NeuronId, act_sign: ActSign, num_samples: int = 20
    ) -> list[ActivationRecord]:
        ci_mapping = neuron_ci_mapping.get(neuron_id, {})
        all_ci_indices = list(range(len(cis)))
        sample_size = min(num_samples, len(all_ci_indices))
        sampled_indices = random.sample(all_ci_indices, sample_size)
        epsilon = 1e-6
        pool: list[ActivationRecord] = []
        for ci_idx in sampled_indices:
            if ci_idx in ci_mapping:
                rec = ci_mapping[ci_idx]
                if act_sign == "pos":
                    new_acts = [max(epsilon, act) for act in rec.activations]
                else:
                    new_acts = [abs(min(-epsilon, act)) for act in rec.activations]
                pool.append(
                    ActivationRecord(
                        tokens=rec.tokens, token_ids=rec.token_ids, activations=new_acts
                    )
                )
            else:
                ci = cis[ci_idx]
                tokens = [tokenizer.decode([tid]) for tid in ci]
                pool.append(
                    ActivationRecord(
                        tokens=tokens, token_ids=list(ci), activations=[epsilon] * len(tokens)
                    )
                )
        return pool

    # ========== PHASE 1: Generate attr explanations ==========
    attr_explanations_by_neuron: dict[NeuronId, dict[str, list[str]]] = {}
    attr_exemplars_by_neuron: dict[NeuronId, dict[str, list[dict[str, Any]]]] = {}

    if skip_attr:
        logger.info("PHASE 1: Skipping attr explanations (skip_attr=True)")
        for neuron_id, _ in sampled_neurons:
            attr_explanations_by_neuron[neuron_id] = {"pos": [], "neg": []}
            attr_exemplars_by_neuron[neuron_id] = {"pos": [], "neg": []}
    elif attr_backend == "api":
        logger.info(
            "PHASE 1: Generating attr explanations via API (%s) for %d neurons",
            attr_api_model_name,
            len(sampled_neurons),
        )
        api_attr_explainer = AnthropicAttrExplainer(
            model_name=attr_api_model_name,
        )

        for neuron_id, activation_records in tqdm(
            sampled_neurons, desc="Generating attr explanations (API)"
        ):
            attr_explanations_by_neuron[neuron_id] = {"pos": [], "neg": []}
            attr_exemplars_by_neuron[neuron_id] = {"pos": [], "neg": []}

            for sign in ("pos", "neg"):
                act_sign: ActSign = sign  # type: ignore[assignment]
                random_pool = build_attr_random_pool(neuron_id, act_sign)
                pool, percentiles, mh, exemplar_dicts = build_attr_exemplar_pool(
                    activation_records,
                    act_sign,
                    min_highlights=min_highlights,
                    random_pool_records=random_pool,
                    threshold_mode=threshold_mode,
                )
                if not pool:
                    continue

                explanations = api_attr_explainer.generate_explanations(
                    pool,
                    percentiles,
                    mh,
                    num_samples=num_expl_samples,
                    threshold_mode=threshold_mode,
                    enforce_top_exemplars=enforce_top_exemplars,
                )
                attr_explanations_by_neuron[neuron_id][sign] = explanations
                attr_exemplars_by_neuron[neuron_id][sign] = exemplar_dicts

                if verbose:
                    for ei, e_str in enumerate(explanations):
                        logger.info("  [attr %s] %s expl %d: %s", sign, neuron_id, ei, e_str)
    else:
        logger.info("PHASE 1: Generating attr explanations for %d neurons", len(sampled_neurons))
        explainer = VLLMExplainer(model_name=attr_explainer_name, gpu_idx=gpu_idx)

        for neuron_id, activation_records in tqdm(
            sampled_neurons, desc="Generating attr explanations"
        ):
            attr_explanations_by_neuron[neuron_id] = {"pos": [], "neg": []}
            attr_exemplars_by_neuron[neuron_id] = {"pos": [], "neg": []}

            for sign in ("pos", "neg"):
                act_sign_v: ActSign = sign  # type: ignore[assignment]
                random_pool = build_attr_random_pool(neuron_id, act_sign_v)
                explanations, exemplar_dicts = generate_attr_explanations(
                    explainer,
                    activation_records,
                    act_sign_v,
                    num_samples=num_expl_samples,
                    min_highlights=min_highlights,
                    random_pool_records=random_pool,
                    threshold_mode=threshold_mode,
                    enforce_top_exemplars=enforce_top_exemplars,
                )
                attr_explanations_by_neuron[neuron_id][sign] = explanations
                attr_exemplars_by_neuron[neuron_id][sign] = exemplar_dicts

                if verbose:
                    for ei, e_str in enumerate(explanations):
                        logger.info("  [attr %s] %s expl %d: %s", sign, neuron_id, ei, e_str)

        # Unload explainer
        explainer.cleanup()
        del explainer

    # ========== PHASE 2: Generate contrib explanations via API ==========
    contrib_explanations_by_neuron: dict[NeuronId, dict[str, list[str]]] = {}
    contrib_exemplars_by_neuron: dict[NeuronId, dict[str, list[dict[str, Any]]]] = {}

    if skip_contrib:
        logger.info("PHASE 2: Skipping contrib explanations (skip_contrib=True)")
        for neuron_id, _records in sampled_neurons:
            contrib_explanations_by_neuron[neuron_id] = {"combined": []}
            contrib_exemplars_by_neuron[neuron_id] = {"combined": []}
    else:
        logger.info(
            "PHASE 2: Generating contrib explanations (minibatch) for %d neurons",
            len(sampled_neurons),
        )
        contrib_explainer = AnthropicContribExplainer(model_name=contrib_model_name)

        for neuron_id, activation_records in tqdm(
            sampled_neurons, desc="Generating contrib explanations"
        ):
            has_contrib = any(
                rec.contrib_map is not None and rec.output_logits is not None
                for rec in activation_records
            )
            if not has_contrib:
                contrib_explanations_by_neuron[neuron_id] = {"combined": []}
                contrib_exemplars_by_neuron[neuron_id] = {"combined": []}
                continue

            minibatch_data = build_contrib_minibatch(activation_records, tokenizer, max_prompts=20)
            if not minibatch_data:
                contrib_explanations_by_neuron[neuron_id] = {"combined": []}
                contrib_exemplars_by_neuron[neuron_id] = {"combined": []}
                continue

            explanations = contrib_explainer.generate_explanations(
                minibatch_data, num_samples=num_expl_samples
            )
            contrib_explanations_by_neuron[neuron_id] = {"combined": explanations}
            contrib_exemplars_by_neuron[neuron_id] = {
                "combined": build_contrib_exemplar_dicts(minibatch_data)
            }

            if verbose:
                for ei, e_str in enumerate(explanations):
                    logger.info("  [contrib combined] %s expl %d: %s", neuron_id, ei, e_str)

    # If not scoring, return explanations directly
    if not score_explanations:
        attr_final: ExplanationResults = {}
        contrib_final: ExplanationResults = {}
        attr_exemplars_final: ExemplarResults = {}
        contrib_exemplars_final: ExemplarResults = {}
        all_neuron_ids = (
            set(attr_explanations_by_neuron.keys())
            | set(contrib_explanations_by_neuron.keys())
            | set(neuron_data.keys())
        )
        for nid in all_neuron_ids:
            attr_final[nid] = attr_explanations_by_neuron.get(nid, {"pos": [], "neg": []})
            contrib_final[nid] = contrib_explanations_by_neuron.get(nid, {"combined": []})
            attr_exemplars_final[nid] = attr_exemplars_by_neuron.get(nid, {"pos": [], "neg": []})
            contrib_exemplars_final[nid] = contrib_exemplars_by_neuron.get(nid, {"combined": []})
        if return_neuron_data:
            return (
                attr_final,
                contrib_final,
                attr_exemplars_final,
                contrib_exemplars_final,
                neuron_data,
            )
        return attr_final, contrib_final, attr_exemplars_final, contrib_exemplars_final

    # ========== PHASE 3: Score attr explanations ==========
    attr_results_scored: ExplanationResults = {}
    if skip_attr:
        logger.info("PHASE 3: Skipping attr scoring (skip_attr=True)")
        for neuron_id in attr_explanations_by_neuron:
            attr_results_scored[neuron_id] = {"pos": [], "neg": []}
    elif attr_backend == "api":
        logger.info("PHASE 3: Scoring attr explanations via API (%s)", attr_api_model_name)
        api_attr_scorer = AnthropicAttrScorer(model_name=attr_api_model_name)

        for neuron_id, sign_explanations in tqdm(
            attr_explanations_by_neuron.items(), desc="Scoring attr (API)"
        ):
            attr_results_scored[neuron_id] = {"pos": [], "neg": []}
            records = neuron_data[neuron_id]

            raw_records = [
                ActivationRecord(
                    tokens=rec.tokens, token_ids=rec.token_ids, activations=rec.activations
                )
                for rec in records
            ]

            for sign_key in ["pos", "neg"]:
                explanation_strs = sign_explanations.get(sign_key, [])
                if not explanation_strs:
                    continue

                try:
                    act_sign_s: ActSign = sign_key  # type: ignore[assignment]
                    scored = api_attr_scorer.score_explanations(
                        explanation_strs,
                        raw_records,
                        act_sign_s,
                        keep_only_top_predictions=(sign_key == "pos"),
                        max_exemplars_per_batch=max_exemplars_per_batch,
                    )

                    # Negate neg scores for combined eval
                    if sign_key == "neg":
                        for se in scored:
                            if se.score is not None:
                                se.score = -se.score

                    scored.sort(
                        key=lambda x: x.score if x.score is not None else float("-inf"),
                        reverse=True,
                    )

                    # Strip predictions from non-top neg
                    if sign_key == "neg" and len(scored) > 1:
                        for i in range(1, len(scored)):
                            se = scored[i]
                            scored[i] = ScoredExplanation(
                                explanation=se.explanation,
                                score=se.score,
                                rsquared=se.rsquared,
                                predictions=None,
                            )

                    attr_results_scored[neuron_id][sign_key] = scored

                    if verbose:
                        for se in scored:
                            logger.info(
                                "  [attr %s scored] %s: %.3f | %s",
                                sign_key,
                                neuron_id,
                                se.score if se.score is not None else float("nan"),
                                se.explanation,
                            )

                except Exception as e:
                    logger.error(
                        "Error scoring %s attr for %s: %s", sign_key, neuron_id, e, exc_info=True
                    )
                    attr_results_scored[neuron_id][sign_key] = [
                        ScoredExplanation(explanation=exp, score=None, rsquared=None)
                        for exp in explanation_strs
                    ]
    else:
        logger.info("PHASE 3: Scoring attr explanations")
        simulator = FinetunedSimulator(model_name=attr_simulator_name, gpu_idx=gpu_idx)

        for neuron_id, sign_explanations in tqdm(
            attr_explanations_by_neuron.items(), desc="Scoring attr (combined)"
        ):
            attr_results_scored[neuron_id] = {"pos": [], "neg": []}
            records = neuron_data[neuron_id]

            # Use raw records for combined eval
            raw_records = [
                ActivationRecord(
                    tokens=rec.tokens, token_ids=rec.token_ids, activations=rec.activations
                )
                for rec in records
            ]

            for sign_key in ["pos", "neg"]:
                explanation_strs = sign_explanations.get(sign_key, [])
                if not explanation_strs:
                    continue

                try:
                    act_sign_v: ActSign = sign_key  # type: ignore[assignment]
                    scored = score_attr_explanations(
                        simulator,
                        explanation_strs,
                        raw_records,
                        act_sign_v,
                        use_raw_activations=True,
                        keep_only_top_predictions=(sign_key == "pos"),
                    )

                    # Negate neg scores for combined eval
                    if sign_key == "neg":
                        for se in scored:
                            if se.score is not None:
                                se.score = -se.score

                    scored.sort(
                        key=lambda x: x.score if x.score is not None else float("-inf"),
                        reverse=True,
                    )

                    # Strip predictions from non-top neg
                    if sign_key == "neg" and len(scored) > 1:
                        for i in range(1, len(scored)):
                            se = scored[i]
                            scored[i] = ScoredExplanation(
                                explanation=se.explanation,
                                score=se.score,
                                rsquared=se.rsquared,
                                predictions=None,
                            )

                    attr_results_scored[neuron_id][sign_key] = scored

                    if verbose:
                        for se in scored:
                            logger.info(
                                "  [attr %s scored] %s: %.3f | %s",
                                sign_key,
                                neuron_id,
                                se.score if se.score is not None else float("nan"),
                                se.explanation,
                            )

                except Exception as e:
                    logger.error(
                        "Error scoring %s attr for %s: %s", sign_key, neuron_id, e, exc_info=True
                    )
                    attr_results_scored[neuron_id][sign_key] = [
                        ScoredExplanation(explanation=exp, score=None, rsquared=None)
                        for exp in explanation_strs
                    ]

        # Unload simulator
        simulator.cleanup()
        del simulator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========== PHASE 4: Score contrib explanations via API ==========
    contrib_results_scored: ExplanationResults = {}
    if skip_contrib:
        logger.info("PHASE 4: Skipping contrib scoring (skip_contrib=True)")
        for neuron_id in contrib_explanations_by_neuron:
            contrib_results_scored[neuron_id] = {"combined": []}
    else:
        logger.info("PHASE 4: Scoring contrib explanations")
        contrib_scorer = AnthropicContribScorer(model_name=contrib_model_name)

        for neuron_id, sign_explanations in tqdm(
            contrib_explanations_by_neuron.items(), desc="Scoring contrib (minibatch)"
        ):
            explanation_strs = sign_explanations.get("combined", [])
            if not explanation_strs or neuron_id not in neuron_data:
                contrib_results_scored[neuron_id] = {"combined": []}
                continue

            try:
                minibatch_data = build_contrib_minibatch(
                    neuron_data[neuron_id], tokenizer, max_prompts=20
                )
                if not minibatch_data:
                    contrib_results_scored[neuron_id] = {"combined": []}
                    continue

                scored = contrib_scorer.score_explanations(
                    explanation_strs,
                    minibatch_data,
                    max_exemplars_per_batch=max_exemplars_per_batch,
                )
                contrib_results_scored[neuron_id] = {"combined": scored}

                if verbose:
                    for se in scored:
                        logger.info(
                            "  [contrib scored] %s: %.3f | %s",
                            neuron_id,
                            se.score if se.score is not None else float("nan"),
                            se.explanation,
                        )

            except Exception as e:
                logger.error("Error scoring contrib for %s: %s", neuron_id, e, exc_info=True)
                contrib_results_scored[neuron_id] = {
                    "combined": [
                        ScoredExplanation(explanation=exp, score=None, rsquared=None)
                        for exp in explanation_strs
                    ]
                }

    # Fill in missing neurons
    all_neuron_ids = (
        set(attr_results_scored.keys())
        | set(contrib_results_scored.keys())
        | set(neuron_data.keys())
    )
    attr_exemplars_final: ExemplarResults = {}
    contrib_exemplars_final: ExemplarResults = {}
    for nid in all_neuron_ids:
        if nid not in attr_results_scored:
            attr_results_scored[nid] = {"pos": [], "neg": []}
        if nid not in contrib_results_scored:
            contrib_results_scored[nid] = {"combined": []}
        attr_exemplars_final[nid] = attr_exemplars_by_neuron.get(nid, {"pos": [], "neg": []})
        contrib_exemplars_final[nid] = contrib_exemplars_by_neuron.get(nid, {"combined": []})

    logger.info("Scoring complete")
    result = (
        attr_results_scored,
        contrib_results_scored,
        attr_exemplars_final,
        contrib_exemplars_final,
    )
    if return_neuron_data:
        return (*result, neuron_data)
    return result


async def summarize_clusters_rich(
    cluster_attr_descs: dict[str, str],
    cluster_contrib_descs: dict[str, str],
    cluster_neuron_descs: dict[str, list[tuple[str, float]]],
    cluster_exemplars: dict[str, list[str]] | None = None,
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_layers: int = 32,
    summary_model: str = "claude-opus-4-6",
    max_neurons_per_cluster: int = 20,
    neurons_only: bool = False,
    attr_only: bool = False,
) -> dict[str, str]:
    """Produce short labels for clusters using attr + contrib + individual neuron descriptions.

    Sends one API call per cluster (in parallel) so each gets full context.

    Args:
        cluster_attr_descs: {cluster_name: attr_description}
        cluster_contrib_descs: {cluster_name: contrib_description}
        cluster_neuron_descs: {cluster_name: [(neuron_desc, avg_attribution), ...]}
            Already sorted by attribution (descending).
        cluster_exemplars: {cluster_name: [label_1, label_2, ...]} top dataset examples
            where this cluster is most active. Optional.
        model_id: Model name for context in the prompt.
        num_layers: Number of layers for context in the prompt.
        summary_model: Anthropic model to use for summarization.
        max_neurons_per_cluster: Max individual neuron descriptions to include.
        neurons_only: If True, only use individual neuron descriptions (skip attr/contrib).
        attr_only: If True, use only attr + contrib descriptions (skip neuron descriptions).

    Returns:
        {cluster_name: short_label}
    """
    from anthropic import AsyncAnthropic
    from circuits.descriptions.prompts import (
        CLUSTER_ATTR_CONTRIB_ONLY_SUMMARY_PROMPT,
        CLUSTER_NEURONS_ONLY_SUMMARY_PROMPT,
        CLUSTER_RICH_SUMMARY_PROMPT,
    )

    client = AsyncAnthropic()

    all_clusters = sorted(
        set(cluster_attr_descs) | set(cluster_contrib_descs) | set(cluster_neuron_descs)
    )

    if cluster_exemplars is None:
        cluster_exemplars = {}

    async def _summarize_one(cluster_name: str) -> tuple[str, str, str]:
        neuron_entries = cluster_neuron_descs.get(cluster_name, [])[:max_neurons_per_cluster]
        if neuron_entries:
            neuron_lines = []
            for desc, score in neuron_entries:
                neuron_lines.append(f"  - [{score:+.4f}] {desc}")
            neuron_block = "\n".join(neuron_lines)
        else:
            neuron_block = "  (no individual neuron descriptions available)"

        # Build exemplars block
        exemplars = cluster_exemplars.get(cluster_name, [])
        if exemplars:
            exemplar_lines = [f"  - {ex}" for ex in exemplars]
            exemplar_block = "\n".join(exemplar_lines)
        else:
            exemplar_block = ""

        if neurons_only:
            prompt = CLUSTER_NEURONS_ONLY_SUMMARY_PROMPT.format(
                model_id=model_id,
                num_layers=num_layers,
                neuron_block=neuron_block,
            )
        elif attr_only:
            attr_desc = cluster_attr_descs.get(cluster_name, "(none)")
            contrib_desc = cluster_contrib_descs.get(cluster_name, "(none)")
            prompt = CLUSTER_ATTR_CONTRIB_ONLY_SUMMARY_PROMPT.format(
                model_id=model_id,
                num_layers=num_layers,
                attr_desc=attr_desc,
                contrib_desc=contrib_desc,
            )
        else:
            attr_desc = cluster_attr_descs.get(cluster_name, "(none)")
            contrib_desc = cluster_contrib_descs.get(cluster_name, "(none)")
            prompt = CLUSTER_RICH_SUMMARY_PROMPT.format(
                model_id=model_id,
                num_layers=num_layers,
                attr_desc=attr_desc,
                contrib_desc=contrib_desc,
                neuron_block=neuron_block,
            )

        # Append exemplars if available
        if exemplar_block:
            prompt += (
                "\n\n## Top dataset examples (where this cluster is most active)\n\n"
                + exemplar_block
            )

        kwargs: dict = {
            "model": summary_model,
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": prompt}],
        }

        for attempt in range(2):
            try:
                response = await client.messages.create(**kwargs)
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning("Retrying %s after error: %s", cluster_name, e)
                    await asyncio.sleep(1)
                else:
                    raise

        # Extract text and thinking from response
        label = cluster_name
        thinking = ""
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                label = block.text.strip()
        return cluster_name, label, thinking

    tasks = [_summarize_one(cl) for cl in all_clusters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    labels: dict[str, str] = {}
    thinking_traces: dict[str, str] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.error("Error summarizing cluster: %s", r)
            continue
        cluster_name, label, thinking = r
        labels[cluster_name] = label
        if thinking:
            thinking_traces[cluster_name] = thinking

    # Fill missing
    for cl in all_clusters:
        if cl not in labels:
            labels[cl] = cl

    return labels, thinking_traces
