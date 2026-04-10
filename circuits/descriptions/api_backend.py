"""Anthropic API backend for contrib and attr explanation generation and scoring."""

import asyncio
import logging
import random
import re
from typing import Any

from circuits.descriptions.exemplars import sample_and_format_exemplars
from circuits.descriptions.prompts import (
    ATTR_API_EXAMPLES,
    ATTR_API_SIMULATOR_SYSTEM_PROMPT,
    ATTR_API_SIMULATOR_USER_TEMPLATE,
    ATTR_API_SYSTEM_PROMPT,
    CONTRIB_SIMULATOR_SYSTEM_PROMPT,
    CONTRIB_SIMULATOR_USER_TEMPLATE,
    CONTRIB_SYSTEM_PROMPT,
    CONTRIB_USER_TEMPLATE,
)
from circuits.descriptions.scoring import compute_correlation_and_rsquared
from circuits.descriptions.types import ActivationRecord, ActSign, ScoredExplanation

logger = logging.getLogger(__name__)


def _unshuffle_scores(
    predicted_shuffled: list[float],
    order: list[int],
    minibatch_data: list[dict[str, Any]],
) -> list[float]:
    """Reorder predicted scores from shuffled exemplar order back to original order."""
    if not predicted_shuffled:
        return predicted_shuffled
    shuffled_offsets: dict[int, tuple[int, int]] = {}
    offset = 0
    for orig_idx in order:
        n = len(minibatch_data[orig_idx]["continuations"])
        shuffled_offsets[orig_idx] = (offset, n)
        offset += n
    if len(predicted_shuffled) != offset:
        return predicted_shuffled
    result: list[float] = []
    for orig_idx in range(len(minibatch_data)):
        start, n = shuffled_offsets[orig_idx]
        result.extend(predicted_shuffled[start : start + n])
    return result


class AnthropicContribExplainer:
    """Generates contrib explanations via Anthropic API (minibatch format)."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_concurrency: int = 50):
        from anthropic import AsyncAnthropic

        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.client = AsyncAnthropic()

    def _format_exemplar(
        self,
        tokens: list[str],
        continuations_with_scores: list[tuple[str, int]],
        index: int = 0,
    ) -> str:
        prompt_str = "".join(tokens)
        cont_str = repr(continuations_with_scores)
        return f'{index + 1}:\nPrompt: "{prompt_str}"\nContinuations: {cont_str}'

    def generate_explanations(
        self,
        minibatch_data: list[dict[str, Any]],
        num_samples: int = 5,
        num_exemplars_range: tuple[int, int] | None = (10, 20),
    ) -> list[str]:
        """Generate num_samples explanations from the minibatch exemplars."""
        messages_list: list[list[dict[str, str]]] = []
        for _ in range(num_samples):
            sample = self._subsample(minibatch_data, num_exemplars_range)
            random.shuffle(sample)
            exemplar_strs: list[str] = []
            for i, data in enumerate(sample):
                conts = [(c["token"], c["normalized_score"]) for c in data["continuations"]]
                exemplar_strs.append(self._format_exemplar(data["tokens"], conts, index=i))
            exemplars_text = "\n\n".join(exemplar_strs)
            user_msg = CONTRIB_USER_TEMPLATE.format(exemplars=exemplars_text)
            messages_list.append([{"role": "user", "content": user_msg}])

        responses = asyncio.run(
            self._get_responses(messages_list, CONTRIB_SYSTEM_PROMPT, temperature=1.0)
        )

        explanations: list[str] = []
        for resp in responses:
            if resp is None:
                continue
            if "[EXPLANATION]:" in resp:
                expl = resp.split("[EXPLANATION]:")[-1].strip()
            else:
                expl = resp.strip()
            if expl:
                explanations.append(expl)

        return explanations

    def _subsample(
        self,
        minibatch_data: list[dict[str, Any]],
        num_exemplars_range: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        if num_exemplars_range is None or len(minibatch_data) <= num_exemplars_range[0]:
            return list(minibatch_data)
        lo, hi = num_exemplars_range
        hi = min(hi, len(minibatch_data))
        lo = min(lo, hi)
        k = random.randint(lo, hi)
        return random.sample(minibatch_data, k)

    async def _get_responses(
        self,
        messages_list: list[list[dict[str, str]]],
        system_prompt: str,
        temperature: float = 0.0,
    ) -> list[str | None]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _call(messages: list[dict[str, str]]) -> str | None:
            async with sem:
                try:
                    resp = await self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=4096,
                        temperature=temperature,
                    )
                    return resp.content[0].text  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning("Anthropic API error: %s", e)
                    return None

        return await asyncio.gather(*[_call(m) for m in messages_list])


class AnthropicAttrExplainer:
    """Generates attr explanations via Anthropic API with chain-of-thought prompting."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_concurrency: int = 10):
        from anthropic import AsyncAnthropic

        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.client = AsyncAnthropic()

    def generate_explanations(
        self,
        pool: list[ActivationRecord],
        percentiles: dict[float, float],
        min_highlights: int,
        num_samples: int = 5,
        num_exemplars_range: tuple[int, int] = (10, 20),
        threshold_mode: str = "quantile",
        enforce_top_exemplars: int = 0,
    ) -> list[str]:
        """Generate num_samples attr explanations using the Anthropic API.

        Uses the same pool/sampling infrastructure as the VLLM path, but sends
        exemplars to the Anthropic API with few-shot examples and CoT prompting.
        """
        rng = random.Random()

        messages_list: list[list[dict[str, str]]] = []
        for _ in range(num_samples):
            exemplar_strs = sample_and_format_exemplars(
                pool,
                percentiles,
                min_highlights,
                rng,
                num_exemplars_range,
                threshold_mode=threshold_mode,
                enforce_top_exemplars=enforce_top_exemplars,
            )
            user_content = "".join(f"Excerpt {i + 1}:{s}\n" for i, s in enumerate(exemplar_strs))

            # Build messages with few-shot examples
            messages: list[dict[str, str]] = []
            for example_user, example_assistant in ATTR_API_EXAMPLES:
                messages.append({"role": "user", "content": example_user})
                messages.append({"role": "assistant", "content": example_assistant})
            messages.append({"role": "user", "content": user_content})
            messages_list.append(messages)

        responses = asyncio.run(
            self._get_responses(messages_list, ATTR_API_SYSTEM_PROMPT, temperature=1.0)
        )

        explanations: list[str] = []
        for resp in responses:
            if resp is None:
                continue
            if "[DESCRIPTION]:" in resp:
                expl = resp.split("[DESCRIPTION]:")[-1].strip()
            elif "[DESCRIPTION]" in resp:
                expl = resp.split("[DESCRIPTION]")[-1].strip("\n :")
            else:
                expl = resp.strip()
            if expl:
                explanations.append(expl)

        return explanations

    async def _get_responses(
        self,
        messages_list: list[list[dict[str, str]]],
        system_prompt: str,
        temperature: float = 0.0,
    ) -> list[str | None]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _call(messages: list[dict[str, str]]) -> str | None:
            async with sem:
                try:
                    resp = await self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=4096,
                        temperature=temperature,
                    )
                    return resp.content[0].text  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning("Anthropic API error: %s", e)
                    return None

        return await asyncio.gather(*[_call(m) for m in messages_list])


class AnthropicAttrScorer:
    """Scores attr explanations via Anthropic API by simulating activation predictions."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_concurrency: int = 10):
        from anthropic import AsyncAnthropic

        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.client = AsyncAnthropic()

    def _format_exemplar_with_blanks(
        self,
        tokens: list[str],
        index: int = 0,
    ) -> str:
        """Format an excerpt as indexed token triples with [blank] placeholders."""
        sequence = "".join(tokens)
        template = ", ".join(f"({i}, {tok!r}, [blank])" for i, tok in enumerate(tokens))
        return f"Excerpt {index + 1}:\n" f"Input sequence: {sequence}\n" f"Template: [{template}]"

    def score_explanations(
        self,
        explanations: list[str],
        records: list[ActivationRecord],
        act_sign: ActSign,
        keep_only_top_predictions: bool = True,
        max_exemplars_per_batch: int = 5,
    ) -> list[ScoredExplanation]:
        """Score attr explanations by predicting per-token activations and correlating."""
        if not explanations or not records:
            return [
                ScoredExplanation(explanation=e, score=None, rsquared=None) for e in explanations
            ]

        # Filter activations by sign
        filtered_records: list[ActivationRecord] = []
        for rec in records:
            if act_sign == "pos":
                new_acts = [max(0.0, act) for act in rec.activations]
            else:
                new_acts = [abs(min(0.0, act)) for act in rec.activations]
            filtered_records.append(
                ActivationRecord(tokens=rec.tokens, token_ids=rec.token_ids, activations=new_acts)
            )

        # Split into chunks
        if len(filtered_records) <= max_exemplars_per_batch:
            chunks = [filtered_records]
        else:
            chunks = [
                filtered_records[i : i + max_exemplars_per_batch]
                for i in range(0, len(filtered_records), max_exemplars_per_batch)
            ]

        num_chunks = len(chunks)

        # Build all messages
        messages_list: list[list[dict[str, str]]] = []
        call_meta: list[tuple[int, list[int], list[ActivationRecord]]] = []

        for explanation in explanations:
            for chunk_idx, chunk in enumerate(chunks):
                order = list(range(len(chunk)))
                random.shuffle(order)
                exemplar_strs: list[str] = []
                for i, idx in enumerate(order):
                    exemplar_strs.append(
                        self._format_exemplar_with_blanks(chunk[idx].tokens, index=i)
                    )
                exemplars_text = "\n\n".join(exemplar_strs)
                user_msg = ATTR_API_SIMULATOR_USER_TEMPLATE.format(
                    description=explanation, exemplars_with_blanks=exemplars_text
                )
                messages_list.append([{"role": "user", "content": user_msg}])
                call_meta.append((chunk_idx, order, chunk))

        responses = asyncio.run(
            self._get_responses(messages_list, ATTR_API_SIMULATOR_SYSTEM_PROMPT, temperature=0.0)
        )

        # Collect true activations
        true_all: list[float] = []
        for rec in filtered_records:
            true_all.extend(rec.activations)

        scored_results: list[ScoredExplanation] = []
        for expl_idx, explanation in enumerate(explanations):
            start = expl_idx * num_chunks
            predicted_all: list[float] = []
            all_chunks_ok = True

            for chunk_offset in range(num_chunks):
                resp_idx = start + chunk_offset
                _chunk_idx, order, chunk = call_meta[resp_idx]
                response = responses[resp_idx]

                # Parse response (in shuffled order)
                shuffled_predicted = self._parse_response(response, [chunk[i] for i in order])

                # Unshuffle back to original chunk order
                if shuffled_predicted is not None:
                    unshuffled: list[list[float]] = [[] for _ in range(len(chunk))]
                    for shuffled_pos, orig_idx in enumerate(order):
                        if shuffled_pos < len(shuffled_predicted):
                            unshuffled[orig_idx] = shuffled_predicted[shuffled_pos]
                        else:
                            all_chunks_ok = False
                            break
                    if all_chunks_ok:
                        for rec_scores in unshuffled:
                            predicted_all.extend(rec_scores)
                else:
                    all_chunks_ok = False

                if not all_chunks_ok:
                    break

            if all_chunks_ok and len(predicted_all) == len(true_all):
                score, rsquared = compute_correlation_and_rsquared(true_all, predicted_all)
            else:
                score, rsquared = None, None
                predicted_all = []

            # Build per-exemplar predictions
            predictions: list[dict[str, Any]] = []
            pred_offset = 0
            for rec in filtered_records:
                n_tokens = len(rec.tokens)
                pred_slice = (
                    predicted_all[pred_offset : pred_offset + n_tokens]
                    if predicted_all
                    else [None] * n_tokens
                )
                predictions.append(
                    {
                        "tokens": rec.tokens,
                        "true": rec.activations,
                        "predicted": pred_slice,
                    }
                )
                pred_offset += n_tokens

            scored_results.append(
                ScoredExplanation(
                    explanation=explanation,
                    score=score,
                    rsquared=rsquared,
                    predictions=predictions,
                )
            )

        # Sort by score descending
        scored_results.sort(
            key=lambda x: x.score if x.score is not None else float("-inf"), reverse=True
        )

        # Keep only top predictions
        if keep_only_top_predictions and len(scored_results) > 1:
            for i in range(1, len(scored_results)):
                scored_results[i] = ScoredExplanation(
                    explanation=scored_results[i].explanation,
                    score=scored_results[i].score,
                    rsquared=scored_results[i].rsquared,
                    predictions=None,
                )

        return scored_results

    def _parse_response(
        self,
        response: str | None,
        records: list[ActivationRecord],
    ) -> list[list[float]] | None:
        """Parse (index, token, activation) triples from API response.

        Handles the format:
            Excerpt 1: [(0, 'token', score), (1, 'token', score), ...]
            Excerpt 2: [(0, 'token', score), ...]

        Parses per-excerpt, matching by index within each excerpt. Falls back
        to flat parsing if excerpt boundaries aren't found.
        Returns a list of score lists (one per record), or None on failure.
        """
        if response is None:
            return None

        # Strip [SIMULATION] prefix if present
        if "[SIMULATION]" in response:
            response = response.split("[SIMULATION]")[-1].strip().lstrip(":")

        # Pattern for (index, token, score) triples
        triple_pattern = r"\(\s*(\d+)\s*,\s*['\"]([^'\"]*?)['\"]\s*,\s*(\d+)\s*\)"

        # Try per-excerpt parsing first
        excerpt_pattern = r"Excerpt\s+(\d+)\s*:"
        excerpt_splits = re.split(excerpt_pattern, response)

        if len(excerpt_splits) > 1:
            # excerpt_splits: ['prefix', '1', 'content1', '2', 'content2', ...]
            excerpt_contents: dict[int, str] = {}
            for i in range(1, len(excerpt_splits), 2):
                excerpt_num = int(excerpt_splits[i])
                excerpt_contents[excerpt_num] = excerpt_splits[i + 1]

            result: list[list[float]] = []
            for rec_idx, rec in enumerate(records):
                n = len(rec.tokens)
                excerpt_text = excerpt_contents.get(rec_idx + 1, "")
                matches = re.findall(triple_pattern, excerpt_text)
                if matches:
                    # Build scores indexed by position
                    scores_by_idx: dict[int, float] = {}
                    for idx_str, _tok, score_str in matches:
                        scores_by_idx[int(idx_str)] = float(score_str)
                    # Fill in order 0..n-1, defaulting to 0 for missing
                    scores = [scores_by_idx.get(i, 0.0) for i in range(n)]
                else:
                    # Fallback: try (token, score) pairs
                    pair_pattern = r"\(\s*['\"]([^'\"]*?)['\"]\s*,\s*(\d+)\s*\)"
                    pair_matches = re.findall(pair_pattern, excerpt_text)
                    if pair_matches and len(pair_matches) >= n:
                        scores = [float(s) for _, s in pair_matches[:n]]
                    else:
                        scores = [0.0] * n
                result.append(scores)
            return result

        # Flat fallback: extract all triples and split by record lengths
        matches = re.findall(triple_pattern, response)
        if not matches:
            pattern2 = r"\(\s*['\"]([^'\"]*?)['\"]\s*,\s*(\d+)\s*\)"
            matches2 = re.findall(pattern2, response)
            if not matches2:
                return None
            all_scores = [float(s) for _, s in matches2]
        else:
            all_scores = [float(s) for _, _, s in matches]

        # Split scores into per-record lists
        result = []
        offset = 0
        for rec in records:
            n = len(rec.tokens)
            if offset + n > len(all_scores):
                # Pad remaining with zeros rather than failing
                scores = all_scores[offset:] + [0.0] * (n - (len(all_scores) - offset))
                result.append(scores)
                offset = len(all_scores)
            else:
                result.append(all_scores[offset : offset + n])
                offset += n

        return result

    async def _get_responses(
        self,
        messages_list: list[list[dict[str, str]]],
        system_prompt: str,
        temperature: float = 0.0,
    ) -> list[str | None]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _call(messages: list[dict[str, str]]) -> str | None:
            async with sem:
                try:
                    resp = await self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=4096,
                        temperature=temperature,
                    )
                    return resp.content[0].text  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning("Anthropic API error: %s", e)
                    return None

        return await asyncio.gather(*[_call(m) for m in messages_list])


class AnthropicContribScorer:
    """Scores contrib explanations via Anthropic API."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001", max_concurrency: int = 50):
        from anthropic import AsyncAnthropic

        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.client = AsyncAnthropic()

    def _format_exemplar_with_blanks(
        self,
        tokens: list[str],
        continuation_tokens: list[str],
        index: int = 0,
    ) -> str:
        prompt_str = "".join(tokens)
        cont_str = repr([(tok, "[blank]") for tok in continuation_tokens])
        return f'{index + 1}:\nPrompt: "{prompt_str}"\nContinuations: {cont_str}'

    def _build_chunk_messages(
        self,
        explanation: str,
        chunk: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[int]]:
        """Build API messages for a single (explanation, chunk) pair.

        Returns (messages, shuffle_order) where shuffle_order maps shuffled
        positions back to original chunk indices.
        """
        order = list(range(len(chunk)))
        random.shuffle(order)
        exemplar_strs: list[str] = []
        for i, idx in enumerate(order):
            data = chunk[idx]
            cont_tokens = [c["token"] for c in data["continuations"]]
            exemplar_strs.append(
                self._format_exemplar_with_blanks(data["tokens"], cont_tokens, index=i)
            )
        exemplars_text = "\n\n".join(exemplar_strs)
        user_msg = CONTRIB_SIMULATOR_USER_TEMPLATE.format(
            description=explanation, exemplars_with_blanks=exemplars_text
        )
        return [{"role": "user", "content": user_msg}], order

    def score_explanations(
        self,
        explanations: list[str],
        minibatch_data: list[dict[str, Any]],
        keep_only_top_predictions: bool = True,
        max_exemplars_per_batch: int = 25,
    ) -> list[ScoredExplanation]:
        """Score explanations by predicting scores and computing correlation.

        When len(minibatch_data) > max_exemplars_per_batch, splits exemplars into
        chunks, fires all API calls concurrently, then concatenates predictions
        before computing correlation.
        """
        if not explanations or not minibatch_data:
            return [
                ScoredExplanation(explanation=e, score=None, rsquared=None) for e in explanations
            ]

        # Split minibatch_data into chunks
        if len(minibatch_data) <= max_exemplars_per_batch:
            chunks = [minibatch_data]
        else:
            chunks = [
                minibatch_data[i : i + max_exemplars_per_batch]
                for i in range(0, len(minibatch_data), max_exemplars_per_batch)
            ]

        num_chunks = len(chunks)

        # Build all messages: flat list ordered as
        # [expl_0_chunk_0, expl_0_chunk_1, ..., expl_1_chunk_0, ...]
        messages_list: list[list[dict[str, str]]] = []
        # Per-call metadata: (chunk_index, shuffle_order, chunk_data)
        call_meta: list[tuple[int, list[int], list[dict[str, Any]]]] = []

        for explanation in explanations:
            for chunk_idx, chunk in enumerate(chunks):
                msgs, order = self._build_chunk_messages(explanation, chunk)
                messages_list.append(msgs)
                call_meta.append((chunk_idx, order, chunk))

        responses = asyncio.run(
            self._get_responses(messages_list, CONTRIB_SIMULATOR_SYSTEM_PROMPT, temperature=0.0)
        )

        true_scores: list[float] = []
        for data in minibatch_data:
            for c in data["continuations"]:
                true_scores.append(float(c["normalized_score"]))

        scored_results: list[ScoredExplanation] = []
        for expl_idx, explanation in enumerate(explanations):
            # Gather responses for this explanation's chunks
            start = expl_idx * num_chunks
            predicted_scores: list[float] = []
            all_chunks_ok = True

            for chunk_offset in range(num_chunks):
                resp_idx = start + chunk_offset
                chunk_idx, order, chunk = call_meta[resp_idx]
                response = responses[resp_idx]

                shuffled_data = [chunk[i] for i in order]
                predicted_shuffled = self._parse_minibatch_response(response, shuffled_data)
                predicted_unshuffled = _unshuffle_scores(predicted_shuffled, order, chunk)

                expected_chunk_total = sum(len(d["continuations"]) for d in chunk)
                if not predicted_unshuffled or len(predicted_unshuffled) != expected_chunk_total:
                    all_chunks_ok = False
                    break

                predicted_scores.extend(predicted_unshuffled)

            if all_chunks_ok and len(predicted_scores) == len(true_scores):
                score, rsquared = compute_correlation_and_rsquared(true_scores, predicted_scores)
            else:
                score, rsquared = None, None
                predicted_scores = []

            # Build per-exemplar predictions
            predictions: list[dict[str, Any]] = []
            pred_idx = 0
            for data in minibatch_data:
                pred_conts: list[dict[str, Any]] = []
                for c in data["continuations"]:
                    pred_val = (
                        predicted_scores[pred_idx]
                        if predicted_scores and pred_idx < len(predicted_scores)
                        else None
                    )
                    pred_conts.append(
                        {
                            "token": c["token"],
                            "true": float(c["normalized_score"]),
                            "predicted": pred_val,
                        }
                    )
                    pred_idx += 1
                predictions.append({"tokens": data["tokens"], "continuations": pred_conts})

            scored_results.append(
                ScoredExplanation(
                    explanation=explanation, score=score, rsquared=rsquared, predictions=predictions
                )
            )

        # Sort by score descending
        scored_results.sort(
            key=lambda x: x.score if x.score is not None else float("-inf"), reverse=True
        )

        # Keep only top predictions
        if keep_only_top_predictions and len(scored_results) > 1:
            for i in range(1, len(scored_results)):
                scored_results[i] = ScoredExplanation(
                    explanation=scored_results[i].explanation,
                    score=scored_results[i].score,
                    rsquared=scored_results[i].rsquared,
                    predictions=None,
                )

        return scored_results

    def _parse_minibatch_response(
        self, response: str | None, minibatch_data: list[dict[str, Any]]
    ) -> list[float]:
        if response is None:
            return []
        predicted: list[float] = []
        pattern = r'\(\s*"([^"]*?)"\s*,\s*(-?\d+)\s*\)'
        matches = re.findall(pattern, response)
        if matches:
            for _tok, score_str in matches:
                try:
                    predicted.append(float(score_str))
                except ValueError:
                    predicted.append(0.0)

        expected_total = sum(len(d["continuations"]) for d in minibatch_data)
        if len(predicted) < expected_total:
            pattern2 = r"\(\s*'([^']*?)'\s*,\s*(-?\d+)\s*\)"
            matches2 = re.findall(pattern2, response)
            if len(matches2) > len(predicted):
                predicted = []
                for _tok, score_str in matches2:
                    try:
                        predicted.append(float(score_str))
                    except ValueError:
                        predicted.append(0.0)

        return predicted

    async def _get_responses(
        self,
        messages_list: list[list[dict[str, str]]],
        system_prompt: str,
        temperature: float = 0.0,
    ) -> list[str | None]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _call(messages: list[dict[str, str]]) -> str | None:
            async with sem:
                try:
                    resp = await self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=4096,
                        temperature=temperature,
                    )
                    return resp.content[0].text  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning("Anthropic API error: %s", e)
                    return None

        return await asyncio.gather(*[_call(m) for m in messages_list])
