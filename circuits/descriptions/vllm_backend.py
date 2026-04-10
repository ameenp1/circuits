"""VLLM-based explainer and HF-based finetuned simulator (no observatory dependency)."""

import gc
import logging
import random
from typing import Any

import torch
from circuits.descriptions.exemplars import build_attr_exemplar_pool, sample_and_format_exemplars
from circuits.descriptions.prompts import ATTR_EXPLAINER_SYSTEM_PROMPT, ATTR_SIMULATOR_PREFIX
from circuits.descriptions.scoring import compute_correlation_and_rsquared
from circuits.descriptions.types import (
    ActivationRecord,
    ActivationRecordWithContrib,
    ActSign,
    ScoredExplanation,
)

logger = logging.getLogger(__name__)

MAX_NORMALIZED_ACTIVATION = 10
VALID_ACTIVATION_TOKENS = [str(i) for i in range(MAX_NORMALIZED_ACTIVATION + 1)]


class VLLMExplainer:
    """Generates explanations using a VLLM-loaded model."""

    def __init__(
        self,
        model_name: str = "Transluce/llama_8b_explainer",
        gpu_idx: int = 0,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Load the model via VLLM
        logger.info("Loading VLLM explainer %s (gpu=%d)", model_name, gpu_idx)
        import os

        old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        self.llm = LLM(model=model_name, trust_remote_code=True)
        if old_env is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_env
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    def generate(
        self,
        pool_records: list[Any],
        percentiles: dict[float, float],
        min_highlights: int,
        num_samples: int = 5,
        num_exemplars_range: tuple[int, int] = (10, 20),
        rng: random.Random | None = None,
        threshold_mode: str = "quantile",
        enforce_top_exemplars: int = 0,
    ) -> list[str]:
        """Generate explanation samples with per-prompt exemplar subset sampling.

        - Each sample gets its own prompt with a different random subset of exemplars
        - Subset size is randomly drawn from num_exemplars_range per prompt
        - Per-subset threshold computed using threshold_mode
        - Uses "Excerpt N:" format (not "Example N:")
        - Does NOT append [DESCRIPTION] to user message (model generates it)
        - Postprocesses by extracting text after [DESCRIPTION]

        Args:
            pool_records: Full pool of ActivationRecord objects for subset sampling.
            percentiles: Pre-computed percentiles from full pool.
            min_highlights: Min unique highlighted token strings per subset.
            num_samples: Number of explanation samples to generate.
            num_exemplars_range: (min, max) exemplars to sample per prompt.
            rng: Random instance for sampling and shuffling.
            threshold_mode: "topk" (sort activations, pick k-th) or "quantile" (walk quantiles).
            enforce_top_exemplars: If > 0, always include this many top records (by max
                activation) in every subset, filling remaining slots randomly.

        Returns:
            List of explanation strings.
        """
        from vllm import SamplingParams

        if rng is None:
            rng = random.Random()

        tokenizer = self.llm.get_tokenizer()
        params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=1,
        )

        def _build_prompts(n: int) -> list[str]:
            """Build n prompts, each with a different random exemplar subset."""
            prompts: list[str] = []
            for _ in range(n):
                exemplar_strs = sample_and_format_exemplars(
                    pool_records,
                    percentiles,
                    min_highlights,
                    rng,
                    num_exemplars_range,
                    threshold_mode=threshold_mode,
                    enforce_top_exemplars=enforce_top_exemplars,
                )
                user_content = "".join(
                    f"Excerpt {i + 1}:{s}\n" for i, s in enumerate(exemplar_strs)
                )
                messages = [
                    {"role": "system", "content": ATTR_EXPLAINER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
            return prompts

        # Retry loop matching observatory (up to 10 iterations)
        explanations: list[str] = []
        for attempt in range(10):
            if len(explanations) >= num_samples:
                break
            if attempt > 0:
                logger.debug("Retrying explainer generation (attempt %d)", attempt + 1)

            prompts = _build_prompts(num_samples)
            outputs = self.llm.generate(prompts, params)

            for output in outputs:
                for completion in output.outputs:
                    text = completion.text.strip()
                    # Postprocess: extract after [DESCRIPTION] (matching observatory)
                    if "[DESCRIPTION]" in text:
                        text = text.split("[DESCRIPTION]")[-1].strip("\n :")
                        if text:
                            explanations.append(text)

        return explanations

    def cleanup(self) -> None:
        """Unload the model and free GPU memory."""
        logger.info("Cleaning up VLLM explainer...")
        try:
            if hasattr(self.llm, "llm_engine"):
                self.llm.llm_engine.shutdown()
        except Exception as e:
            logger.debug("Could not shutdown llm_engine: %s", e)
        del self.llm
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FinetunedSimulator:
    """Scores explanations using a finetuned HF simulator model.

    Matches the observatory's FinetunedSimulator behavior:
    - Left-padded tokenization with a dedicated <PAD> token
    - Special token ID remapping (add_special_tokens=True)
    - position_ids adjusted for left-padding
    - Logits read at position of each token (not position-1)
    - Expected activation in [0, 10] range (not normalized to [0, 1])
    """

    # Special token remapping (same as observatory's UPD_MAPPING)
    UPD_MAPPING = {
        128000: 128257,  # <|begin_of_text|>
        128006: 128258,  # <|start_header_id|>
        128007: 128259,  # <|end_header_id|>
        128009: 128260,  # <|eot_id|>
    }

    def __init__(
        self,
        model_name: str = "Transluce/llama_8b_simulator",
        gpu_idx: int = 0,
        add_special_tokens: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading finetuned simulator %s (gpu=%d)", model_name, gpu_idx)
        self.gpu_idx = gpu_idx
        self.add_special_tokens = add_special_tokens
        device = torch.device(f"cuda:{gpu_idx}")

        # Match observatory: left-padding with a dedicated <PAD> token
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        if add_special_tokens:
            self.tokenizer.add_tokens(
                [
                    "<||begin_of_text||>",
                    "<||start_header_id||>",
                    "<||end_header_id||>",
                    "<||eot_id||>",
                ]
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # Resize embeddings to match observatory (original 128256 + PAD + optional special tokens)
        original_vocab_size = 128256
        additional_tokens = 1 + (4 if add_special_tokens else 0)
        self.model.resize_token_embeddings(original_vocab_size + additional_tokens)
        self.model.config.pad_token_id = original_vocab_size
        self.model.to(device)
        self.model.eval()
        self.device = device

        # The finetuned simulator outputs predictions in the first 11 vocab slots
        # (indices 0 through 10), NOT the token IDs for digit strings "0"-"10".
        self.activation_token_ids = list(range(MAX_NORMALIZED_ACTIVATION + 1))

    @torch.inference_mode()
    def simulate(
        self,
        explanations: list[str],
        tokens: list[str],
        token_ids: list[int],
    ) -> list[list[float]]:
        """Simulate activations for one sequence with multiple explanations.

        Returns list of expected_activations (one per explanation), each a list of
        floats in [0, 10] scale (matching observatory's convention).
        """
        batch_size = len(explanations)

        # Build prompt prefixes
        prompt_prefixes = [ATTR_SIMULATOR_PREFIX.format(explanation=exp) for exp in explanations]
        prefix_tokens = self.tokenizer(prompt_prefixes, padding=True, return_tensors="pt")

        # Concatenate prefix with sequence token_ids
        input_ids = torch.cat(
            (prefix_tokens["input_ids"], torch.tensor([token_ids] * batch_size)),
            dim=1,
        ).to(self.device)

        # Remap special token IDs (match observatory behavior)
        if self.add_special_tokens:
            for old_id, new_id in self.UPD_MAPPING.items():
                input_ids[input_ids == old_id] = new_id

        attention_mask = torch.cat(
            [prefix_tokens["attention_mask"], torch.ones((batch_size, len(token_ids)))],
            dim=1,
        ).to(self.device)

        # Compute position_ids accounting for left-padding (match observatory)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits.float()  # (batch, seq_len, vocab_size), match observatory .float()

        # For each token position in the sequence, extract logits over activation tokens.
        # Observatory reads at token_indices[i] = prefix_len + (i-1) for the i-th token
        # (1-indexed), which equals prefix_len + t_idx for 0-indexed t_idx.
        prefix_len = prefix_tokens["input_ids"].shape[1]
        results: list[list[float]] = []

        for b in range(batch_size):
            expected_activations: list[float] = []
            for t_idx in range(len(token_ids)):
                # Read logit at the position of the token itself (matching observatory)
                logit_pos = prefix_len + t_idx
                if logit_pos < 0 or logit_pos >= logits.shape[1]:
                    expected_activations.append(0.0)
                    continue

                # Extract logits for activation tokens (0-10)
                act_logits = logits[b, logit_pos, self.activation_token_ids]
                probs = torch.softmax(act_logits, dim=0)

                # Expected value in [0, 10] range (matching observatory)
                values = torch.arange(
                    MAX_NORMALIZED_ACTIVATION + 1, dtype=torch.float32, device=self.device
                )
                expected = float(torch.sum(values * probs))
                expected_activations.append(expected)

            results.append(expected_activations)

        return results

    def simulate_batch(
        self,
        explanations: list[str],
        activation_records: list[ActivationRecord],
    ) -> dict[int, list[list[float]]]:
        """Simulate all (explanation, sequence) pairs.

        Returns dict mapping seq_idx -> list of expected_activations (one per explanation).
        """
        results: dict[int, list[list[float]]] = {}
        for seq_idx, act_rec in enumerate(activation_records):
            if act_rec.token_ids is None:
                continue
            results[seq_idx] = self.simulate(explanations, act_rec.tokens, act_rec.token_ids)
        return results

    def cleanup(self) -> None:
        """Unload the model and free GPU memory."""
        logger.info("Cleaning up finetuned simulator...")
        del self.model
        del self.tokenizer
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_attr_explanations(
    explainer: VLLMExplainer,
    records: list[ActivationRecord] | list[ActivationRecordWithContrib],
    act_sign: ActSign,
    num_samples: int = 5,
    min_highlights: int = 1,
    max_exemplars: int = 20,
    num_exemplars_range: tuple[int, int] = (10, 20),
    random_pool_records: list[ActivationRecord] | None = None,
    rng: random.Random | None = None,
    threshold_mode: str = "quantile",
    enforce_top_exemplars: int = 0,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate attr explanations using VLLM explainer.

    Builds a pool of ~100 records, computes threshold on top max_exemplars,
    then samples a different subset per prompt.

    Returns (explanation_strings, exemplar_dicts).
    """
    if rng is None:
        rng = random.Random()

    pool, percentiles, mh, exemplar_dicts = build_attr_exemplar_pool(
        records,
        act_sign,
        max_exemplars=max_exemplars,
        min_highlights=min_highlights,
        random_pool_records=random_pool_records,
        threshold_mode=threshold_mode,
    )

    if not pool:
        return [], []

    explanations = explainer.generate(
        pool,
        percentiles,
        mh,
        num_samples=num_samples,
        num_exemplars_range=num_exemplars_range,
        rng=rng,
        threshold_mode=threshold_mode,
        enforce_top_exemplars=enforce_top_exemplars,
    )
    return explanations, exemplar_dicts


def score_attr_explanations(
    simulator: FinetunedSimulator,
    explanations: list[str],
    records: list[ActivationRecord],
    act_sign: ActSign,
    use_raw_activations: bool = False,
    keep_only_top_predictions: bool = True,
    also_score_last_token_only: bool = False,
) -> list[ScoredExplanation]:
    """Score attr explanations using finetuned simulator.

    Returns list of ScoredExplanation.
    """
    if not explanations or not records:
        return [ScoredExplanation(explanation=e, score=None, rsquared=None) for e in explanations]

    def filter_activations(activations: list[float]) -> list[float]:
        if use_raw_activations:
            return list(activations)
        if act_sign == "pos":
            return [max(0, a) for a in activations]
        else:
            return [abs(min(0, a)) for a in activations]

    # Get simulation results
    sim_results = simulator.simulate_batch(explanations, records)

    # Compute scores per explanation
    scored_results: list[ScoredExplanation] = []
    for expl_idx, explanation in enumerate(explanations):
        all_true: list[float] = []
        all_sim: list[float] = []
        last_true: list[float] = []
        last_sim: list[float] = []
        per_exemplar: list[dict[str, Any]] = []

        for seq_idx, act_rec in enumerate(records):
            if seq_idx not in sim_results:
                continue
            true_acts = filter_activations(act_rec.activations)
            sim_acts = sim_results[seq_idx][expl_idx]
            toks = act_rec.tokens

            # Handle length mismatch
            if len(true_acts) != len(sim_acts):
                min_len = min(len(true_acts), len(sim_acts))
                true_acts = true_acts[-min_len:]
                sim_acts = sim_acts[-min_len:]
                toks = toks[-min_len:]

            per_exemplar.append({"tokens": toks, "true": true_acts, "predicted": sim_acts})
            all_true.extend(true_acts)
            all_sim.extend(sim_acts)

            if also_score_last_token_only and true_acts and sim_acts:
                last_true.append(true_acts[-1])
                last_sim.append(sim_acts[-1])

        if not all_true:
            scored_results.append(
                ScoredExplanation(explanation=explanation, score=None, rsquared=None)
            )
            continue

        score, rsquared = compute_correlation_and_rsquared(all_true, all_sim)
        score_lt, rsquared_lt = (
            compute_correlation_and_rsquared(last_true, last_sim)
            if also_score_last_token_only
            else (None, None)
        )

        scored_results.append(
            ScoredExplanation(
                explanation=explanation,
                score=score,
                rsquared=rsquared,
                predictions=per_exemplar,
                score_last_token=score_lt,
                rsquared_last_token=rsquared_lt,
            )
        )

    # Keep only top predictions
    if keep_only_top_predictions and scored_results:
        best_idx = max(
            range(len(scored_results)),
            key=lambda i: (
                scored_results[i].score if scored_results[i].score is not None else float("-inf")
            ),
        )
        for i in range(len(scored_results)):
            if i != best_idx:
                scored_results[i] = ScoredExplanation(
                    explanation=scored_results[i].explanation,
                    score=scored_results[i].score,
                    rsquared=scored_results[i].rsquared,
                    predictions=None,
                    score_last_token=scored_results[i].score_last_token,
                    rsquared_last_token=scored_results[i].rsquared_last_token,
                )

    return scored_results
