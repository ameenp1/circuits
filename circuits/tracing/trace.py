"""
High-level circuit tracing: prepare inputs, run CLJA, and produce a CircuitData artifact.
"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from circuits.tracing.clja import ADAGConfig, get_all_pairs_cl_ja_effects_with_attributions
from circuits.tracing.utils import Edge, Node
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class CircuitData:
    """Complete output of circuit tracing — everything needed for downstream analysis."""

    df_node: pd.DataFrame
    df_edge: pd.DataFrame
    cis: list[list[int]]
    attention_masks: list[list[int]]
    labels: list[str]
    target_logits: list[list[int]]
    target_logit_probs: list[list[float]]
    k: int
    config: ADAGConfig
    model_id: str = ""
    traced_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def merge(cls, shards: list["CircuitData"]) -> "CircuitData":
        """Merge multiple CircuitData shards (from parallel workers) into one.

        Re-indexes the label suffixes (___N) in df_node/df_edge so they are globally unique.
        """
        if len(shards) == 1:
            return shards[0]

        all_df_node = []
        all_df_edge = []
        global_offset = 0

        for shard in shards:
            shard_size = len(shard.labels)
            if shard_size == 0:
                continue

            # Re-index label suffixes: replace ___<local_idx> with ___<global_idx>
            df_node = shard.df_node.copy()
            df_edge = shard.df_edge.copy()

            def reindex_label(label: str, offset: int) -> str:
                parts = label.rsplit("___", 1)
                if len(parts) == 2:
                    return f"{parts[0]}___{int(parts[1]) + offset}"
                return label

            df_node["label"] = df_node["label"].apply(lambda l: reindex_label(l, global_offset))
            df_edge["label"] = df_edge["label"].apply(lambda l: reindex_label(l, global_offset))

            all_df_node.append(df_node)
            all_df_edge.append(df_edge)
            global_offset += shard_size

        return cls(
            df_node=pd.concat(all_df_node, ignore_index=True),
            df_edge=pd.concat(all_df_edge, ignore_index=True),
            cis=[ci for shard in shards for ci in shard.cis],
            attention_masks=[am for shard in shards for am in shard.attention_masks],
            labels=[l for shard in shards for l in shard.labels],
            target_logits=[tl for shard in shards for tl in shard.target_logits],
            target_logit_probs=[tp for shard in shards for tp in shard.target_logit_probs],
            k=shards[0].k,
            config=shards[0].config,
            model_id=shards[0].model_id,
        )

    def save_to_pickle(self, path: str) -> None:
        """Save CircuitData to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_pickle(cls, path: str) -> "CircuitData":
        """Load CircuitData from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)


# Copied from util.chat_input — removes default system preamble ("Cutting Knowledge Date: ...")
# while keeping the system header structure.
STRIPPED_LLAMA_CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

# Model-specific chat template overrides. Models not listed here use tokenizer.chat_template.
CHAT_TEMPLATES: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": STRIPPED_LLAMA_CHAT_TEMPLATE,
}

# Tokens marking the end of a header per model family, used for rollout position detection.
HEADER_END_TOKENS: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "<|end_header_id|>",
}


def get_chat_template(tokenizer: PreTrainedTokenizer) -> str:
    """Get the appropriate chat template for the tokenizer's model."""
    model_id = getattr(tokenizer, "name_or_path", "")
    if model_id in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[model_id]
    # Fall back to the tokenizer's built-in chat template
    return tokenizer.chat_template


def get_header_end_token(tokenizer: PreTrainedTokenizer) -> str | None:
    """Get the header-end token string for rollout position detection, or None."""
    model_id = getattr(tokenizer, "name_or_path", "")
    return HEADER_END_TOKENS.get(model_id)


def _strip_starting_at_rindex_in_place(arr: list, value: object) -> list:
    """Strips everything including and after the final occurrence of `value` within `arr`."""
    try:
        rindex = arr[::-1].index(value)
        index = len(arr) - 1 - rindex
        del arr[index:]
    except ValueError:
        pass
    return arr


def prepare_ci(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str,
    k: int,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a single chat input.
    """
    # Handle [EMPTY] sentinel: treat as a real seed for template purposes,
    # then strip the encoded "[EMPTY]" tokens from the end of the ci.
    is_empty_seed = seed_response == "[EMPTY]"
    if is_empty_seed:
        seed_response = ""

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    has_seed = seed_response is not None and len(seed_response) > 0
    if has_seed or is_empty_seed:
        messages.append({"role": "assistant", "content": seed_response})

    if use_chat_format:
        token_ids: list[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not has_seed and not is_empty_seed,
            chat_template=get_chat_template(tokenizer),
        )
        if has_seed or is_empty_seed:
            _strip_starting_at_rindex_in_place(token_ids, tokenizer.eos_token_id)
    else:
        # No chat template (base models, e.g. google/gemma-2-2b): concatenate the
        # question and the seed response as plain text. Without this the seed
        # ("Answer:") is dropped, so the trace targets the token after the bare
        # question (a newline) instead of the intended answer.
        text = question + " " + seed_response if has_seed else question
        token_ids = tokenizer(text)["input_ids"]

    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]
    if true_answers is not None:
        # then we create the topk using the first token of every true answer
        topk = [tokenizer.encode(answer)[1] for answer in true_answers]
        topk_probs = [0.0] * len(topk)  # no probs available for true_answers
    else:
        input_ids = torch.tensor([token_ids], device=next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
        topk_result = torch.topk(logits, k)
        topk = topk_result.indices.tolist()
        probs = torch.softmax(logits, dim=-1)
        topk_probs = probs[topk_result.indices].tolist()

    if verbose:
        print("Prepared:", question, seed_response, "->", tokenizer.decode(topk[0]))

    return token_ids, topk, topk_probs


def prepare_cis(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str],
    k: int = 5,
    system_prompt: str | None = None,
    true_answers: list[str] | list[None] | None = None,
    use_chat_format: bool = True,
    verbose: bool = False,
):
    """
    Prepare a list of chat inputs.
    """
    if true_answers is None:
        true_answers = [None] * len(questions)
    res = [
        prepare_ci(model, tokenizer, q, sr, k, system_prompt, ta, use_chat_format, verbose)
        for q, sr, ta in zip(questions, seed_responses, true_answers)
    ]
    cis = [r[0] for r in res]
    topks = [r[1] for r in res]
    topk_probs_list = [r[2] for r in res]
    max_length = max(len(ci) for ci in cis)

    attention_masks = []
    focus_tokens = []
    focus_probs = []
    for topk, probs in zip(topks, topk_probs_list):
        focus_tokens.append(list(topk))
        focus_probs.append(list(probs))

    # pad on left
    starts = []
    padded_cis: list[list[int]] = []
    for ci in cis:
        starts.append(max_length - len(ci))
        attention_mask = [0] * (max_length - len(ci)) + [1] * len(ci)
        padded_cis.append([tokenizer.pad_token_id] * (max_length - len(ci)) + ci)
        attention_masks.append(attention_mask)
    cis = padded_cis

    # keep all tokens
    keep_pos = []
    for i in range(max_length):
        keep_pos.append(i)
    if verbose:
        print(keep_pos)
        print(attention_masks)
        print(focus_tokens)
    return cis, attention_masks, focus_tokens, focus_probs, keep_pos, starts


def prepare_ci_with_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    seed_response: str | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a single chat input.
    """
    messages = [{"role": "user", "content": question}]
    has_seed = seed_response is not None
    if has_seed:
        messages.append({"role": "assistant", "content": seed_response})

    token_ids: list[int] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=not has_seed,
        chat_template=get_chat_template(tokenizer),
    )
    if has_seed:
        _strip_starting_at_rindex_in_place(token_ids, tokenizer.eos_token_id)
    if seed_response is not None and seed_response.endswith(" "):
        space_token = tokenizer.encode(" ")[1]
        token_ids = token_ids + [space_token]

    # generate additional tokens
    input_ids = torch.tensor([token_ids], device=next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    rollout_token_ids = output_ids[0].tolist()[len(token_ids) :]

    if len(rollout_token_ids) != max_new_tokens:
        raise ValueError(
            f"rollout token ids length {len(rollout_token_ids)} != max_new_tokens {max_new_tokens}"
        )

    if verbose:
        print("Prepared:", question, "->", tokenizer.decode(rollout_token_ids))

    return token_ids + rollout_token_ids


def prepare_cis_with_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    seed_responses: list[str] | None = None,
    max_new_tokens: int = 1,
    verbose: bool = True,
):
    """
    Prepare a list of chat inputs.
    """
    if seed_responses is None:
        seed_responses = [None] * len(questions)
    cis = [
        prepare_ci_with_rollout(model, tokenizer, q, seed_response, max_new_tokens, verbose)
        for q, seed_response in zip(questions, seed_responses)
    ]
    max_length = max(len(ci) for ci in cis)
    all_attention_masks = []
    all_focus_tokens = []
    all_tgt_tokens = []

    new_cis: list[list[int]] = []
    starts = []
    for ci in cis:
        starts.append(max_length - len(ci))
        header_end = get_header_end_token(tokenizer)
        if header_end is not None:
            end_token = tokenizer.encode(header_end)[-1]
            positions = [i for i, t in enumerate(ci) if t == end_token]
            start_assistant = positions[2] + 2
        else:
            # For models without explicit header tokens (e.g. Qwen3), find the start
            # of assistant content by looking for the last assistant turn marker.
            # The assistant content starts after "<|im_start|>assistant\n"
            im_start_token = tokenizer.encode("<|im_start|>")[-1]
            positions = [i for i, t in enumerate(ci) if t == im_start_token]
            start_assistant = positions[-1] + 2  # skip "assistant" and "\n" tokens
        # offset by 1 because next token prediction
        tgt_tokens = [i - 1 for i in range(start_assistant, len(ci))]
        # could be different length
        focus_tokens = [ci[i + 1] for i in tgt_tokens]

        attention_masks = [0] * (max_length - len(ci)) + [1] * len(ci)
        padded_ci = [tokenizer.pad_token_id] * (max_length - len(ci)) + ci
        offset_tgt_tokens = [p + (max_length - len(ci)) for p in tgt_tokens]

        all_attention_masks.append(attention_masks)
        all_focus_tokens.append(focus_tokens)
        all_tgt_tokens.append(offset_tgt_tokens)
        new_cis.append(padded_ci)

    # keep all tokens except the last token due to offset
    keep_pos = []
    for i in range(max_length - 1):
        keep_pos.append(i)

    if verbose:
        print(keep_pos)
        print(all_attention_masks)
        print(all_focus_tokens)
        print(all_tgt_tokens)

    # compute focus probs by running a forward pass on the padded sequences
    all_focus_probs: list[list[float]] = []
    device = next(model.parameters()).device
    input_ids = torch.tensor(new_cis, device=device)
    attn_mask = torch.tensor(all_attention_masks, device=device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attn_mask).logits
    for batch_i in range(len(new_cis)):
        probs_for_ci: list[float] = []
        for tgt_pos, focus_tok in zip(all_tgt_tokens[batch_i], all_focus_tokens[batch_i]):
            token_probs = torch.softmax(logits[batch_i, tgt_pos], dim=-1)
            probs_for_ci.append(token_probs[focus_tok].item())
        all_focus_probs.append(probs_for_ci)

    # the reason why we return all_tgt_tokens[0] is because tgt token positions are the same for all
    return (
        new_cis,
        all_attention_masks,
        all_focus_tokens,
        all_focus_probs,
        all_tgt_tokens[0],
        keep_pos,
        starts,
    )


def compute_circuits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    k: int = 1,
    bs: int = 4,
    max_new_tokens: int = 1,
    use_rollout: bool = False,
    system_prompt: str | None = None,
    true_answers: list[str] | None = None,
):
    """
    Compute CLSO graphs for all datapoints in a list of prompts, batched.
    """
    # set up data
    prompts = prompts if isinstance(prompts, list) else [prompts]
    if seed_responses is None:
        seed_responses = [None] * len(prompts)
    seed_responses = seed_responses if isinstance(seed_responses, list) else [seed_responses]

    # storage
    all_nodes, all_edges, all_labels, all_focus, all_starts = [], [], [], [], []
    all_cis, all_attention_masks = [], []
    all_focus_tokens: list[list[int]] = []
    all_focus_probs: list[list[float]] = []

    for i in tqdm(range(0, len(prompts), bs), desc="Processing batches"):
        if use_rollout:
            cis, attention_masks, focus_tokens, focus_probs, tgt_tokens, keep_pos, starts = (
                prepare_cis_with_rollout(
                    model,
                    tokenizer,
                    prompts[i : i + bs],
                    seed_responses[i : i + bs],
                    max_new_tokens=max_new_tokens,
                    verbose=config.verbose,
                )
            )
        else:
            cis, attention_masks, focus_tokens, focus_probs, keep_pos, starts = prepare_cis(
                model,
                tokenizer,
                prompts[i : i + bs],
                seed_responses[i : i + bs],
                k=k,
                system_prompt=system_prompt,
                true_answers=true_answers,
                verbose=config.verbose,
                use_chat_format=tokenizer.chat_template is not None,
            )
        nodes, edges = get_all_pairs_cl_ja_effects_with_attributions(
            model=model,
            tokenizer=tokenizer,
            cis=cis,
            config=config,
            attention_masks=attention_masks,
            focus_logits=focus_tokens,
            src_tokens=keep_pos,
            tgt_tokens=[max(keep_pos) for _ in range(k)] if not use_rollout else tgt_tokens,
        )
        all_nodes.append(nodes)
        all_edges.append(edges)
        all_focus.append([_ for _ in range(len(focus_tokens))])
        all_starts.append(starts)
        all_focus_tokens.extend(focus_tokens)
        all_focus_probs.extend(focus_probs)
        all_cis.extend(cis)
        all_attention_masks.extend(attention_masks)
        if config.verbose:
            print("focus_tokens:", focus_tokens)
            print("starts:", starts)

    return (
        all_nodes,
        all_edges,
        all_labels,
        all_focus,
        all_starts,
        all_cis,
        all_attention_masks,
        all_focus_tokens,
        all_focus_probs,
    )


def compute_cohens_d_loo(vals_x: list[float], all_vals: list[float]) -> float:
    # vals_y is all_vals without vals_x
    vals_y = all_vals[::]
    for val in vals_x:
        vals_y.remove(val)

    std_x = np.std(vals_x, ddof=1) if len(vals_x) > 1 else 0
    std_y = np.std(vals_y, ddof=1) if len(vals_y) > 1 else 0
    s = (
        np.sqrt(((len(vals_x) - 1) * std_x + (len(vals_y) - 1) * std_y) / (len(all_vals) - 2))
        if len(all_vals) > 2
        else 0
    )
    return (np.mean(vals_x) - np.mean(vals_y)) / s if s != 0 else 0


def convert_circuit_to_dataframes(
    nodes: list[list[Node]],
    edges: list[list[Edge]],
    labels: list[str],
    starts: list[list[int]],
    bs: int = 4,
    ignore_bos: bool = False,
    percentage_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process CLSO graph data into a clean dataframe.
    """
    dfs_node, dfs_edge = [], []
    for batch_idx in range(len(nodes)):
        actual_bs = len(starts[batch_idx])
        for idx in range(actual_bs):
            start = starts[batch_idx][idx] + (1 if ignore_bos else 0)

            def extract_map(
                map_tensor: torch.Tensor | None, start_idx: int, is_attr: bool
            ) -> list[float] | None:
                """Extract raw attr_map/contrib_map values (no normalization)."""
                if map_tensor is None:
                    return None
                if is_attr:
                    return map_tensor[idx, start_idx:].tolist()
                else:
                    return map_tensor[idx].tolist()

            d = [
                (
                    node.layer,
                    node.token,
                    node.neuron,
                    node.final_attribution[idx].sum().item(),
                    node.activation[idx].item(),
                    extract_map(node.attr_map, start, is_attr=True),
                    extract_map(node.contrib_map, start, is_attr=False),
                )
                for node in nodes[batch_idx]
                if node.token >= start
            ]
            df_node = pd.DataFrame(
                d,
                columns=[
                    "layer",
                    "token",
                    "neuron",
                    "attribution",
                    "activation",
                    "attr_map",
                    "contrib_map",
                ],
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")
            d = [
                (
                    f"{edge.src.layer}->{edge.tgt.layer}",
                    f"{edge.src.token}->{edge.tgt.token}",
                    f"{edge.src.neuron}->{edge.tgt.neuron}",
                    edge.final_attribution[idx].sum().item(),
                    edge.weight[idx].item(),
                )
                for edge in edges[batch_idx]
                if edge.src.token >= start and edge.tgt.token >= start
            ]
            df_edge = pd.DataFrame(
                d, columns=["layer", "token", "neuron", "attribution", "weight"]
            ).assign(label=labels[batch_idx * bs + idx] + f"___{batch_idx * bs + idx}")

            # normalize attribution by sum of goals
            total_last_layer_attribution = df_node[
                df_node.layer == df_node.layer.max()
            ].attribution.sum()
            df_node.loc[:, "attribution"] = (
                df_node.loc[:, "attribution"] / total_last_layer_attribution
            )
            df_edge.loc[:, "attribution"] = (
                df_edge.loc[:, "attribution"] / total_last_layer_attribution
            )

            # add to df
            dfs_node.append(df_node)
            dfs_edge.append(df_edge)

    # merge dfs
    df_node = pd.concat(dfs_node)
    df_edge = pd.concat(dfs_edge)
    # drop nodes below threshold per-item (with batching, neurons important in
    # any batch item get traced for all items, so we need per-item filtering here)
    if percentage_threshold is not None:
        # Only apply threshold to MLP neurons, not embedding (layer -1) or final layer nodes
        max_layer = df_node["layer"].max()
        is_exempt = (df_node["layer"] < 0) | (df_node["layer"] == max_layer)
        df_node = df_node[is_exempt | (df_node.attribution.abs() >= percentage_threshold)]
    else:
        df_node = df_node[df_node.attribution != 0]
    df_edge = df_edge[df_edge.attribution != 0].dropna(subset=["attribution"])

    # prune edges whose src or tgt node was removed
    surviving_nodes = set(
        zip(df_node["layer"], df_node["token"], df_node["neuron"], df_node["label"])
    )
    # edge columns are "src->tgt" strings, split to check membership
    edge_src = df_edge["layer"].str.split("->").str[0].astype(int)
    edge_src_tok = df_edge["token"].str.split("->").str[0].astype(int)
    edge_src_neu = df_edge["neuron"].str.split("->").str[0].astype(int)
    edge_tgt = df_edge["layer"].str.split("->").str[1].astype(int)
    edge_tgt_tok = df_edge["token"].str.split("->").str[1].astype(int)
    edge_tgt_neu = df_edge["neuron"].str.split("->").str[1].astype(int)
    edge_label = df_edge["label"]
    src_alive = pd.Series(
        [
            (l, t, n, lb) in surviving_nodes
            for l, t, n, lb in zip(edge_src, edge_src_tok, edge_src_neu, edge_label)
        ],
        index=df_edge.index,
    )
    tgt_alive = pd.Series(
        [
            (l, t, n, lb) in surviving_nodes
            for l, t, n, lb in zip(edge_tgt, edge_tgt_tok, edge_tgt_neu, edge_label)
        ],
        index=df_edge.index,
    )
    df_edge = df_edge[src_alive & tgt_alive]
    return df_node, df_edge


def convert_inputs_to_circuits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    config: ADAGConfig,
    seed_responses: list[str] | None = None,
    labels: list[str] | None = None,
    num_datapoints: int | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 1,
    k: int = 1,
    # TODO: topk_logits: int = 1,
    ignore_bos: bool = False,
    system_prompt: str | None = None,
    use_rollout: bool = False,
    true_answers: list[str] | None = None,
) -> CircuitData:
    """
    Convert a list of prompts and seed responses into a CircuitData artifact.
    """
    # num datapoints
    if num_datapoints is None:
        num_datapoints = len(prompts)
        assert len(prompts) == len(labels) == len(seed_responses)
    else:
        assert len(prompts) >= num_datapoints
        assert len(labels) >= num_datapoints
        assert len(seed_responses) >= num_datapoints

    # grab inputs
    prompts = prompts[:num_datapoints]
    labels = labels[:num_datapoints]
    if seed_responses is not None and not use_rollout:
        seed_responses = seed_responses[:num_datapoints]

    print("Prompt:", prompts[0])
    if seed_responses is not None and not use_rollout:
        print("Seed response:", seed_responses[0].replace(" ", "_"))
    print("Number of datapoints:", len(prompts))

    # compute circuits
    nodes, edges, _, focus, starts, cis, attention_masks, focus_tokens, focus_probs = (
        compute_circuits(
            model,
            tokenizer,
            prompts,
            config=config,
            seed_responses=seed_responses,
            k=k,
            bs=batch_size,
            max_new_tokens=max_new_tokens,
            use_rollout=use_rollout,
            system_prompt=system_prompt,
            true_answers=true_answers,
        )
    )

    # convert to dataframes
    df_node, df_edge = convert_circuit_to_dataframes(
        nodes,
        edges,
        labels,
        starts,
        bs=batch_size,
        ignore_bos=ignore_bos,
        percentage_threshold=config.percentage_threshold,
    )

    return CircuitData(
        df_node=df_node,
        df_edge=df_edge,
        cis=cis,
        attention_masks=attention_masks,
        labels=labels,
        target_logits=focus_tokens,
        target_logit_probs=focus_probs,
        k=k,
        config=config,
        model_id=model.config._name_or_path,
    )
