import asyncio
from typing import Literal

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


from circuits.descriptions.prompts import CLUSTER_SUMMARY_BATCH_PROMPT

CLUSTERING_PROMPT = """
You are given a list of natural language descriptions from a large dataset. Your task is to generate a short summary of the labels.{top_examples}

Here are the descriptions you must summarize:
{node_descriptions}

Your description should be less than 5 words long. Only output the summary, no other text.
""".strip()

TOP_EXAMPLE_PROMPT = """
Additionally, you will be provided with the top example for each description. If there is a clear trend in the top examples, prioritise it in your summary; otherwise, focus on descriptions.
""".strip()

STEERING_AND_DESC_PROMPT = """
We are trying to describe what role a cluster of neurons in a language model play in its behavior. We have observed the effect of steering this cluster of neurons on the outputs of the model, over a dataset of examples. Additionally, we have automatically computed natural-language descriptions for the neurons in the cluster, but these may be noisy.

Your task is to produce a short label for the cluster, given the steering results and the neuron descriptions.
- The steering results show various statistics about the model's output when some multiplier is applied to the neurons in the cluster of interesting (so, 0x means turning off all neurons in the cluster; 1x is default behavior; 2x means doubling the strength of the cluster; etc.). The statistics show a next-token prediction that has the maximum score on some metric (e.g. probability, logit difference).
- The neuron descriptions are natural-language descriptions of the neurons in the cluster, sorted by the importance of the neuron to the output (so, the first description is the most salient). But remember that these descriptions may be noisy.

## Steering Results

{steering_results}

## Neuron Descriptions

{neuron_descriptions}

## Task

Your label should just be a summary of the effect of increasing the cluster strength. The label should be **a very short phrase** that is **very specific** and *informative* about the effect of strengthening the cluster.

Only output your label, no other text or formatting or delimiters, in lowercase.
""".strip()

LABEL_SUMMARY_PROMPT = """
We are doing language model interpretability research. You are given two labels: one which describes the effect of strengthening a component on the output text of the language model, and one which describes the effect of weakening the component.

**Positive Label**: {positive_label}
**Negative Label**: {negative_label}

You must produce **a single word** that captures the key interesting or relevant concept in either or both of the two labels. **Be specific**, no vague or generic words especially if a named entity is mentioned in the labels.

Only output your summarized label, no other text or formatting or delimiters. If the word is not a named entity, make it lowercase; else, make it capitalised.
""".strip()

LABEL_EVAL_PROMPT = """
We are doing language model interpretability research. You are provided two things:

- A **label** which claims to describe the effect of strengthening a component on the output text of the language model.
- Actual **steering results** which show various statistics about the model's output when some multiplier is applied to the neurons in the cluster of interesting (so, 0x means turning off all neurons in the cluster; 1x is default behavior; 2x means doubling the strength of the cluster; etc.). The statistics show a next-token prediction that has the maximum score on some metric (e.g. probability, logit difference).

Your task is to give **a score between 0 and 10** for how well the label describes the effect of strengthening the component on the output text of the language model.

## Label

{label}

## Steering Results

{steering_results}

## Task

Only output your score (a number between 0 and 10) about how well the label describes the steering results, no other text or formatting or delimiters.
""".strip()


async def generate_cluster_labels(
    node_descriptions: list[str] | None,
    give_top_example: bool = False,
    top_examples: list[str] | None = None,
    steering_results: str | None = None,
    positive_and_negative_labels: tuple[str, str] | None = None,
    model: str = "gpt-4o-mini",
):
    # node_descriptions = node_descriptions[:20]
    if give_top_example and top_examples and node_descriptions:
        # top_examples = top_examples[:20]
        prompt_content = [
            f"(top label: {top_example}) {desc}"
            for (desc, top_example) in zip(node_descriptions, top_examples)
        ]
        prompt_content = "- " + "\n- ".join(prompt_content)
    elif node_descriptions:
        prompt_content = "- " + "\n- ".join(node_descriptions)
    else:
        prompt_content = ""

    if steering_results is not None:
        prompt = STEERING_AND_DESC_PROMPT.format(
            steering_results=steering_results,
            neuron_descriptions=prompt_content,
            top_examples="\n\n" + TOP_EXAMPLE_PROMPT if give_top_example else "",
        )
    elif positive_and_negative_labels is not None:
        prompt = LABEL_SUMMARY_PROMPT.format(
            positive_label=positive_and_negative_labels[0],
            negative_label=positive_and_negative_labels[1],
        )
    else:
        prompt = CLUSTERING_PROMPT.format(
            node_descriptions=prompt_content,
            top_examples="\n\n" + TOP_EXAMPLE_PROMPT if give_top_example else "",
        )

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2048,
    )

    result = response.choices[0].message.content
    if not result:
        print("???")
        return "???"
    result = result.strip()
    return result


async def generate_label_eval(
    label: str,
    steering_results: str,
    model: str = "gpt-4o-mini",
):
    prompt = LABEL_EVAL_PROMPT.format(
        label=label,
        steering_results=steering_results,
    )

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2048,
    )

    result = response.choices[0].message.content
    if not result:
        print("???")
        return "???"
    result = result.strip()
    return float(result)


def prepare_table_from_steering_results(
    cluster_to_output: pd.DataFrame,
    cluster: int | str,
    hide_labels: bool = True,
):
    subset = cluster_to_output[cluster_to_output.cluster == cluster]
    subset = subset[
        [
            "cluster",
            "label",
            "top_tokens",
            "top_tokens_probs",
            "top_logit_diffs",
            "top_logit_diffs_logits",
            "bottom_logit_diffs",
            "bottom_logit_diffs_logits",
            "multiplier",
        ]
    ]
    subset.head()
    subset["top_1_prob"] = subset.apply(
        lambda x: f"{x['top_tokens'][0]} ({x['top_tokens_probs'][0]:.2%})", axis=1
    )
    subset["top_1_logit_diff"] = subset.apply(
        lambda x: f"{x["top_logit_diffs"][0]} ({x['top_logit_diffs_logits'][0]:.2f})", axis=1
    )
    subset["bottom_1_logit_diff"] = subset.apply(
        lambda x: f"{x["bottom_logit_diffs"][0]} ({x['bottom_logit_diffs_logits'][0]:.2f})", axis=1
    )
    result = (
        subset.groupby(["label", "multiplier"])[
            ["top_1_prob", "top_1_logit_diff", "bottom_1_logit_diff"]
        ]
        .first()
        .to_dict()
    )
    multipliers = [x for x in sorted(subset.multiplier.unique()) if x != 1]
    names = {
        "top_1_prob": "Top Raw Probability\n\nThese may be uninformative since the neuron cluster may be too weak to change the default (multiplier = 1) behavior.",
        "top_1_logit_diff": "Top Logit Effect\n\nThis tells you what next token prediction is most promoted when the cluster is steered.",
        "bottom_1_logit_diff": "Bottom Logit Effect\n\nThis tells you what next token prediction is most demoted when the cluster is steered.",
    }
    final_txt = ""
    for metric in ["top_1_logit_diff", "bottom_1_logit_diff"]:
        final_txt += f"### {names[metric]}\n\n"
        final_txt += (
            f"| {'Label':<20} | " + " | ".join([f"{x:<20.1f}" for x in multipliers]) + " |\n"
        )
        for i, label in enumerate(subset.label.unique()):
            clean_label = label.split("___")[0] if not hide_labels else f"Example {i+1}"
            final_txt += f"| {clean_label:<20} | "
            final_txt += " | ".join(
                [f"{result[metric][(label, multiplier)]:>20}" for multiplier in multipliers]
            )
            final_txt += " |\n"
        final_txt += "\n"
    return final_txt


async def label_clusters(
    df_node: pd.DataFrame,
    cluster_to_output: pd.DataFrame | None = None,
    give_top_example: bool = False,
    model: str = "gpt-4o-mini",
):
    cluster_labels = {}

    # Collect all labeling tasks to run concurrently
    labeling_tasks = []
    cluster_info = {}
    df_node.loc[:, "layer"] = df_node.input_variable.apply(lambda x: x.layer)
    df_node.loc[:, "token"] = df_node.input_variable.apply(lambda x: x.token)
    df_node.loc[:, "neuron"] = df_node.input_variable.apply(lambda x: x.neuron)

    for cluster in sorted(df_node.cluster.unique()):
        if cluster == -1:
            continue
        cluster_nodes = df_node[
            (df_node.cluster == cluster)
            & (~df_node.layer.isin([-1, df_node.layer.max()]))  # assumes max layer is logits
        ]
        if len(cluster_nodes) == 0:
            continue
        cluster_nodes = (
            cluster_nodes.groupby(["layer", "neuron", "description"])["Average"].sum().reset_index()
        )
        sorted_nodes = cluster_nodes.sort_values(
            by="Average", ascending=False, key=lambda x: x.apply(lambda y: np.abs(y))
        )

        # get descriptions
        descriptions = [
            (
                node.description[1:]
                if ("⁺" in node.description or "⁻" in node.description)
                else node.description
            )
            for _, node in sorted_nodes.iterrows()
            if node.description != "N.A."
        ]

        # Store cluster info for later processing
        cluster_info[cluster] = descriptions

        # get steering results and create tasks
        if len(descriptions) == 0 and cluster_to_output is None:
            cluster_labels[int(cluster)] = f"Cluster {cluster}"
        elif cluster_to_output is not None:
            positive_steering_results = prepare_table_from_steering_results(
                cluster_to_output[cluster_to_output.multiplier >= 1.0], cluster
            )
            negative = cluster_to_output[cluster_to_output.multiplier <= 1.0].copy()
            negative.multiplier = negative.multiplier.apply(lambda x: 2 - x)
            negative_steering_results = prepare_table_from_steering_results(negative, cluster)

            # Create concurrent tasks for positive and negative labels
            positive_task = generate_cluster_labels(
                descriptions,
                give_top_example,
                steering_results=positive_steering_results,
                model=model,
            )
            negative_task = generate_cluster_labels(
                descriptions,
                give_top_example,
                steering_results=negative_steering_results,
                model=model,
            )
            labeling_tasks.append((cluster, "steering", positive_task, negative_task))
        else:
            task = generate_cluster_labels(
                descriptions,
                give_top_example,
                model=model,
            )
            labeling_tasks.append((cluster, "simple", task))

    # Execute all labeling tasks concurrently
    if labeling_tasks:
        # Flatten all tasks for concurrent execution
        all_tasks = []
        task_mapping = []

        for task_info in labeling_tasks:
            if task_info[1] == "steering":
                cluster, task_type, positive_task, negative_task = task_info
                all_tasks.extend([positive_task, negative_task])
                task_mapping.append((cluster, task_type, len(all_tasks) - 2, len(all_tasks) - 1))
            else:
                cluster, task_type, task = task_info
                all_tasks.append(task)
                task_mapping.append((cluster, task_type, len(all_tasks) - 1, None))

        # Run all tasks concurrently
        results = await asyncio.gather(*all_tasks)

        # Process results
        for cluster, task_type, pos_idx, neg_idx in task_mapping:
            if task_type == "steering":
                positive_label = results[pos_idx]
                negative_label = results[neg_idx]
                cluster_labels[int(cluster)] = f"{positive_label} | {negative_label}"
            else:
                label = results[pos_idx]
                cluster_labels[int(cluster)] = f"{label}"
            print(f"Labelled {cluster}: {cluster_labels[int(cluster)]}")

    # summarise pos + neg labels
    labeling_tasks = []
    detailed_cluster_labels = [
        cluster_labels.get(c, f"Cluster {c}") for c in range(len(cluster_labels))
    ]  # keep original long labels
    positive_labels = [x.split(" | ")[0] for x in detailed_cluster_labels]
    negative_labels = [x.split(" | ")[1] for x in detailed_cluster_labels]
    if cluster_to_output is not None:
        for cluster, label in cluster_labels.items():
            positive_label = label.split(" | ")[0]
            negative_label = label.split(" | ")[1]
            summary_task = generate_cluster_labels(
                None,
                give_top_example=False,
                positive_and_negative_labels=(positive_label, negative_label),
                model=model,
            )
            labeling_tasks.append((cluster, "summary", summary_task))

    # execute summary tasks
    if labeling_tasks:
        # Extract just the coroutines for asyncio.gather
        summary_coroutines = [task_info[2] for task_info in labeling_tasks]
        results = await asyncio.gather(*summary_coroutines)

        # Process results
        for i, (cluster, task_type, _) in enumerate(labeling_tasks):
            cluster_labels[int(cluster)] = results[i]
            print(f"Summarised {cluster}: {cluster_labels[int(cluster)]}")

    # convert to list
    cluster_labels = [cluster_labels.get(c, f"Cluster {c}") for c in range(len(cluster_labels))]
    return cluster_labels, detailed_cluster_labels, positive_labels, negative_labels


async def eval_labels_on_steering(
    labels: list[str],
    cluster_to_output: pd.DataFrame,
    model: str = "gpt-4o-mini",
    polarity: Literal["positive", "negative"] = "positive",
):
    eval_tasks = []
    for label, cluster in zip(labels, list(range(len(labels)))):
        steering_results = prepare_table_from_steering_results(
            (
                cluster_to_output[cluster_to_output.multiplier >= 1.0]
                if polarity == "positive"
                else cluster_to_output[cluster_to_output.multiplier <= 1.0]
            ),
            cluster,
        )
        eval_task = generate_label_eval(
            label,
            steering_results,
            model=model,
        )
        eval_tasks.append(eval_task)

    results = await asyncio.gather(*eval_tasks)
    return results, sum(results) / len(results)


async def summarize_attr_contrib_descriptions(
    cluster_descriptions: dict[str, dict[str, str]],
    model: str = "claude-sonnet-4-20250514",
    cluster_exemplars: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Produce short, distinctive summary labels from attr + contrib descriptions.

    Sends all clusters in a single prompt so the LLM can differentiate them.

    Args:
        cluster_descriptions: {cluster_name: {"attr": str, "contrib": str}}.
            Either key may be missing or empty.
        model: Anthropic model to use for summarization.
        cluster_exemplars: {cluster_name: [prompt_text, ...]} top examples per cluster.

    Returns:
        {cluster_name: short_label}
    """
    from anthropic import AsyncAnthropic

    if cluster_exemplars is None:
        cluster_exemplars = {}

    client = AsyncAnthropic()

    # Build the cluster block for the batch prompt
    cluster_names = sorted(cluster_descriptions.keys())
    lines: list[str] = []
    included: list[str] = []
    for name in cluster_names:
        descs = cluster_descriptions[name]
        attr_desc = descs.get("attr", "").strip()
        contrib_desc = descs.get("contrib", "").strip()
        if not attr_desc and not contrib_desc:
            continue
        included.append(name)
        block = f"### {name}\n"
        if attr_desc:
            block += f"- Attribution: {attr_desc}\n"
        if contrib_desc:
            block += f"- Contribution: {contrib_desc}\n"
        exemplars = cluster_exemplars.get(name, [])
        if exemplars:
            block += "- Top examples:\n"
            for ex in exemplars:
                block += f"  - {ex}\n"
        lines.append(block)

    if not included:
        return {}

    cluster_block = "\n".join(lines)
    prompt = CLUSTER_SUMMARY_BATCH_PROMPT.format(
        n_clusters=len(included),
        cluster_block=cluster_block,
    )

    response = await client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse "CLUSTER_ID: label" lines
    labels: dict[str, str] = {}
    content = response.content[0].text if response.content else ""
    for line in content.strip().splitlines():
        line = line.strip()
        if ":" in line:
            cid, lbl = line.split(":", 1)
            cid = cid.strip()
            lbl = lbl.strip()
            if cid in cluster_descriptions:
                labels[cid] = lbl

    # Fill in any missing clusters
    for name in included:
        if name not in labels:
            labels[name] = name

    return labels
