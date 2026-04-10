"""All prompt templates inlined (no observatory dependency).

Prompt categories:
- ATTR_*: attribution explanation generation/simulation (finetuned VLLM models)
- CONTRIB_*: contribution explanation generation/simulation (Anthropic API)
- CLUSTER_SUMMARY_*: cluster label summarization (Anthropic API)
"""

# System prompt for the finetuned attr explainer (observatory's SYS_PROMPT_NO_COT)
ATTR_EXPLAINER_SYSTEM_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts. Your overall task is to describe features of text excerpts that cause the neuron to strongly activate.

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the neuron does not activate on any tokens in the excerpt.

Note: Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation."""

# Finetuned simulator prompt prefix format
ATTR_SIMULATOR_PREFIX = "## Neuron Description: {explanation}\n\n ## Input: "

# Minibatch contrib system prompt (for Anthropic API)
CONTRIB_SYSTEM_PROMPT = """\
You are a meticulous AI researcher studying a neuron inside a language model. Your overall task is to describe features of continuations that a neuron contributes to, given a specific dataset.

You will receive prompts from a particular dataset followed by possible continuation tokens, along with how strongly the neuron contributed to each (scores from -10 to 10, where positive = promotes, negative = suppresses, 0 = no effect).

**Required output properties**: Make your final explanations as concise as possible, using as few words as possible to describe text features that the neuron contributes to.

Output format: [EXPLANATION]: <your explanation>
"""

# Minibatch contrib user template
CONTRIB_USER_TEMPLATE = """\
{exemplars}
"""

# Minibatch simulator system prompt (for Anthropic API)
CONTRIB_SIMULATOR_SYSTEM_PROMPT = """\
You are a meticulous AI researcher studying a neuron inside a language model. You are simulating a neuron's contribution to continuations of given prompts, taken from a dataset. Given a proposed description of the types of continuations that the neuron promotes and suppresses, along with a dataset of prompts with possible continuations, predict scores from -10 to 10 for each continuation for each prompt (where positive = promotes, negative = suppresses, 0 = no effect).

Output format: Respond with a paragraph for each prompt, with the first line being "Prompt: [prompt]" where you repeat the prompt you are scoring for, followed by a second line of the format "Continuations: [('[cont_1]', [score]), ...]" where you repeat each continuation option and your integer score, based on the provided description.
"""

# Minibatch simulator user template
CONTRIB_SIMULATOR_USER_TEMPLATE = """\
Description: {description}

{exemplars_with_blanks}
"""

# --- Cluster summary prompts (for summarize_attr_contrib_descriptions) ---

# Batch prompt: all clusters at once for distinctive labels
CLUSTER_SUMMARY_BATCH_PROMPT = """\
You are labeling clusters of neurons in a language model. Each cluster has two descriptions:

- **Attribution**: what input tokens cause this cluster to activate.
- **Contribution**: what output tokens this cluster promotes or suppresses.

Below are all {n_clusters} clusters, which you will label.

## Your task

Produce a very short, **distinctive** label (1-3 words) for each one.

## Output format

- Be as specific as possible — mention concrete entities or patterns from the examples.
- It is OK for multiple clusters to have similar labels if they are genuinely similar, but try to differentiate where possible.
- DO NOT use vague words like “focus”, “answer”, “response”, “promotion”, “lookup”, “recall”, “mapping”.
- If the cluster has an inhibitory effect, wrap the label in a negative function like this: not[label]
- Labels should be lowercase, except for proper nouns (e.g. "Texas geography", "not[New York cities]").
- Labels should be a space-separated and coherent English phrase.
- Output exactly one label per line, in the format:
CLUSTER_ID: label
- Output all {n_clusters} labels and nothing else.

## Examples

- Attribution: primarily Dallas and some other terms only in Texas-specific context
- Contribution: promotes Texas and cities in Texas, does nothing in other cases
Result: Texas

## Clusters

{cluster_block}"""

# Single-cluster fallback prompts
CLUSTER_SUMMARY_PROMPT = """\
You are summarizing what a cluster of neurons in a language model does. You are given two descriptions:

- **Attribution description**: what input tokens cause this cluster to activate.
- **Contribution description**: what output tokens this cluster promotes or suppresses.

**Attribution**: {attr_desc}
**Contribution**: {contrib_desc}

Produce a very short label (1-3 words) that captures what this cluster does. Be specific — mention key concepts, entities, or patterns. Only output the label, no other text."""

CLUSTER_SUMMARY_PROMPT_ATTR_ONLY = """\
You are summarizing what a cluster of neurons in a language model does. You are given a description of what input tokens cause this cluster to activate.

**Attribution**: {attr_desc}

Produce a very short label (1-3 words) that captures the core function of this cluster. Be specific — mention the key concept, entity, or pattern if one exists. Only output the label, no other text."""

CLUSTER_SUMMARY_PROMPT_CONTRIB_ONLY = """\
You are summarizing what a cluster of neurons in a language model does. You are given a description of what output tokens this cluster promotes or suppresses.

**Contribution**: {contrib_desc}

Produce a very short label (1-3 words) that captures the core function of this cluster. Be specific — mention the key concept, entity, or pattern if one exists. Only output the label, no other text."""

# --- Per-cluster rich summary prompt (attr + contrib + neuron descs) ---

CLUSTER_RICH_SUMMARY_PROMPT = """\
You are labeling a cluster of neurons in {model_id}, a {num_layers}-layer transformer language model. \
You are given three pieces of information about this cluster:

1. **Input attribution description**: what input tokens cause this cluster (as a whole) to activate. \
This is computed by taking the gradient of the neuron activations with respect to the input embeddings.
2. **Output contribution description**: what output tokens this cluster promotes or suppresses. \
This is computed by taking the gradient of the output logits with respect to the neuron activations.
3. **Individual neuron descriptions**: descriptions of the individual neurons in this cluster, \
sorted by their average attribution score (most important first). These are from an automated \
neuron description database and may be noisy.

## Cluster information

**Input attribution**: {attr_desc}

**Output contribution**: {contrib_desc}

**Individual neurons** (sorted by avg attribution, score shown):
{neuron_block}

## Your task

Produce a very short, **distinctive** label (1-3 words) for this cluster.

- Be as specific as possible — mention concrete entities, numbers, or patterns from the descriptions and examples.
- DO NOT use vague words like "focus", "answer", "response", "promotion", "lookup", "recall", "mapping".
- If the cluster has an inhibitory effect, wrap the label in a negative function like this: not[label]
- Labels should be lowercase, except for proper nouns (e.g. "Texas geography", "not[New York cities]").
- Only output the label, no other text."""

CLUSTER_ATTR_CONTRIB_ONLY_SUMMARY_PROMPT = """\
You are labeling a cluster of neurons in {model_id}, a {num_layers}-layer transformer language model. \
You are given two pieces of information about this cluster:

1. **Input attribution description**: what input tokens cause this cluster (as a whole) to activate. \
This is computed by taking the gradient of the neuron activations with respect to the input embeddings.
2. **Output contribution description**: what output tokens this cluster promotes or suppresses. \
This is computed by taking the gradient of the output logits with respect to the neuron activations.

## Cluster information

**Input attribution**: {attr_desc}

**Output contribution**: {contrib_desc}

## Your task

Produce a very short, **distinctive** label (1-3 words) for this cluster.

- Be as specific as possible — mention concrete entities, numbers, or patterns from the descriptions and examples.
- DO NOT use vague words like "focus", "answer", "response", "promotion", "lookup", "recall", "mapping".
- If the cluster has an inhibitory effect, wrap the label in a negative function like this: not[label]
- Labels should be lowercase, except for proper nouns (e.g. "Texas geography", "not[New York cities]").
- Only output the label, no other text."""

CLUSTER_NEURONS_ONLY_SUMMARY_PROMPT = """\
You are labeling a cluster of neurons in {model_id}, a {num_layers}-layer transformer language model. \
You are given descriptions of the individual neurons in this cluster, \
sorted by their average attribution score (most important first). These are from an automated \
neuron description database and may be noisy.

**Individual neurons** (sorted by avg attribution, score shown):
{neuron_block}

## Your task

Produce a very short, **distinctive** label (1-3 words) for this cluster.

- Be as specific as possible — mention concrete entities, numbers, or patterns from the descriptions and examples.
- DO NOT use vague words like "focus", "answer", "response", "promotion", "lookup", "recall", "mapping".
- If the cluster has an inhibitory effect, wrap the label in a negative function like this: not[label]
- Labels should be lowercase, except for proper nouns (e.g. "Texas geography", "not[New York cities]").
- Only output the label, no other text."""

# --- API-based attr explanation prompts (chain-of-thought, inspired by bzl) ---

ATTR_API_SYSTEM_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text excerpts. Your overall task is to describe features of text excerpts that cause the neuron to strongly activate.

You will receive a list of text excerpts on which the neuron activates. Tokens causing activation will appear between delimiters like {{this}}. Consecutive activating tokens will also be accordingly delimited {{just like this}}. If no tokens are highlighted with {{}}, then the neuron does not activate on any tokens in the excerpt.

Note: Neurons activate on a word-by-word basis. Also, neuron activations can only depend on words before the word it activates on, so the description cannot depend on words that come after, and should only depend on words that come before the activation.

Please follow these steps in your analysis:
1. Analyze each excerpt, focusing on the highlighted tokens.
2. Look for common patterns across excerpts — consider semantic, syntactic, and positional features.
3. Summarize what text features cause this neuron to activate.
4. Formulate a concise description.

**Required output properties**: Make your final description as concise as possible, using as few words as possible.

Output format: [DESCRIPTION]: <your description>"""

# Few-shot examples for API-based attr explainer (user/assistant pairs)
ATTR_API_EXAMPLES: list[tuple[str, str]] = [
    (
        # User message: Pacific Northwest example
        """\
Excerpt 1: living in the {{Pacific Northwest}} region, where
Excerpt 2: traveled to {{Seattle}} and {{Portland}} last
Excerpt 3: the {{Oregon}} coast is beautiful in
Excerpt 4: hiking in {{Washington}} state during the
Excerpt 5: {{Vancouver}} BC is just across the""",
        # Assistant response
        """\
Let me analyze the highlighted tokens across these excerpts.

1. "Pacific Northwest" — a geographic region in the northwestern US/Canada
2. "Seattle" and "Portland" — major cities in the Pacific Northwest
3. "Oregon" — a Pacific Northwest state
4. "Washington" — a Pacific Northwest state
5. "Vancouver" — a major city in the Pacific Northwest (BC, Canada)

All highlighted tokens are place names associated with the Pacific Northwest region of North America.

[DESCRIPTION]: references to Pacific Northwest locations (cities, states, and the region itself)""",
    ),
    (
        # User message: "an" prediction example
        """\
Excerpt 1: she picked up {{an}} apple from
Excerpt 2: it was {{an}} unusual situation that
Excerpt 3: he saw {{an}} elephant at the
Excerpt 4: we need {{an}} answer to this
Excerpt 5: there is {{an}} opening in the
Excerpt 6: what {{an}} incredible view from""",
        # Assistant response
        """\
Let me analyze the highlighted tokens.

1-6: The highlighted token is always "an" — the indefinite article used before vowel sounds.

Looking at context: each "an" precedes a word starting with a vowel sound ("apple", "unusual", "elephant", "answer", "opening", "incredible"). The neuron activates on the word "an" specifically when it is used as the indefinite article before a vowel-initial word.

[DESCRIPTION]: the indefinite article "an" before words beginning with a vowel sound""",
    ),
    (
        # User message: closing punctuation example
        """\
Excerpt 1: the end of the sentence{{.}}
Excerpt 2: she asked him a question{{?}}
Excerpt 3: what a surprise{{!}}
Excerpt 4: he finished his statement{{.}}
Excerpt 5: is that really true{{?}}""",
        # Assistant response
        """\
Analyzing the highlighted tokens:

1. "." — period at end of sentence
2. "?" — question mark at end of sentence
3. "!" — exclamation mark at end of sentence
4. "." — period at end of sentence
5. "?" — question mark at end of sentence

The neuron consistently activates on sentence-ending punctuation marks.

[DESCRIPTION]: sentence-ending punctuation marks (periods, question marks, exclamation marks)""",
    ),
]

# API-based attr simulator/scorer prompts (adapted from bzl's prompt-based simulator)
ATTR_API_SIMULATOR_SYSTEM_PROMPT = """\
You are a meticulous AI researcher conducting an important investigation into a specific neuron inside a language model that activates in response to text inputs.

Your overall task is to simulate the activations of a neuron given a sequence of input tokens.
Prior to this, you have studied neuron activation patterns given a large corpus of text and summarized the behavior of a neuron in a sentence or two. Look at the description and try to predict how it will fire on each token.

You will be given the following inputs:
- A description of when the target neuron fires
- One or more input sequences with their tokens
- A template for your response, where [blank] should be filled with your predictions for the activations of the neuron on each token.

Please output your final response in the following format:
```
[SIMULATION]:
Excerpt 1: [(0, token_0, predicted_activation_0), (1, token_1, predicted_activation_1), ...]
Excerpt 2: [(0, token_0, predicted_activation_0), ...]
```
where `predicted_activation_i` is the predicted activation of the neuron on `token_i` as an integer from 0 to 10.

It's extremely important that you only fill in the [blank] with integers from 0 to 10. Do not change anything else in the template."""

ATTR_API_SIMULATOR_USER_TEMPLATE = """\
Description: {description}

{exemplars_with_blanks}"""
