# -*- coding: utf-8 -*-
"""Neuronpedia gemma-2-2b graph prompts (base-model completion style).

Generated from custom_automation/prompts/neuronpedia_graphs.csv by
custom_automation/fetch_neuronpedia_prompts.py, which fetches each slug's
canonical prompt from the Neuronpedia graph API
(https://www.neuronpedia.org/api/graph/gemma-2-2b/<slug>). Regenerate with:

    python custom_automation/fetch_neuronpedia_prompts.py

Each prompt is the Neuronpedia prompt with the leading ``<bos>`` stripped (the
tokenizer re-adds it) and trailing spaces preserved, so tracing reproduces the
same target token Neuronpedia used. gemma-2-2b is a base model, so seed_responses
are empty and the trace targets the model's own next-token prediction. labels are
the Neuronpedia slugs, so each exported graph file maps back to its source graph.

The third tuple field (answer) is documentation only — it does not steer tracing.
"""

# (slug, prompt, answer)
_DATA: list[tuple[str, str, str]] = [
    ("gemma-G", "The International Advanced Security Group (IAS", "G"),
    ("gemma-addition", "3 + 5 = ", "8"),
    ("gemma-addition2", "2 + 1 = ", "3"),
    ("gemma-basket", "Fait: Michael Jordan joue au", "basket"),
    ("gemma-dollar", "Mexico:peso :: US:", "dollar"),
    ("gemma-english", "Mexico:Spanish :: US:", "English"),
    ("gemma-euro", "Mexico:peso :: Europe:", "euro"),
    ("gemma-girl-is", "The girl that the teacher sees", "is"),
    ("gemma-girls-are", "The girls that the teacher sees", "are"),
    ("gemma-gp-nps", "The guitarist knew the song", ". / is"),
    ("gemma-keys-cabinet", "The keys on the cabinet", "are"),
    ("gemma-michael-jordan", "Fact: Michael Jordan plays the sport of", "basketball"),
    ("gemma-michael-jordan-es", "Hecho: Michael Jordan juega al", "baloncesto"),
    ("gemma-saison", "La saison après le printemps s'apelle l'", "été"),
    ("gemma-verano", "La estación después de la primavera se llama el", "verano"),
]

prompts = [prompt for _slug, prompt, _answer in _DATA]
seed_responses = [""] * len(prompts)  # base model: no seed, trace the next token
labels = [slug for slug, _prompt, _answer in _DATA]
