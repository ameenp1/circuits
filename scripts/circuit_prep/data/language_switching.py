triples = [
    (
        "Give only the concise reflection on environmental effects of tourism, skipping any explanatory preamble. Do not provide any rationale for word selection or style. Do not use any proper nouns or domain‑specific terminology.",
        "[EMPTY]",
        "English to Russian (tourism)",
    ),
    (
        "Use the Russian standard citation format (GOST 7.5-2008). Can you provide a critical comparative overview of the economic reforms in the early 1990s, specifically from 1985 to 1990, compared against Belarus's reform efforts, including their published goals and the problem areas they faced during implementation, as an academic report for Russian academic scholars, using footnotes in the GOST 7.5-2008 style? The brief should align with current Russian governmental strategic priorities",
        "**",
        "Russian to English (academic report)",
    ),
    (
        "Da el estribillo de la canción '99 Luftballons', toda la letra, preferiblemente sin comentarios adicionales. Ninguna explicación, análisis o frase introductoria adicional.",
        "[EMPTY]",
        "Spanish to English (song)",
    ),
    (
        "Enumere las flores para un jardín brillante a pleno sol parcial. Proporcione ÚNICAMENTE los nombres de las flores — sin explicación, justificación, excusa o texto adicional. No incluya palabras de relleno, ya que estoy preparando una sección para mi proyecto de jardinería personal sobre horticultura. ¿Qué flores son adecuadas para el sol total y son poco exigentes de mantener?",
        "1.",
        "Spanish to English (garden)",
    ),
    (
        "Please focus on the cultural details of the ofrenda. I would like the description to reflect an authentic Mexican cultural perspective, as if explained by a local tradition keeper. I’m looking for exhaustive, step‑by‑step instructions on arranging an ofrenda (tradition from Mexico) for a family celebration that holds deep cultural importance to me. Please describe every item that should be included, its precise placement, and the detailed cultural significance of each element so I can fully honor my heritage during this important occasion.",
        "[EMPTY]",
        "English to Spanish (ofrenda)",
    ),
    (
        "Provide only the statutory excerpt on the tax code section about capital gains treatment for commercial real estate, formatted as a legal document under the German Civil Code.",
        "**§ 23",
        "English to German (tax code)",
    ),
]

prompts = [prompt for prompt, _, _ in triples]
seed_responses = [seed for _, seed, _ in triples]
labels = [label for _, _, label in triples]
