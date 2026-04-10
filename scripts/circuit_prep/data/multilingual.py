"""Multilingual antonyms dataset: opposite-of prompts across 9 languages."""

lang_to_prompt = {
    "en": "What is the opposite of {x}?",
    "zh": "{x}的反义词是什么？",
    "fr": "Quel est le contraire de {x} ?",
    "de": "Was ist das Gegenteil von {x}?",
    "es": "¿Cuál es el opuesto de {x}?",
    "it": "Qual è il contrario di {x}?",
    "ru": "Что является антонимом слова {x}?",
    "hi": "{x} का विपरीतार्थक शब्द क्या है?",
    "ar": "ما هو عكس كلمة {x}؟",
}
lang_to_seed_response = {
    "en": "Answer:",
    "zh": "答案：",
    "fr": "Réponse :",
    "de": "Antwort:",
    "es": "Respuesta:",
    "it": "Risposta:",
    "ru": "Ответ:",
    "hi": "उत्तर:",
    "ar": "الإجابة:",
}
word_to_lang = {
    "big": {
        "en": "big",
        "zh": "大",
        "fr": "grand",
        "de": "groß",
        "es": "grande",
        "it": "grande",
        "ru": "большой",
        "hi": "बड़ा",
        "ar": "كبير",
    },
    "small": {
        "en": "small",
        "zh": "小",
        "fr": "petit",
        "de": "klein",
        "es": "pequeño",
        "it": "piccolo",
        "ru": "маленький",
        "hi": "छोटा",
        "ar": "صغير",
    },
    "fast": {
        "en": "fast",
        "zh": "快",
        "fr": "rapide",
        "de": "schnell",
        "es": "rápido",
        "it": "veloce",
        "ru": "быстрый",
        "hi": "तेज़",
        "ar": "سريع",
    },
    "slow": {
        "en": "slow",
        "zh": "慢",
        "fr": "lent",
        "de": "langsam",
        "es": "lento",
        "it": "lento",
        "ru": "медленный",
        "hi": "धीमा",
        "ar": "بطيء",
    },
    "hot": {
        "en": "hot",
        "zh": "热",
        "fr": "chaud",
        "de": "heiß",
        "es": "caliente",
        "it": "caldo",
        "ru": "горячий",
        "hi": "गर्म",
        "ar": "حار",
    },
    "cold": {
        "en": "cold",
        "zh": "冷",
        "fr": "froid",
        "de": "kalt",
        "es": "frío",
        "it": "freddo",
        "ru": "холодный",
        "hi": "ठंडा",
        "ar": "بارد",
    },
}

prompts: list[str] = []
seed_responses: list[str] = []
labels: list[str] = []
for lang, prompt_template in lang_to_prompt.items():
    for word, lang_to_word in word_to_lang.items():
        if lang in lang_to_word:
            prompts.append(prompt_template.format(x=lang_to_word[lang]))
            seed_responses.append(lang_to_seed_response[lang])
            labels.append(f"{word}___{lang}")
