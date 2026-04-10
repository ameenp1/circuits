"""Capitals dataset: prompts asking for the capital of the state containing a given city."""

city_state_capital = [
    ("Dallas", "Texas", "Austin"),
]

prompts = [
    f"What is the capital of the state containing {city}?" for city, _, _ in city_state_capital
]
seed_responses = ["Answer:"] * len(prompts)
labels = [" " + capital for _, _, capital in city_state_capital]
