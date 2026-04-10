"""Math addition dataset: "What is X + Y?" for X, Y in [0, 100)."""

prompts = [f"What is {x} + {y}?" for x in range(100) for y in range(100)]
seed_responses = ["Answer: "] * len(prompts)
labels = [f"{x} + {y} = {x + y}" for x in range(100) for y in range(100)]
