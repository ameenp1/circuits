"""Capitals dataset with system prompt instructing NONE for Southern states.

The system prompt tells the model to reply NONE for cities in the South.
Single-example dataset: Dallas, Texas (expected answer: NONE).
"""

prompts = ["What is the capital of the state containing Dallas?"]
seed_responses = ["[EMPTY]"]
labels = [" NONE"]
