"""
Shared configuration for the custom_automation pipeline.

This is the ADAG (raw MLP neuron) variant of the pipeline: the description and
grouping steps operate directly on the per-graph `graph_*.json` files produced by
`scripts/circuit_prep/batch_export_neurons.py` (each carries `prompt`, `target`,
and a top-N `neurons` list). There is no transcoder feature store, no remote
viewer, and no fixed artifact files — every step reads and writes the graph JSON
in place, so this module only holds the LLM knobs + logging.

Override any constant via the matching environment variable.
"""

import logging
import os
import sys
from pathlib import Path

# Make stdout/stderr tolerant of non-ASCII tokens on Windows terminals.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the custom_automation package (directory containing this file).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# ---------------------------------------------------------------------------
# Models (OpenAI) — used by generate_description.py / generate_supernodes.py
# ---------------------------------------------------------------------------

# Model for generating per-neuron descriptions.
DESCRIPTION_MODEL: str = os.environ.get("DESCRIPTION_MODEL", "gpt-5-mini")

# Model for semantic clustering into supernodes.
GROUPING_MODEL: str = os.environ.get("GROUPING_MODEL", "gpt-5-mini")

# ---------------------------------------------------------------------------
# Grouping (generate_supernodes.py)
# ---------------------------------------------------------------------------

# Which grouping prompt variant to use (a0=neutral baseline, a1=structural rules,
# a2=a1 + description-aware naming, a3=a2 + reader test). Override with env var.
GROUPING_VARIANT: str = os.environ.get("GROUPING_VARIANT", "a2")

# Phase 1 seeds groups from the top-K most influential features; Phase 2 assigns
# the remaining (lower-influence) features in batches of GROUPING_BATCH_SIZE.
GROUPING_TOP_K_SEED: int = int(os.environ.get("GROUPING_TOP_K_SEED", "50"))
GROUPING_BATCH_SIZE: int = int(os.environ.get("GROUPING_BATCH_SIZE", "50"))

# Max concurrent Phase-2 batch requests in flight at once.
GROUPING_MAX_CONCURRENCY: int = int(os.environ.get("GROUPING_MAX_CONCURRENCY", "32"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int | None = None) -> logging.Logger:
    """Configure and return the pipeline-wide logger."""
    if level is None:
        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("custom_automation")
