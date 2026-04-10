"""Shared pytest fixtures for circuit tests."""

import pytest
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR

TEXAS_PICKLE = RESULTS_DIR / "case_studies/texas_circuit.pkl"


@pytest.fixture(scope="session")
def texas_circuit() -> Circuit | None:
    """Load Texas circuit from pickle.

    Returns None if the pickle file doesn't exist (allows tests to skip gracefully).
    """
    if not TEXAS_PICKLE.exists():
        return None
    return Circuit.load_from_pickle(str(TEXAS_PICKLE))


@pytest.fixture(scope="session")
def texas_circuit_required(texas_circuit: Circuit | None) -> Circuit:
    """Same as texas_circuit but skips if not available."""
    if texas_circuit is None:
        pytest.skip(f"Texas circuit pickle not found at {TEXAS_PICKLE}")
    return texas_circuit
