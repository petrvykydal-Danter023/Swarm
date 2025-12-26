import pytest
import jax
import jax.numpy as jnp
from entropy.core.world import WorldState, create_initial_state

@pytest.fixture
def rng():
    """Reproducible RNG for test."""
    return jax.random.PRNGKey(42)

@pytest.fixture
def small_world(rng):
    """Small world for quick tests."""
    return create_initial_state(num_agents=5)

