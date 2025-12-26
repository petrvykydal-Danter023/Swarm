import pytest
import jax.numpy as jnp
from entropy.core.physics import physics_step

def test_physics_step_runs(small_world):
    """Smoke test: physics_step runs without error."""
    actions = jnp.zeros((5, 2))
    state = physics_step(small_world, actions)
    assert state.timestep == 1

def test_agent_motion(small_world):
    """Smoke test: agents move given action."""
    actions = jnp.ones((5, 2)) # Full speed ahead
    state = physics_step(small_world, actions)
    
    # Should have moved from 0,0
    assert not jnp.allclose(state.agent_positions, small_world.agent_positions)
    assert jnp.all(state.agent_positions[:, 0] > 0) # Moved roughly positive x/y dep on angle

def test_no_nans(small_world):
    """Smoke test: no NaNs produced."""
    actions = jnp.ones((5, 2))
    state = physics_step(small_world, actions)
    assert not jnp.any(jnp.isnan(state.agent_positions))
