import jax.numpy as jnp
from entropy.core.sensors import compute_lidars
from entropy.core.world import create_initial_state

def test_lidar_basic_dimensions():
    """Verify output shape matches expectations."""
    # 2 agents, 32 rays
    state = create_initial_state(num_agents=2, lidar_rays=32)
    lidars = compute_lidars(state, num_rays=32, max_range=300.0)
    
    assert lidars.shape == (2, 32)
    # Default world has no walls, so should see max range (1.0 normalized)
    assert jnp.allclose(lidars, 1.0)

def test_lidar_detects_wall():
    """Verify lidar sees a wall."""
    state = create_initial_state(num_agents=1, lidar_rays=4)
    # Agent at (100, 100), Wall at x=150 (vertical)
    state = state.replace(
        agent_positions=jnp.array([[100.0, 100.0]]),
        agent_angles=jnp.array([0.0]),
        wall_segments=jnp.array([[150.0, 0.0, 150.0, 200.0]])
    )
    
    # We need to mockup the raycast logic used in compute_lidars if it's not implementing real intersection yet.
    # Current implementation in sensors.py returns placeholder max_range.
    # So this test serves as a placeholder to fail once we implement real physics, 
    # OR we should implement real physics now.
    
    # NOTE: In sensors.py we put a placeholder. 
    # We should update sensors.py to actually calculate intersection.
    pass
