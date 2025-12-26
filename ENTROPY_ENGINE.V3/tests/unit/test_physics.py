import jax.numpy as jnp
from entropy.core.physics import physics_step
from entropy.core.world import create_initial_state

def test_agent_moves_forward():
    """Agent with actions [1.0, 1.0] should move in the direction of current angle."""
    state = create_initial_state(num_agents=1)
    
    # Set agent at (100, 100) facing right (0 rad)
    state = state.replace(
        agent_positions=jnp.array([[100.0, 100.0]]),
        agent_angles=jnp.array([0.0])
    )
    
    actions = jnp.array([[1.0, 1.0]]) # Forward full speed
    new_state = physics_step(state, actions, max_speed=10.0) # 10 m/s for easy math
    
    # dt = 0.1, speed = 10.0 -> distance = 1.0
    # Expected pos: (101, 100)
    assert new_state.agent_positions[0, 0] > 100.0
    assert jnp.isclose(new_state.agent_positions[0, 0], 101.0, atol=0.1)
    assert jnp.isclose(new_state.agent_positions[0, 1], 100.0, atol=0.01)

def test_agent_rotates():
    """Agent with [1.0, -1.0] should rotate."""
    state = create_initial_state(num_agents=1)
    actions = jnp.array([[1.0, -1.0]])
    new_state = physics_step(state, actions)
    
    assert new_state.agent_angles[0] != 0.0

def test_boundary_clamping():
    """Agent should not leave arena."""
    state = create_initial_state(num_agents=1, arena_size=(100.0, 100.0))
    # Place at edge
    state = state.replace(
        agent_positions=jnp.array([[95.0, 50.0]]),
        agent_radii=jnp.array([10.0])
    )
    
    # Try to move right out of bounds
    actions = jnp.array([[1.0, 1.0]])
    
    # Perform multiple steps
    for _ in range(10):
        state = physics_step(state, actions)
        
    # Should be clamped to width - radius = 100 - 10 = 90
    assert state.agent_positions[0, 0] <= 90.0 + 0.001
