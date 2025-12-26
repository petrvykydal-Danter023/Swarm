import jax
import jax.numpy as jnp
import pytest
from entropy.core.world import create_initial_state
from entropy.core.physics import physics_step
from entropy.core.sensors import compute_lidars
from entropy.brain.communication import TransformerContextDecoder

def test_full_simulation_step():
    """
    Simulates one complete step of the engine:
    1. Physics (Move agents)
    2. Sensors (Lidar raycast)
    3. Communication (Process messages)
    """
    print("\n🚀 Starting End-to-End Test...")
    
    # === 1. SETUP ===
    num_agents = 5
    state = create_initial_state(num_agents=num_agents)
    rng = jax.random.PRNGKey(42)
    
    # Initialize Comm Model
    comm_model = TransformerContextDecoder(msg_dim=36, context_dim=64)
    dummy_msgs = jnp.zeros((1, num_agents, 36))
    params = comm_model.init(rng, dummy_msgs)
    
    # === 2. SIMULATION LOOP (1 Step) ===
    
    # A. ACTIONS
    # Random motor actions
    actions = jnp.ones((num_agents, 2)) # Full speed forward
    
    # B. PHYSICS STEP
    # Agent 0 starts at (0,0) facing 0. Should move to ~ (1,0) (speed 10 * dt 0.1)
    state = physics_step(state, actions, max_speed=10.0)
    
    assert state.timestep == 1
    assert state.agent_positions[0, 0] > 0.5 # Has moved
    print("✅ Physics: Agents moved.")
    
    # C. SENSORS STEP
    lidars = compute_lidars(state)
    assert lidars.shape == (num_agents, 32)
    assert jnp.all(lidars > 0.0) # No weird zeros in empty space
    print("✅ Sensors: Lidars computed.")
    
    # D. COMMUNICATION STEP
    # Assume agents generated messages (mocked here)
    current_messages = jax.random.normal(rng, (1, num_agents, 36))
    
    # Decode context
    # Context should be [1, num_agents, 64]
    contexts = comm_model.apply(params, current_messages)
    
    assert contexts.shape == (1, num_agents, 64)
    print("✅ Communication: Contexts decoded.")
    
    print("🎉 End-to-End Test PASSED!")

if __name__ == "__main__":
    test_full_simulation_step()
