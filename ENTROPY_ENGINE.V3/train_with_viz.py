"""
Entropy Engine V3 - Training with Visualization Integration
Demonstrates how to use RenderServer during training.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial

from entropy.core.world import WorldState, create_initial_state
from entropy.core.physics import physics_step
from entropy.core.sensors import compute_lidars
from entropy.render.server import RenderServer
from entropy.render.schema import RenderFrame

def world_state_to_render_frame(
    state: WorldState, 
    rewards: np.ndarray = None,
    fps: float = 0.0
) -> RenderFrame:
    """
    Convert JAX WorldState to numpy RenderFrame for visualization.
    """
    return RenderFrame(
        timestep=int(state.timestep),
        agent_positions=np.array(state.agent_positions),
        agent_angles=np.array(state.agent_angles),
        agent_colors=np.random.rand(state.num_agents, 3),  # Random for demo
        agent_messages=np.array(state.agent_messages),
        agent_radii=np.array(state.agent_radii),
        agent_velocities=np.array(state.agent_velocities),
        lidar_readings=np.array(state.lidar_readings),
        goal_positions=np.array(state.goal_positions),
        object_positions=np.array(state.object_positions),
        object_types=np.array(state.object_types),
        wall_segments=np.array(state.wall_segments),
        rewards=rewards,
        fps=fps
    )


def train_with_visualization(
    num_agents: int = 20,
    max_steps: int = 500,
    render_every: int = 5,
    enable_render: bool = True
):
    """
    Run a simple training loop with optional rendering.
    
    Args:
        num_agents: Number of agents
        max_steps: Steps per episode
        render_every: Broadcast frame every N steps (to reduce overhead)
        enable_render: Whether to start RenderServer
    """
    # Initialize
    render_server = RenderServer(port=5555) if enable_render else None
    rng = jax.random.PRNGKey(42)
    
    # Create initial state
    state = create_initial_state(num_agents=num_agents)
    
    # Randomize positions
    rng, pos_rng, goal_rng = jax.random.split(rng, 3)
    state = state.replace(
        agent_positions=jax.random.uniform(pos_rng, (num_agents, 2)) * jnp.array([800, 600]),
        goal_positions=jax.random.uniform(goal_rng, (num_agents, 2)) * jnp.array([800, 600])
    )
    
    print("Training with visualization started.")
    if enable_render:
        print("Run 'python -m entropy.render.viewer' in another terminal to see the visualization.")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        for step in range(max_steps):
            # Random actions for demo
            rng, action_rng = jax.random.split(rng)
            actions = jax.random.uniform(action_rng, (num_agents, 2), minval=-1, maxval=1)
            
            # Physics step
            state = physics_step(state, actions)
            
            # Compute lidars (for debug viz)
            lidars = compute_lidars(state)
            state = state.replace(lidar_readings=lidars)
            
            # Compute pseudo-rewards (just distance to goal)
            dists = jnp.linalg.norm(state.agent_positions - state.goal_positions, axis=1)
            rewards = -dists / 100.0
            
            # Broadcast to renderer
            if render_server and step % render_every == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                frame = world_state_to_render_frame(state, np.array(rewards), fps)
                render_server.publish_frame(frame)
                frame_count += 1
            
            # Small delay to simulate real training time
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    finally:
        if render_server:
            render_server.close()
        
    elapsed = time.time() - start_time
    print(f"Completed {max_steps} steps in {elapsed:.2f}s ({max_steps/elapsed:.1f} steps/sec)")


if __name__ == "__main__":
    train_with_visualization(
        num_agents=30,
        max_steps=1000,
        render_every=5,
        enable_render=True
    )
