import jax
import jax.numpy as jnp
import os
import time
from entropy.config import ExperimentConfig, IntentConfig, SafetyConfig
from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.visuals.recorder import Recorder

def main():
    print("🎬 Generating Safety + Intent Visualization Demo...")
    
    # 1. Config
    cfg = ExperimentConfig()
    cfg.agent.num_agents = 5
    cfg.agent.lidar_rays = 32
    
    # Enable Hybrid Architecture
    cfg.intent = IntentConfig(enabled=True)
    cfg.safety = SafetyConfig(
        enabled=True, 
        safety_radius=40.0,
        repulsion_radius=30.0, 
        enable_repulsion=True
    )
    
    # 2. Setup
    env = EntropyGymWrapper(cfg)
    recorder = Recorder(output_dir="outputs/viz_demo", fps=20)
    
    rng = jax.random.PRNGKey(42)
    state, obs = env.reset(rng)
    
    frames = []
    
    # 3. Simulate Scenario
    print("Simulating...")
    
    # Scenario: Agents moving towards center to trigger collision checks
    # Actions: Target Mode -> Point to (400, 300)
    
    for t in range(100):
        # Create Intent Actions
        # [N, 3 + Rest]
        # Type = 1.0 (Target)
        # P1 = RelX, P2 = RelY
        
        # Calculate relative vectors to center (400, 300)
        center_metrics = jnp.array([400.0, 300.0]) - state.agent_positions
        
        # In local frame? 
        # For simplicity in this test, let's assume P1/P2 are WORLD coordinates if our processor supported it,
        # OR we just map World -> Local roughly for the intent.
        # But wait, our processor assumes RelX/RelY (Egocentric).
        # So we need to rotate world vector into body frame.
        
        # Rotate vector to body frame
        dx = center_metrics[:, 0]
        dy = center_metrics[:, 1]
        dist = jnp.sqrt(dx**2 + dy**2)
        angle_to_target = jnp.arctan2(dy, dx)
        rel_angle = angle_to_target - state.agent_angles
        
        # Local x (forward), y (left)
        local_x = dist * jnp.cos(rel_angle)
        local_y = dist * jnp.sin(rel_angle)
        
        # Construct Action
        # Type=1.0, P1=local_x, P2=local_y
        intent_actions = jnp.stack([
            jnp.full((5,), 1.0), # Type Target
            local_x,
            local_y,
            jnp.zeros(5), # Gate
            jnp.zeros(5), # ...
        ], axis=1)
        
        # Pad with zeros to match action_dim
        # Current action_dim is likely around 6-40 depending on comms
        act_dim = env.action_dim
        if intent_actions.shape[1] < act_dim:
             padding = jnp.zeros((5, act_dim - intent_actions.shape[1]))
             intent_actions = jnp.concatenate([intent_actions, padding], axis=1)
             
        rng, step_rng = jax.random.split(rng)
        state, obs, _, _, _ = env.step(state, intent_actions, step_rng)
        
        # Create Frame manually or use recorder helper?
        # Recorder.record_episode does full loop. We want single step.
        # Let's use internal renderer directly to test render implementation
        # But we need RenderFrame.
        # Let's verify via Recorder but using a mock "Actor"
        
    print("Saving via Recorder...")
    # Since we can't easily hook into loop, let's just use record_episode with a dummy actor
    
    class DummyActor:
        def apply(self, params, obs):
            # Return intent to center
            # We can't access state here easily to do perfect math, 
            # so let's just do random intents to verify VISUALIZATION works.
            
            N = obs.shape[0] if len(obs.shape) > 1 else 1 # obs is batch?
            # Obs shape is [N, O]
            
            # Random switching between Velocity and Target
            # 50% chance
            key = jax.random.PRNGKey(int(time.time_ns()))
            type_rnd = jax.random.uniform(key, (N, 1), minval=-1.0, maxval=1.0)
            
            vals = jax.random.uniform(key, (N, 2), minval=-1.0, maxval=1.0)
            
            # [Type, P1, P2, ...]
            act = jnp.concatenate([type_rnd, vals, jnp.zeros((N, act_dim-3))], axis=1)
            return act, None

    save_path = recorder.record_episode(
        env, DummyActor(), None, "safety_viz_test.mp4", max_steps=100
    )
    print(f"Video saved to: {save_path}")

if __name__ == "__main__":
    main()
