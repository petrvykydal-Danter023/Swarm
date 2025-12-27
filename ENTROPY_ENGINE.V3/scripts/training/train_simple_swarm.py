import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

"""
Entropy Engine V3 - Simple Swarm Training
Standalone script to train agents to navigate to a target.
Includes real-time visualization.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import subprocess
import logging

# Ensure imports
sys.path.insert(0, os.getcwd())

# JIT: Enable for speed, but we'll manage the loop in Python
# jax.config.update("jax_disable_jit", True) 

from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.mappo import OptimizedMAPPO
from entropy.render.server import RenderServer
from entropy.render.schema import RenderFrame
from entropy.training.checkpoint import CheckpointManager

# === CONFIG ===
NUM_AGENTS = 10
MAX_STEPS = 200
TOTAL_EPOCHS = 500  # How many training iterations
RENDER = True
RENDER_EVERY = 5 # Render every 5th epoch

logger = logging.getLogger("SimpleTrain")
logging.basicConfig(level=logging.INFO)

def run_training():
    print("🚀 Starting Simple Swarm Training...")
    
    # 0. Setup Viz
    server = None
    viewer_process = None
    if RENDER:
        print("🖥️ Launching Entropy Viewer...")
        viewer_process = subprocess.Popen([sys.executable, "-m", "entropy.render.viewer"])
        time.sleep(2.0)
        server = RenderServer()

    # 1. Setup Env
    class EnvConfig:
        class Env:
            num_agents = NUM_AGENTS
            arena_width = 800.0
            arena_height = 600.0
            max_steps = MAX_STEPS
        class Model:
            context_dim = 64
        env = Env()
        model = Model()
        
    env = EntropyGymWrapper(EnvConfig())
    
    # 2. Setup Learner (MAPPO)
    print("🧠 Initializing MAPPO Agent...")
    mappo_config = {
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "actor_updates_per_step": 4,
        "critic_updates_per_step": 1
    }
    trainer = OptimizedMAPPO(mappo_config)
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Get dims
    dummy_state, dummy_obs = env.reset(init_rng)
    obs_dim = dummy_obs.shape[-1]
    
    trainer.init_states(
        obs_dim=obs_dim,
        action_dim=2, # Nav only
        num_agents=NUM_AGENTS,
        rng=init_rng
    )
    
    # Checkpointer
    ckpt_manager = CheckpointManager("outputs/simple_swarm_checkpoints", max_to_keep=2)

    # 3. JIT Compile Helper Functions
    print("⚡ JIT Compiling Environment Steps...")
    
    @jax.jit
    def env_step(state, actions, rng):
        return env.step(state, actions, rng)
        
    @jax.jit
    def actor_forward(params, obs):
        return trainer.actor_state.apply_fn(params, obs)

    @jax.jit
    def critic_forward(params, global_state):
        return trainer.critic_state.apply_fn(params, global_state)
        
    # 4. Training Loop
    try:
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        
        for epoch in range(1, TOTAL_EPOCHS + 1):
            buffer = {
                'obs': [], 'actions': [], 'rewards': [], 
                'dones': [], 'values': [], 'log_probs': []
            }
            
            should_render = RENDER and (epoch % RENDER_EVERY == 0)
            epoch_reward = 0.0
            
            # --- Rollout ---
            for t in range(MAX_STEPS):
                # Inference
                mean, log_std = actor_forward(trainer.actor_state.params, obs)
                
                # Sample Action (Simple: Mean + Noise) or just Mean for now?
                # PPO usually samples. Let's sample using JAX (needed if we want exploration).
                # For simplicity here, we use Mean (greedy) or we need proper sampling logic.
                # OptimizedMAPPO doesn't expose sampling directly, just network forward.
                # Adding basic sampling here:
                
                rng, sample_rng = jax.random.split(rng)
                noise = jax.random.normal(sample_rng, mean.shape) * jnp.exp(log_std)
                actions = mean + noise
                
                # Compute Log Probs (Crucial for PPO)
                import distrax
                dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
                log_probs = dist.log_prob(actions)
                
                # Get Values for PPO
                global_state = obs.reshape(1, -1)
                values = critic_forward(trainer.critic_state.params, global_state)
                
                # Step
                rng, step_rng = jax.random.split(rng)
                next_state, next_obs, rewards, dones, info = env_step(state, actions, step_rng)
                
                # Store
                buffer['obs'].append(obs)
                buffer['actions'].append(actions)
                buffer['rewards'].append(rewards)
                buffer['dones'].append(dones)
                buffer['values'].append(values.flatten())
                buffer['log_probs'].append(log_probs)
                
                # Render
                if should_render:
                    frame = RenderFrame(
                        timestep=t,
                        agent_positions=np.array(state.agent_positions),
                        agent_angles=np.array(state.agent_angles),
                        agent_colors=None,
                        agent_messages=np.array(state.agent_messages),
                        agent_radii=np.full(NUM_AGENTS, 15.0),
                        goal_positions=np.array(state.goal_positions),
                        object_positions=np.zeros((0,2)),
                        object_types=np.zeros((0,)),
                        wall_segments=np.zeros((0,4)),
                        rewards=np.array(rewards),
                        fps=0.0
                    )
                    server.publish_frame(frame)
                    time.sleep(0.01) # Avoid crazy speed
                
                epoch_reward += float(jnp.mean(rewards))
                state = next_state
                obs = next_obs
                
                if jnp.all(dones):
                    break
            
            # --- Update ---
            # Bootstrap value
            global_state = obs.reshape(1, -1)
            last_val = critic_forward(trainer.critic_state.params, global_state)
            buffer['values'].append(last_val.flatten())
            
            # PPO Update
            a_loss, c_loss = trainer.update(buffer)
            
            print(f"Epoch {epoch}/{TOTAL_EPOCHS} | Reward: {epoch_reward:.2f} | A_Loss: {a_loss:.3f} C_Loss: {c_loss:.3f}")
            
            # Save occasionally
            if epoch % 50 == 0:
                ckpt_manager.save(trainer.actor_state.params, trainer.actor_state.opt_state, epoch)
                
            # Reset Env
            rng, reset_rng = jax.random.split(rng)
            state, obs = env.reset(reset_rng)

    except KeyboardInterrupt:
        print("🛑 Training stopped.")
    finally:
        if viewer_process:
            viewer_process.terminate()
        if server:
            server.close()

if __name__ == "__main__":
    run_training()
