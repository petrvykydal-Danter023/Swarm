
"""
Entropy Engine V3 - Fast Swarm Training
Uses jax.lax.scan for 1000x faster rollouts, with Replay Visualization.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import subprocess
import logging
import distrax

# Ensure imports
sys.path.insert(0, os.getcwd())

# JIT MUST be enabled for scan
# jax.config.update("jax_disable_jit", False)

from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.mappo import OptimizedMAPPO
# from entropy.render.server import RenderServer
# from entropy.render.schema import RenderFrame
from entropy.visuals.recorder import Recorder
from entropy.training.checkpoint import CheckpointManager

# === CONFIG ===
NUM_AGENTS = 20
MAX_STEPS = 200
TOTAL_EPOCHS = 50000 # Increased for long run
RENDER = True
RENDER_EVERY = 1000  # Render every 1000th epoch (User Request)
OUTPUT_DIR = "outputs/fast_swarm"

logger = logging.getLogger("FastTrain")
logging.basicConfig(level=logging.INFO)

def run_training():
    print("🚀 Starting FAST Swarm Training (JIT + Scan + GIF Recording)...")
    
    # 0. Setup Recorder
    recorder = None
    if RENDER:
        print("📹 Initializing Recorder...")
        recorder = Recorder(output_dir=OUTPUT_DIR, fps=20)

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
        
    env_wrapper = EntropyGymWrapper(EnvConfig())
    
    # 2. Setup Learner
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
    
    dummy_state, dummy_obs = env_wrapper.reset(init_rng)
    obs_dim = dummy_obs.shape[-1]
    
    trainer.init_states(obs_dim, 2, NUM_AGENTS, init_rng)
    
    ckpt_manager = CheckpointManager("outputs/fast_swarm_checkpoints", max_to_keep=2)

    # 3. Define Scan Rollout (The Speed Secret)
    print("⚡ JIT Compiling Rollout Loop...")

    def rollout_scan_fn(carry, x):
        state, obs, rng = carry
        
        # Split RNG
        rng, a_key, s_key = jax.random.split(rng, 3)
        
        # Inference
        mean, log_std = trainer.actor_state.apply_fn(trainer.actor_state.params, obs)
        noise = jax.random.normal(a_key, mean.shape) * jnp.exp(log_std)
        actions = mean + noise
        
        # Log Prob
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        log_probs = dist.log_prob(actions)
        
        # Critic
        global_state = obs.reshape(1, -1)
        values = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
        
        # Step
        next_state, next_obs, rewards, dones, info = env_wrapper.step(state, actions, s_key)
        
        # Collect Data
        step_data = {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values.flatten(),
            'log_probs': log_probs,
            # For Viz (Save positions if rendering needed, but scan needs array outputs)
            'viz_positions': state.agent_positions,
            'viz_angles': state.agent_angles,
            'viz_goals': state.goal_positions
        }
        
        return (next_state, next_obs, rng), step_data
    
    @jax.jit
    def run_epoch_pipeline(rng, state, obs):
        # 1. Rollout with Scan
        carry_in = (state, obs, rng)
        carry_out, trajectory = jax.lax.scan(rollout_scan_fn, carry_in, None, length=MAX_STEPS)
        
        last_state, last_obs, last_rng = carry_out
        
        # 2. Bootstrap Value
        global_state = last_obs.reshape(1, -1)
        last_val = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
        # Reshape last_val to [1, NumAgents] to match trajectory['values'] [Time, NumAgents]
        last_val_expanded = last_val.reshape(1, -1) 
        
        # Prepare Buffer
        buffer = {
            'obs': trajectory['obs'],
            'actions': trajectory['actions'],
            'rewards': trajectory['rewards'],
            'dones': trajectory['dones'],
            'values': jnp.concatenate([trajectory['values'], last_val_expanded], axis=0), 
            'log_probs': trajectory['log_probs']
        }
        
        # 3. Update (Gradients)
        # Note: trainer.update expects lists usually, we need to ensure it handles arrays.
        # OptimizedMAPPO.update takes 'buffer' dict of arrays.
        # But wait, `buffer['values']` logic in my wrapper (python) handled appending.
        # `update` in `mappo.py` handles arrays.
        # But `OptimizedMAPPO._calculate_gae` uses numpy loops... so we can't JIT `update` fully yet?
        # Checking mappo.py: Yes, `_calculate_gae` uses python loop.
        # So we can JIT rollout, but update might need to be outside OR we fix GAE to be scan-based.
        # Simplest fix: Return TRAJECTORY, do Update in Python (fast enough), but Rollout in JAX.
        
        return (last_state, last_obs, last_rng), trajectory

    # 4. Training Loop
    try:
        rng, reset_rng = jax.random.split(rng)
        state, obs = env_wrapper.reset(reset_rng)
        
        start_time = time.time()
        last_time = time.time()
        
        for epoch in range(1, TOTAL_EPOCHS + 1):
            
            # FAST ROLLOUT
            (state, obs, rng), traj = run_epoch_pipeline(rng, state, obs)
            
            # --- UPDATE ---
            # ... (update logic unchanged) ...
            
            # RE-RUN Critic for Bootstrap (Cheap)
            global_state = obs.reshape(1, -1)
            last_val = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
            last_val = np.array(last_val).reshape(1, -1)
            
            buffer = {
                'obs': traj['obs'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['dones'],
                'values': np.concatenate([traj['values'], last_val], axis=0), 
                'log_probs': traj['log_probs']
            }
            
            # Run Update
            a_loss, c_loss = trainer.update(buffer)
            
            mean_reward = float(jnp.mean(traj['rewards']))
            
            if epoch % 10 == 0:
                curr_time = time.time()
                elapsed = curr_time - start_time
                delta_time = curr_time - last_time
                last_time = curr_time
                
                # FPS Calculation
                # 10 epochs * MAX_STEPS
                steps_done = 10 * MAX_STEPS 
                fps = steps_done / (delta_time + 1e-6)
                
                print(f"Epoch {epoch} | R: {mean_reward:.2f} | Losses: {a_loss:.2f}/{c_loss:.2f} | FPS: {fps:.0f}")
            
            # --- VIZ RECORDING ---
            if RENDER and epoch % RENDER_EVERY == 0:
                print(f"🎥 Recording validation epoch {epoch}...")
                # We can either record the trajectory we just collected (if we have a way to render it purely from data)
                # OR (simpler/robuster) run a fresh validation episode using the Recorder.
                # Recorder runs a separate rollout.
                
                filename = f"epoch_{epoch:04d}.gif"
                # Note: recorder needs env_wrapper (new instance or reset one)
                try:
                    path = recorder.record_episode(
                        env_wrapper, 
                        trainer.actor_state, 
                        trainer.actor_state.params, 
                        filename
                    )
                    print(f"✅ GIF saved: {path}")
                    
                    # Optional: Log to WandB if available
                    # if wandb.run: ...
                except Exception as e:
                    print(f"❌ Recording failed: {e}")
            
    except KeyboardInterrupt:
        print("🛑 Stopped.")
    finally:
        pass # No viewer process to kill

if __name__ == "__main__":
    run_training()
