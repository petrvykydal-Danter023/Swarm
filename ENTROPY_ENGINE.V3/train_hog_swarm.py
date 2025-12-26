
"""
Entropy Engine V3 - Hand of God (HOG) Training
Fast Swarm Training with linearly decaying expert guidance.
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

from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.mappo import OptimizedMAPPO
from entropy.visuals.recorder import Recorder
from entropy.training.checkpoint import CheckpointManager

# === CONFIG ===
NUM_AGENTS = 20
MAX_STEPS = 200
TOTAL_EPOCHS = 50000 
RENDER = True
RENDER_EVERY = 1000  
OUTPUT_DIR = "outputs/hog_swarm"

# === HOG CONFIG ===
HOG_START = 1.0  # Full expert help at start
HOG_END = 0.0    # No help at end
HOG_DECAY_EPOCHS = 5000 # Help vanishes after 5000 epochs (10% of training)

logger = logging.getLogger("HOGTrain")
logging.basicConfig(level=logging.INFO)

def run_training():
    print("🚀 Starting Hand-of-God Swarm Training...")
    print(f"👻 HOG Schedule: {HOG_START*100}% -> {HOG_END*100}% over {HOG_DECAY_EPOCHS} epochs.")
    
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
    
    ckpt_manager = CheckpointManager("outputs/hog_swarm_checkpoints", max_to_keep=2)

    # 3. Define Scan Rollout with HOG
    print("⚡ JIT Compiling HOG Rollout Loop...")

    def get_expert_action(state):
        # Calculate vector to goal
        # agent_pos: [N, 2], goal_pos: [N, 2]
        # Vectors must be JAX arrays
        diff = state.goal_positions - state.agent_positions
        # Normalize
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        direction = diff / dist
        # Speed factor (assume simplistic 1.0 magnitude for expert)
        return direction * 1.5 # Arbitrary speed scaling to match env dynamics approx

    def rollout_scan_fn(carry, x):
        state, obs, rng, hog_weight = carry
        
        # Split RNG
        rng, a_key, s_key = jax.random.split(rng, 3)
        
        # 1. Agent Inference
        mean, log_std = trainer.actor_state.apply_fn(trainer.actor_state.params, obs)
        noise = jax.random.normal(a_key, mean.shape) * jnp.exp(log_std)
        agent_action = mean + noise
        
        # 2. Expert Calculation
        expert_action = get_expert_action(state)
        
        # 3. Hand of God Mixing
        # Linear Interpolation: (1-w) * Agent + w * Expert
        # We assume 'hog_weight' is a scalar float
        final_action = (1.0 - hog_weight) * agent_action + hog_weight * expert_action
        
        # 4. Log Probability for PPO
        # CRITICAL: We calculate the log_prob of the FINAL action under the AGENT'S distribution.
        # This tells PPO: "You took 'final_action'. It was good. Update your 'mean' to do this more often."
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        log_probs = dist.log_prob(final_action)
        
        # Critic
        global_state = obs.reshape(1, -1)
        values = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
        
        # Step
        next_state, next_obs, rewards, dones, info = env_wrapper.step(state, final_action, s_key)
        
        # Collect Data
        step_data = {
            'obs': obs,
            'actions': final_action, # Store what was actually executed
            'rewards': rewards,
            'dones': dones,
            'values': values.flatten(),
            'log_probs': log_probs,
        }
        
        return (next_state, next_obs, rng, hog_weight), step_data
    
    @jax.jit
    def run_epoch_pipeline(rng, state, obs, hog_weight):
        # 1. Rollout with Scan
        # Pass hog_weight into carry
        carry_in = (state, obs, rng, hog_weight)
        carry_out, trajectory = jax.lax.scan(rollout_scan_fn, carry_in, None, length=MAX_STEPS)
        
        last_state, last_obs, last_rng, _ = carry_out
        
        # 2. Bootstrap Value
        global_state = last_obs.reshape(1, -1)
        last_val = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
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
        
        return (last_state, last_obs, last_rng), trajectory, buffer

    # 4. Training Loop
    try:
        rng, reset_rng = jax.random.split(rng)
        state, obs = env_wrapper.reset(reset_rng)
        
        start_time = time.time()
        last_time = time.time()
        
        for epoch in range(1, TOTAL_EPOCHS + 1):
            
            # --- HOG SCHEDULER ---
            if epoch < HOG_DECAY_EPOCHS:
                progress = (epoch - 1) / HOG_DECAY_EPOCHS
                curr_hog = HOG_START - progress * (HOG_START - HOG_END)
            else:
                curr_hog = HOG_END
            
            # Ensure float for JAX
            curr_hog_jax = jnp.array(curr_hog, dtype=jnp.float32)

            # FAST ROLLOUT
            (state, obs, rng), traj, buffer_jax = run_epoch_pipeline(rng, state, obs, curr_hog_jax)
            
            # --- UPDATE ---
            
            # Convert JAX buffer to Python land structure for OptimizedMAPPO.update if needed?
            # Actually run_epoch_pipeline returns a nice dictionary.
            # But the 'values' concatenation was done inside JAX.
            # OptimizedMAPPO update runs on JAX too.
            # However we had that python-side reshaping fix earlier.
            # Let's see if we can just pass 'buffer_jax' directly if it's all arrays.
            # `OptimizedMAPPO.update`:
            #   obs = jnp.array(buffer['obs']) -> Works on JAX arrays too.
            #   _calculate_gae -> uses numpy loops. So we DO need to block/convert to numpy (implicitly handled by JAX->NumPy bridge usually).
            # But passing JAX arrays to numpy loop might trigger forced syncs.
            # Let's keep it simple.
            
            # We must fix the dimension issue we saw in `train_fast_swarm.py`
            # In `train_fast_swarm.py` we did:
            #   values = np.concatenate(...)
            # Here I did `values`: jnp.concatenate inside JAX.
            # That's actually BETTER/FASTER.
            # The error in `train_fast_swarm` was because I Mixed JAX arrays and Numpy arrays in `run_epoch_pipeline` return? No.
            # It was because `last_val` had wrong shape. I fixed that in `run_epoch_pipeline` above (`last_val_expanded`).
            # So `buffer_jax` is fully consistent.
            
            # Run Update
            a_loss, c_loss = trainer.update(buffer_jax)
            
            mean_reward = float(jnp.mean(traj['rewards']))
            
            if epoch % 10 == 0:
                curr_time = time.time()
                elapsed = curr_time - start_time
                delta_time = curr_time - last_time
                last_time = curr_time
                
                steps_done = 10 * MAX_STEPS 
                fps = steps_done / (delta_time + 1e-6)
                
                print(f"Epoch {epoch} | HOG: {curr_hog:.2f} | R: {mean_reward:.2f} | L: {a_loss:.2f}/{c_loss:.2f} | FPS: {fps:.0f}")
            
            # --- VIZ RECORDING ---
            if RENDER and epoch % RENDER_EVERY == 0:
                print(f"🎥 Recording validation epoch {epoch} (HOG: {curr_hog:.2f})...")
                filename = f"epoch_{epoch:04d}_hog.gif"
                try:
                    # Note: Recorder uses its OWN internal rollout.
                    # Does Recorder utilize HOG?
                    # The `recorder.record_episode` calls `actor_state.apply_fn`.
                    # **It uses purely the learned policy.**
                    # Usefully, this shows us if the agent has ACTUALLY learned, or is just being pushed by HOG.
                    # If the GIF shows random movement while Reward is high, it means HOG is doing the work.
                    # If GIF shows smart movement, the Agent learned from HOG.
                    
                    path = recorder.record_episode(
                        env_wrapper, 
                        trainer.actor_state, 
                        trainer.actor_state.params, 
                        filename
                    )
                    print(f"✅ GIF saved: {path}")
                    
                except Exception as e:
                    print(f"❌ Recording failed: {e}")
            
    except KeyboardInterrupt:
        print("🛑 Stopped.")
    finally:
        pass 

if __name__ == "__main__":
    run_training()
