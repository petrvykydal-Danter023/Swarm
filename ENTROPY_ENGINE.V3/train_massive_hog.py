
"""
Entropy Engine V3 - MASSIVE HOG Training
Combines:
1. Massive Parallelism (64 Envs x 20 Agents = 1280 Agents) via jax.vmap
2. Hand of God (HOG) Expert Guidance with decay
3. JAX Scan for fast rollouts
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
NUM_ENVS = 64       # Massive Parallelism
NUM_AGENTS = 20     # Per Env
MAX_STEPS = 200
TOTAL_EPOCHS = 50000 
RENDER = True
RENDER_EVERY = 1000  
OUTPUT_DIR = "outputs/massive_hog"

# === HOG CONFIG ===
HOG_START = 1.0  
HOG_END = 0.0    
HOG_DECAY_EPOCHS = 5000 

logger = logging.getLogger("MassiveHOG")
logging.basicConfig(level=logging.INFO)

def run_training():
    print("🚀 Starting MASSIVE HOG Swarm Training...")
    print(f"🌍 Environments: {NUM_ENVS}")
    print(f"🤖 Total Agents: {NUM_ENVS * NUM_AGENTS}")
    print(f"👻 HOG Schedule: {HOG_START*100}% -> {HOG_END*100}%")
    
    # 0. Setup Recorder
    recorder = None
    if RENDER:
        recorder = Recorder(output_dir=OUTPUT_DIR, fps=20)

    # 1. Setup Env Config
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
        
    # We use ONE wrapper class, but we will vmap its methods
    env_wrapper = EntropyGymWrapper(EnvConfig())
    
    # 2. Setup Learner
    print("🧠 Initializing MAPPO Agent...")
    mappo_config = {
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "actor_updates_per_step": 4, # Standard PPO
        "critic_updates_per_step": 1
    }
    trainer = OptimizedMAPPO(mappo_config)
    
    # Init Random Keys
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Init One Instance to get Dims
    dummy_state, dummy_obs = env_wrapper.reset(init_rng)
    obs_dim = dummy_obs.shape[-1]
    
    trainer.init_states(obs_dim, 2, NUM_AGENTS, init_rng)
    
    ckpt_manager = CheckpointManager("outputs/massive_hog_checkpoints", max_to_keep=2)

    # 3. Define Logic
    
    # --- Single Env Logic (Same as before) ---
    def get_expert_action(state):
        diff = state.goal_positions - state.agent_positions
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        direction = diff / dist
        return direction * 1.5 

    def rollout_scan_fn(carry, x):
        state, obs, rng, hog_weight = carry
        rng, a_key, s_key = jax.random.split(rng, 3)
        
        # Inference
        mean, log_std = trainer.actor_state.apply_fn(trainer.actor_state.params, obs)
        noise = jax.random.normal(a_key, mean.shape) * jnp.exp(log_std)
        agent_action = mean + noise
        
        # HOG
        expert_action = get_expert_action(state)
        final_action = (1.0 - hog_weight) * agent_action + hog_weight * expert_action
        
        # Log Prob (of Final Action)
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        log_probs = dist.log_prob(final_action)
        
        # Critic
        global_state = obs.reshape(1, -1)
        values = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
        
        # Step
        next_state, next_obs, rewards, dones, info = env_wrapper.step(state, final_action, s_key)
        
        step_data = {
            'obs': obs,
            'actions': final_action,
            'rewards': rewards,
            'dones': dones,
            'values': values.flatten(),
            'log_probs': log_probs,
        }
        return (next_state, next_obs, rng, hog_weight), step_data

    # --- Massive Parallel Logic ---
    # We vmap the rollout function over the 'carry' (states)
    
    @jax.jit
    def run_epoch_pipeline(rng, states, obss, hog_weight):
        # inputs: [NumEnvs, ...], [NumEnvs, ...], [NumEnvs, ...]
        
        # We need to scan over time... but we need vmap over Envs.
        # Strategy:
        # Define a function 'single_env_rollout' that does scan.
        # Then vmap THAT function.
        
        def single_env_rollout(state, obs, r, h):
            carry_in = (state, obs, r, h)
            carry_out, trajectory = jax.lax.scan(rollout_scan_fn, carry_in, None, length=MAX_STEPS)
            
            # Bootstrap value logic inside single env
            last_state, last_obs, last_rng, _ = carry_out
            global_state = last_obs.reshape(1, -1)
            last_val = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
            last_val_expanded = last_val.reshape(1, -1)
            
            # Combine values
            values_full = jnp.concatenate([trajectory['values'], last_val_expanded], axis=0) # [T+1, N]
            
            # Return slightly restructured trajectory for easier stacking
            return (last_state, last_obs, last_rng), trajectory, values_full

        # VMAP IT!
        # Maps over (states, obss, rngs)
        # hog_weight is broadcasted (None)
        
        # rng needs to be split for each env
        rngs = jax.random.split(rng, NUM_ENVS)
        
        (last_states, last_obss, last_rngs), trajectories, all_values = jax.vmap(
            single_env_rollout, in_axes=(0, 0, 0, None)
        )(states, obss, rngs, hog_weight)
        
        return (last_states, last_obss, last_rngs[0]), trajectories, all_values

    # 4. Training Loop
    try:
        # Reset All Envs
        # We can vmap reset too
        reset_rngs = jax.random.split(rng, NUM_ENVS)
        states, obss = jax.vmap(env_wrapper.reset)(reset_rngs)
        
        print("⚡ JIT Compiling Massive Pipeline...")
        # Warmup?
        # _ = run_epoch_pipeline(rng, states, obss, 1.0)
        
        start_time = time.time()
        last_time = time.time()
        
        for epoch in range(1, TOTAL_EPOCHS + 1):
            
            # Scheduler
            if epoch < HOG_DECAY_EPOCHS:
                progress = (epoch - 1) / HOG_DECAY_EPOCHS
                curr_hog = HOG_START - progress * (HOG_START - HOG_END)
            else:
                curr_hog = HOG_END
            curr_hog_jax = jnp.array(curr_hog, dtype=jnp.float32)

            # --- MASSIVE ROLLOUT ---
            (states, obss, rng), trajs, all_values = run_epoch_pipeline(rng, states, obss, curr_hog_jax)
            
            # --- PREPARE DATA ---
            # trajs structure: {key: [NumEnvs, Steps, ...]}
            # we need flattened: [NumEnvs * Steps, ...]
            
            # Helper to flatten: [E, T, N, D] -> [E*T, N, D]
            # Wait, OptimizedMAPPO handles [Steps, N, D].
            # Actually, `update` treats first dim as batch.
            # So we can just concatenate E and T.
            # obs: [NumEnvs, Steps, Agents, Dims] -> [NumEnvs*Steps, Agents, Dims]
            
            def flatten_batch(x):
                # x is [E, T, ...]
                # reshape to [E*T, ...]
                s = x.shape
                return x.reshape((s[0] * s[1],) + s[2:])
            
            buffer = {
                'obs': flatten_batch(trajs['obs']),
                'actions': flatten_batch(trajs['actions']),
                'rewards': flatten_batch(trajs['rewards']),
                'dones': flatten_batch(trajs['dones']),
                'log_probs': flatten_batch(trajs['log_probs']),
                # Values: [E, T+1, N] -> we need compatible format.
                # OptimizedMAPPO _calculate_gae expects [T, N] and [T+1, N].
                # It does NOT handle batches of episodes easily in the current loop implementation 
                # because GAE needs strict temporal order (t+1 depends on t).
                # We CANNOT flatten E and T together for GAE calculation.
                # We must modify `update` or run GAE per Env.
            }
            
            # Hack: Run Update manually here? Or extend OptimizedMAPPO?
            # Easiest: Let's do GAE computation HERE, inside Python loop, for each Env, then concatenate.
            # This is safer.
            
            # Extract as numpy for GAE
            rew_np = np.array(trajs['rewards'])   # [E, T, N]
            val_np = np.array(all_values)         # [E, T+1, N]
            don_np = np.array(trajs['dones'])     # [E, T, N]
            
            all_advantages = []
            
            # Compute GAE for each Env
            # This loop is fast (64 iters)
            for i in range(NUM_ENVS):
                adv = trainer._calculate_gae(rew_np[i], val_np[i], don_np[i])
                all_advantages.append(adv)
                
            all_advantages = np.concatenate(all_advantages, axis=0) # [E*T, N]
            
            # Now we can update using flattened buffers
            # But wait, trainer.update normally calculates GAE itself.
            # We should bypass that or feed it something it likes.
            # OptimizedMAPPO.update is designed for single trajectory usually.
            # Actually, looking at mappo.py:
            # `update(self, buffer)`:
            #   obs = buffer['obs'] ...
            #   advantages = self._calculate_gae(...)
            # It assumes `rewards` is [T, N].
            # If we pass [E*T, N], GAE will treat it as one looooong episode.
            # Steps E*T and (E*T)-1 (which is from different env) would be linked. 
            # THIS IS WRONG. GAE breaks at boundaries.
            
            # FIX: We calculated correct GAEs above (`all_advantages`).
            # We should subclass or modify `update` to accept pre-calculated advantages?
            # Or temporarily monkey-patch?
            # Or just update `mappo.py`... 
            # Let's add a `update_with_advantages` method or similar.
            # OR simpler: OptimizedMAPPO is ours. Let's modify it to handle batch dim if present?
            # Or just hack it:
            # We can't easily change mappo.py right now safely without breaking other things.
            
            # ALTERNATIVE: Use the Trainer's private methods directly.
            # `update` does:
            # 1. GAE
            # 2. Critic Update (loop)
            # 3. Actor Update (loop)
            
            # We can replicate this logic here.
            
            # 1. Targets (Returns) = Adv + Values[:-1]
            # val_np is [E, T+1, N]. We need [E*T, N] corresponding to rewards.
            values_trimmed = val_np[:, :-1, :].reshape(-1, NUM_AGENTS) # [E*T, N]
            targets = all_advantages + values_trimmed
            
            # Normalize Adv
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
            all_advantages = jnp.array(all_advantages)
            targets = jnp.array(targets)
            
            # 2. Critic Update
            obs_all = jnp.array(buffer['obs']) # [E*T, N, D]
            total_steps = obs_all.shape[0]
            # global state
            obs_flat = obs_all.reshape(total_steps, -1) # [E*T, N*D]
            
            critic_loss_val = 0.0
            for _ in range(trainer.critic_updates_per_step):
                trainer.critic_state, loss, _ = trainer._train_critic(
                    trainer.critic_state, obs_flat, targets.flatten(), NUM_AGENTS
                )
                critic_loss_val += loss
                
            # 3. Actor Update
            actions_all = jnp.array(buffer['actions'])
            log_probs_all = jnp.array(buffer['log_probs'])
            
            obs_flat_actor = obs_all.reshape(total_steps * NUM_AGENTS, -1)
            actions_flat = actions_all.reshape(total_steps * NUM_AGENTS, -1)
            log_probs_flat = log_probs_all.reshape(total_steps * NUM_AGENTS)
            adv_flat = all_advantages.reshape(total_steps * NUM_AGENTS)
            
            actor_loss_val = 0.0
            for _ in range(trainer.actor_updates_per_step):
                trainer.actor_state, loss, _ = trainer._train_actor(
                    trainer.actor_state, obs_flat_actor, actions_flat, log_probs_flat, adv_flat
                )
                actor_loss_val += loss
                
            # --- LOGGING ---
            mean_reward = float(np.mean(rew_np))
            
            if epoch % 10 == 0:
                curr_time = time.time()
                elapsed = curr_time - start_time
                delta_time = curr_time - last_time
                last_time = curr_time
                
                # FPS: Total Agent Steps / Time
                # Steps = 10 * MAX_STEPS * NUM_ENVS
                steps_done = 10 * MAX_STEPS * NUM_ENVS
                fps = steps_done / (delta_time + 1e-6)
                
                print(f"Epoch {epoch} | HOG: {curr_hog:.2f} | R: {mean_reward:.2f} | L: {actor_loss_val:.2f}/{critic_loss_val:.2f} | FPS: {fps:.0f}")
            
            # --- VIZ RECORDING ---
            if RENDER and epoch % RENDER_EVERY == 0:
                print(f"🎥 Recording validation epoch {epoch}...")
                # Use Recorder with First Env setup
                # Just take the model state and run a single discrete episode
                filename = f"epoch_{epoch:04d}_massive.gif"
                try:
                    path = recorder.record_episode(
                        env_wrapper, trainer.actor_state, trainer.actor_state.params, filename
                    )
                    print(f"✅ GIF saved: {path}")
                except Exception as e:
                     print(f"Recording failed: {e}")

    except KeyboardInterrupt:
        print("🛑 Stopped.")
    finally:
        pass 

if __name__ == "__main__":
    run_training()
