
"""
Entropy Engine V3 - MASSIVE HYBRID HOG Training
"The Holy Grail" Architecture
Combines:
1. Massive Parallelism (64 Envs)
2. Hybrid Architecture (Intent + Safety)
3. Hand of God (Expert Intent Guidance)
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import logging
import distrax

# Ensure imports
sys.path.insert(0, os.getcwd())

from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.mappo import OptimizedMAPPO
from entropy.visuals.recorder import Recorder
from entropy.training.checkpoint import CheckpointManager
from entropy.config import ExperimentConfig
from entropy.training.hand_of_god.expert import IntentNavigator, DirectNavigator

# === CONFIGURATION ===
# We use the Universal ExperimentConfig for everything
from dataclasses import replace

def create_hybrid_config():
    cfg = ExperimentConfig()
    
    # 1. Simulation Setup
    cfg.sim.num_envs = 64
    cfg.sim.max_steps = 200
    cfg.total_epochs = 10000 
    
    # 2. Agent Setup
    cfg.agent.num_agents = 20
    
    # 3. Hybrid Mode (The Core)
    cfg.intent.enabled = True
    cfg.safety.enabled = True
    cfg.safety.log_metrics = True # Monitor safety interventions
    
    # 4. HOG Setup
    cfg.hog.enabled = True
    cfg.hog.start_weight = 1.0
    cfg.hog.end_weight = 0.0
    cfg.hog.decay_epochs = 2000
    
    # 5. Rewards (Living Penalty is Key)
    cfg.reward.w_living_penalty = -0.001
    cfg.reward.w_dist = 1.0
    cfg.reward.w_reach = 10.0
    
    # 6. Render
    cfg.render.enabled = True
    cfg.render.fps = 20
    cfg.render.output_dir = "outputs/massive_hybrid"
    
    return cfg

def run_training():
    cfg = create_hybrid_config()
    print("🚀 Starting MASSIVE HYBRID HOG Training (The Holy Grail)...")
    print(f"🌍 Environments: {cfg.sim.num_envs}")
    print(f"🤖 Total Agents: {cfg.sim.num_envs * cfg.agent.num_agents}")
    print(f"🛡️ Safety Layer: {cfg.safety.enabled}")
    print(f"🧠 Intent System: {cfg.intent.enabled}")
    
    # 0. Setup Recorder
    recorder = None
    if cfg.render.enabled:
        recorder = Recorder(output_dir=cfg.render.output_dir, fps=cfg.render.fps)

    # 1. Initialize Wrapper
    env_wrapper = EntropyGymWrapper(cfg)
    
    # 2. Setup Learner
    print("🧠 Initializing MAPPO Agent...")
    mappo_config = {
        "lr_actor": cfg.ppo.lr_actor,
        "lr_critic": cfg.ppo.lr_critic,
        "actor_updates_per_step": cfg.ppo.actor_updates,
        "critic_updates_per_step": cfg.ppo.critic_updates,
        "clip_eps": cfg.ppo.clip_eps,
        "agent": cfg.agent
    }
    trainer = OptimizedMAPPO(mappo_config)
    
    # Init Random Keys
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Init One Instance to get Dims
    dummy_state, dummy_obs = env_wrapper.reset(init_rng)
    obs_dim = dummy_obs.shape[-1]
    action_dim = env_wrapper.action_dim
    print(f"📊 Obs Dim: {obs_dim} | Action Dim: {action_dim} (Intent Based)")
    
    trainer.init_states(obs_dim, action_dim, cfg.agent.num_agents, init_rng)
    
    # Recurrent State Initialization
    init_actor_carries = jnp.zeros((cfg.sim.num_envs, cfg.agent.num_agents, trainer.hidden_dim))
    
    ckpt_manager = CheckpointManager(f"{cfg.render.output_dir}_checkpoints", max_to_keep=2)

    # 3. EXPERT SELECTION
    if cfg.intent.enabled:
        print("👴 Expert: IntentNavigator (Outputting Goal Intents)")
        expert_policy = IntentNavigator(num_agents=cfg.agent.num_agents)
    else:
        print("👴 Expert: DirectNavigator (Outputting Motor Commands)")
        expert_policy = DirectNavigator()

    # 4. Logic Definitions
    
    def get_expert_action_batch(state):
        # Wrapper for ExpertPolicy.act
        rng_exp = jax.random.PRNGKey(0) 
        expert_act = expert_policy.act(state, rng_exp) # [N, ExpertDim]
        
        # No padding needed if dims match (Comms Disabled)
        # if expert_act.shape[-1] < action_dim:
        #    padding = jnp.zeros((cfg.agent.num_agents, action_dim - expert_act.shape[-1]))
        #    expert_act = jnp.concatenate([expert_act, padding], axis=-1)
            
        return expert_act

    def rollout_scan_fn(carry, x):
        # Unpack carry: state, obs, actor_carry, rng, hog_weight
        state, obs, actor_carry, rng, hog_weight = carry
        rng, a_key, s_key = jax.random.split(rng, 3)
        
        # Inference (Recurrent)
        # Apply function returns: new_carry, mean, log_std
        new_actor_carry, mean, log_std = trainer.actor_state.apply_fn(
            trainer.actor_state.params, obs, actor_carry
        )
        
        noise = jax.random.normal(a_key, mean.shape) * jnp.exp(log_std)
        agent_action = mean + noise
        
        # HOG Blend
        expert_action = get_expert_action_batch(state)
        final_action = (1.0 - hog_weight) * agent_action + hog_weight * expert_action
        
        # Log Prob
        dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        log_probs = dist.log_prob(final_action)
        
        # Critic
        global_state = obs.reshape(1, -1)
        values = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
        
        # Step
        next_state, next_obs, rewards, dones, info = env_wrapper.step(state, final_action, s_key)
        
        # Mask hidden state for done agents
        dones_mask = dones[..., None].astype(jnp.float32)
        next_actor_carry = new_actor_carry * (1.0 - dones_mask)
        
        step_data = {
            'obs': obs,
            'actions': final_action,
            'rewards': rewards,
            'dones': dones.astype(jnp.float32),
            'values': values.flatten(),
            'log_probs': log_probs,
            'actor_states': actor_carry # Save for training
        }
        return (next_state, next_obs, next_actor_carry, rng, hog_weight), step_data

    @jax.jit
    def run_epoch_pipeline(rng, states, obss, actor_carries, hog_weight):
        def single_env_rollout(state, obs, ac_carry, r, h):
            carry_in = (state, obs, ac_carry, r, h)
            carry_out, trajectory = jax.lax.scan(rollout_scan_fn, carry_in, None, length=cfg.sim.max_steps)
            last_state, last_obs, last_ac_carry, last_rng, _ = carry_out
            
            global_state = last_obs.reshape(1, -1)
            last_val = trainer.critic_state.apply_fn(trainer.critic_state.params, global_state)
            last_val_expanded = last_val.reshape(1, -1)
            
            values_full = jnp.concatenate([trajectory['values'], last_val_expanded], axis=0)
            return (last_state, last_obs, last_ac_carry, last_rng), trajectory, values_full

        rngs = jax.random.split(rng, cfg.sim.num_envs)
        (last_states, last_obss, last_ac_carries, last_rngs), trajectories, all_values = jax.vmap(
            single_env_rollout, in_axes=(0, 0, 0, 0, None)
        )(states, obss, actor_carries, rngs, hog_weight)
        
        return (last_states, last_obss, last_ac_carries, last_rngs[0]), trajectories, all_values

    # 5. Training Loop
    try:
        reset_rngs = jax.random.split(rng, cfg.sim.num_envs)
        states, obss = jax.vmap(env_wrapper.reset)(reset_rngs)
        actor_carries = init_actor_carries
        
        print("⚡ JIT Compiling Massive Pipeline...")
        start_time = time.time()
        last_time = time.time()
        
        for epoch in range(1, cfg.total_epochs + 1):
            
            # Scheduler
            if epoch < cfg.hog.decay_epochs:
                progress = (epoch - 1) / cfg.hog.decay_epochs
                curr_hog = cfg.hog.start_weight - progress * (cfg.hog.start_weight - cfg.hog.end_weight)
            else:
                curr_hog = cfg.hog.end_weight
            curr_hog_jax = jnp.array(curr_hog, dtype=jnp.float32)

            # Rollout
            (states, obss, actor_carries, rng), trajs, all_values = run_epoch_pipeline(
                rng, states, obss, actor_carries, curr_hog_jax
            )
            
            # GAE Calculation (Per Env loop for safety)
            rew_np = np.array(trajs['rewards'])
            val_np = np.array(all_values)
            don_np = np.array(trajs['dones'])
            
            all_advantages = []
            for i in range(cfg.sim.num_envs):
                adv = trainer._calculate_gae(rew_np[i], val_np[i], don_np[i])
                all_advantages.append(adv)
            all_advantages = np.concatenate(all_advantages, axis=0) # [E*T, N]
            
            # --- UPDATE ---
            values_trimmed = val_np[:, :-1, :].reshape(-1, cfg.agent.num_agents)
            targets = all_advantages + values_trimmed
            
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
            all_advantages = jnp.array(all_advantages)
            targets = jnp.array(targets)
            
            # Critic
            obs_all = jnp.array(trajs['obs'])
            total_steps = obs_all.shape[0] * obs_all.shape[1] # E * T
            obs_flat = obs_all.reshape(total_steps, -1)
            
            critic_loss_val = 0.0
            for _ in range(cfg.ppo.critic_updates):
                trainer.critic_state, loss, _ = trainer._train_critic(
                    trainer.critic_state, obs_flat, targets.flatten(), cfg.agent.num_agents
                )
                critic_loss_val += loss
                
            # Actor
            actions_all = jnp.array(trajs['actions'])
            log_probs_all = jnp.array(trajs['log_probs'])
            carries_all = jnp.array(trajs['actor_states']) # [E, T, N, H]
            
            obs_flat_actor = obs_all.reshape(total_steps * cfg.agent.num_agents, -1)
            actions_flat = actions_all.reshape(total_steps * cfg.agent.num_agents, -1)
            log_probs_flat = log_probs_all.reshape(total_steps * cfg.agent.num_agents)
            adv_flat = all_advantages.reshape(total_steps * cfg.agent.num_agents)
            carries_flat = carries_all.reshape(total_steps * cfg.agent.num_agents, -1)
            
            actor_loss_val = 0.0
            for _ in range(cfg.ppo.actor_updates):
                trainer.actor_state, loss, _ = trainer._train_actor(
                    trainer.actor_state, 
                    obs_flat_actor, 
                    actions_flat, 
                    log_probs_flat, 
                    adv_flat,
                    carries_flat
                )
                actor_loss_val += loss
                
            # Logging
            mean_reward = float(np.mean(rew_np))
            if epoch % 10 == 0:
                curr_time = time.time()
                delta_time = curr_time - last_time
                last_time = curr_time
                steps_done = 10 * cfg.sim.max_steps * cfg.sim.num_envs
                fps = steps_done / (delta_time + 1e-6)
                print(f"Ep {epoch} | HOG: {curr_hog:.2f} | R: {mean_reward:.2f} | FPS: {fps:.0f} | L: {actor_loss_val:.2f}/{critic_loss_val:.2f}")

            # Render
            if epoch % 50 == 0 and cfg.render.enabled:
                print(f"🎥 Recording validation epoch {epoch}...")
                filename = f"epoch_{epoch:04d}_hybrid.gif"
                try:
                    path = recorder.record_episode(
                        env_wrapper, trainer.actor_state, trainer.actor_state.params, filename
                    )
                    print(f"✅ GIF saved: {path}")
                except Exception as e:
                    print(f"Recording failed: {e}")

    except KeyboardInterrupt:
        print("🛑 Stopped.")

if __name__ == "__main__":
    run_training()
