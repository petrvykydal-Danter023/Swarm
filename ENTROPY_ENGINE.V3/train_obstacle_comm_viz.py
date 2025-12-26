"""
Entropy Engine V3 - Obstacle Avoidance & Communication Training
=================================================================
Trénink zaměřený na:
1. Vyhýbání se překážkám (Safety Layer + Repulsion)
2. Komunikaci mezi agenty (Spatial + Pheromones + Hierarchy)
3. Plnou vizualizaci pro debug (zprávy, trajektorie, senzory)

Hardware Target: AMD Ryzen 5 5600X + RTX 3060 (12GB)
Duration Target: ~20 minut tréninku
"""
import os
import sys

# =============================================================================
# 🎮 GPU VYNUCENÍ - MUSÍ BÝT PŘED IMPORTEM JAX!
# =============================================================================
# Toto musí být nastaveno PŘED importem JAX
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # Vynutí GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Nezabírat veškerou VRAM
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"  # Max 85% VRAM

import jax
import jax.numpy as jnp
import numpy as np
import time
import logging
import distrax

# === GPU KONTROLA ===
def check_gpu():
    """Zkontroluje dostupnost GPU a ukončí se pokud není."""
    devices = jax.devices()
    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
    
    print("\n" + "=" * 60)
    print("🎮 GPU KONTROLA")
    print("=" * 60)
    print(f"Nalezená zařízení: {devices}")
    
    if not gpu_devices:
        print("\n❌ CHYBA: GPU NENÍ DOSTUPNÉ!")
        print("=" * 60)
        print("JAX běží na CPU. Pro GPU potřebujete:")
        print()
        print("1. Odinstalujte stávající JAX:")
        print("   pip uninstall jax jaxlib -y")
        print()
        print("2. Nainstalujte JAX s CUDA (pro RTX 3060 / CUDA 12):")
        print("   pip install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print()
        print("3. Nebo pro CUDA 11:")
        print("   pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        print()
        print("Zkontrolujte verzi CUDA: nvcc --version")
        print("=" * 60)
        # FALLBACK TO CPU
        print("⚠️  Vynucuji běh na CPU (Pure JAX mode)...")
        return jax.devices()[0]
    
    print(f"✅ GPU nalezeno: {gpu_devices[0]}")
    
    # Test GPU paměti
    try:
        test_arr = jnp.ones((1000, 1000))
        _ = test_arr @ test_arr  # Force computation
        print(f"✅ GPU compute test: OK")
    except Exception as e:
        print(f"⚠️  GPU test selhal: {e}")
    
    print("=" * 60 + "\n")
    return gpu_devices[0]

# Spustit kontrolu hned při importu
GPU_DEVICE = check_gpu()

sys.path.insert(0, os.getcwd())

from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.mappo import OptimizedMAPPO
from entropy.visuals.recorder import Recorder
from entropy.training.checkpoint import CheckpointManager
from entropy.config import (
    ExperimentConfig, SimConfig, AgentConfig, CommConfig,
    RewardConfig, PPOConfig, HogConfig, RenderConfig,
    SafetyConfig, IntentConfig
)
from entropy.training.hand_of_god.expert import IntentNavigator, DirectNavigator
from entropy.render.server import RenderServer
from entropy.render.schema import RenderFrame

# === CONSTANTS ===
# Přibližný odhad epoch pro 20 minut:
# Na RTX 3060 s 32 envy a 20 agenty očekáváme ~200 FPS (full pipeline)
# 20 min = 1200 sekund
# 200 FPS * max_steps(300) / 300 = ~200 epoch/s... 
# Realisticky: ~30-50 epoch/s (s úplnou vizualizací a logováním)
# 1200s * 40 = ~48000 epoch... ale s rendererem bude pomalejší
# Konzervativní: 5000-8000 epoch pro 20min s vizualizací

TARGET_TRAINING_MINUTES = 20
ESTIMATED_EPOCHS_PER_MINUTE = 300  # Konzervativní odhad s vizualizací


def create_obstacle_comm_config():
    """
    Konfigurace optimalizovaná pro vyhýbání se překážkám a komunikaci.
    """
    cfg = ExperimentConfig()
    cfg.name = "obstacle_avoidance_communication"
    
    # === 1. SIMULATION ===
    # Balancovaná paralelizace pro RTX 3060 (12GB VRAM)
    cfg.sim = SimConfig(
        arena_width=800.0,
        arena_height=600.0,
        max_steps=300,       # Delší epizody pro lepší komunikaci
        num_envs=32,         # 32 prostředí - sweet spot pro 12GB VRAM
        dt=0.1
    )
    
    # Dynamický výpočet epoch
    cfg.total_epochs = int(TARGET_TRAINING_MINUTES * ESTIMATED_EPOCHS_PER_MINUTE)
    print(f"📊 Cílový čas: {TARGET_TRAINING_MINUTES} min → {cfg.total_epochs} epoch")
    
    # === 2. AGENTS ===
    cfg.agent = AgentConfig(
        num_agents=16,           # 16 agentů - dobré pro komunikaci
        lidar_rays=32,           # 32 paprsků pro detailní detekci překážek
        lidar_range=250.0,       # Delší dosah pro včasnou reakci
        use_communication=True,  # ✅ KOMUNIKACE ZAPNUTA
        vocab_size=10,           # 10 tokenů pro rozmanitost
        context_dim=64,
        
        # Pokročilá komunikace
        comm=CommConfig(
            mode="spatial",      # Prostorová komunikace
            msg_dim=16,          # 16D zprávy
            max_neighbors=5,     # Top-5 sousedů
            
            # Dual-Channel Attention
            dual_attention=True,
            local_radius=200.0,  # Lokální taktické zprávy
            local_heads=2,
            global_heads=2,
            
            # Event-Triggered Communication
            surprise_gating=True,
            surprise_threshold=0.05,  # Nižší práh = více komunikace
            info_gain_reward=0.15,    # Odměna za užitečné zprávy
            
            # Virtual Pheromones (Stigmergy)
            pheromones_enabled=True,
            pheromone_radius=60.0,
            pheromone_ttl=150,        # Delší životnost
            max_pheromones=120,
            pheromone_dim=8,
            
            # Dynamic Hierarchy (Squads & Leaders)
            hierarchy_enabled=True,
            squad_size=4,             # 16/4 = 4 squady
            leader_election_mode="proximity",
            leader_broadcast_only=False,  # Všichni mohou mluvit
            
            # Gating & Penalty
            gating_threshold=0.4,     # Nižší práh = více mluvení
            spam_penalty=-0.005,      # Mírnější penalizace
            comm_warmup_epochs=500    # Kratší warmup
        )
    )
    
    # === 3. SAFETY (Vyhýbání překážkám!) ===
    cfg.safety = SafetyConfig(
        enabled=True,  # ✅ SAFETY ZAPNUTO
        
        # Collision Avoidance - KLÍČOVÉ
        safety_radius=50.0,           # Větší safety zóna
        min_distance=15.0,            # Hard stop vzdálenost
        collision_check_radius=80.0,
        
        # Repulsion (Liquid Swarm) - AKTIVNÍ
        enable_repulsion=True,
        repulsion_radius=35.0,
        repulsion_force=0.8,          # Silnější odpuzování
        
        # Speed Limits
        max_speed=8.0,                # Pomalejší = bezpečnější
        emergency_brake_dist=8.0,
        
        # Communication Limits
        msg_rate_limit=8,             # Více zpráv povoleno
        msg_rate_window=15,
        
        # Energy (vypnuto pro tento experiment)
        energy_enabled=False,
        
        # Watchdog (Anti-Stalemate)
        watchdog_enabled=True,
        stalemate_window=80,
        stalemate_min_distance=8.0,
        stalemate_random_duration=25,
        stalemate_random_speed=0.6,
        
        # Geo-Fence
        geofence_enabled=True,
        geofence_push_distance=40.0,
        geofence_push_force=1.2,
        
        # Override
        allow_ai_override=False,      # AI nemůže obejít safety
        
        # Metrics - DŮLEŽITÉ PRO DEBUG
        log_metrics=True,
        log_interval=100              # Log každých 100 kroků (ne moc často)
    )
    
    # === 4. INTENT ===
    cfg.intent = IntentConfig(
        enabled=True,
        pid_pos_kp=2.5,
        pid_pos_kd=0.6,
        pid_rot_kp=6.0,
        pid_rot_kd=0.6,
        max_linear_accel=4.0,
        max_angular_accel=8.0
    )
    
    # === 5. REWARDS ===
    cfg.reward = RewardConfig(
        w_dist=0.8,              # Vzdálenost k cíli
        w_reach=15.0,            # Bonus za dosažení cíle
        w_energy=-0.005,         # Mírná energetická penalizace
        w_living_penalty=-0.002, # Time pressure
        shared_goal=False        # Každý má svůj cíl
    )
    
    # === 6. PPO ===
    cfg.ppo = PPOConfig(
        lr_actor=2e-4,           # Stabilnější learning rate
        lr_critic=8e-4,
        actor_updates=3,
        critic_updates=2,
        clip_eps=0.2,
        gamma=0.99
    )
    
    # === 7. HAND OF GOD ===
    cfg.hog = HogConfig(
        enabled=True,
        start_weight=0.9,        # 90% expert na začátku
        end_weight=0.1,          # 10% expert na konci
        decay_epochs=int(cfg.total_epochs * 0.6),  # 60% času = decay
        adaptive=False,
        target_reward=0.0
    )
    
    # === 8. RENDER ===
    cfg.render = RenderConfig(
        enabled=True,
        render_every=100,        # GIF každých 100 epoch
        fps=25,
        output_dir="outputs/obstacle_comm_viz"
    )
    
    return cfg


def run_training():
    """Hlavní tréninková smyčka s live vizualizací."""
    
    cfg = create_obstacle_comm_config()
    
    print("=" * 60)
    print("🚀 ENTROPY ENGINE V3 - Obstacle Avoidance & Communication")
    print("=" * 60)
    print(f"🎯 Cíl: Vyhýbání překážkám + Komunikace")
    print(f"⏱️  Odhadovaný čas: {TARGET_TRAINING_MINUTES} minut")
    print(f"🌍 Prostředí: {cfg.sim.num_envs}")
    print(f"🤖 Agenti/env: {cfg.agent.num_agents}")
    print(f"🤖 Celkem agentů: {cfg.sim.num_envs * cfg.agent.num_agents}")
    print(f"📡 Komunikace: {cfg.agent.use_communication}")
    print(f"🛡️  Safety Layer: {cfg.safety.enabled}")
    print(f"🧠 Intent System: {cfg.intent.enabled}")
    print(f"🐜 Pheromones: {cfg.agent.comm.pheromones_enabled}")
    print(f"👑 Hierarchy: {cfg.agent.comm.hierarchy_enabled}")
    print("=" * 60)
    
    # === SETUP ===
    
    # 1. Recorder pro GIF
    recorder = None
    if cfg.render.enabled:
        recorder = Recorder(output_dir=cfg.render.output_dir, fps=cfg.render.fps)
    
    # 2. Live Render Server (volitelné - pro real-time viewer)
    render_server = None
    try:
        render_server = RenderServer(port=5555)
        print("📺 Live Viewer dostupný na port 5555")
        print("   Spusťte: python -m entropy.render.viewer")
    except Exception as e:
        print(f"⚠️  Live viewer není k dispozici: {e}")
    
    # 3. Environment Wrapper
    env_wrapper = EntropyGymWrapper(cfg)
    
    # 4. MAPPO Agent
    print("🧠 Inicializace MAPPO agenta...")
    mappo_config = {
        "lr_actor": cfg.ppo.lr_actor,
        "lr_critic": cfg.ppo.lr_critic,
        "actor_updates_per_step": cfg.ppo.actor_updates,
        "critic_updates_per_step": cfg.ppo.critic_updates,
        "clip_eps": cfg.ppo.clip_eps,
        "agent": cfg.agent
    }
    trainer = OptimizedMAPPO(mappo_config)
    
    # 5. Keys & Inits
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_state, dummy_obs = env_wrapper.reset(init_rng)
    obs_dim = dummy_obs.shape[-1]
    action_dim = env_wrapper.action_dim
    print(f"📊 Obs Dim: {obs_dim} | Action Dim: {action_dim}")
    
    trainer.init_states(obs_dim, action_dim, cfg.agent.num_agents, init_rng)
    
    # Recurrent states
    init_actor_carries = jnp.zeros((cfg.sim.num_envs, cfg.agent.num_agents, trainer.hidden_dim))
    
    # 6. Checkpoint Manager
    ckpt_manager = CheckpointManager(f"{cfg.render.output_dir}_checkpoints", max_to_keep=3)
    
    # 7. Expert Navigator
    if cfg.intent.enabled:
        print("👴 Expert: IntentNavigator")
        expert_policy = IntentNavigator(num_agents=cfg.agent.num_agents)
    else:
        print("👴 Expert: DirectNavigator")
        expert_policy = DirectNavigator()
    
    # === FUNCTIONS ===
    
    def get_expert_action_batch(state):
        rng_exp = jax.random.PRNGKey(0)
        expert_act = expert_policy.act(state, rng_exp)  # [N, 3] intent only
        
        # Padding pro komunikaci - expert mlčí (nulové komunikační akce)
        # Action Space: [Intent(3)] + [Comm(msg_dim + gating + spatial)] = 23
        if expert_act.shape[-1] < action_dim:
            comm_padding = jnp.zeros((cfg.agent.num_agents, action_dim - expert_act.shape[-1]))
            expert_act = jnp.concatenate([expert_act, comm_padding], axis=-1)
        
        return expert_act
    
    def publish_debug_frame(state, rewards, fps_val, epoch):
        """Publikuje frame do live vieweru."""
        if render_server is None:
            return
            
        try:
            frame = RenderFrame(
                timestep=int(state.timestep),
                agent_positions=np.array(state.agent_positions),
                agent_angles=np.array(state.agent_angles),
                agent_colors=None,
                agent_messages=np.array(state.agent_messages),
                agent_radii=np.full(cfg.agent.num_agents, 12.0),
                goal_positions=np.array(state.goal_positions),
                object_positions=np.zeros((0, 2)),
                object_types=np.zeros((0,)),
                wall_segments=np.array([
                    [[0, 0], [cfg.sim.arena_width, 0]],
                    [[cfg.sim.arena_width, 0], [cfg.sim.arena_width, cfg.sim.arena_height]],
                    [[cfg.sim.arena_width, cfg.sim.arena_height], [0, cfg.sim.arena_height]],
                    [[0, cfg.sim.arena_height], [0, 0]]
                ]),
                rewards=np.array(rewards) if rewards is not None else None,
                fps=fps_val,
                # Pheromones
                pheromone_positions=np.array(state.pheromone_positions) if hasattr(state, 'pheromone_positions') else None,
                pheromone_ttls=np.array(state.pheromone_ttls) if hasattr(state, 'pheromone_ttls') else None,
                pheromone_valid=np.array(state.pheromone_valid) if hasattr(state, 'pheromone_valid') else None,
                pheromone_max_ttl=float(cfg.agent.comm.pheromone_ttl),
                pheromone_radius=float(cfg.agent.comm.pheromone_radius),
                # Hierarchy
                agent_squad_ids=np.array(state.agent_squad_ids) if hasattr(state, 'agent_squad_ids') else None,
                agent_is_leader=np.array(state.agent_is_leader) if hasattr(state, 'agent_is_leader') else None,
                # Safety
                safety_enabled=cfg.safety.enabled,
                safety_radius=cfg.safety.safety_radius,
                safety_repulsion_radius=cfg.safety.repulsion_radius,
                # Intent
                intent_enabled=cfg.intent.enabled
            )
            render_server.publish_frame(frame)
        except Exception as e:
            pass  # Silent fail - nechceme zpomalovat trénink
    
    def rollout_scan_fn(carry, x):
        state, obs, actor_carry, rng, hog_weight = carry
        rng, a_key, s_key = jax.random.split(rng, 3)
        
        # Inference
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
        
        # Mask for done agents
        dones_mask = dones[..., None].astype(jnp.float32)
        next_actor_carry = new_actor_carry * (1.0 - dones_mask)
        
        step_data = {
            'obs': obs,
            'actions': final_action,
            'rewards': rewards,
            'dones': dones.astype(jnp.float32),
            'values': values.flatten(),
            'log_probs': log_probs,
            'actor_states': actor_carry
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
    
    # === TRAINING LOOP ===
    
    try:
        reset_rngs = jax.random.split(rng, cfg.sim.num_envs)
        states, obss = jax.vmap(env_wrapper.reset)(reset_rngs)
        actor_carries = init_actor_carries
        
        print("\n⚡ JIT Kompilace pipeline... (může trvat 30-60s)")
        start_time = time.time()
        last_time = time.time()
        
        # Statistiky
        total_goals_reached = 0
        total_collisions_avoided = 0
        total_messages_sent = 0
        
        for epoch in range(1, cfg.total_epochs + 1):
            
            # HOG Schedule
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
            
            # GAE Calculation
            rew_np = np.array(trajs['rewards'])
            val_np = np.array(all_values)
            don_np = np.array(trajs['dones'])
            
            all_advantages = []
            for i in range(cfg.sim.num_envs):
                adv = trainer._calculate_gae(rew_np[i], val_np[i], don_np[i])
                all_advantages.append(adv)
            all_advantages = np.concatenate(all_advantages, axis=0)
            
            # Update
            values_trimmed = val_np[:, :-1, :].reshape(-1, cfg.agent.num_agents)
            targets = all_advantages + values_trimmed
            
            all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
            all_advantages = jnp.array(all_advantages)
            targets = jnp.array(targets)
            
            # Critic Update
            obs_all = jnp.array(trajs['obs'])
            total_steps = obs_all.shape[0] * obs_all.shape[1]
            obs_flat = obs_all.reshape(total_steps, -1)
            
            critic_loss_val = 0.0
            for _ in range(cfg.ppo.critic_updates):
                trainer.critic_state, loss, _ = trainer._train_critic(
                    trainer.critic_state, obs_flat, targets.flatten(), cfg.agent.num_agents
                )
                critic_loss_val += loss
            
            # Actor Update
            actions_all = jnp.array(trajs['actions'])
            log_probs_all = jnp.array(trajs['log_probs'])
            carries_all = jnp.array(trajs['actor_states'])
            
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
            
            # === LOGGING ===
            mean_reward = float(np.mean(rew_np))
            
            # Publish to live viewer (každý 10. epoch)
            if epoch % 10 == 0 and render_server:
                curr_time = time.time()
                delta = curr_time - last_time if curr_time > last_time else 1
                fps = (10 * cfg.sim.max_steps * cfg.sim.num_envs) / delta
                
                # Publikuj stav prvního prostředí
                first_state = jax.tree_util.tree_map(lambda x: x[0], states)
                first_rewards = rew_np[0, -1]  # Poslední reward prvního envu
                publish_debug_frame(first_state, first_rewards, fps, epoch)
            
            # Console logging
            if epoch % 25 == 0:
                curr_time = time.time()
                delta_time = curr_time - last_time
                last_time = curr_time
                steps_done = 25 * cfg.sim.max_steps * cfg.sim.num_envs
                fps = steps_done / (delta_time + 1e-6)
                
                elapsed_min = (curr_time - start_time) / 60
                eta_min = (cfg.total_epochs - epoch) / (epoch / (elapsed_min + 1e-6)) if epoch > 0 else 0
                
                # Komunikační statistiky
                msgs_np = np.array(states.agent_messages)
                msgs_active = np.sum(np.abs(msgs_np) > 0.1)
                
                print(f"Ep {epoch:5d}/{cfg.total_epochs} | "
                      f"HOG: {curr_hog:.2f} | "
                      f"R: {mean_reward:+.3f} | "
                      f"FPS: {fps:6.0f} | "
                      f"Msgs: {msgs_active:3d} | "
                      f"Loss: {actor_loss_val:.3f}/{critic_loss_val:.3f} | "
                      f"ETA: {eta_min:.1f}min")
            
            # === RENDERING ===
            if epoch % cfg.render.render_every == 0 and cfg.render.enabled:
                print(f"\n🎥 Nahrávám GIF epoch {epoch}...")
                filename = f"epoch_{epoch:05d}_obstacle_comm.gif"
                try:
                    path = recorder.record_episode(
                        env_wrapper, trainer.actor_state, trainer.actor_state.params, filename
                    )
                    print(f"✅ Uloženo: {path}\n")
                except Exception as e:
                    print(f"⚠️  Nahrávání selhalo: {e}\n")
            
            # === CHECKPOINTING ===
            if epoch % 500 == 0:
                try:
                    ckpt_manager.save(
                        epoch,
                        trainer.actor_state.params,
                        trainer.critic_state.params,
                        {"hog_weight": curr_hog, "epoch": epoch}
                    )
                    print(f"💾 Checkpoint uložen (epoch {epoch})")
                except Exception as e:
                    print(f"⚠️  Checkpoint selhal: {e}")
        
        # === FINAL ===
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("🏁 TRÉNINK DOKONČEN!")
        print("=" * 60)
        print(f"⏱️  Celkový čas: {total_time/60:.1f} minut")
        print(f"📊 Epoch: {cfg.total_epochs}")
        print(f"📈 Finální HOG: {curr_hog:.2f}")
        print(f"🎯 Finální Reward: {mean_reward:+.3f}")
        
        # Finální GIF
        if cfg.render.enabled:
            print("\n🎬 Generuji finální GIF...")
            try:
                path = recorder.record_episode(
                    env_wrapper, trainer.actor_state, trainer.actor_state.params,
                    "FINAL_obstacle_comm.gif", max_steps=500
                )
                print(f"✅ Finální GIF: {path}")
            except Exception as e:
                print(f"⚠️  Finální GIF selhal: {e}")
        
        # Finální checkpoint
        try:
            ckpt_manager.save(
                cfg.total_epochs,
                trainer.actor_state.params,
                trainer.critic_state.params,
                {"hog_weight": curr_hog, "epoch": cfg.total_epochs, "final": True}
            )
            print(f"💾 Finální checkpoint uložen")
        except Exception as e:
            print(f"⚠️  Finální checkpoint selhal: {e}")
    
    except KeyboardInterrupt:
        print("\n🛑 Trénink přerušen uživatelem.")
        
        # Uložení při přerušení
        try:
            ckpt_manager.save(
                epoch,
                trainer.actor_state.params,
                trainer.critic_state.params,
                {"hog_weight": curr_hog, "epoch": epoch, "interrupted": True}
            )
            print(f"💾 Checkpoint při přerušení uložen (epoch {epoch})")
        except:
            pass
    
    finally:
        if render_server:
            render_server.close()


if __name__ == "__main__":
    print("\n" + "🔥" * 30)
    print("ENTROPY ENGINE V3 - Obstacle Avoidance & Communication Training")
    print("🔥" * 30 + "\n")
    
    # Kontrola GPU
    try:
        devices = jax.devices()
        print(f"📟 Dostupná zařízení: {devices}")
        for d in devices:
            print(f"   - {d}")
    except:
        print("⚠️  Nelze zjistit zařízení")
    
    print()
    run_training()
