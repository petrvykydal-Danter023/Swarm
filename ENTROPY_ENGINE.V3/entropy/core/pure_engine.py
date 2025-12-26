"""
Entropy Engine V4 - Pure JAX Engine ⚡
The functional heart of the simulation.
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

from entropy.core.world import create_initial_state
from entropy.core.pure_structures import EnvState, EnvParams, EnvStep
from entropy.core.physics import physics_step
from entropy.core.sensors import compute_lidars

# Safety & Intent (We use adapters or direct calls)
from entropy.safety.reflexes import apply_collision_reflex
from entropy.safety.intent import process_intent
from entropy.safety.metrics import compute_safety_metrics, SafetyMetrics

class PureEntropyEngine:
    
    @staticmethod
    def reset(rng: jax.Array, params: EnvParams) -> EnvStep:
        """
        Pure functional reset.
        """
        # Create World
        # We need to pass static params. 
        # Since params are baked in JIT, we can just use them.
        world = create_initial_state(
            num_agents=params.num_agents,
            arena_size=(params.arena_width, params.arena_height),
            dt=params.dt,
            lidar_rays=params.lidar_rays,
            msg_dim=params.msg_dim
        )
        
        # Randomize Positions
        rng, pos_key, ang_key, goal_key = jax.random.split(rng, 4)
        
        # Simple Logic for now (can expand later)
        positions = jax.random.uniform(pos_key, (params.num_agents, 2)) * jnp.array([params.arena_width, params.arena_height])
        angles = jax.random.uniform(ang_key, (params.num_agents,)) * 2 * jnp.pi
        
        # Goals
        goals = jax.random.uniform(goal_key, (params.num_agents, 2)) * jnp.array([params.arena_width, params.arena_height])
        
        # Update World
        world = world.replace(
            agent_positions=positions,
            agent_angles=angles,
            goal_positions=goals,
            timestep=0
        )
        
        # Init State
        # Ensure metrics are JAX arrays for scan consistency
        metrics = SafetyMetrics(
            speed_reductions=jnp.array(0),
            hard_stops=jnp.array(0),
            repulsion_activations=jnp.array(0),
            messages_blocked=jnp.array(0),
            tokens_depleted=jnp.array(0),
            stalemates_detected=jnp.array(0),
            random_walks_triggered=jnp.array(0),
            boundary_pushes=jnp.array(0),
            safety_overrides=jnp.array(0)
        )
        
        state = EnvState(
            world=world,
            rng=rng,
            step_count=0,
            safety_metrics=metrics
        )
        
        # Initial Obs
        obs = PureEntropyEngine._compute_obs(world, params)
        
        return EnvStep(
            obs=obs,
            state=state,
            reward=jnp.zeros(params.num_agents),
            done=jnp.zeros(params.num_agents, dtype=bool),
            info=jnp.array(0.0) # Placeholder
        )

    @staticmethod
    def step(rng: jax.Array, state: EnvState, action: jnp.ndarray, params: EnvParams) -> EnvStep:
        """
        The massive JAX Kernel.
        """
        world = state.world
        
        # === 0. Unified Distance Matrix ===
        # Compute ONCE for use in Physics (if implemented), Safety, and Rewards
        pos = world.agent_positions
        dist_matrix = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        
        # === 1. Intent Translation ===
        # Create simple duck-typed config for intent
        class IntentCfg:
            enabled = True 
            # PID params
            pid_pos_kp = 2.0
            pid_pos_kd = 0.5
            pid_rot_kp = 5.0
            pid_rot_kd = 0.5
            max_linear_accel = 5.0
            max_angular_accel = 10.0
            
        # For strict correctness we should check params.safety_enabled logic for intent too if they are coupled,
        # but Intent is "Brain", Safety is "Lizard".
        intent_cfg = IntentCfg()
        
        # Raw -> Motor
        motor_actions = process_intent(world, action, intent_cfg)
        
        # === 2. Safety Layer (Reflexes) ===
        def safe_fn(w, a, d_mat):
            class SafetyCfg:
                safety_radius = params.agent_radius * 1.5 
                collision_check_radius = params.agent_radius * 3.0
                min_distance = params.agent_radius * 0.5
                repulsion_force = 0.5
                enable_repulsion = True
                max_speed = 100.0 
                repulsion_radius = params.agent_radius * 2.5
                emergency_brake_dist = params.agent_radius * 0.2 # Standard max speed
                
            # Pass pre-computed distance matrix
            return apply_collision_reflex(w, a, SafetyCfg(), dist_matrix=d_mat)
            
        def unsafe_fn(w, a, d_mat): return a
        
        final_actions = jax.lax.cond(
            params.safety_enabled,
            safe_fn, unsafe_fn,
            world, motor_actions, dist_matrix
        )
        
        # === 3. Physics ===
        # params are baked 
        next_world = physics_step(world, final_actions)
        
        # === 4. Rewards ===
        # Compute Rewards (Manually here to avoid dependency hell)
        # Dist Reward
        dists = jnp.linalg.norm(next_world.agent_positions - next_world.goal_positions, axis=1)
        prev_dists = jnp.linalg.norm(world.agent_positions - world.goal_positions, axis=1)
        r_dist = (prev_dists - dists) * params.w_dist
        
        # Reach Reward
        reached = dists < next_world.goal_radii
        r_reach = reached.astype(jnp.float32) * params.w_reach
        
        # Collision Reward
        # We can use the dist_matrix (from PREVIOUS step - wait, we need CURRENT step collision for reward)
        # So we technically need to re-compute or approximate.
        # Physics step moved agents.
        # Re-calc minimal dists for collision penalty.
        pos_new = next_world.agent_positions
        dist_matrix_new = jnp.linalg.norm(pos_new[:, None, :] - pos_new[None, :, :], axis=-1)
        # Mask self
        dist_matrix_new = dist_matrix_new + jnp.eye(params.num_agents) * 1e9
        min_dists = jnp.min(dist_matrix_new, axis=1)
        
        # Threshold for collision (2 * radius)
        collision_threshold = params.agent_radius * 2.0
        in_collision = min_dists < collision_threshold
        r_coll = in_collision.astype(jnp.float32) * params.w_collision
        
        # Living Penalty
        r_live = jnp.where(reached, 0.0, params.w_living_penalty)
        
        reward = r_dist + r_reach + r_coll + r_live
        
        # === 5. Dones & Resets ===
        step_count = state.step_count + 1
        done_timelimit = step_count >= params.max_steps
        
        # Auto-Reset Logic logic usually handled by `training/env_wrapper` using `auto_reset`.
        # In Pure Engine, we might want to return `done` and let caller handle reset, 
        # OR implementation `auto_reset` inside step (common in JAX RL).
        
        # Let's do explicit Auto-Reset for "Endless Episode" training paradigm.
        # "The One Kernel" usually implies we just keep rolling.
        # But for 'done' signal propagation:
        
        # If done, reset world but keep RNG flow?
        # Actually, standard PPO usually masks gradients at done.
        # Let's return done flag and next_state.
        # If massive parallelism, we usually just reset environment index that is done.
        
        # Complication: lax.scan needs consistent state shape.
        # So we MUST reset inside step if done.
        
        done = jnp.broadcast_to(done_timelimit, (params.num_agents,))
        # Or individual done? Shared step count -> All done at once.
        
        # === 6. Obs ===
        obs = PureEntropyEngine._compute_obs(next_world, params)
        
        # Update State
        # If done, we conceptually reset step_count.
        # Note: True Reset logic involves generating new positions.
        # Implementing `jax.lax.cond(done, reset_fn, step_fn)` is needed.
        # But strict `step` function just returns next. Reset wrapper handles it?
        # No, for `lax.scan` across episode, we don't reset usually (fixed horizon).
        # We just finish.
        
        new_state = EnvState(
            world=next_world,
            rng=rng,
            step_count=step_count,
            safety_metrics=state.safety_metrics # Update these later
        )
        
        return EnvStep(
            obs=obs,
            state=new_state,
            reward=reward,
            done=done,
            info=jnp.array(0.0)
        )

    @staticmethod
    def _compute_obs(world: Any, params: EnvParams) -> jnp.ndarray:
        """
        Compute observations (Lidars + Vectors).
        """
        # 1. Self State
        # [VelX, VelY, GoalRelX, GoalRelY]
        vel = world.agent_velocities
        goal_rel = world.goal_positions - world.agent_positions
        
        # 2. Lidars
        lidars = compute_lidars(world) # Optim: Takes world, uses internal logic.
        
        # 3. Comms (If enabled)
        # Placeholder for strict Pure Engine without heavy Comms logic yet.
        # User Tip: "Unified Dist Matrix".
        # compute_lidars might re-calc.
        # Ideally pass dists to compute_lidars.
        
        obs = jnp.concatenate([lidars, vel, goal_rel], axis=-1)
        return obs
