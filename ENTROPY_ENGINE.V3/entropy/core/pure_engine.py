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

# ... imports
from entropy.core.reward_system import calculate_universal_reward

class PureEntropyEngine:
    
    @staticmethod
    def reset(rng: jax.Array, params: EnvParams) -> EnvStep:
        """
        Pure functional reset.
        """
        # Create World
        world = create_initial_state(
            num_agents=params.num_agents,
            arena_size=(params.arena_width, params.arena_height),
            dt=params.dt,
            lidar_rays=params.lidar_rays,
            msg_dim=params.msg_dim
        )
        
        # Randomize Positions
        rng, pos_key, ang_key, goal_key = jax.random.split(rng, 4)
        
        positions = jax.random.uniform(pos_key, (params.num_agents, 2)) * jnp.array([params.arena_width, params.arena_height])
        angles = jax.random.uniform(ang_key, (params.num_agents,)) * 2 * jnp.pi
        
        # Goals (Used for Nav task)
        goals = jax.random.uniform(goal_key, (params.num_agents, 2)) * jnp.array([params.arena_width, params.arena_height])
        
        # Update World
        world = world.replace(
            agent_positions=positions,
            agent_angles=angles,
            goal_positions=goals,
            timestep=0
        )
        
        # Init State
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
        
        # New State Fields Initialization
        state = EnvState(
            world=world,
            rng=rng,
            step_count=0,
            safety_metrics=metrics,
            
            # History
            prev_pos=positions,           # Initial prev_pos = current pos (no movement)
            prev_action=jnp.zeros((params.num_agents, params.action_dim)), # No previous action
            
            # Universal Task Fields
            target=goals,                 # Map world goals to universal target field [N, 2]
            box_pos=jnp.zeros((1, 2)),    # Placeholder
            prev_box_pos=jnp.zeros((1, 2)), # Placeholder
            target_visible=jnp.ones((params.num_agents, 1)) # Assume visible for now
        )
        
        # Initial Obs
        obs = PureEntropyEngine._compute_obs(world, params)
        
        return EnvStep(
            obs=obs,
            state=state,
            reward=jnp.zeros(params.num_agents),
            done=jnp.zeros(params.num_agents, dtype=bool),
            info=jnp.array(0.0) 
        )

    @staticmethod
    def step(rng: jax.Array, state: EnvState, action: jnp.ndarray, params: EnvParams, progress: float = 0.0) -> EnvStep:
        """
        The massive JAX Kernel.
        """
        world = state.world
        
        # === 0. History Backup for PBRS ===
        # We need to save the state BEFORE physics update
        prev_pos_backup = world.agent_positions
        prev_box_pos_backup = state.box_pos # If we had boxes moving logic
        
        # === 1. Unified Distance Matrix ===
        pos = world.agent_positions
        dist_matrix = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        
        # === 2. Intent & Safety ===
        # Intent Translation
        class IntentCfg:
            enabled = True 
            pid_pos_kp = 2.0
            pid_pos_kd = 0.5
            pid_rot_kp = 5.0
            pid_rot_kd = 0.5
            max_linear_accel = 5.0
            max_angular_accel = 10.0
        
        motor_actions = process_intent(world, action, IntentCfg())
        
        # Safety Layer
        def safe_fn(w, a, d_mat):
            class SafetyCfg:
                safety_radius = params.agent_radius * 1.5 
                collision_check_radius = params.agent_radius * 3.0
                min_distance = params.agent_radius * 0.5
                repulsion_force = 0.5
                enable_repulsion = True
                max_speed = 100.0 
                repulsion_radius = params.agent_radius * 2.5
                emergency_brake_dist = params.agent_radius * 0.2
                
            return apply_collision_reflex(w, a, SafetyCfg(), dist_matrix=d_mat)
            
        def unsafe_fn(w, a, d_mat): return a
        
        final_actions = jax.lax.cond(
            params.safety_enabled,
            safe_fn, unsafe_fn,
            world, motor_actions, dist_matrix
        )
        
        # === 3. Physics Update ===
        next_world = physics_step(world, final_actions)
        
        # === 4. Update State Context ===
        # We must create a temporary state object that holds the NEW world
        # but logically we are preparing to calculate reward based on transition from OLD to NEW.
        # But `calculate_universal_reward` expects `state` to have `pos` (NEW) and `prev_pos` (OLD).
        
        # Create Next State foundation
        step_count = state.step_count + 1
        
        # Construct the state that represents "After Physics Steps"
        # Crucially, we store the OLD positions into `prev_pos`.
        # And we store the CURRENT action into `prev_action` so next step sees it as previous.
        next_state = EnvState(
            world=next_world,
            rng=rng,
            step_count=step_count,
            safety_metrics=state.safety_metrics, # Should update metrics if safety layer returned them
            
            # History Update
            prev_pos=prev_pos_backup,     # The position BEFORE this step
            prev_action=action,           # The action we JUST took
            
            # Task Fields (Carry over or update if dynamic)
            target=state.target,
            box_pos=state.box_pos, # Update if box moved
            prev_box_pos=prev_box_pos_backup,
            target_visible=state.target_visible
        )
        
        # === 5. Universal Reward Calculation ===
        # Passing `next_state` which contains:
        # - pos: New positions (from next_world)
        # - prev_pos: Old positions
        reward = calculate_universal_reward(next_state, action, params, progress)
        
        # === 6. Dones ===
        done_timelimit = step_count >= params.max_steps
        done = jnp.broadcast_to(done_timelimit, (params.num_agents,))
        
        # === 7. Obs ===
        obs = PureEntropyEngine._compute_obs(next_world, params)
        
        return EnvStep(
            obs=obs,
            state=next_state,
            reward=reward,
            done=done,
            info=jnp.array(0.0)
        )

    @staticmethod
    def _compute_obs(world: Any, params: EnvParams) -> jnp.ndarray:
        """
        Compute observations (Lidars + Vectors).
        """
        # [VelX, VelY, GoalRelX, GoalRelY]
        vel = world.agent_velocities
        goal_rel = world.goal_positions - world.agent_positions
        
        lidars = compute_lidars(world)
        
        obs = jnp.concatenate([lidars, vel, goal_rel], axis=-1)
        return obs
