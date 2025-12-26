"""
Entropy Engine V3 - RL Environment Wrapper
Bridges the functional Core Physics with the PPO Training Loop.
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

from entropy.core.world import WorldState, create_initial_state
from entropy.core.physics import physics_step
from entropy.core.sensors import compute_lidars
from entropy.brain.communication import (
    TransformerContextDecoder, 
    compute_bandwidth_penalty, 
    MSG_DIM,
    Token
)

# Safety Layer imports (top-level for JIT optimization)
from entropy.safety.reflexes import apply_collision_reflex
from entropy.safety.geofence import apply_geofence
from entropy.safety.comm_limiter import apply_comm_limit, TokenBucketState
from entropy.safety.watchdog import apply_watchdog, WatchdogState
from entropy.safety.intent import process_intent
from entropy.safety.metrics import compute_safety_metrics

class EntropyGymWrapper:
    """
    JAX-compatible environment wrapper for PPO.
    Manages state transitions, observations, and rewards.
    """
    def __init__(self, cfg):
        # Support both old style object config and new ExperimentConfig
        if hasattr(cfg, 'sim'):
            # New Universal Config
            self.num_agents = cfg.agent.num_agents
            self.arena_size = (cfg.sim.arena_width, cfg.sim.arena_height)
            self.max_steps = cfg.sim.max_steps
            
            # Comms Config
            self.use_comms = cfg.agent.use_communication
            if self.use_comms:
                self.comm_cfg = cfg.agent.comm
                self.msg_dim = self.comm_cfg.msg_dim
                self.vocab_size = 0 # Not used in spatial
                
                # Dynamic Action: [Motor(2), Gate(1), Chan(1), Angle(1), Dist(1), Msg(D)]
                self.action_dim = 2 + 1 + 1 + 1 + 1 + self.msg_dim
                
                # Dynamic Obs: 
                # Lidar(R) + Vel(2) + Goal(2) + 
                # InboxMsgs(K*D) + InboxMeta(K*3) + InboxMask(K) + Pheromone?
                K = self.comm_cfg.max_neighbors
                inbox_size = K * (self.msg_dim + 3 + 1)
                
                phero_size = 0
                if hasattr(self.comm_cfg, 'pheromones_enabled') and self.comm_cfg.pheromones_enabled:
                    phero_size = self.comm_cfg.pheromone_dim
                
                hierarchy_size = 0
                if hasattr(self.comm_cfg, 'hierarchy_enabled') and self.comm_cfg.hierarchy_enabled:
                    # OneHot(S) + IsLeader(1) + RelCentroid(2)
                    max_squads = self.num_agents // self.comm_cfg.squad_size + 1 
                    # Use N for OneHot dimension for safety and static shape
                    self.max_squads = self.num_agents 
                    hierarchy_size = self.max_squads + 1 + 2
                
                self.obs_dim = cfg.agent.lidar_rays + 4 + inbox_size + phero_size + hierarchy_size
            else:
                self.comm_cfg = None
                self.action_dim = 2
                self.obs_dim = cfg.agent.lidar_rays + 4 # Lidar + Vel + Goal
                
            self.lidar_rays = cfg.agent.lidar_rays
            self.reward_cfg = cfg.reward
            self.shared_goal = cfg.reward.shared_goal
            self.safety_cfg = cfg.safety if hasattr(cfg, 'safety') else None
            self.intent_cfg = cfg.intent if hasattr(cfg, 'intent') else None
            
            # Action space adjustment for Intent Mode
            # Direct: [MotL, MotR, Gate, ...]
            # Intent: [Type, P1, P2, Gate, ...] -> +1 Dimension (Type)
            # Actually, MotL/MotR are replaced by Type/P1/P2.
            # Direct: 2 + Rest
            # Intent: 3 + Rest
            if self.intent_cfg and self.intent_cfg.enabled:
                self.action_dim += 1 # 2 params replace 2 motors, +1 for type

            
        else:
            # Legacy Config
            self.num_agents = cfg.env.num_agents
            self.arena_size = (cfg.env.arena_width, cfg.env.arena_height)
            self.max_steps = cfg.env.max_steps
            self.lidar_rays = 32
            
            self.use_comms = False
            self.action_dim = 2
            self.obs_dim = 32 + 4
            self.reward_cfg = None 
            self.shared_goal = False

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> Tuple[WorldState, jnp.ndarray]:
        """
        Resets the environment to initial state.
        Returns: (state, obs)
        """
        # Create initial state
        # In a real scenario, we would randomize positions using rng
        # For now, create standard empty state and potentially scatter agents
        # Prepare Pheromone Config
        p_dim = 8
        max_pheromones = 100
        if self.use_comms and self.comm_cfg and hasattr(self.comm_cfg, 'pheromones_enabled') and self.comm_cfg.pheromones_enabled:
             p_dim = self.comm_cfg.pheromone_dim
             max_pheromones = self.comm_cfg.max_pheromones

        max_squads = 64
        if hasattr(self, 'max_squads'):
             max_squads = self.max_squads
             
        state = create_initial_state(
            num_agents=self.num_agents,
            arena_size=self.arena_size,
            ctx_dim=self.context_dim if hasattr(self, 'context_dim') else 64,
            pheromone_dim=p_dim,
            max_pheromones=max_pheromones,
            max_squads=max_squads
        )
        
        # Randomize agent positions
        rng, pos_rng, angle_rng, goal_rng = jax.random.split(rng, 4)
        
        # Simple random placement
        positions = jax.random.uniform(pos_rng, (self.num_agents, 2)) * jnp.array(self.arena_size)
        angles = jax.random.uniform(angle_rng, (self.num_agents,)) * 2 * jnp.pi
        
        # Handle Shared vs Unique Goals
        if self.shared_goal:
            single_goal = jax.random.uniform(goal_rng, (1, 2)) * jnp.array(self.arena_size)
            goals = jnp.tile(single_goal, (self.num_agents, 1))
        else:
            goals = jax.random.uniform(goal_rng, (self.num_agents, 2)) * jnp.array(self.arena_size)

        # Initialize Safety State (Staggered Tokens)
        last_refill = jnp.zeros(self.num_agents)
        if hasattr(self, 'safety_cfg') and self.safety_cfg and self.safety_cfg.enabled:
             # Stagger over window
             rng, stagger_rng = jax.random.split(rng)
             offsets = jax.random.randint(stagger_rng, (self.num_agents,), 0, self.safety_cfg.msg_rate_window)
             last_refill = -offsets.astype(jnp.float32)
        
        state = state.replace(
            agent_positions=positions,
            agent_angles=angles,
            goal_positions=goals,
            safety_last_refill=last_refill
        )
        
        # Initial Obs
        # We need default values for inbox since no communication happened yet
        if self.use_comms:
             # Create empty inboxes
             K = self.comm_cfg.max_neighbors
             D = self.msg_dim
             N = self.num_agents
             
             inbox_msgs = jnp.zeros((N, K, D))
             inbox_meta = jnp.zeros((N, K, 3))
             inbox_mask = jnp.zeros((N, K))
             
             obs = self._get_obs(state, inbox_msgs, inbox_meta, inbox_mask)
        else:
             obs = self._get_obs(state)
             
        return state, obs

    def step(
        self, 
        state: WorldState, 
        actions: jnp.ndarray, 
        rng: jax.Array
    ) -> Tuple[WorldState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict]:
        
        # ========== 0. INTENT TRANSLATION (Phase 2) ==========
        # Convert High-Level Intent -> Low-Level Motor Actions
        # BEFORE Safety Layer (Safety checks the MOTOR outputs)
        motor_actions_raw = actions
        if hasattr(self, 'intent_cfg') and self.intent_cfg and self.intent_cfg.enabled:
             motor_actions_raw = process_intent(state, actions, self.intent_cfg)
        
        # ========== SAFETY LAYER ==========
        safe_actions = motor_actions_raw
        if hasattr(self, 'safety_cfg') and self.safety_cfg and self.safety_cfg.enabled:
            # 1. Override Check
            override_mask = None
            if self.safety_cfg.allow_ai_override and actions.shape[-1] > self.base_action_dim if hasattr(self, 'base_action_dim') else False:
                 pass

            # 2. Collision Reflex (Squad-Aware & Local)
            safe_actions = apply_collision_reflex(state, safe_actions, self.safety_cfg)
            
            # 3. Geo-Fence
            if self.safety_cfg.geofence_enabled:
                 safe_actions = apply_geofence(state, safe_actions, [], self.safety_cfg)

            # 4. Watchdog (Anti-Stalemate)
            if self.safety_cfg.watchdog_enabled:
                wd_state = WatchdogState(
                    position_old=state.safety_watchdog_pos_old,
                    steps_since_snapshot=state.safety_watchdog_steps,
                    random_walk_remaining=state.safety_watchdog_walk
                )
                rng, wd_rng = jax.random.split(rng)
                safe_actions, new_wd = apply_watchdog(state, safe_actions, wd_state, self.safety_cfg, wd_rng)
                
                # Update Watchdog State in WorldState
                state = state.replace(
                    safety_watchdog_pos_old=new_wd.position_old,
                    safety_watchdog_steps=new_wd.steps_since_snapshot,
                    safety_watchdog_walk=new_wd.random_walk_remaining
                )
            
            # 5. Comm Limiter (Staggered)
            # Only if comms are used (gate is index 2)
            if self.use_comms:
                tb_state = TokenBucketState(
                    tokens=state.safety_tokens,
                    last_refill=state.safety_last_refill
                )
                safe_actions, new_tb = apply_comm_limit(state, safe_actions, tb_state, self.safety_cfg)
                
                state = state.replace(
                    safety_tokens=new_tb.tokens,
                    safety_last_refill=new_tb.last_refill
                )
        
        # Calculate Safety Metrics (Telemetry)
        safety_stats = {}
        if hasattr(self, 'safety_cfg') and self.safety_cfg and self.safety_cfg.log_metrics:
            metrics = compute_safety_metrics(motor_actions_raw, safe_actions, state, self.safety_cfg)
            safety_stats = metrics.to_dict()
                
        # Use SAFE actions for physics and logic
        final_actions = safe_actions

        # 1. Physics Step (Only use motor actions)
        motor_actions = final_actions[:, :2]
        next_state = physics_step(state, motor_actions)
        
        # 2. Communication Step (Compute Inboxes)
        if self.use_comms:
            # Import router here to avoid circular dep at top level if needed
            from entropy.core.comms_router import route_messages
            
            inbox_msgs, inbox_meta, inbox_mask = route_messages(
                state.agent_positions,
                final_actions,
                self.comm_cfg,
                rng,
                squad_ids=state.agent_squad_ids,
                is_leader=state.agent_is_leader
            )
            
            # --- PHEROMONES ---
            if hasattr(self.comm_cfg, 'pheromones_enabled') and self.comm_cfg.pheromones_enabled:
                from entropy.core.pheromones import decay_pheromones, place_pheromone
                
                # A. Decay
                next_state = decay_pheromones(next_state)
                
                # B. Place Logic (Pheromones)
                # ... (Keep existing logic if needed, or simply assume it runs as part of loop)
                # Re-inserting Pheromone loop logic here for completeness as per previous state
                place_threshold = 2.0
                channel_logits = final_actions[:, 3]
                is_placing = channel_logits > place_threshold
                
                addr_angle = final_actions[:, 4] * jnp.pi
                addr_dist = jax.nn.softplus(final_actions[:, 5])
                
                target_vec_x = jnp.cos(addr_angle) * addr_dist
                target_vec_y = jnp.sin(addr_angle) * addr_dist
                target_points = state.agent_positions + jnp.stack([target_vec_x, target_vec_y], axis=1)
                
                comm_msgs = final_actions[:, 6:] 
                p_dim = self.comm_cfg.pheromone_dim
                p_msgs = comm_msgs[:, :p_dim]
                
                def place_body(i, s):
                    should_place = is_placing[i]
                    def true_branch(st):
                        return place_pheromone(
                             st, 
                             target_points[i], 
                             p_msgs[i], 
                             self.comm_cfg.pheromone_ttl, 
                             self.comm_cfg.max_pheromones
                        )
                    def false_branch(st):
                        return st
                    return jax.lax.cond(should_place, true_branch, false_branch, s)
                
                next_state = jax.lax.fori_loop(0, self.num_agents, place_body, next_state)

            # --- DYNAMIC HIERARCHY ---
            if hasattr(self.comm_cfg, 'hierarchy_enabled') and self.comm_cfg.hierarchy_enabled:
                from entropy.brain.hierarchy import assign_squads_proximity, compute_squad_centroids, elect_leaders
                
                # Re-elect logic: e.g. every 50 steps
                # Or if timestep is 0 (initial)
                re_elect_interval = 50
                is_election_step = (state.timestep % re_elect_interval) == 0
                
                def update_hierarchy(s):
                    # 1. Assign Squads
                    squad_ids = assign_squads_proximity(s.agent_positions, self.comm_cfg.squad_size)
                    
                    # 2. Compute Centroids
                    # We need max_squads constant. Derive from num_agents or config?
                    # Using hardcoded max or self.max_squads (need to init)
                    # Let's assume max_squads = num_agents (safe upper bound)
                    safe_max_squads = self.num_agents 
                    centroids = compute_squad_centroids(s.agent_positions, squad_ids, safe_max_squads)
                    
                    s = s.replace(agent_squad_ids=squad_ids, squad_centroids=centroids)
                    
                    # 3. Elect Leaders
                    is_leader = elect_leaders(s, self.comm_cfg.squad_size, self.comm_cfg.leader_election_mode)
                    s = s.replace(agent_is_leader=is_leader)
                    return s

                next_state = jax.lax.cond(is_election_step, update_hierarchy, lambda s: s, next_state)
            
        else:
            inbox_msgs, inbox_meta, inbox_mask = None, None, None

        # 3. Compute Observations
        obs = self._get_obs(next_state, inbox_msgs, inbox_meta, inbox_mask)
        
        # 4. Compute Rewards
        rewards = self._compute_rewards(next_state, final_actions)
        
        # 5. Check Termination
        timeout = next_state.timestep >= self.max_steps
        dones = next_state.goal_reached | timeout
        
        info = {
            "goal_reached": next_state.goal_reached.astype(jnp.float32),
            "max_steps": jnp.full((self.num_agents,), timeout, dtype=jnp.float32),
            # Add safety stats (broadcasted or single val?)
            # Metrics are scalar sums (int). Put them in info.
            **{k: jnp.array(v, dtype=jnp.float32) for k, v in safety_stats.items()}
        }
        
        return next_state, obs, rewards, dones, info

    def _get_obs(self, state: WorldState, inbox_msgs=None, inbox_meta=None, inbox_mask=None) -> jnp.ndarray:
        """Constructs observation vectors."""
        # 1. Lidars [N, 32]
        lidars = compute_lidars(state)
        
        # 2. Velocity & RelGoal
        velocities = state.agent_velocities / 100.0
        rel_goals = (state.goal_positions - state.agent_positions) / 1000.0
        
        # 3. Communication Inbox (Flattened)
        comm_obs = jnp.zeros((state.num_agents, 0))
        if self.use_comms and inbox_msgs is not None:
             # Flatten: [N, K, D] -> [N, K*D]
             N = state.agent_positions.shape[0]
             flat_msgs = inbox_msgs.reshape(N, -1)
             flat_meta = inbox_meta.reshape(N, -1)
             flat_mask = inbox_mask.reshape(N, -1)
             
             comm_obs = jnp.concatenate([flat_msgs, flat_meta, flat_mask], axis=-1)
             
        # 4. Pheromone Signal
        phero_signal = jnp.zeros((state.num_agents, 0))
        if self.use_comms and self.comm_cfg and hasattr(self.comm_cfg, 'pheromones_enabled') and self.comm_cfg.pheromones_enabled:
             from entropy.core.pheromones import read_nearby_pheromones
             phero_signal = read_nearby_pheromones(
                 state, 
                 state.agent_positions, 
                 self.comm_cfg.pheromone_radius
             )
        
        # 5. Hierarchy Info
        hierarchy_obs = jnp.zeros((state.num_agents, 0))
        if self.use_comms and self.comm_cfg and hasattr(self.comm_cfg, 'hierarchy_enabled') and self.comm_cfg.hierarchy_enabled:
            # Squad OneHot
            # Ensure max_squads matches init
            max_s_dim = self.num_agents # Must match init calculation
            squad_onehot = jax.nn.one_hot(state.agent_squad_ids, max_s_dim)
            
            # Is Leader
            is_leader = state.agent_is_leader.astype(jnp.float32)[:, None]
            
            # Rel Centroid
            my_centroids = state.squad_centroids[state.agent_squad_ids]
            rel_centroid = (my_centroids - state.agent_positions) / 1000.0
            
            hierarchy_obs = jnp.concatenate([squad_onehot, is_leader, rel_centroid], axis=-1)

        # Concat: [Core, Hierarchy, Phero, Inbox] 
        # Making sure RecurrentActor handles this.
        # Core is usually fixed (Lidar+Vel+Goal). 
        # Hierarchy is "Core-like" (intrinsic state).
        return jnp.concatenate([lidars, velocities, rel_goals, hierarchy_obs, phero_signal, comm_obs], axis=-1)

    def _compute_rewards(self, state: WorldState, actions: jnp.ndarray) -> jnp.ndarray:
        """Computes dense reward signal."""
        # Get weights from config or defaults
        w_dist = self.reward_cfg.w_dist if self.reward_cfg else 1.0
        w_reach = self.reward_cfg.w_reach if self.reward_cfg else 10.0
        w_energy = self.reward_cfg.w_energy if self.reward_cfg else -0.01
        w_living = self.reward_cfg.w_living_penalty if self.reward_cfg and hasattr(self.reward_cfg, 'w_living_penalty') else -0.001
        
        # 1. Progress to Goal
        dist = jnp.linalg.norm(state.agent_positions - state.goal_positions, axis=1)
        r_dist = w_dist * (-dist / 1000.0) 
        
        # 2. Goal Reached Bonus
        r_reached = state.goal_reached.astype(jnp.float32) * w_reach
        
        # 3. Energy Penalty
        # Motor actions are always first 2 dims
        motor_actions = actions[:, :2]
        r_energy = jnp.mean(jnp.abs(motor_actions), axis=1) * w_energy
        
        # 4. Living Penalty (Time Pressure)
        # Penalize every step where agent is NOT at goal
        not_at_goal = ~state.goal_reached
        r_living = not_at_goal.astype(jnp.float32) * w_living
        
        # 5. Bandwidth Penalty (Communication)
        if self.use_comms and state.agent_messages.shape[-1] >= 32:
             tokens = jnp.argmax(state.agent_messages[:, :32], axis=1)
             r_bandwidth = compute_bandwidth_penalty(tokens, Token.SILENCE, penalty=-0.01)
        else:
             r_bandwidth = 0.0
             
        return r_dist + r_reached + r_energy + r_living + r_bandwidth
