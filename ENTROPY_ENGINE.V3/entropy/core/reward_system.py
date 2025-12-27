
import jax
import jax.numpy as jnp
from entropy.core.pure_structures import EnvState, EnvParams

def calculate_universal_reward(state: EnvState, action: jnp.ndarray, params: EnvParams, progress: float) -> jnp.ndarray:
    """
    Vypočítá reward pro všechny agenty najednou.
    
    Args:
        state: EnvState (s novou pozicí a uloženou prev_pos)
        action: Aktuálně aplikovaná akce [N, action_dim]
        params: EnvParams (agent_radius, target_radius, gamma...)
        progress: 0.0-1.0 (pro curriculum decay)
    """
    
    # === A. PHYSICS REGULARIZATION (Vždy aktivní) ===
    
    # 1. Action Smoothness (Tlumí škubání)
    action_diff = jnp.sum(jnp.square(action - state.prev_action), axis=-1)
    smoothness_penalty = action_diff * params.w_smooth

    # 2. Energy Cost
    energy_penalty = jnp.sum(jnp.square(action), axis=-1) * params.w_energy

    # 3. Existence Penalty (Motivace k rychlosti)
    time_penalty = params.w_living_penalty # e.g. -0.01

    # === B. COLLISION DETECTION (On-the-fly) ===
    # Access positions from world
    pos = state.world.agent_positions
    N = pos.shape[0]
    
    # Use higher precision for distance calc if needed, but float32 usually fine.
    diff = pos[:, None] - pos[None, :]
    dist_matrix = jnp.linalg.norm(diff, axis=-1)
    
    # Add large value to diagonal to ignore self-collision
    dist_matrix = dist_matrix + jnp.eye(N) * 1e6  
    
    min_dist = jnp.min(dist_matrix, axis=-1)
    # Collision threshold = 2 * radius (touching)
    collision_mask = min_dist < (params.agent_radius * 2.0)
    
    # Hard penalty.
    collision_penalty_val = collision_mask * params.w_collision

    # === C. TASK REWARDS (Switch) ===

    # ÚKOL 0: NAVIGACE
    def reward_nav(st: EnvState):
        current_pos = st.world.agent_positions
        # Current distance to target
        curr_dist = jnp.linalg.norm(current_pos - st.target, axis=-1)
        # Previous distance to target (from st.prev_pos which describes state BEFORE this step)
        prev_dist = jnp.linalg.norm(st.prev_pos - st.target, axis=-1)
        
        # PBRS Shaping
        shaping = (params.gamma * (-curr_dist) - (-prev_dist)) * params.w_dist 
        
        # Sparse Goal
        at_goal = curr_dist < params.target_radius
        sparse = at_goal * params.w_reach
        
        return shaping + sparse

    # ÚKOL 1: SEARCH (Exploration)
    def reward_search(st: EnvState):
        is_visible = st.target_visible.squeeze() 
        found = is_visible * 100.0
        movement = jnp.linalg.norm(st.world.agent_velocities, axis=-1) * 0.05
        return found + movement

    # ÚKOL 2: PUSH (Manipulace)
    def reward_push(st: EnvState):
        current_pos = st.world.agent_positions
        # We need broadcasted targets if box is single
        box_curr = jnp.linalg.norm(st.box_pos - st.target[0], axis=-1) 
        box_prev = jnp.linalg.norm(st.prev_box_pos - st.target[0], axis=-1)
        
        box_shaping = (params.gamma * (-box_curr) - (-box_prev)) * 2.0
        
        agent_to_box = jnp.linalg.norm(current_pos - st.box_pos, axis=-1) 
        agent_shaping = -agent_to_box * 0.05
        return box_shaping + agent_shaping

    # Výběr rewardu podle Task ID
    # We pass the WHOLE state to the lambda, but switch needs same signature?
    # Switch takes index and list of branches. Branches must take operands?
    # Correct usage: jax.lax.switch(index, branches, *operands)
    task_reward = jax.lax.switch(
        params.task_id,
        [reward_nav, reward_search, reward_push],
        state
    )

    # === D. CURRICULUM (Postupné odbourávání) ===
    # Decay from 1.0 to 0.2
    alpha = 1.0 - (0.8 * progress)
    
    # Auxiliary rewards (Collisions)
    # We want to penalized collisions, so we subtract valid penalty
    # If collision_penalty_val is 10.0, we want -10.0.
    aux_reward = alpha * (-collision_penalty_val)

    # === FINAL SUM ===
    total_reward = (
        task_reward + 
        aux_reward - 
        smoothness_penalty - 
        energy_penalty + 
        time_penalty # time_penalty is usually negative in params
    )
    
    return total_reward
