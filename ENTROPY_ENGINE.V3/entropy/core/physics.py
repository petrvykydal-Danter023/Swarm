import jax
import jax.numpy as jnp
from functools import partial
from .world import WorldState

@partial(jax.jit, static_argnums=(2,))
def physics_step(state: WorldState, actions: jnp.ndarray, wheel_base: float = 20.0, max_speed: float = 100.0) -> WorldState:
    """
    Jeden krok fyzikální simulace.
    
    Args:
        state: Aktuální WorldState
        actions: [N, 2] - (left_motor, right_motor) in range [-1, 1]
    
    Returns:
        Nový WorldState
    """
    dt = state.dt
    
    # === 1. PARSING ACTIONS (Differential Drive) ===
    # Actions are in range [-1, 1]
    left_motor = jnp.clip(actions[:, 0], -1.0, 1.0)
    right_motor = jnp.clip(actions[:, 1], -1.0, 1.0)
    
    v_left = left_motor * max_speed
    v_right = right_motor * max_speed
    
    # Kinematics
    linear_v = (v_left + v_right) / 2.0
    angular_v = (v_right - v_left) / wheel_base
    
    # === 2. INTEGRATION (Euler) for AGENTS ===
    new_angles = state.agent_angles + angular_v * dt
    # Normalize angles to [-pi, pi] for cleanliness (optional but good)
    new_angles = jnp.arctan2(jnp.sin(new_angles), jnp.cos(new_angles))
    
    vx = linear_v * jnp.cos(new_angles)
    vy = linear_v * jnp.sin(new_angles)
    new_velocities = jnp.stack([vx, vy], axis=1)
    
    new_positions = state.agent_positions + new_velocities * dt
    
    # === 3. ARENA BOUNDARY COLLISION (Agents) ===
    # Simple clamping
    w, h = state.arena_size
    radii = state.agent_radii
    
    # Clamp x
    new_positions = new_positions.at[:, 0].set(
        jnp.clip(new_positions[:, 0], radii, w - radii)
    )
    # Clamp y
    new_positions = new_positions.at[:, 1].set(
        jnp.clip(new_positions[:, 1], radii, h - radii)
    )
    
    # === 4. UPDATE CRATES (Physics & Interactions) ===
    # Check if we have crates
    if state.crate_positions.shape[0] > 0:
        # Resolve collisions between agents and crates
        # Agents push crates, crates push back (simplified: agents push crates, positions updated)
        
        # Identify "pushing" contact: Agent colliding with crate
        # We need to compute total force on each crate from all agents
        
        # 4.1 Compute Forces on Crates from Agents
        # Vector from Agent -> Crate
        # Force = overlap * stiffness (spring model) or just impulse
        
        # Simplification for grid/continuous hybrid:
        # 1. Resolve overlaps ( Agents cannot overlap Crates )
        # 2. If overlap exists, Agent imparts force on Crate
        
        # Let's use the explicit blueprint logic:
        # "Total force = sum(forces_on_crate)"
        # Force from agent = Agent's "Push Force" if in contact and moving towards crate
        
        # Naive implementation for V3 phase 1:
        # Treat crates as dynamic circles for collision resolution
        crate_radii = jnp.full((state.crate_positions.shape[0],), 15.0) # Assume 15.0 radius
        
        # Resolve Agent <-> Crate Collisions
        # Updates both positions to separate them
        new_positions, new_crate_positions = _resolve_agent_crate_collisions(
            new_positions, state.agent_radii,
            state.crate_positions, crate_radii
        )
        
        # 4.2 Crate Friction Logic (The "Heavy Object" logic)
        # Calculate displacement (velocity)
        crate_displacement = new_crate_positions - state.crate_positions
        crate_velocity = crate_displacement / dt
        
        # Apply Friction Threshold
        # If velocity is too small (force small), set to zero (Static Friction)
        # We need per-crate params. For now assume uniform or from state if we added arrays (we added crate_masses)
        
        # Simply: if displacement < threshold, reset position.
        # But real friction works on FORCE. Here we have position-based dynamics (PBD).
        # We can simulate friction by damping velocity.
        
        # Damping
        crate_velocity = crate_velocity * 0.9 # Kinetic friction
        new_crate_positions = state.crate_positions + crate_velocity * dt
        
        # Boundary Crate
        new_crate_positions = new_crate_positions.at[:, 0].set(jnp.clip(new_crate_positions[:, 0], 15.0, w - 15.0))
        new_crate_positions = new_crate_positions.at[:, 1].set(jnp.clip(new_crate_positions[:, 1], 15.0, h - 15.0))
        
    else:
        new_crate_positions = state.crate_positions
        crate_velocity = state.crate_velocities

    # === 5. AGENT COLLISIONS (Agent <-> Agent) ===
    new_positions = _resolve_agent_collisions(new_positions, state.agent_radii)
    
    # === 6. GOAL CHECK ===
    dist_to_goal = jnp.linalg.norm(new_positions - state.goal_positions, axis=1)
    reached = dist_to_goal < state.goal_radii
    
    # === 7. ZONE TRIGGERS ===
    # Check if agents are in zones
    # We won't change state here (e.g. energy) as `physics_step` mainly handles transform
    # But we could return a mask "in_zone" for reward system
    
    # === 8. UPDATE STATE ===
    return WorldState(
        agent_positions=new_positions,
        agent_velocities=new_velocities,
        agent_angles=new_angles,
        agent_ang_velocities=angular_v,
        agent_radii=state.agent_radii,
        lidar_readings=state.lidar_readings,
        
        # Interactions
        agent_carrying=state.agent_carrying,
        
        # Crates
        crate_positions=new_crate_positions,
        crate_velocities=crate_velocity,
        crate_masses=state.crate_masses,
        crate_values=state.crate_values,
        
        # Zones
        zone_types=state.zone_types,
        zone_bounds=state.zone_bounds,
        
        agent_messages=state.agent_messages,
        agent_contexts=state.agent_contexts,
        
        # Hierarchy
        agent_squad_ids=state.agent_squad_ids,
        agent_is_leader=state.agent_is_leader,
        squad_centroids=state.squad_centroids,
        
        goal_positions=state.goal_positions,
        goal_radii=state.goal_radii,
        goal_reached=reached,
        
        wall_segments=state.wall_segments,
        
        # Pheromones (Pass-through)
        pheromone_positions=state.pheromone_positions,
        pheromone_messages=state.pheromone_messages,
        pheromone_ttls=state.pheromone_ttls,
        pheromone_valid=state.pheromone_valid,
        pheromone_write_ptr=state.pheromone_write_ptr,
        
        # Legacy fields to keep compatibility if any (can remove if verified)
        object_positions=state.object_positions,
        object_types=state.object_types,
        object_carried_by=state.object_carried_by,
        
        timestep=state.timestep + 1,
        dt=state.dt,
        arena_size=state.arena_size
    )

def _resolve_agent_collisions(positions: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """Vectorized inter-agent collision resolution."""
    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    radii_sum = radii[:, None] + radii[None, :]
    overlap = radii_sum - dist
    mask = (overlap > 0) & (jnp.eye(n) == 0)
    safe_dist = jnp.where(dist > 1e-6, dist, 1e-6)
    direction = diff / safe_dist[:, :, None]
    repulsion = direction * (overlap[:, :, None] / 2.0) * mask[:, :, None]
    total_repulsion = jnp.sum(repulsion, axis=1)
    return positions + total_repulsion

def _resolve_agent_crate_collisions(
    agent_pos: jnp.ndarray, 
    agent_radii: jnp.ndarray,
    crate_pos: jnp.ndarray,
    crate_radii: jnp.ndarray
):
    """
    Resolve collisions between N agents and M crates.
    Returns updated (agent_pos, crate_pos).
    """
    # Simply push them apart based on overlap
    # Broadcasting: [N, 1, 2] - [1, M, 2] = [N, M, 2] difference
    diff = agent_pos[:, None, :] - crate_pos[None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1) # [N, M]
    
    # Combined radii
    radii_sum = agent_radii[:, None] + crate_radii[None, :] # [N, M]
    
    overlap = radii_sum - dist # [N, M]
    mask = overlap > 0
    
    # Directions
    safe_dist = jnp.where(dist > 1e-6, dist, 1e-6)
    direction = diff / safe_dist[:, :, None] # [N, M, 2] (From Crate TO Agent)
    
    # Repulsion half-half? 
    # Or based on mass? Agents usually heavier or crates?
    # Let's say Agents are dynamic kinematic, Crates are dynamic.
    # We put 100% update on Crate to simulate "Pushing" if agent is moving towards it?
    # For stability, we separate them 50/50
    
    correction = direction * (overlap[:, :, None] / 2.0) * mask[:, :, None]
    
    # Agent correction (sum over crates)
    agent_correction = jnp.sum(correction, axis=1) 
    
    # Crate correction (sum over agents) -> Note sign flip!
    # Direction is Crate->Agent. So Agent moves +correction. Crate moves -correction.
    crate_correction = jnp.sum(-correction, axis=0)
    
    return agent_pos + agent_correction, crate_pos + crate_correction
