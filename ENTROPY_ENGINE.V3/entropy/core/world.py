from flax import struct
from typing import Tuple
import jax.numpy as jnp

@struct.dataclass
class WorldState:
    """
    Simulační stav světa v Structure-of-Arrays (SoA) formátu.
    Plně kompatibilní s JAX transformacemi (JIT, VMAP).
    Immutable - každá změna vrací novou instanci.
    """
    # === AGENTI [N, ...] ===
    # Agents (SoA)
    agent_positions: jnp.ndarray    # [N, 2]
    agent_angles: jnp.ndarray       # [N]
    agent_velocities: jnp.ndarray   # [N, 2]
    lidar_readings: jnp.ndarray     # [N, 32]
    agent_ang_velocities: jnp.ndarray # [N] - rad/s
    agent_radii: jnp.ndarray          # [N] - meters
    
    # Interactions
    agent_carrying: jnp.ndarray     # [N] -1 if empty, >=0 crate_id
    
    # Objects (Crates)
    crate_positions: jnp.ndarray    # [M, 2]
    crate_velocities: jnp.ndarray   # [M, 2]
    crate_masses: jnp.ndarray       # [M]
    crate_values: jnp.ndarray       # [M]
    
    # Zones
    zone_types: jnp.ndarray         # [Z]
    zone_bounds: jnp.ndarray        # [Z, 4] distance (0-1)
    
    # === KOMUNIKACE [N, ...] ===
    agent_messages: jnp.ndarray       # [N, MSG_DIM] - current broadcast
    agent_contexts: jnp.ndarray       # [N, CTX_DIM] - decoded context from others
    
    # === CÍLE [N, ...] ===
    goal_positions: jnp.ndarray       # [N, 2]
    goal_radii: jnp.ndarray           # [N]
    goal_reached: jnp.ndarray         # [N] - bool/int mask
    
    # === PROSTŘEDÍ ===
    wall_segments: jnp.ndarray        # [W, 4] - (x1, y1, x2, y2)
    
    # === OBJEKTY [O, ...] ===
    object_positions: jnp.ndarray     # [O, 2] - x, y
    object_types: jnp.ndarray         # [O] - int enum (RESOURCE=0, OBSTACLE=1, ...)
    object_carried_by: jnp.ndarray    # [O] - agent ID or -1
    
    # === INVENTORY [N, ...] ===
    agent_carrying: jnp.ndarray       # [N] - object ID or -1
    
    # === METADATA ===
    timestep: int
    dt: float  # Simulation step size
    arena_size: Tuple[float, float]   # (width, height)
    
    # === PHEROMONES [P, ...] ===
    pheromone_positions: jnp.ndarray  # [P, 2]
    pheromone_messages: jnp.ndarray   # [P, D]
    pheromone_ttls: jnp.ndarray       # [P]
    pheromone_valid: jnp.ndarray      # [P] bool
    pheromone_write_ptr: int          # Scalar
    
    # === HIERARCHY [N, ...] ===
    agent_squad_ids: jnp.ndarray      # [N] int
    agent_is_leader: jnp.ndarray      # [N] bool
    squad_centroids: jnp.ndarray      # [MaxSquads, 2]

    # === SAFETY: Token Bucket ===
    safety_tokens: jnp.ndarray        # [N]
    safety_last_refill: jnp.ndarray   # [N]

    # === SAFETY: Watchdog ===
    safety_watchdog_pos_old: jnp.ndarray  # [N, 2]
    safety_watchdog_steps: int            # scalar
    safety_watchdog_walk: jnp.ndarray     # [N]

    @property
    def num_agents(self) -> int:
        return self.agent_positions.shape[0]

def create_initial_state(
    num_agents: int, 
    num_objects: int = 0,
    num_crates: int = 0,
    num_zones: int = 0,
    arena_size: Tuple[float, float] = (800.0, 600.0),
    dt: float = 0.1,
    msg_dim: int = 36,
    ctx_dim: int = 64,
    lidar_rays: int = 32,
    max_pheromones: int = 100,
    pheromone_dim: int = 8,
    max_squads: int = 64 # Reasonable default max
) -> WorldState:
    """Vytvoří inicializovaný prázdný stav."""
    return WorldState(
        agent_positions=jnp.zeros((num_agents, 2)),
        agent_velocities=jnp.zeros((num_agents, 2)),
        agent_angles=jnp.zeros(num_agents),
        agent_ang_velocities=jnp.zeros(num_agents),
        agent_radii=jnp.full(num_agents, 10.0),
        
        lidar_readings=jnp.zeros((num_agents, lidar_rays)),
        
        agent_messages=jnp.zeros((num_agents, msg_dim)),
        agent_contexts=jnp.zeros((num_agents, ctx_dim)),
        
        # Hierarchy
        agent_squad_ids=jnp.zeros(num_agents, dtype=jnp.int32),
        agent_is_leader=jnp.zeros(num_agents, dtype=bool),
        squad_centroids=jnp.zeros((max_squads, 2)),
        
        goal_positions=jnp.zeros((num_agents, 2)),
        goal_radii=jnp.full(num_agents, 15.0),
        goal_reached=jnp.zeros(num_agents, dtype=bool),
        
        wall_segments=jnp.zeros((0, 4)),
        
        # Crates
        crate_positions=jnp.zeros((num_crates, 2)),
        crate_velocities=jnp.zeros((num_crates, 2)),
        crate_masses=jnp.ones(num_crates),
        crate_values=jnp.ones(num_crates),
        
        # Zones
        zone_types=jnp.zeros(num_zones, dtype=jnp.int32),
        zone_bounds=jnp.zeros((num_zones, 4)),
        
        object_positions=jnp.zeros((num_objects, 2)),
        object_types=jnp.zeros(num_objects, dtype=jnp.int32),
        object_carried_by=jnp.full(num_objects, -1, dtype=jnp.int32),
        
        agent_carrying=jnp.full(num_agents, -1, dtype=jnp.int32),
        
        # Pheromones
        pheromone_positions=jnp.zeros((max_pheromones, 2)),
        pheromone_messages=jnp.zeros((max_pheromones, pheromone_dim)),
        pheromone_ttls=jnp.zeros(max_pheromones, dtype=jnp.int32),
        pheromone_valid=jnp.zeros(max_pheromones, dtype=bool),
        pheromone_write_ptr=0,
        
        # Safety
        safety_tokens=jnp.full(num_agents, 5.0),
        safety_last_refill=jnp.zeros(num_agents), # Will be staggered in env_wrapper or init
        safety_watchdog_pos_old=jnp.zeros((num_agents, 2)),
        safety_watchdog_steps=0,
        safety_watchdog_walk=jnp.zeros(num_agents, dtype=jnp.int32),
        
        timestep=0,
        dt=dt,
        arena_size=arena_size
    )

