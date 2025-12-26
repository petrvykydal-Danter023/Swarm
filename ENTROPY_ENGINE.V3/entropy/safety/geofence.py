import jax.numpy as jnp
from flax import struct
from entropy.core.world import WorldState
from entropy.config import SafetyConfig

@struct.dataclass
class GeoFenceZone:
    """Definuje zakázanou/povolenou zónu."""
    center: jnp.ndarray      # [2] střed zóny
    radius: float            # poloměr
    is_forbidden: bool       # True = zakázaná, False = povinná (musí zůstat uvnitř)
    
def apply_geofence(
    state: WorldState,
    actions: jnp.ndarray,
    zones: list,  # List of GeoFenceZone (static)
    config: SafetyConfig
) -> jnp.ndarray:
    """
    Aplikuje virtuální sílu směrem od/do zón.
    
    - Forbidden zones: Push agents OUT
    - Required zones (arena): Push agents IN
    """
    N = state.agent_positions.shape[0]
    total_force = jnp.zeros((N, 2))
    
    # Arena bounds (required zone - must stay inside)
    arena_w, arena_h = state.arena_size
    pos = state.agent_positions
    
    # Distance to each wall
    dist_left = pos[:, 0]
    dist_right = arena_w - pos[:, 0]
    dist_bottom = pos[:, 1]
    dist_top = arena_h - pos[:, 1]
    
    # Push force when close to wall (exponential falloff)
    push_threshold = config.geofence_push_distance
    push_strength = config.geofence_push_force
    
    # Left wall -> push right (+X)
    left_force = jnp.where(
        dist_left < push_threshold,
        push_strength * (1.0 - dist_left / push_threshold),
        0.0
    )
    # Right wall -> push left (-X)
    right_force = jnp.where(
        dist_right < push_threshold,
        -push_strength * (1.0 - dist_right / push_threshold),
        0.0
    )
    # Bottom wall -> push up (+Y)
    bottom_force = jnp.where(
        dist_bottom < push_threshold,
        push_strength * (1.0 - dist_bottom / push_threshold),
        0.0
    )
    # Top wall -> push down (-Y)
    top_force = jnp.where(
        dist_top < push_threshold,
        -push_strength * (1.0 - dist_top / push_threshold),
        0.0
    )
    
    total_force = total_force.at[:, 0].add(left_force + right_force)
    total_force = total_force.at[:, 1].add(bottom_force + top_force)
    
    # Apply force to motor commands
    # Use .at[].add() for JAX compatibility
    safe_actions = actions.at[:, 0].add(total_force[:, 0])
    safe_actions = safe_actions.at[:, 1].add(total_force[:, 1])
    
    return safe_actions
