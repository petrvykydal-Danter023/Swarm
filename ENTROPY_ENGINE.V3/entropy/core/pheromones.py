
"""
Entropy Engine V3 - Virtual Pheromones
Handles placement, decay, and sensing of chemical markers (Stigmergy).
"""
import jax
import jax.numpy as jnp
from entropy.core.world import WorldState

def place_pheromone(
    state: WorldState, 
    position: jnp.ndarray, 
    message: jnp.ndarray, 
    ttl: int, 
    max_buffer: int
) -> WorldState:
    """
    Places a new pheromone in the circular buffer.
    
    Args:
        state: WorldState
        position: [2]
        message: [D]
        ttl: int (steps)
        max_buffer: int (constant)
        
    Returns:
        Updated WorldState
    """
    idx = state.pheromone_write_ptr
    
    new_pos = state.pheromone_positions.at[idx].set(position)
    new_msgs = state.pheromone_messages.at[idx].set(message)
    new_ttls = state.pheromone_ttls.at[idx].set(ttl)
    new_valid = state.pheromone_valid.at[idx].set(True)
    
    new_ptr = (idx + 1) % max_buffer
    
    return state.replace(
        pheromone_positions=new_pos,
        pheromone_messages=new_msgs,
        pheromone_ttls=new_ttls,
        pheromone_valid=new_valid,
        pheromone_write_ptr=new_ptr
    )

def decay_pheromones(state: WorldState) -> WorldState:
    """
    Decrements TTL for all valid pheromones.
    Invalidates those that reach 0.
    """
    ttls = state.pheromone_ttls
    valid = state.pheromone_valid
    
    # Decrement only valid ones (optional optimization, or just all)
    new_ttls = jnp.maximum(0, ttls - 1)
    
    # Invalidate if TTL hits 0
    still_valid = (new_ttls > 0) & valid
    
    return state.replace(
        pheromone_ttls=new_ttls,
        pheromone_valid=still_valid
    )

def read_nearby_pheromones(
    state: WorldState, 
    agent_positions: jnp.ndarray, 
    radius: float
) -> jnp.ndarray:
    """
    Calculates aggregated pheromone signal, weighted by distance.
    Signal = Sum(Message * (1 - dist/radius)) / Sum(Weights) ? 
    Or just Sum(Message * Weight).
    
    Returns:
        signal: [N, D]
    """
    # [N, 1, 2] vs [1, P, 2]
    ag_pos = agent_positions[:, None, :]
    ph_pos = state.pheromone_positions[None, :, :]
    
    dists = jnp.linalg.norm(ag_pos - ph_pos, axis=-1) # [N, P]
    
    # Filter by Radius and Validity
    in_range = dists < radius
    is_valid = state.pheromone_valid[None, :] # [1, P]
    mask = in_range & is_valid
    
    # Weight: Linear falloff (1.0 at dist=0, 0.0 at dist=radius)
    # Avoid div by zero if radius is tiny ?
    weight = jnp.maximum(0.0, 1.0 - (dists / radius))
    
    # Mask weights
    masked_weights = jnp.where(mask, weight, 0.0)
    
    # Weighted Sum of Messages
    # [N, P] @ [P, D] -> [N, D]
    signal = masked_weights @ state.pheromone_messages
    
    # Normalize? 
    # Plan says: "Weighted average" or just "Sum".
    # Implementation plan said: "Sensing: Agents perceive a weighted average..."
    # Usually sum is better for "intensity". Average dilutes if many weak signals?
    # Let's stick to Sum for now (Intensity acumulation). 
    # Or normalize by sum of weights?
    # "Stronger signal if more pheromones" -> Sum.
    
    return signal
