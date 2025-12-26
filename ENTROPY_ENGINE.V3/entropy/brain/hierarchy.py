
"""
Entropy Engine V3 - Dynamic Hierarchy
Handles squad formation and leader election.
"""
import jax
import jax.numpy as jnp
from entropy.core.world import WorldState

def assign_squads_proximity(positions: jnp.ndarray, squad_size: int) -> jnp.ndarray:
    """
    Groups agents into squads based on X-coordinate proximity.
    Simple and deterministic.
    
    Args:
        positions: [N, 2]
        squad_size: int (Agents per squad)
        
    Returns:
        squad_ids: [N] (int32)
    """
    N = positions.shape[0]
    num_squads = max(1, N // squad_size)
    
    # Sort by X coordinate
    sort_idx = jnp.argsort(positions[:, 0])
    
    # Create squad assignments in sorted order
    # Group 0: Agents 0..K
    sorted_squad_ids = jnp.arange(N) // squad_size
    
    # Cap at num_squads-1 (handle remainder)
    sorted_squad_ids = jnp.minimum(sorted_squad_ids, num_squads - 1)
    
    # squad_ids[sort_idx[i]] = sorted_squad_ids[i]
    squad_ids = jnp.zeros(N, dtype=jnp.int32)
    squad_ids = squad_ids.at[sort_idx].set(sorted_squad_ids)
    
    return squad_ids

def compute_squad_centroids(positions: jnp.ndarray, squad_ids: jnp.ndarray, max_squads: int) -> jnp.ndarray:
    """
    Calculates centroid (mean position) for each squad.
    """
    one_hot = jax.nn.one_hot(squad_ids, max_squads) # [N, S]
    
    sum_pos = jnp.einsum('ns,nd->sd', one_hot, positions)
    counts = jnp.sum(one_hot, axis=0) # [S]
    
    # Avoid div by zero
    counts = jnp.maximum(counts, 1.0)
    
    centroids = sum_pos / counts[:, None]
    return centroids

def elect_leaders(state: WorldState, squad_size: int, mode: str = "proximity") -> jnp.ndarray:
    """
    Elects one leader per squad.
    
    Args:
        state: WorldState
        squad_size: int
        mode: str ("proximity", "random")
        
    Returns:
        is_leader: [N] bool
    """
    N = state.agent_positions.shape[0]
    num_squads = max(1, N // squad_size)
    
    if mode == "proximity":
        # Dist to Own Centroid
        my_centroid = state.squad_centroids[state.agent_squad_ids]
        
        diff = state.agent_positions - my_centroid
        dist = jnp.linalg.norm(diff, axis=1) # [N]
        
        # We need, for each squad ID, the agent with MIN dist.
        # Use simple Scan over num_squads
        def leader_scan_body(carry, squad_idx):
            mask = state.agent_squad_ids == squad_idx
            # Set dists of non-members to infinity
            masked_dists = jnp.where(mask, dist, jnp.inf)
            leader_idx = jnp.argmin(masked_dists)
            return carry, leader_idx

        _, leader_indices = jax.lax.scan(leader_scan_body, None, jnp.arange(num_squads))
        
        # Set True at indices
        is_leader = jnp.zeros(N, dtype=bool)
        is_leader = is_leader.at[leader_indices].set(True)
        return is_leader
        
    elif mode == "random":
        pass
        
    return jnp.zeros(N, dtype=bool)
