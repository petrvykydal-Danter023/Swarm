
"""
Entropy Engine V3 - Communication Router
Handles Spatial Addressing, Gating, and Inbox Management.
"""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any

# Constants
# -----------------------------------------------------------------------------
CHANNEL_DIRECT = 0
CHANNEL_BROADCAST = 1

def route_messages(
    agent_positions: jnp.ndarray,  # [N, 2]
    actions: jnp.ndarray,          # [N, ActionDim]
    config: Any,                   # CommConfig
    rng: jax.Array,
    squad_ids: jnp.ndarray = None, # [N] (Optional for Hierarchy)
    is_leader: jnp.ndarray = None  # [N] (Optional for Hierarchy)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Processes actions to fill agent Inboxes.
    Supports Spatial and Hierarchical Routing.
    """
    N = agent_positions.shape[0]
    K = config.max_neighbors
    D = config.msg_dim
    
    # 1. Parse Actions
    gate_logits = actions[:, 2]
    channel_logits = actions[:, 3]
    addr_angle = actions[:, 4] * jnp.pi 
    addr_dist = jax.nn.softplus(actions[:, 5])
    messages = actions[:, 6:]
    
    # 2. Gating Check
    is_speaking = jax.nn.sigmoid(gate_logits) > config.gating_threshold
    
    # 3. Hierarchy Logic (Filtering)
    hierarchy_enabled = hasattr(config, 'hierarchy_enabled') and config.hierarchy_enabled and squad_ids is not None
    
    if hierarchy_enabled:
        # Channel 0: Squad Direct (Intra-Squad)
        # Channel 1: Broadcast (Inter-Squad / Global)
        # Channel 2: Pheromone (Handled elsewhere) or Leader-to-Leader?
        
        is_broadcast = channel_logits > 0.5
        
        # Rule: Only Leader can Broadcast (if restricted)
        if config.leader_broadcast_only:
            # If not leader, mask broadcast -> force to squad direct?
            # Or silence it?
            # Let's silence it effectively by treating as "Invalid Broadcast" 
            # OR better: Non-leaders trying to broadcast just fallback to Intra-Squad.
            # "Leader converts broadcast to squad-direct" logic from plan.
            can_broadcast = is_leader
            is_broadcast = is_broadcast & can_broadcast
            
        # 4. Routing Matrix Construction
        # We need masks:
        # Same Squad Mask [N, N]
        same_squad = squad_ids[:, None] == squad_ids[None, :]
        
        # If Direct (Not Broadcast): Must match Squad ID?
        # Plan says: "Channel 0 = Direct: Message goes to squad members only"
        # So we penalize cross-squad edges for Direct messages.
        
        # We handle this by adding penalty to distance score where (not is_broadcast) AND (not same_squad).
        
    else:
        is_broadcast = channel_logits > 0
        same_squad = jnp.ones((N, N), dtype=bool) # Everyone is same "group" effectively
        
    
    # 5. Calculate Target Points
    target_vec_x = jnp.cos(addr_angle) * addr_dist
    target_vec_y = jnp.sin(addr_angle) * addr_dist
    target_points = agent_positions + jnp.stack([target_vec_x, target_vec_y], axis=1)
    
    # 6. Distance Matrices
    # Send_Targets: [N, 1, 2]
    # Recv_Pos:     [1, N, 2]
    targets_exp = target_points[:, None, :]
    pos_exp = agent_positions[None, :, :]
    
    dists_target_recv = jnp.linalg.norm(targets_exp - pos_exp, axis=-1) 
    
    eye_mask = jnp.eye(N, dtype=jnp.bool_)
    infinity = 1e9
    dists_target_recv = jnp.where(eye_mask, infinity, dists_target_recv)
    
    dists_sender_recv = jnp.linalg.norm(agent_positions[:, None, :] - pos_exp, axis=-1)
    dists_sender_recv = jnp.where(eye_mask, infinity, dists_sender_recv)
    
    # 7. Apply Broadcast/Direct Logic + Hierarchy
    # Effective Distance
    raw_score = jnp.where(is_broadcast[:, None], dists_sender_recv, dists_target_recv)
    
    if hierarchy_enabled:
        # Penalize cross-squad for Direct
        # If Sender is NOT broadcasting, Receiver MUST be in same squad.
        # Penalty = Infinity if (not Broadcast) and (not SameSquad)
        
        # Broadcast check per sender: is_broadcast [N]
        # Same squad matrix: [N, N]
        
        valid_connection = is_broadcast[:, None] | same_squad
        
        # Apply strict filtering (Infinity penalty)
        final_score = jnp.where(valid_connection, raw_score, infinity)
    else:
        final_score = raw_score
    
    # 8. Top-K Selection
    final_score = jnp.where(is_speaking[:, None], final_score, infinity)
    
    neg_score = -final_score
    best_vals, best_indices = jax.lax.top_k(neg_score.T, K) 
    inbox_dists = -best_vals
    
    # 9. Construct Inbox
    inbox_msgs = messages[best_indices] 
    inbox_mask = (inbox_dists < (infinity / 2)).astype(jnp.float32)
    
    sender_pos = agent_positions[best_indices]
    receiver_pos = agent_positions[:, None, :] 
    rel_vec = sender_pos - receiver_pos
    rel_dist = jnp.linalg.norm(rel_vec, axis=-1, keepdims=True)
    rel_angle = jnp.arctan2(rel_vec[..., 1], rel_vec[..., 0])[..., None]
    sender_channels = is_broadcast[best_indices].astype(jnp.float32)[..., None]
    
    inbox_meta = jnp.concatenate([rel_dist, rel_angle, sender_channels], axis=-1)
    
    return inbox_msgs, inbox_meta, inbox_mask

