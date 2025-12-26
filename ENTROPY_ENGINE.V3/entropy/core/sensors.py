import jax
import jax.numpy as jnp
from functools import partial
from entropy.core.world import WorldState

@partial(jax.jit, static_argnums=(1,))
def compute_lidars(state: WorldState, num_rays: int = 32, max_range: float = 300.0) -> jnp.ndarray:
    """
    Vypočítá lidar readings pro všechny agenty.
    
    Args:
        state: WorldState
        num_rays: Počet paprsků (musí odpovídat state.lidar_readings.shape[1])
        max_range: Maximální dosah lidaru
        
    Returns:
        jnp.ndarray [N, num_rays] - normalizované vzdálenosti (0.0 = kolize, 1.0 = max range)
    """
    num_agents = state.agent_positions.shape[0]
    
    # 1. Připrav paprsky
    # Úhly paprsků relativně k agentovi [0, 2pi)
    ray_angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False) # [R]
    
    # Globální úhly paprsků pro každého agenta [N, R]
    # shape: [N, 1] + [1, R] -> [N, R] (broadcasting)
    global_angles = state.agent_angles[:, None] + ray_angles[None, :]
    
    # Start pozice (na obvodu agenta)
    # [N, R, 2]
    # cos/sin of global angles
    ray_dirs = jnp.stack([jnp.cos(global_angles), jnp.sin(global_angles)], axis=-1)
    
    starts = state.agent_positions[:, None, :] + ray_dirs * state.agent_radii[:, None, None]
    ends = state.agent_positions[:, None, :] + ray_dirs * max_range
    
    # 2. Raycast proti zdím
    # [N, R] -> distance
    wall_dists = _raycast_walls(starts, ends, state.wall_segments, max_range)
    
    # 3. Raycast proti ostatním agentům
    # Pro zjednodušení ve fázi 2 zatím vynecháme inter-agent raycasting nebo přidáme později.
    # Blueprint zmiňuje _batch_raycast_agents. Prozatím returns wall_dists.
    
    final_dists = wall_dists
    
    # Normalizace na [0, 1]
    return final_dists / max_range

@jax.jit
def _raycast_walls(starts, ends, walls, max_range):
    """
    Vektorizovaný raycast proti úsečkám zdí.
    starts: [N, R, 2]
    ends: [N, R, 2]
    walls: [W, 4] (x1, y1, x2, y2)
    
    Returns: [N, R] distance
    """
    # Expand dims for broadcasting:
    # starts/ends: [N, R, 1, 2]
    p1 = starts[:, :, None, :]
    p2 = ends[:, :, None, :]
    
    # walls: [1, 1, W, 2] for start and end
    w1 = walls[None, None, :, 0:2]
    w2 = walls[None, None, :, 2:4]
    
    # Implement intersection logic
    # Line 1: p1 + t*(p2-p1)
    # Line 2: w1 + u*(w2-w1)
    
    dp = p2 - p1
    dw = w2 - w1
    
    det = dp[..., 0] * dw[..., 1] - dp[..., 1] * dw[..., 0]
    
    # Avoid division by zero (parallel lines)
    det = jnp.where(jnp.abs(det) < 1e-6, 1e-6, det)
    
    d_pw = p1 - w1
    
    t = (d_pw[..., 1] * dw[..., 0] - d_pw[..., 0] * dw[..., 1]) / (-det)
    u = (dp[..., 1] * d_pw[..., 0] - dp[..., 0] * d_pw[..., 1]) / (-det)
    
    # Intersection valid if 0 <= t <= 1 and 0 <= u <= 1
    valid = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    
    # Distance is t * length of ray (which is max_range)
    # OR simpler: distance = t * |p2-p1|. Since |p2-p1| = max_range, dist = t * max_range.
    # If not valid, dist = infinity
    
    dist = jnp.where(valid, t * max_range, jnp.inf)
    
    # Minimum across all walls [N, R, W] -> [N, R]
    # Check if we have walls, otherwise return inf
    has_walls = walls.shape[0] > 0
    
    # We need a safe reduction.
    # If W=0, min over axis -1 is invalid.
    # Logic: return min(dist) if has_walls else inf
    
    # NOTE: In JAX JIT, shapes are static. If we trace with W=0, this branch is known at compile time.
    # However, standard jnp.min() on empty axis raises ValueError.
    
    # Approach: Use explicit check if W > 0
    if walls.shape[0] > 0:
         min_dist = jnp.min(dist, axis=-1)
    else:
         # Maintain shape [N, R]
         min_dist = jnp.full(starts.shape[:2], jnp.inf)
    
    # Clip to max_range (if no wall hit, min_dist is inf)
    return jnp.minimum(min_dist, max_range)
