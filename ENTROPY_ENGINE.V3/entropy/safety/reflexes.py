import jax
import jax.numpy as jnp
from functools import partial
from entropy.core.world import WorldState
from entropy.config import SafetyConfig

@partial(jax.jit, static_argnums=(2,))
def apply_collision_reflex(
    state: WorldState,
    raw_actions: jnp.ndarray,  # [N, ActionDim] from AI
    config: SafetyConfig
) -> jnp.ndarray:
    """
    Modifikuje akce AI tak, aby se zabránilo kolizím.
    
    OPTIMALIZACE:
    - Squad-aware: Neodpuzuje agenty ve stejném squadu (formace fungují!)
    - Lokální radius: Pro velké swarmy počítá jen blízké sousedy
    - JIT-safe: Používá jax.lax.cond místo Python if
    
    Logika:
    1. Spočítej vzdálenosti k nejbližším překážkám (agenti + zdi)
    2. FILTRUJ: Ignoruj členy stejného squadu (pokud hierarchy enabled)
    3. Pokud dist < safety_radius: Sniž rychlost + přidej repulzi
    """
    N = state.agent_positions.shape[0]
    pos = state.agent_positions
    
    # ========== 1. Agent-Agent Distances [N, N] ==========
    dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    dists = dists + jnp.eye(N) * 1e9  # Ignore self
    
    # ========== 2. SQUAD-AWARE FILTERING ==========
    # Neodpuzuj členy stejného squadu → formace zůstanou intaktní
    squad_ids = getattr(state, 'agent_squad_ids', None)
    if squad_ids is not None:
        same_squad = squad_ids[:, None] == squad_ids[None, :]
        # Nastavíme vzdálenost na INF pro členy stejného squadu
        dists = jnp.where(same_squad, 1e9, dists)
    
    # ========== 3. LOKÁLNÍ SOUSEDSTVÍ (pro škálování) ==========
    # Ignoruj agenty dál než collision_check_radius (default: 2x safety_radius)
    check_radius = getattr(config, 'collision_check_radius', config.safety_radius * 2)
    dists = jnp.where(dists > check_radius, 1e9, dists)
    
    min_agent_dist = jnp.min(dists, axis=1)  # [N]
    
    # ========== 4. Wall Distances ==========
    arena_w, arena_h = state.arena_size
    dist_left = pos[:, 0]
    dist_right = arena_w - pos[:, 0]
    dist_bottom = pos[:, 1]
    dist_top = arena_h - pos[:, 1]
    
    min_wall_dist = jnp.minimum(
        jnp.minimum(dist_left, dist_right),
        jnp.minimum(dist_bottom, dist_top)
    )
    
    min_obstacle = jnp.minimum(min_agent_dist, min_wall_dist)
    
    # ========== 5. Speed Reduction Factor ==========
    # 1.0 at dist >= safety_radius, 0.0 at dist = 0
    speed_factor = jnp.clip(min_obstacle / config.safety_radius, 0.0, 1.0)
    
    # ========== 6. Apply to Motor Actions ==========
    safe_actions = raw_actions.at[:, 0].multiply(speed_factor)
    safe_actions = safe_actions.at[:, 1].multiply(speed_factor)
    
    # ========== 7. Repulsion Force (Squad-Aware) ==========
    def apply_repulsion(actions):
        # Find nearest NON-SQUAD agent
        nearest_idx = jnp.argmin(dists, axis=1)
        nearest_pos = pos[nearest_idx]
        repel_vec = pos - nearest_pos
        # Normalize with epsilon to avoid division by zero
        repel_vec = repel_vec / (jnp.linalg.norm(repel_vec, axis=1, keepdims=True) + 1e-6)
        
        # Repulsion strength (falls off with distance)
        repel_strength = jnp.where(
            min_agent_dist < config.repulsion_radius,
            config.repulsion_force * (1.0 - min_agent_dist / config.repulsion_radius),
            0.0
        )
        
        # Apply to motor commands
        return actions.at[:, 0].add(repel_vec[:, 0] * repel_strength).at[:, 1].add(repel_vec[:, 1] * repel_strength)
    
    # JIT-safe conditional
    safe_actions = jax.lax.cond(
        config.enable_repulsion,
        apply_repulsion,
        lambda x: x,
        safe_actions
    )
    
    return safe_actions
