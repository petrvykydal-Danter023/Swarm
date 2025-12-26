import jax
import jax.numpy as jnp
from flax import struct
from entropy.core.world import WorldState
from entropy.config import SafetyConfig

@struct.dataclass
class WatchdogState:
    """
    Sleduje pohyb agentů pro detekci zacyklení.
    OPTIMALIZACE: Pouze 2 pozice místo celé historie!
    """
    position_old: jnp.ndarray           # [N, 2] pozice před window kroky
    steps_since_snapshot: int           # Kolik kroků od posledního snapshotu
    random_walk_remaining: jnp.ndarray  # [N] steps remaining in random walk
    
def create_watchdog_state(num_agents: int) -> WatchdogState:
    return WatchdogState(
        position_old=jnp.zeros((num_agents, 2)),
        steps_since_snapshot=0,
        random_walk_remaining=jnp.zeros(num_agents, dtype=jnp.int32)
    )

def apply_watchdog(
    state: WorldState,
    actions: jnp.ndarray,
    watchdog: WatchdogState,
    config: SafetyConfig,
    rng: jax.Array
) -> tuple:
    """
    Detekuje zaseknuté agenty a vynutí náhodný pohyb.
    OPTIMALIZACE: Místo circular bufferu používá snapshot každých N kroků.
    
    Returns:
        (safe_actions, new_watchdog_state)
    """
    N = state.agent_positions.shape[0]
    current_pos = state.agent_positions
    
    # ========== 1. Check if it's time to evaluate (every window steps) ==========
    is_check_step = watchdog.steps_since_snapshot >= config.stalemate_window
    
    # ========== 2. Calculate movement since last snapshot ==========
    total_movement = jnp.linalg.norm(current_pos - watchdog.position_old, axis=1)
    
    # ========== 3. Detect stalemate (only on check steps) ==========
    is_stuck = is_check_step & (total_movement < config.stalemate_min_distance)
    
    # ========== 4. Start random walk for stuck agents ==========
    should_start_walk = is_stuck & (watchdog.random_walk_remaining <= 0)
    new_walk_remaining = jnp.where(
        should_start_walk,
        config.stalemate_random_duration,
        watchdog.random_walk_remaining
    )
    
    # ========== 5. Apply random walk override ==========
    is_walking = new_walk_remaining > 0
    
    rng, walk_rng = jax.random.split(rng)
    random_angles = jax.random.uniform(walk_rng, (N,), minval=-jnp.pi, maxval=jnp.pi)
    random_motors = jnp.stack([
        jnp.cos(random_angles) * config.stalemate_random_speed,
        jnp.sin(random_angles) * config.stalemate_random_speed
    ], axis=1)
    
    # Blend: walking agents use random, others use original
    safe_actions = actions.at[:, :2].set(
        jnp.where(is_walking[:, None], random_motors, actions[:, :2])
    )
    
    # ========== 6. Update state ==========
    # Decrement walk timer
    new_walk_remaining = jnp.maximum(0, new_walk_remaining - 1)
    
    # Reset snapshot on check step
    new_position_old = jax.lax.cond(
        is_check_step,
        lambda: current_pos,
        lambda: watchdog.position_old
    )
    new_steps = jax.lax.cond(
        is_check_step,
        lambda: 0,
        lambda: watchdog.steps_since_snapshot + 1
    )
    
    new_watchdog = WatchdogState(
        position_old=new_position_old,
        steps_since_snapshot=new_steps,
        random_walk_remaining=new_walk_remaining
    )
    
    return safe_actions, new_watchdog
