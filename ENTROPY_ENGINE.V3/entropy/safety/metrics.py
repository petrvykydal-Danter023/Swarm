import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, Any
from entropy.core.world import WorldState
from entropy.config import SafetyConfig

@struct.dataclass
class SafetyMetrics:
    """Metriky pro monitoring Safety Layer."""
    
    # Collision Avoidance
    speed_reductions: int = 0       # Kolikrát byla snížena rychlost
    hard_stops: int = 0             # Kolikrát došlo k úplnému zastavení
    repulsion_activations: int = 0  # Kolikrát se aktivovala repulze
    
    # Communication
    messages_blocked: int = 0       # Zprávy zahozené kvůli rate limit
    tokens_depleted: int = 0        # Kolikrát došly tokeny
    
    # Watchdog
    stalemates_detected: int = 0    # Počet detekovaných zacyklení
    random_walks_triggered: int = 0 # Počet spuštěných random walks
    
    # Geo-Fence
    boundary_pushes: int = 0        # Počet pushů od hranic
    
    # Override
    safety_overrides: int = 0       # Kolikrát AI přepsala safety
    
    def to_dict(self) -> Dict[str, int]:
        return self.__dict__
    
    def log_summary(self, epoch: int):
        print(f"[SAFETY] Epoch {epoch} | "
              f"SpeedRed: {self.speed_reductions} | "
              f"Stops: {self.hard_stops} | "
              f"MsgBlocked: {self.messages_blocked} | "
              f"Stalemates: {self.stalemates_detected}")

def compute_safety_metrics(
    raw_actions: jnp.ndarray,
    safe_actions: jnp.ndarray,
    state: WorldState,
    config: SafetyConfig
) -> SafetyMetrics:
    """
    Počítá metriky z rozdílu raw vs safe actions.
    """
    # Speed reduction detection
    raw_speed = jnp.linalg.norm(raw_actions[:, :2], axis=1)
    safe_speed = jnp.linalg.norm(safe_actions[:, :2], axis=1)
    speed_reduced = (raw_speed - safe_speed) > 0.1
    
    # Hard stop detection
    hard_stopped = (safe_speed < 0.01) & (raw_speed > 0.1)
    
    # Gate changes (message blocking)
    raw_gate = jax.nn.sigmoid(raw_actions[:, 2]) > 0.5
    safe_gate = jax.nn.sigmoid(safe_actions[:, 2]) > 0.5
    msg_blocked = raw_gate & ~safe_gate
    
    return SafetyMetrics(
        speed_reductions=jnp.sum(speed_reduced),
        hard_stops=jnp.sum(hard_stopped),
        messages_blocked=jnp.sum(msg_blocked)
    )
