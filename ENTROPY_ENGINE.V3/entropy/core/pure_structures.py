"""
Entropy Engine V4 - Pure JAX Data Structures ⚡
Defines the State and Parameters for the functional engine.
"""
from flax import struct
import jax.numpy as jnp
from typing import Tuple, Any

from entropy.core.world import WorldState
from entropy.safety.metrics import SafetyMetrics

@struct.dataclass
class EnvParams:
    """
    Static environment configuration.
    Values here are "baked in" during JIT compilation.
    """
    # === SIMULATION ===
    num_agents: int = 20
    arena_width: float = 800.0
    arena_height: float = 600.0
    max_steps: int = 200
    dt: float = 0.1
    
    # === AGENT ===
    lidar_rays: int = 32
    lidar_range: float = 200.0
    agent_radius: float = 10.0
    
    # === COMMUNICATION ===
    use_comms: bool = False
    msg_dim: int = 8
    max_neighbors: int = 5
    comm_radius: float = 150.0
    
    # === SAFETY LAYER ===
    safety_enabled: bool = True
    collision_reflex: bool = True
    monitor_metrics: bool = True
    
    # === REWARDS ===
    w_dist: float = 1.0
    w_reach: float = 10.0
    w_collision: float = -1.0
    w_living_penalty: float = -0.001
    
    # === HOG ===
    hog_density: float = 0.0 # Placeholder for HOG integration param? 
                             # Actually HOG weight is dynamic per step, not static param.

@struct.dataclass
class EnvState:
    """
    Dynamic container for the environment.
    Passed between pure function steps.
    """
    world: WorldState
    rng: jnp.ndarray
    step_count: int
    
    # Safety Metrics (Aggregated)
    # We store them here to carry over steps
    safety_metrics: SafetyMetrics
    
    @property
    def done(self) -> bool:
        # Check if max steps reached
        return self.step_count >= self.world.max_steps # Assuming world stores max_steps or we use params
        # Wait, max_steps is in Params usually.
        # But for 'done' property convenience, we might need access. 
        # Better to compute 'done' in the step function.
        pass

@struct.dataclass
class EnvStep:
    """
    Output of a single environment step.
    """
    obs: jnp.ndarray
    state: EnvState
    reward: jnp.ndarray # [N] (or scalar if shared)
    done: jnp.ndarray   # [N] (or scalar)
    info: Any # Dict-like struct or just jnp array of metrics
