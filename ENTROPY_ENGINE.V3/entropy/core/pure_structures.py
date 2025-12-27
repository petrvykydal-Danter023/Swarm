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
    action_dim: int = 2 # Dimension of action vector
    
    # === COMMUNICATION ===
    use_comms: bool = False
    msg_dim: int = 8
    max_neighbors: int = 5
    comm_radius: float = 150.0
    
    # === SAFETY LAYER ===
    safety_enabled: bool = True
    collision_reflex: bool = True
    monitor_metrics: bool = True
    
    # === REWARD HYPERPARAMETERS (Universal System) ===
    gamma: float = 0.99
    w_dist: float = 1.0
    w_reach: float = 200.0 # Sparse reward magnitude
    w_energy: float = 0.01
    w_smooth: float = 0.1
    w_collision: float = 10.0
    w_living_penalty: float = -0.01
    
    # === TASKS ===
    task_id: int = 0            # 0=Nav, 1=Search, 2=Push
    target_radius: float = 5.0
    
    # === HOG ===
    hog_density: float = 0.0

@struct.dataclass
class EnvState:
    """
    Dynamic container for the environment.
    Passed between pure function steps.
    """
    # === KINEMATICS ===
    world: WorldState
    
    # === HISTORY (For PBRS & Smoothness) ===
    prev_pos: jnp.ndarray     # [N, 2]
    prev_action: jnp.ndarray  # [N, 2] (Differential drive commands)
    
    # === TASK SPECIFIC ===
    # Stores [N, 2] for flexible per-agent target assignment
    target: jnp.ndarray       # [N, 2]
    
    # Optional Push Task fields
    box_pos: jnp.ndarray      # [1, 2]
    prev_box_pos: jnp.ndarray # [1, 2]
    
    # Optional Search Task fields
    target_visible: jnp.ndarray # [N, 1] 0.0 or 1.0
    
    # === META & RNG ===
    rng: jnp.ndarray
    step_count: int
    
    # Safety Metrics (Aggregated)
    safety_metrics: SafetyMetrics
    
    @property
    def done(self) -> bool:
        # Check if max steps reached
        # Note: We need access to params.max_steps to be accurate here, 
        # but this property is mostly for debug/convenience outside of JIT.
        # Inside JIT, use explicit done flag returned by step.
        return False

@struct.dataclass
class EnvStep:
    """
    Output of a single environment step.
    """
    obs: jnp.ndarray
    state: EnvState
    reward: jnp.ndarray # [N]
    done: jnp.ndarray   # [N] boolean
    info: Any # Dict-like struct or metric array
