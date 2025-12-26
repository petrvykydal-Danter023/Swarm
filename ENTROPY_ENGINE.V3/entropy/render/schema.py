"""
Entropy Engine V3 - Rendering Schema
Defines the data structure passed between Core and Visualization.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List, Tuple

@dataclass
class RenderFrame:
    """
    Data payload for a single visualization frame.
    All arrays are numpy arrays (CPU-side) for serialization.
    """
    timestep: int
    
    # Agents (required)
    agent_positions: np.ndarray   # [N, 2]
    agent_angles: np.ndarray      # [N]
    agent_colors: np.ndarray      # [N, 3] RGB
    agent_messages: np.ndarray    # [N, MSG_DIM]
    agent_radii: np.ndarray       # [N]
    
    # Environment (required)
    goal_positions: np.ndarray    # [N, 2]
    object_positions: np.ndarray  # [O, 2]
    object_types: np.ndarray      # [O]
    wall_segments: np.ndarray     # [W, 4]
    
    # Optional fields (must come after required)
    agent_velocities: Optional[np.ndarray] = None  # [N, 2] for debug
    lidar_readings: Optional[np.ndarray] = None    # [N, R] for debug
    rewards: Optional[np.ndarray] = None           # [N]
    fps: float = 0.0
    
    # Pheromone Visualization (Stigmergy)
    pheromone_positions: Optional[np.ndarray] = None  # [P, 2]
    pheromone_ttls: Optional[np.ndarray] = None       # [P] remaining steps
    pheromone_valid: Optional[np.ndarray] = None      # [P] bool mask
    pheromone_max_ttl: float = 100.0                  # For fading calculation
    pheromone_radius: float = 50.0                    # Detection radius
    
    # Hierarchy Visualization (Squads & Leaders)
    agent_squad_ids: Optional[np.ndarray] = None      # [N] int squad index
    agent_is_leader: Optional[np.ndarray] = None      # [N] bool
    
    # Safety Visualization
    safety_enabled: bool = False
    safety_radius: float = 30.0
    safety_repulsion_radius: float = 25.0
    geofence_zones: Optional[List] = None             # List of zone dicts
    
    # Intent Visualization
    intent_enabled: bool = False
    agent_intents: Optional[np.ndarray] = None        # [N, D] Raw intent vector
    intent_targets: Optional[np.ndarray] = None       # [N, 2] computed target (debug)
    
    def to_dict(self):
        """Helper for serialization if needed."""
        return self.__dict__

