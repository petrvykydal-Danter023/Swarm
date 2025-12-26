"""
Privileged Oracle Expert
Has access to full state and uses A* for optimal navigation.
"""
import numpy as np
from typing import Any, Dict, Tuple, Optional, Set
from factory.oracle.pathfinding import AStarNavigator

class PrivilegedOracle:
    """
    The All-Knowing Expert.
    Has access to full state, future predictions, and global map.
    Uses A* for optimal pathfinding.
    """
    def __init__(self, config: Any):
        self.config = config if isinstance(config, dict) else {}
        
        # Initialize A* navigator
        nav_config = self.config.get("navigator", {})
        self.navigator = AStarNavigator(
            resolution=nav_config.get("resolution", 10),
            allow_diagonal=nav_config.get("allow_diagonal", True)
        )
        
        # Current path cache
        self.current_path: Optional[list] = None
        self.current_goal: Optional[Tuple[float, float]] = None

    def act(self, full_state: Any) -> Dict[str, np.ndarray]:
        """
        Compute perfect action based on full state.
        
        Args:
            full_state: Dictionary with keys:
                - agent_position: (x, y)
                - agent_angle: heading in radians
                - goal_position: (x, y) target
                - obstacles: Set of (grid_x, grid_y) blocked cells
                - arena_size: (width, height)
                
        Returns:
            {"motor": [linear_vel, angular_vel], "comm": token_encoding}
        """
        # Handle mock/empty state for testing
        if not full_state or not isinstance(full_state, dict):
            return {"motor": np.zeros(2), "comm": np.zeros(64)}
        
        # Extract state
        agent_pos = full_state.get("agent_position", (50.0, 50.0))
        agent_angle = full_state.get("agent_angle", 0.0)
        goal_pos = full_state.get("goal_position", (90.0, 90.0))
        obstacles = full_state.get("obstacles", set())
        arena_size = full_state.get("arena_size", (100, 100))
        
        # 1. Navigation (A*)
        motor_action = self._navigate(agent_pos, agent_angle, goal_pos, obstacles, arena_size)
        
        # 2. Communication (Ground Truth based on goal direction)
        comm_action = self._communicate(agent_pos, goal_pos)
        
        return {"motor": motor_action, "comm": comm_action}

    def _navigate(
        self,
        agent_pos: Tuple[float, float],
        agent_angle: float,
        goal_pos: Tuple[float, float],
        obstacles: Set[Tuple[int, int]],
        arena_size: Tuple[int, int]
    ) -> np.ndarray:
        """Compute navigation action using A*."""
        
        # Recompute path if goal changed or no path exists
        if self.current_goal != goal_pos or self.current_path is None:
            self.current_path = self.navigator.find_path(
                start=agent_pos,
                goal=goal_pos,
                obstacles=obstacles,
                arena_size=arena_size
            )
            self.current_goal = goal_pos
        
        if self.current_path is None:
            # No path found - stay still
            return np.array([0.0, 0.0])
        
        # Remove passed waypoints
        while len(self.current_path) > 1:
            wp = self.current_path[0]
            dist = np.sqrt((wp[0] - agent_pos[0])**2 + (wp[1] - agent_pos[1])**2)
            if dist < 5.0:  # Within 5 units, move to next waypoint
                self.current_path.pop(0)
            else:
                break
        
        # Compute motor action to follow path
        linear_vel, angular_vel = self.navigator.compute_action(
            current_pos=agent_pos,
            current_angle=agent_angle,
            path=self.current_path,
            max_speed=5.0,
            max_turn_rate=2.0
        )
        
        return np.array([linear_vel, angular_vel])

    def _communicate(self, agent_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> np.ndarray:
        """
        Generate ground truth communication token.
        Encodes direction to goal as a simple 64-dim vector.
        """
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.1:
            # At goal
            comm = np.zeros(64)
            comm[0] = 1.0  # "GOAL_REACHED" token
        else:
            # Encode direction as angle and distance
            angle = np.arctan2(dy, dx)
            comm = np.zeros(64)
            comm[1] = np.cos(angle)  # Direction X component
            comm[2] = np.sin(angle)  # Direction Y component
            comm[3] = min(distance / 100.0, 1.0)  # Normalized distance
        
        return comm

