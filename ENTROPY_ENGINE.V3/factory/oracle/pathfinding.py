"""
A* Pathfinding for Oracle Expert
Provides optimal paths for demonstration generation.
"""
import heapq
from typing import List, Tuple, Set, Optional, Dict, Any
import numpy as np

class AStarNavigator:
    """
    A* pathfinding for 2D grid-based navigation.
    Used by the Oracle to generate perfect trajectories.
    """
    def __init__(self, resolution: int = 10, allow_diagonal: bool = True):
        """
        Args:
            resolution: Grid cell size in world units
            allow_diagonal: Allow 8-directional movement
        """
        self.resolution = resolution
        self.allow_diagonal = allow_diagonal
        
        # Movement directions
        if allow_diagonal:
            self.directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            self.costs = [1.0] * 4 + [1.414] * 4
        else:
            self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            self.costs = [1.0] * 4

    def world_to_grid(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        return (int(pos[0] / self.resolution), int(pos[1] / self.resolution))

    def grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world center."""
        return (
            (cell[0] + 0.5) * self.resolution,
            (cell[1] + 0.5) * self.resolution
        )

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_path(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: Set[Tuple[int, int]],
        arena_size: Tuple[int, int] = (100, 100)
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Find optimal path from start to goal.
        
        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
            obstacles: Set of blocked grid cells
            arena_size: Arena dimensions in world units
            
        Returns:
            List of waypoints in world coordinates, or None if no path exists
        """
        start_cell = self.world_to_grid(start)
        goal_cell = self.world_to_grid(goal)
        
        # Grid bounds
        max_x = arena_size[0] // self.resolution
        max_y = arena_size[1] // self.resolution
        
        # Priority queue: (f_score, g_score, cell)
        open_set = [(0, 0, start_cell)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_cell: 0}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == goal_cell:
                # Reconstruct path
                path = [self.grid_to_world(current)]
                while current in came_from:
                    current = came_from[current]
                    path.append(self.grid_to_world(current))
                return list(reversed(path))
            
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Bounds check
                if not (0 <= neighbor[0] < max_x and 0 <= neighbor[1] < max_y):
                    continue
                    
                # Obstacle check
                if neighbor in obstacles:
                    continue
                
                tentative_g = g_score[current] + self.costs[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal_cell)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None  # No path found

    def compute_action(
        self,
        current_pos: Tuple[float, float],
        current_angle: float,
        path: List[Tuple[float, float]],
        max_speed: float = 5.0,
        max_turn_rate: float = 2.0
    ) -> Tuple[float, float]:
        """
        Compute motor action to follow path.
        
        Args:
            current_pos: Current position
            current_angle: Current heading (radians)
            path: Waypoints to follow
            max_speed: Maximum linear velocity
            max_turn_rate: Maximum angular velocity
            
        Returns:
            (linear_vel, angular_vel) action tuple
        """
        if not path or len(path) < 2:
            return (0.0, 0.0)
        
        # Target next waypoint
        target = path[1] if len(path) > 1 else path[0]
        
        # Direction to target
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 1.0:  # Reached waypoint
            return (0.0, 0.0)
        
        # Target angle
        target_angle = np.arctan2(dy, dx)
        
        # Angle difference (wrapped to [-pi, pi])
        angle_diff = target_angle - current_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # Proportional control
        angular_vel = np.clip(angle_diff * 2.0, -max_turn_rate, max_turn_rate)
        
        # Slow down when turning sharply
        turn_factor = 1.0 - min(abs(angle_diff) / np.pi, 0.5)
        linear_vel = max_speed * turn_factor
        
        return (linear_vel, angular_vel)
