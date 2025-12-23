import numpy as np
import math
from dataclasses import dataclass

@dataclass
class GoalState:
    x: float
    y: float
    type: str = "static"
    radius: float = 5.0
    active: bool = True

class GoalManager:
    """
    Manages goals for the environment.
    Supports different goal types: 'static', 'path', 'zone', 'dynamic'.
    """
    def __init__(self, config: dict):
        self.config = config
        self.goal_type = config.get("goal_type", "static")
        self.goals = []
        self._parse_goals(config)
        
    def _parse_goals(self, config):
        self.goals = []
        special_objects = config.get("special_objects", [])
        
        # Legacy support: look for "goal" types in special_objects
        for obj in special_objects:
            if obj.get("type") == "goal":
                self.goals.append(GoalState(
                    x=obj["x"], 
                    y=obj["y"], 
                    radius=obj.get("radius", 5.0),
                    type="static"
                ))
                
        # Future: If goal_type == "path", parse waypoints etc.
                
    def check_completion(self, agent) -> bool:
        """
        Checks if agent has reached its goal.
        For static goal: dist < radius ?
        """
        # For now, return false as "completion" usually triggers reset or score,
        # but in current env, rewards handle it. 
        # We can implement specific logic here later.
        return False

    def get_goal_for_agent(self, agent):
        """
        Returns the target (x, y) for the specific agent.
        For static single goal, it's the first goal.
        """
        if not self.goals:
            return None
        return self.goals[0] # Default to first goal

    def get_observation(self, agent, mode: str = "nav", detection_radius: float = 100.0) -> np.ndarray:
        """
        Returns the goal vector (dx, dy) for the observation space.
        Applies mode logic (blind search masking etc.)
        """
        target = self.get_goal_for_agent(agent)
        if not target:
            return np.zeros(2, dtype=np.float32)

        dx = target.x - agent.x
        dy = target.y - agent.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0: dist = 0.001

        # Mode Logic
        if mode == "search" and dist > detection_radius:
            return np.zeros(2, dtype=np.float32) # Unknown
            
        if mode == "scent":
            # Return intensity only? But existing obs expects 2 floats.
            # We might return [intensity, 0].
            # For now, keep navigation behavior as default.
            pass

        return np.array([dx / dist, dy / dist], dtype=np.float32)

    def update(self, dt):
        """
        Update dynamic goals (physics).
        """
        pass
