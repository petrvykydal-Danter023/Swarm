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
    # Dynamics for "dynamic" type
    vx: float = 0.0
    vy: float = 0.0
    # For Zone type
    capture_progress: float = 0.0

class GoalManager:
    """
    Manages goals for the environment.
    Supports different goal types: 'static', 'path', 'zone', 'dynamic'.
    """
    def __init__(self, config: dict):
        self.config = config
        self.world_w = config.get("world_width", 100.0)
        self.world_h = config.get("world_height", 100.0)
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
        
        # New Config Logic
        if self.goal_type == "dynamic":
            # Add a dynamic goal if none exist
            if not self.goals:
                self.goals.append(GoalState(
                    x=50, y=50, 
                    type="dynamic", 
                    vx=5.0, vy=3.0, # Slow movement
                    radius=5.0
                ))
            else:
                for g in self.goals:
                    g.type = "dynamic"
                    g.vx = 5.0
                    g.vy = 3.0

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
        # Future: Path/Waypoint logic here
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
            # Scaled intensity: 1.0 at 0 dist, 0.0 at infinity.
            # Typical normalization: 100 / (dist + 1)
            intensity = 100.0 / (dist + 10.0) 
            # We return intensity in first channel, 0 in second (vector shape is 2)
            # Or Gradient?
            return np.array([intensity, 0.0], dtype=np.float32)

        return np.array([dx / dist, dy / dist], dtype=np.float32)

    def get_batch_observation(self, agents_pos: np.ndarray, mode: str = "nav", detection_radius: float = 100.0) -> np.ndarray:
        """
        Vectorized observation for N agents.
        agents_pos: (N, 2) matrix
        Returns: (N, 2) matrix of goal vectors.
        """
        N = agents_pos.shape[0]
        
        # Determine target for each agent
        # For now, mostly single static/dynamic goal supported
        if not self.goals:
            return np.zeros((N, 2), dtype=np.float32)
            
        target = self.goals[0] # Default target
        
        # Diffs: (Goal - Agent)
        # target.x is scalar, agents_pos[:, 0] is vector
        dx = target.x - agents_pos[:, 0]
        dy = target.y - agents_pos[:, 1]
        
        # Distances
        dists = np.sqrt(dx*dx + dy*dy)
        dists = np.maximum(dists, 0.001) # Avoid div/0
        
        # Normalize
        # Stack dx, dy to (N, 2)
        goal_vecs = np.stack([dx / dists, dy / dists], axis=1)
        
        # Mode Logic
        if mode == "search":
             # Mask out agents who are too far
             # dists shape is (N,)
             # mask shape (N, 1) to broadcast to (N, 2)
             mask = (dists <= detection_radius).astype(np.float32)[:, np.newaxis]
             goal_vecs *= mask
             
        elif mode == "scent":
             # Intensity = 100 / (dist + 10)
             intensity = 100.0 / (dists + 10.0)
             # Return (Intensity, 0)
             # Fill col 0 with intensity, col 1 with 0
             goal_vecs = np.zeros((N, 2), dtype=np.float32)
             goal_vecs[:, 0] = intensity
        
        return goal_vecs

    def update(self, dt):
        """
        Update dynamic goals (physics).
        """
        for g in self.goals:
            if g.type == "dynamic":
                g.x += g.vx * dt
                g.y += g.vy * dt
                
                # Bounce Logic
                if g.x < g.radius or g.x > self.world_w - g.radius:
                    g.vx *= -1
                if g.y < g.radius or g.y > self.world_h - g.radius:
                    g.vy *= -1
