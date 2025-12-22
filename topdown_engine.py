import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import time

@dataclass
class Agent:
    """V4 Agent: Pure Top-Down Entity"""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 2.0
    mass: float = 1.0
    color: Tuple[int, int, int] = (100, 150, 255)
    
    # State flags
    is_grabbing: bool = False
    energy: float = 1.0
    
    def to_dict(self):
        return {
            "id": self.id, "x": self.x, "y": self.y,
            "vx": self.vx, "vy": self.vy,
            "is_grabbing": self.is_grabbing, "energy": self.energy
        }

@dataclass
class Obstacle:
    """Static circular obstacle"""
    x: float
    y: float
    radius: float
    type: str = "obstacle" # obstacle, goal

@dataclass
class Payload:
    """Heavy movable object (Circular for V4 simplicity)"""
    id: int
    x: float
    y: float
    radius: float
    mass: float = 10.0
    vx: float = 0.0
    vy: float = 0.0
    color: tuple = (200, 200, 50) # Yellow

class TopDownSwarmEnv(gym.Env):
    """
    Entropy Engine V4: Pure Top-Down Swarm Simulator.
    Features: Zero Gravity, High Friction, Elastic Collisions, Payload Transport.
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # World Params
        self.width = config.get("world_width", 100.0)
        self.height = config.get("world_height", 100.0)
        self.num_agents = config.get("num_agents", 10)
        self.dt = config.get("dt", 0.1)
        
        # Physics Params
        self.friction = config.get("friction", 0.2)
        self.restitution = config.get("restitution", 0.5)
        self.payload_friction = config.get("payload_friction", 0.1) # Heavy objects slide more/less?
        
        # Entities
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []
        self.payloads: List[Payload] = [] # Transportable objects
        
        # Gym Spaces
        self.action_type = config.get("action_type", "continuous")
        if self.action_type == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, 3), dtype=np.float32
            )
        else:
            self.action_space = spaces.MultiDiscrete([6] * self.num_agents)

        self.obs_dim = 14
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, self.obs_dim), dtype=np.float32
        )
        
        self.reward_func = self._compile_reward(config.get("reward_code", "reward = 0.0"))
        
        self._build_world()
        self.current_step = 0
        self.max_steps = config.get("max_steps", 500)

    def _build_world(self):
        """Initialize agents, obstacles, and payloads."""
        self.agents = []
        spawn = self.config.get("spawn_zone", {"x": 10, "y": 10, "w": 10, "h": 80})
        
        for i in range(self.num_agents):
            x = np.random.uniform(spawn["x"], spawn["x"] + spawn["w"])
            y = np.random.uniform(spawn["y"], spawn["y"] + spawn["h"])
            self.agents.append(Agent(id=i, x=x, y=y))
            
        self.obstacles = []
        self.payloads = []
        for obj in self.config.get("special_objects", []):
            if obj["type"] == "payload":
                self.payloads.append(Payload(
                    id=len(self.payloads),
                    x=obj["x"], y=obj["y"], radius=obj.get("radius", 5), mass=obj.get("mass", 5.0)
                ))
            else:
                self.obstacles.append(Obstacle(
                    obj["x"], obj["y"], obj.get("radius", 5), obj["type"]
                ))
    
    # ... reset, step, _apply_control methods mostly same ...
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        self._build_world()
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, actions):
        self.current_step += 1
        self._apply_control(actions)
        self._integrate_physics()
        self._solve_collisions()
        self._enforce_bounds()
        return self._get_obs(), self._compute_rewards(), False, self.current_step >= self.max_steps, {}

    def _apply_control(self, actions):
        speed = 1.0
        for i, agent in enumerate(self.agents):
            if self.action_type == "continuous":
                ax, ay, grab = actions[i]
            else:
                # Basic discrete logic
                sc = actions[i] if np.isscalar(actions[i]) else actions[i].item()
                act = int(sc)
                ax, ay = 0.0, 0.0
                if act == 1: ay = 1.0
                elif act == 2: ay = -1.0
                elif act == 3: ax = -1.0
                elif act == 4: ax = 1.0
            agent.vx += ax * speed * self.dt
            agent.vy += ay * speed * self.dt

    def _integrate_physics(self):
        """Apply velocity and friction to Agents AND Payloads."""
        # Agents
        for agent in self.agents:
            agent.vx *= (1.0 - self.friction)
            agent.vy *= (1.0 - self.friction)
            agent.x += agent.vx
            agent.y += agent.vy
            
        # Payloads
        for p in self.payloads:
            p.vx *= (1.0 - self.payload_friction)
            p.vy *= (1.0 - self.payload_friction)
            p.x += p.vx
            p.y += p.vy

    def _solve_collisions(self):
        """Solve ALL collisions (Agent-Agent, Agent-Obstacle, Agent-Payload, Payload-Obstacle)."""
        
        # 1. Agent vs Everything
        for i in range(self.num_agents):
            a1 = self.agents[i]
            
            # Vs Obstacles
            for obs in self.obstacles:
                if obs.type == "obstacle":
                    self._solve_circle_circle(a1, obs, dynamic_b=False)

            # Vs Payloads (Dynamic-Dynamic with Mass)
            for p in self.payloads:
                self._solve_circle_circle(a1, p, dynamic_b=True)
            
            # Vs Agents
            for j in range(i + 1, self.num_agents):
                self._solve_circle_circle(a1, self.agents[j], dynamic_b=True)

        # 2. Payload vs Environment (Obstacles)
        for p in self.payloads:
             for obs in self.obstacles:
                if obs.type == "obstacle":
                    self._solve_circle_circle(p, obs, dynamic_b=False)

    def _solve_circle_circle(self, c1, c2, dynamic_b=True):
        """Generic elastic collision solver."""
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        dist = math.sqrt(dx*dx + dy*dy)
        min_dist = c1.radius + c2.radius
        
        if dist < min_dist:
            if dist == 0: dist = 0.001
            nx, ny = dx/dist, dy/dist
            overlap = min_dist - dist
            
            # Static B (Obstacle)
            if not dynamic_b:
                c1.x -= nx * overlap
                c1.y -= ny * overlap
                vn = c1.vx * nx + c1.vy * ny
                if vn > 0: return # Moving away
                # Bounce
                c1.vx -= (1 + self.restitution) * vn * nx
                c1.vy -= (1 + self.restitution) * vn * ny
                return

            # Dynamic B (Agent/Payload)
            # Mass handling
            m1 = getattr(c1, 'mass', 1.0)
            m2 = getattr(c2, 'mass', 1.0)
            inv_m1 = 1.0 / m1
            inv_m2 = 1.0 / m2
            total_inv_mass = inv_m1 + inv_m2
            
            # Position Correction (Push apart based on mass)
            move_per_mass = overlap / total_inv_mass
            c1.x -= nx * move_per_mass * inv_m1
            c1.y -= ny * move_per_mass * inv_m1
            c2.x += nx * move_per_mass * inv_m2
            c2.y += ny * move_per_mass * inv_m2
            
            # Velocity Response (Elastic)
            # Relative velocity along normal
            dvx = c2.vx - c1.vx
            dvy = c2.vy - c1.vy
            vn = dvx * nx + dvy * ny
            
            if vn > 0: return # Moving away
            
            # Impulse scalar
            j = -(1 + self.restitution) * vn
            j /= total_inv_mass
            
            impulse_x = j * nx
            impulse_y = j * ny
            
            c1.vx -= impulse_x * inv_m1
            c1.vy -= impulse_y * inv_m1
            c2.vx += impulse_x * inv_m2
            c2.vy += impulse_y * inv_m2

    def _enforce_bounds(self):
        # Agents & Payloads
        for ent in self.agents + self.payloads:
            if ent.x < ent.radius: ent.x = ent.radius; ent.vx *= -0.5
            if ent.x > self.width - ent.radius: ent.x = self.width - ent.radius; ent.vx *= -0.5
            if ent.y < ent.radius: ent.y = ent.radius; ent.vy *= -0.5
            if ent.y > self.height - ent.radius: ent.y = self.height - ent.radius; ent.vy *= -0.5

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        # Populate simplest obs: [x,y,vx,vy] relative
        # To be expanded for real tasks
        return obs
    
    def _compute_rewards(self):
        # Wrapper for user code
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            try:
                env_state = {
                    "agent_idx": i,
                    "agents": [a.to_dict() for a in self.agents],
                    "payloads": [p.__dict__ for p in self.payloads],
                    "goals": [o.__dict__ for o in self.obstacles if o.type == "goal"]
                }
                rewards[i] = self.reward_func(agent.to_dict(), env_state, math, np)
            except: rewards[i] = 0.0
        return rewards

    def _compile_reward(self, code):
        wrapper = f"""
def user_reward(agent, env_state, math, np):
    reward = 0.0
{chr(10).join(['    ' + l for l in code.splitlines()])}
    return float(reward)
"""
        loc = {}
        exec(wrapper, {}, loc)
        return loc["user_reward"]

    def render(self, mode="rgb_array"):
        h, w = int(self.height * 5), int(self.width * 5)
        img = np.zeros((h, w, 3), dtype=np.uint8) + 30
        
        def draw_circle(x, y, r, c):
            cx, cy = int(x*5), int((self.height-y)*5)
            rr = int(r*5)
            y_min, y_max = max(0, cy-rr), min(h, cy+rr)
            x_min, x_max = max(0, cx-rr), min(w, cx+rr)
            # Center block
            img[max(0,cy-rr+1):min(h,cy+rr-1), max(0,cx-rr+1):min(w,cx+rr-1)] = c

        for obs in self.obstacles:
            col = (0, 255, 0) if obs.type == "goal" else (100, 100, 100)
            draw_circle(obs.x, obs.y, obs.radius, col)
            
        for p in self.payloads:
            draw_circle(p.x, p.y, p.radius, p.color)

        for agent in self.agents:
            draw_circle(agent.x, agent.y, agent.radius, agent.color)
            
        return img
