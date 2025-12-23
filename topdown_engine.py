import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
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
    Supports: Dynamic Sensors, Communication, Noise/Interference.
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, config: dict):
        super().__init__()
        # Flatten config if env_params exists (compatibility with nested JSON)
        self.config = config.copy()
        if "env_params" in config:
            self.config.update(config["env_params"])
            
        # World Params
        self.width = self.config.get("world_width", 100.0)
        self.height = self.config.get("world_height", 100.0)
        self.num_agents = self.config.get("num_agents", 10)
        self.dt = self.config.get("dt", 0.1)
        self.observation_type = self.config.get("observation_type", "spatial") # Compatibility with rl_trainer
        
        # Physics Params
        self.friction = self.config.get("friction", 0.2)
        self.restitution = self.config.get("restitution", 0.5)
        self.payload_friction = self.config.get("payload_friction", 0.1)
        
        # Sensory & Comms
        self.sensors = self.config.get("sensors", ["position", "velocity", "goal_vector", "neighbor_vectors"])
        self.enable_communication = self.config.get("enable_communication", False)
        self.obs_noise_std = self.config.get("obs_noise_std", 0.0)
        self.comm_range = self.config.get("comm_range", 1000.0) # Default: practically unlimited
        self.packet_loss_prob = self.config.get("packet_loss_prob", 0.1)
        
        # Entities
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []
        self.payloads: List[Payload] = []
        
        # Comm State
        self.comm_signals = np.zeros(self.num_agents, dtype=np.float32)
        
        # Calculate Obs Dim
        self.obs_dim = self._calculate_obs_dim()
        
        # Gym Spaces
        self.action_type = config.get("action_type", "continuous")
        self.action_space_type = self.action_type # Compatibility
        if self.action_type == "continuous":
            # [vx, vy, grab] + [comm]
            act_dim = 3 + (1 if self.enable_communication else 0)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, act_dim), dtype=np.float32
            )
            # Motor Lag (Action Buffer)
            self.motor_lag = config.get("motor_lag", 0) # 0 = no lag
            self.action_queues = [deque(maxlen=self.motor_lag + 1) for _ in range(self.num_agents)]
            # Fill with zeros initially
            if self.motor_lag > 0:
                for i in range(self.num_agents):
                    for _ in range(self.motor_lag):
                         self.action_queues[i].append(np.zeros(act_dim, dtype=np.float32))
            
            # Action Smoothness (History)
            self.last_actions = np.zeros((self.num_agents, act_dim), dtype=np.float32)

        else:
            # Discrete (No comms support in discrete currently, or need mapping)
            self.action_space = spaces.MultiDiscrete([6] * self.num_agents)
            # Motor lag for discrete not fully impl detail yet, skipping strict queue for now or assuming continuous only for lag
            self.motor_lag = 0 # Force 0 for discrete for now
            self.action_queues = []
            self.last_actions = None # Not supported for discrete yet

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, self.obs_dim), dtype=np.float32
        )
        
        self.reward_func = self._compile_reward(config.get("reward_code", "reward = 0.0"))
        
        self._build_world()
        self.current_step = 0
        self.max_steps = config.get("max_steps", 500)

    def _calculate_obs_dim(self):
        dim = 0
        for s in self.sensors:
            if s == "position": dim += 2
            elif s == "velocity": dim += 2
            elif s == "goal_vector": dim += 2
            elif s == "neighbor_vectors": dim += 6 # 3 neighbors * 2
            elif s == "obstacle_radar": dim += 8
            elif s == "payload_radar": dim += 8
            elif s == "grabbing_state": dim += 1
            elif s == "energy": dim += 1
            elif s == "neighbor_signals": dim += 3 # 3 nearest
        return dim

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
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        self._build_world()
        self._reset_queues() # Clear buffers
        self.current_step = 0
        return self._get_obs(), {}

    def _reset_queues(self):
        if self.motor_lag > 0 and self.action_type == "continuous":
             # [vx, vy, grab] + [comm]
             act_dim = 3 + (1 if self.enable_communication else 0)
             self.action_queues = [deque(maxlen=self.motor_lag + 1) for _ in range(self.num_agents)]
             for i in range(self.num_agents):
                 for _ in range(self.motor_lag):
                      self.action_queues[i].append(np.zeros(act_dim, dtype=np.float32))

    def step(self, actions, verbose=False):
        self.current_step += 1
        
        applied_actions = actions
        if self.motor_lag > 0 and self.action_type == "continuous":
             # Push new actions, Pop old actions to apply
             delayed_actions = []
             for i in range(self.num_agents):
                  self.action_queues[i].append(actions[i])
                  # Debug
                  # print(f"DEBUG: Step {self.current_step}. Queue Len Before Pop={len(self.action_queues[i])}")
                  
                  # Re-init logic slightly:
                  # We just need to popleft and append.
                  old_act = self.action_queues[i].popleft()
                  # self.action_queues[i].append(actions[i]) # REMOVED DUPLICATE
                  delayed_actions.append(old_act)
             applied_actions = np.array(delayed_actions)

        # Split actions if communication enabled
        move_actions = applied_actions
        if self.enable_communication and self.action_type == "continuous":
            move_actions = applied_actions[:, :3]
            self.comm_signals = applied_actions[:, 3]
        
        self._apply_control(move_actions)
        self._integrate_physics()
        self._solve_collisions()
        self._enforce_bounds()
        
        rewards = self._compute_rewards(applied_actions)
        
        # Update history
        if self.action_type == "continuous":
             self.last_actions = applied_actions.copy()
             
        if verbose:
            print(f"--- Step {self.current_step} ---")
            for i, agent in enumerate(self.agents):
                speed = math.sqrt(agent.vx**2 + agent.vy**2)
                act_str = np.array2string(applied_actions[i], precision=2, suppress_small=True)
                print(f"Agent {i}: Spd={speed:.3f} | Act={act_str} | Rew={rewards[i]:.3f}")

        return self._get_obs(), rewards, False, self.current_step >= self.max_steps, {}

    def _apply_control(self, actions):
        speed = 1.0
        for i, agent in enumerate(self.agents):
            if self.action_type == "continuous":
                # actions[i] can be 3 or 4 dims. Grab first 3.
                ax, ay, grab = actions[i][:3]
                agent.is_grabbing = (grab > 0.5)
            else:
                sc = actions[i] if np.isscalar(actions[i]) else actions[i].item()
                act = int(sc)
                ax, ay = 0.0, 0.0
                if act == 1: ay = 1.0
                elif act == 2: ay = -1.0
                elif act == 3: ax = -1.0
                elif act == 4: ax = 1.0
                if act == 5: agent.is_grabbing = not agent.is_grabbing
            
            # Energy Penalty
            if agent.energy < 0.05:
                 ax *= 0.2
                 ay *= 0.2
            
            agent.vx += ax * speed * self.dt
            agent.vy += ay * speed * self.dt
            
            # Energy Consumption & Regen
            # Params (could be config, using defaults for now matching legacy)
            drain = 0.0
            
            # 1. Movement Cost
            current_speed = math.sqrt(agent.vx**2 + agent.vy**2)
            drain += current_speed * 0.005
            
            # 2. Grab Cost
            if agent.is_grabbing:
                drain += 0.002
                
            # 3. Signal Cost
            if self.enable_communication and self.action_type == "continuous":
                 # signal is in self.comm_signals[i], updated in step()
                 sig = self.comm_signals[i]
                 if abs(sig) > 0.01:
                     drain += 0.001 * abs(sig)
            
            # Apply
            agent.energy = max(0.0, min(1.0, agent.energy - drain + 0.001))

    def _integrate_physics(self):
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
        # 1. Agent vs Everything
        for i in range(self.num_agents):
            a1 = self.agents[i]
            # Vs Obstacles
            for obs in self.obstacles:
                if obs.type == "obstacle":
                    self._solve_circle_circle(a1, obs, dynamic_b=False)
            # Vs Payloads
            for p in self.payloads:
                self._solve_circle_circle(a1, p, dynamic_b=True)
            # Vs Agents
            for j in range(i + 1, self.num_agents):
                self._solve_circle_circle(a1, self.agents[j], dynamic_b=True)

        # 2. Payload vs Environment
        for p in self.payloads:
             for obs in self.obstacles:
                if obs.type == "obstacle":
                    self._solve_circle_circle(p, obs, dynamic_b=False)

    def _solve_circle_circle(self, c1, c2, dynamic_b=True):
        dx = c2.x - c1.x
        dy = c2.y - c1.y
        dist = math.sqrt(dx*dx + dy*dy)
        min_dist = c1.radius + c2.radius
        
        if dist < min_dist:
            if dist == 0: dist = 0.001
            nx, ny = dx/dist, dy/dist
            overlap = min_dist - dist
            
            if not dynamic_b:
                c1.x -= nx * overlap
                c1.y -= ny * overlap
                vn = c1.vx * nx + c1.vy * ny
                if vn > 0: return 
                c1.vx -= (1 + self.restitution) * vn * nx
                c1.vy -= (1 + self.restitution) * vn * ny
                return

            m1 = getattr(c1, 'mass', 1.0)
            m2 = getattr(c2, 'mass', 1.0)
            inv_m1 = 1.0 / m1
            inv_m2 = 1.0 / m2
            total_inv_mass = inv_m1 + inv_m2
            
            move_per_mass = overlap / total_inv_mass
            c1.x -= nx * move_per_mass * inv_m1
            c1.y -= ny * move_per_mass * inv_m1
            c2.x += nx * move_per_mass * inv_m2
            c2.y += ny * move_per_mass * inv_m2
            
            # Velocity Response
            dvx = c2.vx - c1.vx
            dvy = c2.vy - c1.vy
            vn = dvx * nx + dvy * ny
            if vn > 0: return 

            # Check for Grabbing (Sticky Collision)
            # Check for Grabbing (Sticky Collision)
            is_grabbing_interaction = False
            
            # Helper to check if obj is an agent that is grabbing
            def is_grabber(obj):
                return hasattr(obj, 'is_grabbing') and obj.is_grabbing

            # Check if c1 is grabbing c2 (and c2 is not another agent with same ID, though ID collision shouldn't happen in sim, test setup might cause it)
            # Actually, we just need to know if ONE is grabbing and the other is dynamic.
            # We want Agent -> Payload OR Agent -> Agent connection.
            
            if is_grabber(c1) and getattr(c2, 'id', -100) != getattr(c1, 'id', -200):
                 is_grabbing_interaction = True
            if is_grabber(c2) and getattr(c1, 'id', -100) != getattr(c2, 'id', -200):
                 is_grabbing_interaction = True

            # Fix for test case where Payload ID might clash with Agent ID (both 0)
            # We can check if types are different.
            if is_grabber(c1) and not is_grabber(c2): is_grabbing_interaction = True
            if is_grabber(c2) and not is_grabber(c1): is_grabbing_interaction = True
            
            # If both are agents and grabbing, they stick.
            if is_grabber(c1) and is_grabber(c2): is_grabbing_interaction = True
            
            if is_grabbing_interaction and dynamic_b:
                # Inelastic Collision (Stick Together)
                # Weighted average velocity (conservation of momentum)
                # v_final = (m1*v1 + m2*v2) / (m1 + m2)
                
                final_vx = (m1 * c1.vx + m2 * c2.vx) / total_inv_mass # Wait, total_inv_mass is 1/m1+1/m2.
                # Mass sum
                total_mass = m1 + m2
                final_vx = (m1 * c1.vx + m2 * c2.vx) / total_mass
                final_vy = (m1 * c1.vy + m2 * c2.vy) / total_mass
                
                c1.vx = final_vx
                c1.vy = final_vy
                c2.vx = final_vx
                c2.vy = final_vy
                return

            j = -(1 + self.restitution) * vn
            j /= total_inv_mass
            c1.vx -= j * nx * inv_m1
            c1.vy -= j * ny * inv_m1
            c2.vx += j * nx * inv_m2
            c2.vy += j * ny * inv_m2

    def _enforce_bounds(self):
        for ent in self.agents + self.payloads:
            if ent.x < ent.radius: ent.x = ent.radius; ent.vx *= -0.5
            if ent.x > self.width - ent.radius: ent.x = self.width - ent.radius; ent.vx *= -0.5
            if ent.y < ent.radius: ent.y = ent.radius; ent.vy *= -0.5
            if ent.y > self.height - ent.radius: ent.y = self.height - ent.radius; ent.vy *= -0.5

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        
        goals = [o for o in self.obstacles if o.type == "goal"]
        
        for i, agent in enumerate(self.agents):
            idx = 0
            for sensor in self.sensors:
                if sensor == "position":
                    obs[i, idx] = (agent.x / self.width) * 2 - 1
                    obs[i, idx+1] = (agent.y / self.height) * 2 - 1
                    idx += 2
                elif sensor == "velocity":
                    obs[i, idx] = np.clip(agent.vx, -1, 1)
                    obs[i, idx+1] = np.clip(agent.vy, -1, 1)
                    idx += 2
                elif sensor == "goal_vector":
                    if goals:
                        # Find nearest goal
                        g = min(goals, key=lambda o: (o.x-agent.x)**2 + (o.y-agent.y)**2)
                        dx, dy = g.x - agent.x, g.y - agent.y
                        dist = math.sqrt(dx*dx + dy*dy) + 0.001
                        obs[i, idx] = dx / dist
                        obs[i, idx+1] = dy / dist
                    idx += 2
                elif sensor == "neighbor_vectors":
                    # 3 nearest
                    others = [(a, math.sqrt((a.x-agent.x)**2 + (a.y-agent.y)**2)) 
                              for a in self.agents if a.id != agent.id]
                    others.sort(key=lambda x: x[1])
                    for k in range(3):
                        if k < len(others):
                            oa, dist = others[k]
                            if dist == 0: dist = 0.001
                            obs[i, idx+k*2] = (oa.x - agent.x) / 50.0 # Norm by some range
                            obs[i, idx+k*2+1] = (oa.y - agent.y) / 50.0
                    idx += 6
                elif sensor == "obstacle_radar":
                    # 8 rays
                    radar = self._compute_radar(agent, [o for o in self.obstacles if o.type=="obstacle"])
                    obs[i, idx:idx+8] = radar
                    idx += 8
                elif sensor == "neighbor_signals":
                    others = [(j, math.sqrt((self.agents[j].x-agent.x)**2 + (self.agents[j].y-agent.y)**2)) 
                              for j in range(self.num_agents) if j != i]
                    others.sort(key=lambda x: x[1])
                    for k in range(3):
                        if k < len(others):
                            idx_other, dist = others[k]
                            
                            # Range Check
                            if dist > self.comm_range:
                                obs[i, idx+k] = 0.0
                                continue
                                
                            # Packet Loss Check
                            if self.packet_loss_prob > 0 and np.random.random() < self.packet_loss_prob:
                                obs[i, idx+k] = 0.0
                                continue
                            
                            obs[i, idx+k] = self.comm_signals[idx_other]
                    idx += 3
                elif sensor == "grabbing_state":
                    obs[i, idx] = 1.0 if agent.is_grabbing else 0.0
                    idx += 1
                elif sensor == "energy":
                    obs[i, idx] = agent.energy
                    idx += 1
        
        # Add Noise (Interference)
        if self.obs_noise_std > 0:
            noise = np.random.normal(0, self.obs_noise_std, obs.shape)
            obs += noise
            
        return obs

    def _compute_radar(self, agent, targets):
        """
        Simulate LiDAR with 8 rays. Raycast against all rigid targets (obstacles).
        Returns distance to ANY first hit (normalized 0..1).
        """
        radar = np.ones(8, dtype=np.float32)
        max_range = 30.0
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        
        # Define rays in world space
        rays = []
        for ang in angles:
             dx = math.cos(ang)
             dy = math.sin(ang)
             rays.append((dx, dy))
             
        # For each ray, find nearest intersection
        for i, (rdx, rdy) in enumerate(rays):
             closest_dist = max_range
             
             for t in targets:
                  # Check if target is close enough to potentially hit
                  # (Optimization: simple dist check first)
                  dist_sq = (t.x - agent.x)**2 + (t.y - agent.y)**2
                  # If target is further than max_range + radius, skip
                  if dist_sq > (max_range + t.radius)**2: 
                      continue
                      
                  hit_dist = self._intersect_ray_circle(agent.x, agent.y, rdx, rdy, t.x, t.y, t.radius)
                  if hit_dist is not None and hit_dist < closest_dist:
                      closest_dist = hit_dist
             
             radar[i] = closest_dist / max_range

        return radar

    def _intersect_ray_circle(self, ox, oy, dx, dy, cx, cy, r):
        """
        Ray-Circle Intersection.
        Ray: P = O + t*D
        Circle: |P - C|^2 = r^2
        """
        fx = ox - cx
        fy = oy - cy
        
        a = dx*dx + dy*dy # Should be 1 if normalized
        b = 2 * (fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - r*r
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return None
            
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        # We want smallest positive t
        t = None
        if t1 >= 0: t = t1
        if t2 >= 0 and (t is None or t2 < t): t = t2
        
        return t

    def _compute_rewards(self, current_actions=None):
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            try:
                # Add action context to agent dict for reward function
                agent_dict = agent.to_dict()
                if current_actions is not None and self.action_type == "continuous":
                     agent_dict['action'] = current_actions[i]
                     agent_dict['last_action'] = self.last_actions[i]
                
                if self.enable_communication:
                    agent_dict['comm'] = self.comm_signals[i]

                env_state = {
                    "agent_idx": i,
                    "agents": [a.to_dict() for a in self.agents],
                    "payloads": [p.__dict__ for p in self.payloads],
                    "goals": [o.__dict__ for o in self.obstacles if o.type == "goal"],
                    "obstacles": [o.__dict__ for o in self.obstacles if o.type == "obstacle"],
                    "comm_signals": self.comm_signals
                }
                rewards[i] = self.reward_func(agent_dict, env_state, math, np)
                
                # Visual Penalty Feedback
                # Reset to Blue
                agent.color = (0, 0, 255)
                # If significant penalty, flash Orange
                if rewards[i] < -1.0:
                    agent.color = (255, 165, 0)
                # If critical penalty (stall/crash), flash Red
                if rewards[i] < -9.0:
                    agent.color = (255, 0, 0)
                    
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
        """
        Robust rendering using OpenCV.
        Draws agents, obstacles, goals, and annotations.
        """
        import cv2
        
        scale = 5
        h, w = int(self.height * scale), int(self.width * scale)
        # Background: Dark Gray/Black for contrast
        img = np.zeros((h, w, 3), dtype=np.uint8) + 30 
        
        # Draw Grid (faint)
        grid_spacing = 10 * scale
        for x in range(0, w, grid_spacing):
            cv2.line(img, (x, 0), (x, h), (40, 40, 40), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(img, (0, y), (w, y), (40, 40, 40), 1)

        # Draw Obstacles & Goals
        for obs in self.obstacles:
            cx, cy = int(obs.x * scale), int((self.height - obs.y) * scale)
            r = int(obs.radius * scale)
            if obs.type == "goal":
                # Green filled + lighter outline
                cv2.circle(img, (cx, cy), r, (0, 150, 0), -1)
                cv2.circle(img, (cx, cy), r, (0, 200, 0), 2)
                # Label
                cv2.putText(img, "GOAL", (cx - 15, cy + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Gray Obstacle
                cv2.circle(img, (cx, cy), r, (100, 100, 100), -1)
                cv2.circle(img, (cx, cy), r, (150, 150, 150), 2)

        # Draw Payloads
        for p in self.payloads:
            cx, cy = int(p.x * scale), int((self.height - p.y) * scale)
            r = int(p.radius * scale)
            cv2.circle(img, (cx, cy), r, p.color, -1)
            cv2.circle(img, (cx, cy), r, (255, 255, 255), 1)

        # Draw Agents
        for agent in self.agents:
            cx, cy = int(agent.x * scale), int((self.height - agent.y) * scale)
            r = int(agent.radius * scale)
            
            # Grabbing Aura
            if agent.is_grabbing:
                cv2.circle(img, (cx, cy), r + 3, (255, 255, 255), 1)
                
            # Body
            # Convert color to BGR for OpenCV if tuple is RGB
            # Assuming agent.color is RGB, cv2 uses BGR.
            # My logic set colors as (255, 0, 0) for Red. In BGR that is (0, 0, 255).
            # So I should flip.
            b, g, r_val = agent.color[2], agent.color[1], agent.color[0]
            # Actually Agent default was (0,0,255) in my init edit?
            # Wait, `Agent(..., color=(0,0,255))`.
            # If I treat that as RGB -> Blue is (0,0,255).
            # If I treat as BGR for CV2 -> Red is (0,0,255).
            # Let's standardize: Internal state is RGB.
            # Render converts to BGR.
            
            # Default Blue (0, 0, 255) RGB.
            # Orange (255, 165, 0) RGB.
            # Red (255, 0, 0) RGB.
            
            color_bgr = (agent.color[2], agent.color[1], agent.color[0])
            
            cv2.circle(img, (cx, cy), r, color_bgr, -1)
            
            # Direction Indicator (Velocity)
            vel_mag = math.sqrt(agent.vx**2 + agent.vy**2)
            if vel_mag > 0.1:
                end_x = int(cx + (agent.vx / vel_mag) * r * 1.5)
                end_y = int(cy - (agent.vy / vel_mag) * r * 1.5) # Flip Y for image coords
                cv2.line(img, (cx, cy), (end_x, end_y), (255, 255, 255), 2)
            
            # ID
            # cv2.putText(img, str(agent.id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        # Overlay Info
        cv2.putText(img, f"Step: {self.current_step}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                   
        return img
