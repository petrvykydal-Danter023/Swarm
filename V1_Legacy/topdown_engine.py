import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
import time
from goals import GoalManager

# NOTE: Agent class is REMOVED/Deprecated internal usage.
# Use dicts for backward compatibility in rewards if needed.

@dataclass
class Obstacle:
    """Static circular obstacle"""
    x: float
    y: float
    radius: float
    type: str = "obstacle" # obstacle, goal

@dataclass
class Payload:
    """Heavy movable object"""
    id: int
    x: float
    y: float
    radius: float
    mass: float = 10.0
    vx: float = 0.0
    vy: float = 0.0
    color: tuple = (200, 200, 50) 
    
class TopDownSwarmEnv(gym.Env):
    """
    Entropy Engine V4 (Vectorized): Pure Top-Down Swarm Simulator.
    Optimized for performance using NumPy matrix operations (SIMD).
    """
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, config: dict):
        super().__init__()
        # Flatten config
        self.config = config.copy()
        if "env_params" in config:
            self.config.update(config["env_params"])
            
        # World Params
        self.width = self.config.get("world_width", 100.0)
        self.height = self.config.get("world_height", 100.0)
        self.num_agents = self.config.get("num_agents", 10)
        self.dt = self.config.get("dt", 0.1)
        self.observation_type = self.config.get("observation_type", "spatial") 
        self.agent_mode = self.config.get("agent_mode", "nav")
        self.detection_radius = self.config.get("detection_radius", 100.0)
        
        # Physics Params
        self.friction = self.config.get("friction", 0.2)
        self.restitution = self.config.get("restitution", 0.5)
        self.payload_friction = self.config.get("payload_friction", 0.1)
        
        # Sensory & Comms
        self.sensors = self.config.get("sensors", ["position", "velocity", "goal_vector", "neighbor_vectors"])
        self.enable_communication = self.config.get("enable_communication", False)
        self.obs_noise_std = self.config.get("obs_noise_std", 0.0)
        self.comm_range = self.config.get("comm_range", 1000.0) 
        self.packet_loss_prob = self.config.get("packet_loss_prob", 0.1)
        
        # --- VECTORIZED STATE ---
        # Position (N, 2)
        self.pos = np.zeros((self.num_agents, 2), dtype=np.float32)
        # Velocity (N, 2)
        self.vel = np.zeros((self.num_agents, 2), dtype=np.float32)
        # State: [radius, mass, energy, is_grabbing] -> Actually handle separaretely for clarity
        self.radius = np.full(self.num_agents, 2.0, dtype=np.float32)
        self.mass = np.full(self.num_agents, 1.0, dtype=np.float32)
        self.energy = np.full(self.num_agents, 1.0, dtype=np.float32)
        self.is_grabbing = np.zeros(self.num_agents, dtype=bool)
        self.colors = np.full((self.num_agents, 3), [100, 150, 255], dtype=np.int32)
        self.ids = np.arange(self.num_agents)
        
        self.obstacles: List[Obstacle] = []
        self.payloads: List[Payload] = [] # Payloads kept as objects for now (usually low count)
        
        # Goal Manager
        self.goal_manager = GoalManager(self.config)
        
        # Comm State (N,)
        self.comm_signals = np.zeros(self.num_agents, dtype=np.float32)
        
        # Calculate Obs Dim
        self.obs_dim = self._calculate_obs_dim()
        
        # Gym Spaces
        self.action_type = config.get("action_type", "continuous")
        self.action_space_type = self.action_type
        if self.action_type == "continuous":
            # [vx, vy, grab] + [comm]
            act_dim = 3 + (1 if self.enable_communication else 0)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, act_dim), dtype=np.float32
            )
            # Motor Lag (Buffer)
            self.motor_lag = config.get("motor_lag", 0) 
            # Vectorized queueing is hard, simpler to just store history tensor?
            # For simplicity in V4 vectorized: Use deque of (N, act_dim) arrays
            self.action_queue = deque(maxlen=self.motor_lag + 1)
            # Fill with zeros
            if self.motor_lag > 0:
                for _ in range(self.motor_lag):
                    self.action_queue.append(np.zeros((self.num_agents, act_dim), dtype=np.float32))
            
            self.last_actions = np.zeros((self.num_agents, act_dim), dtype=np.float32)

        else:
            self.action_space = spaces.MultiDiscrete([6] * self.num_agents)
            self.motor_lag = 0 
            self.action_queue = None
            self.last_actions = None 

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_agents, self.obs_dim), dtype=np.float32
        )
        
        self.reward_func = self._compile_reward(config.get("reward_code", "reward = 0.0"))
        self.reward_type = self.config.get("reward_type", "custom")  # "vectorized" or "custom"
        
        self._build_world()
        self.current_step = 0
        self.max_steps = config.get("max_steps", 500)

    def _calculate_obs_dim(self):
        dim = 0
        for s in self.sensors:
            if s == "position": dim += 2
            elif s == "velocity": dim += 2
            elif s == "goal_vector": dim += 2
            elif s == "neighbor_vectors": dim += 6 
            elif s == "obstacle_radar": dim += 8
            elif s == "payload_radar": dim += 8 # Not impl vectorized yet
            elif s == "grabbing_state": dim += 1
            elif s == "energy": dim += 1
            elif s == "neighbor_signals": dim += 3 
        return dim

    def _build_world(self):
        """Initialize agents arrays."""
        spawn = self.config.get("spawn_zone", {"x": 10, "y": 10, "w": 10, "h": 80})
        
        # Vectorized Spawn
        self.pos[:, 0] = np.random.uniform(spawn["x"], spawn["x"] + spawn["w"], self.num_agents)
        self.pos[:, 1] = np.random.uniform(spawn["y"], spawn["y"] + spawn["h"], self.num_agents)
        self.vel.fill(0.0)
        self.energy.fill(1.0)
        self.is_grabbing.fill(False)
            
        self.obstacles = []
        self.payloads = []
        for obj in self.config.get("special_objects", []):
            if obj["type"] == "payload":
                self.payloads.append(Payload(
                    id=len(self.payloads),
                    x=obj["x"], y=obj["y"], radius=obj.get("radius", 5), mass=obj.get("mass", 5.0)
                ))
            elif obj["type"] == "goal":
                pass
            else:
                self.obstacles.append(Obstacle(
                    obj["x"], obj["y"], obj.get("radius", 5), obj["type"]
                ))
        
        self.goal_manager._parse_goals(self.config)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        self._build_world()
        self._reset_queues() 
        self.current_step = 0
        return self._get_obs(), {}

    def _reset_queues(self):
        if self.motor_lag > 0 and self.action_type == "continuous":
             act_dim = 3 + (1 if self.enable_communication else 0)
             self.action_queue.clear()
             for _ in range(self.motor_lag):
                  self.action_queue.append(np.zeros((self.num_agents, act_dim), dtype=np.float32))

    def step(self, actions):
        self.current_step += 1
        
        # Ensure actions is array (N, dim)
        actions = np.array(actions, dtype=np.float32)
        if len(actions.shape) == 1 and self.action_type == "continuous":
             actions = actions.reshape(self.num_agents, -1)
        
        applied_actions = actions
        if self.motor_lag > 0 and self.action_type == "continuous":
             self.action_queue.append(actions)
             old_act = self.action_queue.popleft()
             applied_actions = old_act

        # Split actions
        # applied_actions: (N, 3 or 4) -> vx, vy, grab, [comm]
        if self.action_type == "continuous":
            move_actions = applied_actions[:, :2] # vx, vy
            grab_actions = applied_actions[:, 2]  # grab
            if self.enable_communication:
                self.comm_signals = applied_actions[:, 3]
            
            # Control Application
            # Energy Penalty (Vectorized)
            low_energy_mask = self.energy < 0.05
            move_actions[low_energy_mask] *= 0.2
            
            speed = 1.0 # scalar
            self.vel += move_actions * speed * self.dt
            
            # Grabbing
            self.is_grabbing = grab_actions > 0.5
            
            # Energy Consumtion
            current_speed = np.linalg.norm(self.vel, axis=1) # (N,)
            drain = current_speed * 0.005
            drain += (self.is_grabbing.astype(float) * 0.002)
            if self.enable_communication:
                drain += np.abs(self.comm_signals) * 0.001
            
            self.energy = np.clip(self.energy - drain + 0.001, 0.0, 1.0)
            
            self.last_actions = applied_actions.copy()

        # Physics Integration (SIMD)
        # Friction
        self.vel *= (1.0 - self.friction)
        # Position
        self.pos += self.vel
        # Payloads (Object Loop)
        for p in self.payloads:
            p.vx *= (1.0 - self.payload_friction); p.vy *= (1.0 - self.payload_friction)
            p.x += p.vx; p.y += p.vy
            
        # Collision (Vectorized Agents vs Obstacles)
        self._solve_collisions_vectorized()
        
        # Bounds (Vectorized)
        # Clip pos, reflect vel
        # Left/Right
        mask_l = self.pos[:, 0] < self.radius
        mask_r = self.pos[:, 0] > self.width - self.radius
        self.pos[mask_l, 0] = self.radius[mask_l]; self.vel[mask_l, 0] *= -0.5
        self.pos[mask_r, 0] = self.width - self.radius[mask_r]; self.vel[mask_r, 0] *= -0.5
        # Top/Bottom
        mask_t = self.pos[:, 1] < self.radius
        mask_b = self.pos[:, 1] > self.height - self.radius
        self.pos[mask_t, 1] = self.radius[mask_t]; self.vel[mask_t, 1] *= -0.5
        self.pos[mask_b, 1] = self.height - self.radius[mask_b]; self.vel[mask_b, 1] *= -0.5
        
        # Goals Update
        self.goal_manager.update(self.dt)
        
        # Rewards (Dual-mode: fast vectorized or custom exec)
        if self.reward_type == "vectorized":
            rewards = self._compute_rewards_fast()
        else:
            rewards = self._compute_rewards_vectorized(applied_actions)
        
        return self._get_obs(), rewards, False, self.current_step >= self.max_steps, {}

    def _solve_collisions_vectorized(self):
        # 1. Agent-Agent (N^2 distance matrix)
        # D[i,j] = dist(i, j)
        # diffs: (N, 1, 2) - (1, N, 2) = (N, N, 2)
        diffs = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Radii sum: (N, 1) + (1, N) = (N, N)
        radii_sum = self.radius[:, np.newaxis] + self.radius[np.newaxis, :]
        
        # Overlap mask: dists < radii_sum AND dists > 0 (avoid self)
        overlap = radii_sum - dists
        mask = (dists < radii_sum) & (dists > 0.0001)
        
        # We only need upper triangle to avoid double counting
        # mask = np.triu(mask, k=1) 
        # Actually for simple impulse it might be better to iterate pairs found by mask
        # indices = np.argwhere(mask) => (K, 2) pairs
        
        # Only process upper triangle
        indices = np.transpose(np.nonzero(np.triu(mask, k=1)))
        
        for idx in indices:
            i, j = idx[0], idx[1]
            # Resolve i, j
            # Physics (Elastic)
            # Normal
            d = dists[i, j]
            nx, ny = diffs[i, j] / d
            o = overlap[i, j]
            
            # Move apart (0.5 each)
            self.pos[i] -= np.array([nx, ny]) * o * 0.5
            self.pos[j] += np.array([nx, ny]) * o * 0.5
            
            # Velocity
            dvx = self.vel[j, 0] - self.vel[i, 0]
            dvy = self.vel[j, 1] - self.vel[i, 1]
            vn = dvx * nx + dvy * ny
            if vn > 0: continue
            
            # Impulse
            impulse = -(1 + self.restitution) * vn
            impulse /= 2.0 # Assume equal mass 1.0
            
            self.vel[i, 0] -= impulse * nx
            self.vel[i, 1] -= impulse * ny
            self.vel[j, 0] += impulse * nx
            self.vel[j, 1] += impulse * ny
            
        # 2. Agent-Obstacle (N * M)
        # Loop over obstacles (usually few)
        for obs in self.obstacles:
            # Vectorized check against one obstacle
            # diff: (N, 2) - (1, 2)
            d_obs = self.pos - np.array([obs.x, obs.y])
            dist_obs = np.linalg.norm(d_obs, axis=1)
            min_dist = self.radius + obs.radius
            
            mask_obs = dist_obs < min_dist
            
            if np.any(mask_obs):
                # Resolve
                # (K, 2)
                d_vec = d_obs[mask_obs]
                d = dist_obs[mask_obs, np.newaxis]
                d = np.maximum(d, 0.001)
                norms = d_vec / d
                overs = (min_dist[mask_obs] - dist_obs[mask_obs])[:, np.newaxis]
                
                # Move agents out
                self.pos[mask_obs] += norms * overs
                
                # Reflect velocity
                # v - 2*(v.n)*n   (Actually bounce logic: vn = -e * vn)
                vels = self.vel[mask_obs]
                # dot product (v . n)
                vns = np.sum(vels * norms, axis=1, keepdims=True)
                # Only reflect if moving towards
                mask_towards = (vns < 0).flatten()
                
                # Apply reflection to subset moving towards
                # J = -(1+e)*vn
                # new_v = old_v + J*n
                # Simplified reflection: v_new = v - (1+e)*(v.n)*n
                
                # We need to update self.vel[mask_obs] where mask_towards is true
                # It's tricky with double masking.
                
                # Simple loop for colliders might be safer/cleaner or use indices
                idxs = np.where(mask_obs)[0]
                for k in range(len(idxs)):
                    idx = idxs[k]
                    if vns[k] < 0: # moving towards
                         self.vel[idx] -= (1 + self.restitution) * vns[k] * norms[k]

        # 3. Agent-Payload (Dynamic collision with push)
        for p in self.payloads:
            # Vectorized check against one payload
            d_pay = self.pos - np.array([p.x, p.y])
            dist_pay = np.linalg.norm(d_pay, axis=1)
            min_dist = self.radius + p.radius
            
            mask_pay = dist_pay < min_dist
            
            if np.any(mask_pay):
                idxs = np.where(mask_pay)[0]
                for k in idxs:
                    # Resolve single agent-payload collision
                    d = dist_pay[k]
                    if d < 0.001: d = 0.001
                    nx, ny = d_pay[k, 0] / d, d_pay[k, 1] / d
                    overlap = min_dist[k] - d
                    
                    # Mass ratio for separation
                    m_agent = self.mass[k]
                    m_payload = p.mass
                    total_mass = m_agent + m_payload
                    
                    # Separate based on mass
                    self.pos[k, 0] += nx * overlap * (m_payload / total_mass)
                    self.pos[k, 1] += ny * overlap * (m_payload / total_mass)
                    p.x -= nx * overlap * (m_agent / total_mass)
                    p.y -= ny * overlap * (m_agent / total_mass)
                    
                    # Velocity exchange (elastic with grabbing = inelastic)
                    dvx = p.vx - self.vel[k, 0]
                    dvy = p.vy - self.vel[k, 1]
                    vn = dvx * nx + dvy * ny
                    
                    if vn > 0: continue # separating
                    
                    # Check grabbing for inelastic
                    if self.is_grabbing[k]:
                        # Inelastic: shared velocity
                        final_vx = (m_agent * self.vel[k, 0] + m_payload * p.vx) / total_mass
                        final_vy = (m_agent * self.vel[k, 1] + m_payload * p.vy) / total_mass
                        self.vel[k, 0] = final_vx
                        self.vel[k, 1] = final_vy
                        p.vx = final_vx
                        p.vy = final_vy
                    else:
                        # Elastic impulse
                        j = -(1 + self.restitution) * vn / (1/m_agent + 1/m_payload)
                        self.vel[k, 0] -= j * nx / m_agent
                        self.vel[k, 1] -= j * ny / m_agent
                        p.vx += j * nx / m_payload
                        p.vy += j * ny / m_payload

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        col = 0
        
        for s in self.sensors:
            if s == "position":
                # Normalize [0, W] -> [-1, 1]
                obs[:, col] = (self.pos[:, 0] / self.width) * 2 - 1
                obs[:, col+1] = (self.pos[:, 1] / self.height) * 2 - 1
                col += 2
            elif s == "velocity":
                 obs[:, col:col+2] = np.clip(self.vel, -1, 1)
                 col += 2
            elif s == "goal_vector":
                 obs[:, col:col+2] = self.goal_manager.get_batch_observation(
                     self.pos, self.agent_mode, self.detection_radius
                 )
                 col += 2
            elif s == "neighbor_vectors":
                 # Vectorized K-Nearest
                 # Diffs: (N, N, 2)
                 diffs = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :] # Target - Self
                 dists = np.linalg.norm(diffs, axis=2)
                 dists[np.arange(self.num_agents), np.arange(self.num_agents)] = 9999.0 # Self is far
                 
                 # Argpartition to get indices of K nearest
                 # We want 3 neighbors.
                 # (N, 3) indices
                 k = 3
                 nearest_indices = np.argpartition(dists, k, axis=1)[:, :k]
                 
                 # Gather vectors
                 # diffs is (N, N, 2). We want diffs[i, nearest_indices[i], :]
                 # Advanced indexing
                 row_indices = np.arange(self.num_agents)[:, np.newaxis]
                 vecs = diffs[row_indices, nearest_indices, :] # (N, 3, 2)
                 
                 # Normalize by 50.0 range
                 vecs /= 50.0
                 obs[:, col:col+6] = vecs.reshape(self.num_agents, 6)
                 col += 6
                 
            elif s == "obstacle_radar":
                # Fallback to loop for radar for now (complex geometry)
                # But parallelized by N agents
                # Or simplify: Just dist to nearest obstacle?
                # Using 8-ray logic in python loop for N agents is better than Python Object loop but still slow.
                # JIT would be best here.
                # For compatibility, keeping loop but optimized access
                obs[:, col:col+8] = self._compute_radar_batch()
                col += 8
                
            elif s == "neighbor_signals":
                 # Use Distance Matrix calculated above?? No, expensive to store/pass
                 # Re-calc dists
                 diffs = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
                 dists = np.linalg.norm(diffs, axis=2)
                 np.fill_diagonal(dists, 9999.0)
                 
                 nearest_indices = np.argsort(dists, axis=1)[:, :3]
                 
                 # Get signals
                 # signals: (N,)
                 # neighbor_sigs: (N, 3)
                 sigs = self.comm_signals[nearest_indices]
                 
                 # Range mask
                 neighbor_dists = dists[np.arange(self.num_agents)[:, np.newaxis], nearest_indices]
                 mask = neighbor_dists <= self.comm_range
                 if self.packet_loss_prob > 0:
                      drop_mask = np.random.random(mask.shape) > self.packet_loss_prob
                      mask &= drop_mask
                 
                 sigs *= mask
                 obs[:, col:col+3] = sigs
                 col += 3
                 
            elif s == "energy":
                obs[:, col] = self.energy
                col += 1
        
        return obs

    def _compute_radar_batch(self):
        # Semi-optimized radar
        # (N, 8)
        radar = np.ones((self.num_agents, 8), dtype=np.float32)
        # Angles
        angles = np.array([0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4])
        # Rays: (8, 2)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        
        max_range = 30.0
        
        # We check circle intersection for each agent ray against obstacles
        # To vectorise fully: (N_agents, 8_rays, M_obstacles)
        # That's big (100 * 8 * 10 = 8000 checks). Doable in NumPy.
        
        # Agent Pos: (N, 1, 2)
        # Obstacle Pos: (1, M, 2)
        # Diff: (N, M, 2)
        if not self.obstacles:
             return radar
        
        obs_pos = np.array([[o.x, o.y] for o in self.obstacles])
        obs_rad = np.array([o.radius for o in self.obstacles])
        
        # Vectorized Ray Sphere Intersection
        # O = agent_pos (N, 1, 2)
        # D = ray_dirs (1, 8, 2) -> (1, 8, 1, 2) ?? No
        # We need (N, 8, M) intersection times
        
        # Let's loop Rays (8 times) and do (N, M) check. Faster than loop N.
        
        # P = O - C (Agent to Obstacle) : (N, M, 2)
        P = self.pos[:, np.newaxis, :] - obs_pos[np.newaxis, :, :]
        
        # Iterate 8 rays
        for i in range(8):
             dx, dy = ray_dirs[i]
             # Project P onto D
             # b = 2 * (P.x * dx + P.y * dy)
             b = 2 * (P[:, :, 0] * dx + P[:, :, 1] * dy) # (N, M)
             # c = |P|^2 - r^2
             c = np.sum(P**2, axis=2) - obs_rad**2 # (N, M)
             # a = 1
             
             delta = b**2 - 4*c
             
             # Valid hits where delta >= 0
             mask = delta >= 0
             
             # t = (-b - sqrt(delta)) / 2
             # We initialize t with max_range
             
             # Use where to calc
             # We handle mask by setting negative/nan to inf
             safe_delta = np.maximum(delta, 0)
             t = (-b - np.sqrt(safe_delta)) / 2.0
             
             # t must be > 0 and < max_range
             # also apply mask
             valid = mask & (t > 0) & (t < max_range)
             
             # For each agent, finding min t across all obstacles
             # set invalid t to infinity
             t_final = np.where(valid, t, np.inf)
             
             # Min over obstacles (axis 1)
             min_t = np.min(t_final, axis=1)
             
             # Normalize
             radar[:, i] = np.minimum(min_t, max_range) / max_range
             
        return radar

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

    def _compute_rewards_vectorized(self, current_actions):
        # We need to constructing the 'env_state' and 'agent_dict' for compatibility
        # because the User's Reward Code expects dictionaries.
        # This is the BOTTLENECK now inside the loop.
        # Ideally we convert User Code to Vectorized Code, but that's hard.
        # We will keep the Loop for Rewards but rely on the fact that Physics was fast.
        # To truly optimize, we need Vectorized Reward Function input.
        
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        # Pre-construct env_state partial
        base_state = {
             "world_width": self.width,
             "world_height": self.height,
             "goals": [g.__dict__ for g in self.goal_manager.goals],
             "obstacles": [o.__dict__ for o in self.obstacles],
             "comm_signals": self.comm_signals
        }
        
        # Loop is necessary for `exec` based dynamic code
        for i in range(self.num_agents):
             # Construct minimal dict
             # Access array directly
             agent_dict = {
                 "id": i, "x": self.pos[i,0], "y": self.pos[i,1],
                 "vx": self.vel[i,0], "vy": self.vel[i,1],
                 "energy": self.energy[i],
                 "is_grabbing": bool(self.is_grabbing[i])
             }
             if self.action_type == "continuous":
                  agent_dict['action'] = current_actions[i]
                  agent_dict['last_action'] = self.last_actions[i]
                  
             if self.enable_communication:
                  agent_dict['comm'] = self.comm_signals[i]
             
             # Env State copy?? No, reused list reference is dangerous if modified? 
             # Reward func usually reads.
             # We can optimize obstacles list creation (it's static usually)
             rewards[i] = self.reward_func(agent_dict, base_state, math, np)
             
             # Visual Color Logic (Vectorized post-loop?)
             # Just set colors
             if rewards[i] < -8.0: self.colors[i] = [255, 0, 0]
             elif rewards[i] < -3.5: self.colors[i] = [255, 165, 0]
             else: self.colors[i] = [0, 0, 255]
             
        return rewards

    def _compute_rewards_fast(self):
        """
        Fully vectorized reward computation. NO Python loops.
        Uses NumPy broadcasting for maximum speed.
        """
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        # Goal-based reward (vectorized)
        if self.goal_manager.goals:
            goal = self.goal_manager.goals[0]
            goal_pos = np.array([goal.x, goal.y])
            
            # Distance to goal for ALL agents (N,)
            dists = np.linalg.norm(self.pos - goal_pos, axis=1)
            
            # 1. Distance Penalty (closer = less penalty)
            rewards -= dists / 50.0
            
            # 2. Goal Reached Bonus
            rewards += (dists < goal.radius) * 20.0
            
            # 3. Velocity Alignment (reward moving towards goal)
            to_goal = goal_pos - self.pos  # (N, 2)
            to_goal_norm = to_goal / (np.linalg.norm(to_goal, axis=1, keepdims=True) + 0.001)
            alignment = np.sum(self.vel * to_goal_norm, axis=1)  # dot product (N,)
            rewards += alignment * 0.5
        
        # Wall Penalty (vectorized)
        margin = 5.0
        wall_penalty = np.zeros(self.num_agents, dtype=np.float32)
        wall_penalty += (self.pos[:, 0] < margin) * 2.0
        wall_penalty += (self.pos[:, 0] > self.width - margin) * 2.0
        wall_penalty += (self.pos[:, 1] < margin) * 2.0
        wall_penalty += (self.pos[:, 1] > self.height - margin) * 2.0
        rewards -= wall_penalty
        
        # Speed Penalty (encourage efficiency)
        speed = np.linalg.norm(self.vel, axis=1)
        rewards -= speed * 0.02
        
        # Payload pushing bonus (if payloads exist)
        for p in self.payloads:
            payload_pos = np.array([p.x, p.y])
            agent_to_payload = np.linalg.norm(self.pos - payload_pos, axis=1)
            
            # Bonus for being close to payload
            rewards += (agent_to_payload < 15) * 0.5
            rewards += (agent_to_payload < 8) * 1.0
            
            # Payload to goal distance
            if self.goal_manager.goals:
                goal = self.goal_manager.goals[0]
                payload_to_goal = np.sqrt((p.x - goal.x)**2 + (p.y - goal.y)**2)
                rewards -= payload_to_goal / 30.0  # Everyone shares payload progress
                
                # Big bonus if payload reaches goal
                if payload_to_goal < goal.radius:
                    rewards += 50.0
        
        # Vectorized color update
        self.colors[:] = [0, 0, 255]  # Default blue
        self.colors[rewards < -3.5] = [255, 165, 0]  # Orange
        self.colors[rewards < -8.0] = [255, 0, 0]  # Red
        
        return rewards

    def render(self, mode="rgb_array"):
         import cv2
         scale = 5
         h, w = int(self.height * scale), int(self.width * scale)
         img = np.zeros((h, w, 3), dtype=np.uint8) + 20 # Dark BG
         
         # Goals
         for g in self.goal_manager.goals:
             cv2.circle(img, (int(g.x*scale), int(g.y*scale)), int(g.radius*scale), (0, 255, 0), -1)
             
         # Obstacles
         for o in self.obstacles:
              cv2.circle(img, (int(o.x*scale), int(o.y*scale)), int(o.radius*scale), (100, 100, 100), -1)
         
         # Payloads (Yellow)
         for p in self.payloads:
              cv2.circle(img, (int(p.x*scale), int(p.y*scale)), int(p.radius*scale), (0, 200, 200), -1) # Yellow in BGR
              
         # Agents
         for i in range(self.num_agents):
              cx, cy = int(self.pos[i,0]*scale), int(self.pos[i,1]*scale)
              clr = (int(self.colors[i][0]), int(self.colors[i][1]), int(self.colors[i][2]))
              # CV2 uses BGR
              clr_bgr = (clr[2], clr[1], clr[0])
              cv2.circle(img, (cx, cy), int(2.0*scale), clr_bgr, -1)
              # Dir
              ex = int(cx + self.vel[i,0]*scale*2)
              ey = int(cy + self.vel[i,1]*scale*2)
              cv2.line(img, (cx, cy), (ex, ey), (200, 200, 200), 1)
              
         return img

