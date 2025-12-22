"""
POLYMORPH Engine - Universal Swarm Environment

A universal multi-agent simulation environment for reinforcement learning.
Supports spatial (2D physics) and grid-based observation modes with
configurable action spaces and dynamic reward functions.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class Agent:
    """Represents a single agent in the simulation."""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    comm_signal: float = 0.0
    energy: float = 1.0 # Battery/Stamina (0.0 to 1.0)
    is_grabbing: bool = False
    is_grounded: bool = False
    role: str = "default"
    
    # Physics V2 properties
    radius: float = 2.0  # Physical size
    mass: float = 1.0    # Mass for collisions
    active_constraints: list = field(default_factory=list)  # IDs of agents I'm holding
    
    def to_dict(self) -> dict:
        """Convert agent state to dictionary for reward code."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "is_grabbing": self.is_grabbing,
            "is_grounded": self.is_grounded,
            "radius": self.radius,
            "mass": self.mass,
            "role": self.role,
        }


@dataclass
class SpecialObject:
    """Represents a special object in the environment (goal, obstacle, gap)."""
    type: str
    x: float = 0.0
    y: float = 0.0
    x1: float = 0.0
    x2: float = 0.0
    radius: float = 0.0
    width: float = 0.0
    height: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert object to dictionary."""
        return {
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "x1": self.x1,
            "x2": self.x2,
            "radius": self.radius,
            "width": self.width,
            "height": self.height,
        }


class UniversalSwarmEnv(gym.Env):
    """
    Universal Multi-Agent Swarm Environment for Reinforcement Learning.
    
    Supports:
    - Observation types: "spatial" (2D physics) or "grid" (2D matrix)
    - Action types: "continuous" (2D vectors) or "discrete" (directional + grab)
    - Dynamic reward functions via Python code strings
    - Configurable world parameters (size, physics, objects)
    
    The environment is designed for swarm AI where multiple agents share
    a single policy network but receive different observations based on
    their individual positions and surroundings.
    """
    
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}
    
    def __init__(self, config: dict, render_mode: str = "rgb_array"):
        """
        Initialize the environment from configuration dictionary.
        
        Args:
            config: Configuration dictionary with task parameters
            render_mode: Rendering mode ("rgb_array" or "human")
        """
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Parse configuration
        self.task_name = config.get("task_name", "unnamed_task")
        self.description = config.get("description", "")
        self.observation_type: Literal["spatial", "grid"] = config.get("observation_type", "spatial")
        self.action_space_type: Literal["continuous", "discrete"] = config.get("action_space_type", "continuous")
        
        # Environment parameters
        env_params = config.get("env_params", {})
        self.world_width = env_params.get("world_width", 100)
        self.world_height = env_params.get("world_height", 100)
        self.num_agents = env_params.get("num_agents", 10)
        
        # Physics parameters
        physics = env_params.get("physics", {})
        self.gravity_y = physics.get("gravity_y", 0.0)
        self.friction = physics.get("friction", 0.1)
        self.time_step = physics.get("time_step", 0.1)
        
        # Parse special objects
        self.special_objects_config = env_params.get("special_objects", [])
        
        # === CONFIGURABLE SENSORS ===
        # Default sensors if not specified
        default_sensors = ["position", "velocity", "goal_vector", "obstacle_radar"]
        self.sensors = env_params.get("sensors", default_sensors)
        
        # Sensor dimensions (for calculating observation space size)
        self.sensor_dims = {
            "position": 2,           # x, y (normalized)
            "velocity": 2,           # vx, vy
            "goal_vector": 2,        # dx, dy to nearest goal
            "goal_distance": 1,      # distance to nearest goal
            "obstacle_radar": 8,     # 8 directions, distance to nearest obstacle
            "neighbor_count": 1,     # count of nearby agents
            "neighbor_vectors": 6,   # vectors to 3 nearest neighbors (3 * 2)
            "neighbor_density": 4,   # density in 4 quadrants
            "wall_distance": 4,      # distance to 4 walls
            "grabbing_state": 1,     # is_grabbing flag
            "energy": 1,             # battery level
            "time_remaining": 1,     # normalized time left
            "neighbor_signals": 3,   # signals from 3 nearest neighbors
        }
        self.enable_communication = env_params.get("enable_communication", False)
        self.sensor_noise_std = env_params.get("sensor_noise_std", 0.0)
        self.enable_sensor_noise = (self.sensor_noise_std > 0.0)
        self.packet_loss_prob = env_params.get("packet_loss_prob", 0.0)
        self.comm_range = env_params.get("comm_range", 1000.0) # Default infinite
        
        # Calculate total observation dimension
        self.obs_dim = sum(self.sensor_dims.get(s, 0) for s in self.sensors)
        if self.obs_dim == 0:
            self.obs_dim = 10  # fallback
        
        # Compile reward code (Optimized Function)
        reward_code_str = config.get("reward_code", "reward = 0.0")
        self.reward_function = self._compile_reward_function(reward_code_str)
        
        # Initialize agents and objects
        self.agents: list[Agent] = []
        self.special_objects: list[SpecialObject] = []
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Simulation state
        self.current_step = 0
        self.max_steps = config.get("training_params", {}).get("max_episode_steps", 500)
        self.action_repeat = config.get("env_params", {}).get("action_repeat", 1)
        self.constraint_iterations = config.get("env_params", {}).get("physics", {}).get("constraint_iterations", 4)
        self.physics_profile = config.get("env_params", {}).get("physics_profile", "realistic")
        self.constraints = []  # Active constraints (joints)
        
        # Optimization: Spatial Hash
        self.spatial_cell_size = 6.0  # Slightly larger than agent diameter (4.0)
        self.spatial_hash = {}  # (cx, cy) -> [agent_indices]

        # For grid mode: grid state
        self.grid: Optional[np.ndarray] = None
        
        # Rendering parameters
        self.render_scale = 5  # pixels per world unit
    
    def _compile_reward_function(self, code_str: str):
        """Compile reward code string into a callable function."""
        # Wrap user code in a function definition
        indented_code = "\n".join(["    " + line for line in code_str.split("\n")])
        
        wrapper_code = f"""
def generated_reward_func(agent, env_state, math, np):
    reward = 0.0
{indented_code}
    return float(reward)
"""
        # Execute the definition to create the function object
        local_scope = {}
        try:
            exec(wrapper_code, {}, local_scope)
            return local_scope["generated_reward_func"]
        except Exception as e:
            raise ValueError(f"Failed to compile reward code: {e}")
            
    def _setup_spaces(self):
        """Setup observation and action spaces based on config."""
        
        if self.observation_type == "spatial":
            # Use configurable sensor dimensions
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, self.obs_dim), dtype=np.float32
            )
        else:  # grid
            # Grid observation: small window around agent position
            # Grid dimensions from config or defaults
            grid_window = 5  # 5x5 window around agent
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.num_agents, grid_window, grid_window), 
                dtype=np.float32
            )
        
        if self.action_space_type == "continuous":
            # Continuous: 2D movement vector per agent (+1 Comm if enabled)
            action_dim = 3 if self.enable_communication else 2
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_agents, action_dim), dtype=np.float32
            )
        else:  # discrete
            # Discrete: 6 actions per agent (none, up, down, left, right, grab)
            self.action_space = spaces.MultiDiscrete([6] * self.num_agents)
    
    def _build_world(self):
        """Build/reset the world: agents and special objects."""
        
        # Get spawn zone from config (default: entire world)
        spawn_zone = self.config.get("env_params", {}).get("spawn_zone", None)
        if spawn_zone:
            spawn_x1, spawn_x2 = spawn_zone.get("x1", 0), spawn_zone.get("x2", self.world_width)
            spawn_y1, spawn_y2 = spawn_zone.get("y1", 0), spawn_zone.get("y2", self.world_height)
        else:
            spawn_x1, spawn_x2 = 0, self.world_width
            spawn_y1, spawn_y2 = 0, self.world_height
        
        # Get physics parameters
        phys_params = self.config.get("env_params", {}).get("physics", {})
        agent_radius = phys_params.get("agent_radius", 2.0)
        agent_mass = phys_params.get("agent_mass", 1.0)
        
        # Initialize agents in spawn zone
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                id=i,
                x=np.random.uniform(spawn_x1, spawn_x2),
                y=np.random.uniform(spawn_y1, spawn_y2),
                vx=0.0,
                vy=0.0,
                is_grabbing=False,
                radius=agent_radius,
                mass=agent_mass,
            )
            self.agents.append(agent)
        
        # Parse special objects from config
        self.special_objects = []
        for obj_config in self.special_objects_config:
            obj = SpecialObject(
                type=obj_config.get("type", "unknown"),
                x=obj_config.get("x", 0.0),
                y=obj_config.get("y", 0.0),
                x1=obj_config.get("x1", 0.0),
                x2=obj_config.get("x2", 0.0),
                radius=obj_config.get("radius", 1.0),
                width=obj_config.get("width", 0.0),
                height=obj_config.get("height", 0.0),
            )
            self.special_objects.append(obj)
            
        # Optimization: Build Static Spatial Hash for obstacles
        self.static_cell_size = 20.0
        self.static_spatial_hash = {}
        for obj in self.special_objects:
            if obj.type == "obstacle":
                cx = int(obj.x / self.static_cell_size)
                cy = int(obj.y / self.static_cell_size)
                key = (cx, cy)
                if key not in self.static_spatial_hash:
                    self.static_spatial_hash[key] = []
                self.static_spatial_hash[key].append(obj)
        
        # For grid mode: initialize grid
        if self.observation_type == "grid":
            self.grid = np.zeros((self.world_height, self.world_width), dtype=np.float32)
            # Mark special objects on grid
            for obj in self.special_objects:
                if obj.type == "goal":
                    gx, gy = int(obj.x), int(obj.y)
                    if 0 <= gx < self.world_width and 0 <= gy < self.world_height:
                        self.grid[gy, gx] = 1.0  # Goal marker
                elif obj.type == "obstacle":
                    ox, oy = int(obj.x), int(obj.y)
                    r = int(obj.radius)
                    for dy in range(-r, r + 1):
                        for dx in range(-r, r + 1):
                            nx, ny = ox + dx, oy + dy
                            if 0 <= nx < self.world_width and 0 <= ny < self.world_height:
                                self.grid[ny, nx] = -1.0  # Obstacle marker
    
    def _get_obs_all(self) -> np.ndarray:
        """Get observations for all agents."""
        
        if self.observation_type == "spatial":
            return self._get_spatial_obs_all()
        else:
            return self._get_grid_obs_all()
    
    def _get_spatial_obs_all(self) -> np.ndarray:
        """Get spatial observations for all agents using configurable sensors."""
        obs = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        
        # Precompute commonly needed data
        goals = [obj for obj in self.special_objects if obj.type == "goal"]
        obstacles = [obj for obj in self.special_objects if obj.type == "obstacle"]
        
        for i, agent in enumerate(self.agents):
            idx = 0  # Current position in observation vector
            
            for sensor in self.sensors:
                if sensor == "position":
                    # Normalized position [-1, 1]
                    obs[i, idx] = (agent.x / self.world_width) * 2 - 1
                    obs[i, idx + 1] = (agent.y / self.world_height) * 2 - 1
                    idx += 2
                    
                elif sensor == "velocity":
                    # Normalized velocity
                    max_speed = 3.0 # Matches physics max_speed (2.0) + buffer for collisions
                    obs[i, idx] = np.clip(agent.vx / max_speed, -1, 1)
                    obs[i, idx + 1] = np.clip(agent.vy / max_speed, -1, 1)
                    idx += 2
                    
                elif sensor == "goal_vector":
                    # Vector to nearest goal
                    if goals:
                        nearest = min(goals, key=lambda g: (g.x - agent.x)**2 + (g.y - agent.y)**2)
                        obs[i, idx] = np.clip((nearest.x - agent.x) / self.world_width, -1, 1)
                        obs[i, idx + 1] = np.clip((nearest.y - agent.y) / self.world_height, -1, 1)
                    idx += 2
                    
                elif sensor == "goal_distance":
                    # Distance to nearest goal
                    if goals:
                        nearest = min(goals, key=lambda g: (g.x - agent.x)**2 + (g.y - agent.y)**2)
                        dist = math.sqrt((nearest.x - agent.x)**2 + (nearest.y - agent.y)**2)
                        max_dist = math.sqrt(self.world_width**2 + self.world_height**2)
                        obs[i, idx] = 1.0 - np.clip(dist / max_dist, 0, 1)
                    idx += 1
                    
                elif sensor == "obstacle_radar":
                    # 8-direction radar for obstacles
                    radar = self._compute_radar(agent, obstacles)
                    obs[i, idx:idx + 8] = radar
                    idx += 8
                    
                elif sensor == "neighbor_count":
                    # Count of nearby agents
                    count = sum(1 for a in self.agents if a.id != agent.id and 
                                math.sqrt((a.x - agent.x)**2 + (a.y - agent.y)**2) < 15)
                    obs[i, idx] = np.clip(count / 5, 0, 1)
                    idx += 1
                    
                elif sensor == "neighbor_vectors":
                    # Vectors to 3 nearest neighbors
                    others = [(a, math.sqrt((a.x - agent.x)**2 + (a.y - agent.y)**2)) 
                              for a in self.agents if a.id != agent.id]
                    others.sort(key=lambda x: x[1])
                    for j in range(3):
                        if j < len(others):
                            a, _ = others[j]
                            obs[i, idx + j*2] = np.clip((a.x - agent.x) / 20, -1, 1)
                            obs[i, idx + j*2 + 1] = np.clip((a.y - agent.y) / 20, -1, 1)
                    idx += 6
                    
                elif sensor == "neighbor_density":
                    # Density in 4 quadrants (NE, NW, SW, SE)
                    for qx, qy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
                        count = sum(1 for a in self.agents if a.id != agent.id and
                                    ((a.x - agent.x) * qx > 0) and ((a.y - agent.y) * qy > 0) and
                                    math.sqrt((a.x - agent.x)**2 + (a.y - agent.y)**2) < 20)
                        obs[i, idx] = np.clip(count / 3, 0, 1)
                        idx += 1
                    
                elif sensor == "wall_distance":
                    # Distance to 4 walls (normalized)
                    obs[i, idx] = agent.x / self.world_width       # left wall
                    obs[i, idx + 1] = 1 - agent.x / self.world_width  # right
                    obs[i, idx + 2] = agent.y / self.world_height     # bottom
                    obs[i, idx + 3] = 1 - agent.y / self.world_height # top
                    idx += 4
                    
                elif sensor == "grabbing_state":
                    obs[i, idx] = 1.0 if agent.is_grabbing else 0.0
                    idx += 1
                    
                elif sensor == "energy":
                    obs[i, idx] = agent.energy
                    idx += 1
                    
                elif sensor == "time_remaining":
                    obs[i, idx] = 1.0 - (self.current_step / self.max_steps)
                    idx += 1
                    
                elif sensor == "neighbor_signals":
                    # Signals from 3 nearest neighbors
                    others = [(a, math.sqrt((a.x - agent.x)**2 + (a.y - agent.y)**2)) 
                              for a in self.agents if a.id != agent.id]
                    others.sort(key=lambda x: x[1])
                    for j in range(3):
                        if j < len(others):
                            a, dist = others[j]
                            
                            # Range Check
                            if dist > self.comm_range:
                                obs[i, idx + j] = 0.0
                                continue
                            
                            # Packet Loss Check
                            if self.packet_loss_prob > 0.0 and np.random.random() < self.packet_loss_prob:
                                obs[i, idx + j] = 0.0
                            else:
                                obs[i, idx + j] = np.clip(a.comm_signal, -1, 1)
                        else:
                            obs[i, idx + j] = 0.0
                    idx += 3
        
        # Add Sensor Noise (Gaussian)
        if self.enable_sensor_noise:
            noise = np.random.normal(0.0, self.sensor_noise_std, size=obs.shape).astype(np.float32)
            obs += noise
            # Clip to valid range? Ideally sensors are [-1, 1], noise can exceed.
            # We let it exceed, network handles it. Simulates wild sensor spikes.
            
        return obs
    
    def _compute_radar(self, agent, obstacles) -> np.ndarray:
        """Compute 8-direction radar for obstacles using Spatial Hash."""
        radar = np.ones(8, dtype=np.float32)  # Default: max distance (1.0)
        # Performance/Design Choice: Radar range limited to 30.0 units
        # This allows O(1) lookup in spatial hash instead of O(N) scan of world
        max_range = 30.0 
        
        # 8 directions: 0, 45, 90, 135, 180, 225, 270, 315 degrees
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 
                  5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        
        # Determine relevant grid cells (3x3 around agent's cell is usually enough for 20.0 cell size and 30.0 range)
        # Actually range 30.0 with cell 20.0 requires 3x3 or 4x4 check.
        # Simple approach: Check cells in spiral
        ax, ay = int(agent.x / self.static_cell_size), int(agent.y / self.static_cell_size)
        
        nearby_obstacles = []
        # Check 3x3 neighborhood (covers 60x60 area centered on agent)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                 key = (ax + dx, ay + dy)
                 if key in self.static_spatial_hash:
                     nearby_obstacles.extend(self.static_spatial_hash[key])
        
        for obstacle in nearby_obstacles:
            dx = obstacle.x - agent.x
            dy = obstacle.y - agent.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < max_range:
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                
                # Find nearest direction bin
                for d in range(8):
                    angle_diff = abs(angles[d] - angle)
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                    if angle_diff < math.pi/4:  # Within 22.5 degrees
                        radar[d] = min(radar[d], dist / max_range)
        
        return radar
    
    def _get_grid_obs_all(self) -> np.ndarray:
        """Get grid observations for all agents (5x5 window around each)."""
        window_size = 5
        half_w = window_size // 2
        obs = np.zeros((self.num_agents, window_size, window_size), dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            ax, ay = int(agent.x), int(agent.y)
            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    gx, gy = ax + dx, ay + dy
                    oy, ox = dy + half_w, dx + half_w
                    if 0 <= gx < self.world_width and 0 <= gy < self.world_height:
                        obs[i, oy, ox] = self.grid[gy, gx]
                    else:
                        obs[i, oy, ox] = -1.0  # Out of bounds = obstacle
        
        return obs
    
    def _build_env_state_for_agent(self, agent_id: int) -> dict:
        """Build environment state dictionary for reward calculation."""
        agent = self.agents[agent_id]
        
        # Get goals
        goals = [obj.to_dict() for obj in self.special_objects if obj.type == "goal"]
        
        # Get neighbors (agents within a certain distance)
        neighbor_distance = 10.0
        neighbors = []
        for other in self.agents:
            if other.id != agent_id:
                dist = math.sqrt((other.x - agent.x)**2 + (other.y - agent.y)**2)
                if dist <= neighbor_distance:
                    neighbors.append(other.to_dict())
        
        # Get obstacles
        obstacles = [obj.to_dict() for obj in self.special_objects if obj.type == "obstacle"]
        
        # Get gaps
        gaps = [obj.to_dict() for obj in self.special_objects if obj.type == "gap"]
        
        return {
        "agent_idx": agent_id,  # Add current agent index
        "goals": goals,
        "neighbors": neighbors,
        "obstacles": obstacles,
        "gaps": gaps,
        "time_step": self.current_step,
        "world_width": self.world_width,
        "world_height": self.world_height,
        "num_agents": self.num_agents,
        "all_agents": [a.to_dict() for a in self.agents],
        "agents": [a.to_dict() for a in self.agents],  # Alias for all_agents
    }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self._build_world()
        self.current_step = 0
        
        obs = self._get_obs_all()
        info = {
            "task_name": self.task_name,
            "num_agents": self.num_agents,
            "step": 0,
        }
        
        # Initialize render history
        self.render_history = []
        return obs, info
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        """Run one timestep of the environment's dynamics with Action Skipping."""
        total_rewards = np.zeros(self.num_agents, dtype=np.float32)
        terminated = False
        truncated = False
        info = {}
        
        # Handle Communication (Split Actions)
        move_actions = actions
        if self.enable_communication and self.action_space_type == "continuous":
             combined = actions.reshape(self.num_agents, -1)
             move_actions = combined[:, :2] # First 2 dims are movement
             comm_signals = combined[:, 2] # 3rd dim is signal
             
             # Update agent signals
             for i, agent in enumerate(self.agents):
                 agent.comm_signal = float(comm_signals[i])

        # Action Repeat Loop (Frame Skip)
        for _ in range(self.action_repeat):
            # 1. Apply actions (sets forces/velocities for this frame)
            # If discrete: Impulse is applied every frame (mimics holding button)
            if self.action_space_type == "continuous":
                self._apply_continuous_actions(move_actions)
            else:
                self._apply_discrete_actions(actions)
            
            # 2. Physics step
            self._apply_physics()
            
            # Record history for rendering (Ghost Trails)
            # Snapshot simple array of positions
            snapshot = np.array([[a.x, a.y] for a in self.agents], dtype=np.float32)
            self.render_history.append(snapshot)
            
            # 3. Calculate rewards (accumulate)
            step_rewards = self._calculate_rewards()
            total_rewards += step_rewards
            
            # 4. Check termination (early exit)
            if self._check_terminated():
                terminated = True
                break
        
        # Increment decision step counter
        self.current_step += 1
        
        # Check truncation (max decision steps reached)
        if self.current_step >= self.max_steps:
             truncated = True
        
        # 5. Get observations (final state after skip)
        obs = self._get_obs_all()
        
        info = {
            "step": self.current_step,
            "mean_reward": float(np.mean(total_rewards)),
        }
        
        return obs, total_rewards, terminated, truncated, info
    
    def _apply_continuous_actions(self, actions: np.ndarray):
        """Apply continuous actions with Energy & Inertia."""
        actions = np.array(actions).reshape(self.num_agents, 2)
        
        # Physics Params
        max_speed = 2.0
        inertia_alpha = 0.2 # low-pass filter (0.2 = heavy inertia, 1.0 = instant)
        energy_drain_rate = 0.005
        energy_regen_rate = 0.001
        
        for i, agent in enumerate(self.agents):
            ax, ay = actions[i]
            
            # 1. Energy Check
            if agent.energy <= 0.05: # Low battery mode
                ax *= 0.2
                ay *= 0.2
                
            # 2. Target Velocity
            target_vx = ax * max_speed
            target_vy = ay * max_speed
            
            # 3. Apply Inertia (Torque Limit)
            agent.vx = inertia_alpha * target_vx + (1 - inertia_alpha) * agent.vx
            agent.vy = inertia_alpha * target_vy + (1 - inertia_alpha) * agent.vy
            
            # 4. Energy Consuption
            speed = math.sqrt(agent.vx**2 + agent.vy**2)
            drain = speed * energy_drain_rate
            agent.energy = max(0.0, min(1.0, agent.energy - drain + energy_regen_rate))
    
    def _apply_discrete_actions(self, actions: np.ndarray):
        """Apply discrete actions (0=none, 1=up, 2=down, 3=left, 4=right, 5=grab)."""
        actions = np.array(actions).flatten()
        base_speed = 2.0
        
        # Physics Profile Logic
        arcade = (self.physics_profile == "arcade")
        
        for i, agent in enumerate(self.agents):
            action = int(actions[i])
            
            # TOP-DOWN: All agents always have control (no grounded check)
            has_control = True
            
            # Movement Speed
            speed = base_speed
            
            if action == 1:  # up (+Y in top-down)
                agent.vy += speed
            elif action == 2:  # down (-Y in top-down)
                agent.vy -= speed
            elif action == 3:  # left (-X)
                agent.vx -= speed
            elif action == 4:  # right (+X)
                agent.vx += speed
            elif action == 5:  # grab
                # Toggle grab intent
                agent.is_grabbing = not agent.is_grabbing

    def _update_spatial_hash(self):
        """Rebuild spatial hash grid for O(N) lookups."""
        self.spatial_hash = {}
        size = self.spatial_cell_size
        for i, agent in enumerate(self.agents):
            cx = int(agent.x / size)
            cy = int(agent.y / size)
            key = (cx, cy)
            if key not in self.spatial_hash:
                self.spatial_hash[key] = []
            self.spatial_hash[key].append(i)

    def _solve_collisions(self):
        """Solve elastic circle-circle collisions using Spatial Hash (O(N))."""
        # Iterate over all agents and check neighbors in grid
        for i in range(self.num_agents):
            a1 = self.agents[i]
            cx = int(a1.x / self.spatial_cell_size)
            cy = int(a1.y / self.spatial_cell_size)
            
            # Check 3x3 grid neighborhood
            for ox in [-1, 0, 1]:
                for oy in [-1, 0, 1]:
                    key = (cx + ox, cy + oy)
                    if key in self.spatial_hash:
                        for j in self.spatial_hash[key]:
                            if i >= j: continue  # Optimized check: only j > i to avoid duplicates/self
                            
                            a2 = self.agents[j]
                            
                            dx = a2.x - a1.x
                            dy = a2.y - a1.y
                            
                            # Fast AABB check first? No, circles are small.
                            dist_sq = dx*dx + dy*dy
                            radii = a1.radius + a2.radius
                            
                            if dist_sq < radii*radii:
                                dist = math.sqrt(dist_sq)
                                if dist == 0: continue
                                
                                # Overlap amount
                                overlap = radii - dist
                                
                                # Normal vector
                                nx = dx / dist
                                ny = dy / dist
                                
                                # Mass weighting
                                inv_mass1 = 1.0 / a1.mass
                                inv_mass2 = 1.0 / a2.mass
                                total_inv_mass = inv_mass1 + inv_mass2
                                
                                if total_inv_mass == 0: continue
                                
                                move_per_mass = overlap / total_inv_mass
                                
                                a1.x -= nx * move_per_mass * inv_mass1
                                a1.y -= ny * move_per_mass * inv_mass1
                                a2.x += nx * move_per_mass * inv_mass2
                                a2.y += ny * move_per_mass * inv_mass2
                                
                                # Elastic velocity exchange
                                dvx = a1.vx - a2.vx
                                dvy = a1.vy - a2.vy
                                vel_along_normal = dvx * nx + dvy * ny
                                
                                if vel_along_normal > 0:
                                    continue
                                    
                                restitution = 0.2
                                j_impulse = -(1 + restitution) * vel_along_normal
                                j_impulse /= total_inv_mass
                                
                                impulse_x = j_impulse * nx
                                impulse_y = j_impulse * ny
                                
                                a1.vx += impulse_x * inv_mass1
                                a1.vy += impulse_y * inv_mass1
                                a2.vx -= impulse_x * inv_mass2
                                a2.vy -= impulse_y * inv_mass2

    def _solve_constraints(self):
        """Solve distance constraints (grabbing)."""
        # Stored in self.constraints: list of (agent_a_id, agent_b_id, target_dist)
        # We iterate and satisfy constraints multiple times for stability
        for _ in range(self.constraint_iterations):
            for (id_a, id_b, target_dist) in self.constraints:
             a1 = self.agents[id_a]
             a2 = self.agents[id_b]
             
             dx = a2.x - a1.x
             dy = a2.y - a1.y
             dist = math.sqrt(dx*dx + dy*dy)
             if dist == 0: continue
             
             # Difference from target
             diff = (dist - target_dist) / dist
             
             # Stiffness (0.1 to 1.0)
             # Higher = stiffer chain, better for bridging
             stiffness = 0.9 
             
             move_x = dx * diff * stiffness * 0.5 # Equal weighting for simplicity
             move_y = dy * diff * stiffness * 0.5
             
             a1.x += move_x
             a1.y += move_y
             a2.x -= move_x
             a2.y -= move_y
             
             # Kill relative velocity (Damping)
             # Simple damping: blend velocities to reduce oscillation
             dvx = a2.vx - a1.vx
             dvy = a2.vy - a1.vy
             damping = 0.1
             a1.vx += dvx * damping
             a1.vy += dvy * damping
             a2.vx -= dvx * damping
             a2.vy -= dvy * damping
    
    def _is_in_gap(self, agent) -> bool:
        """Check if agent is positioned within any gap zone."""
        for obj in self.special_objects:
            if obj.type == "gap":
                # Gap defined by x1, x2, y1, y2 (rectangular zone)
                in_x = obj.x1 <= agent.x <= obj.x2
                in_y = getattr(obj, 'y1', 0) <= agent.y <= getattr(obj, 'y2', self.world_height)
                if in_x and in_y:
                    return True
        return False
    
    def _is_connected_to_ground(self, agent_id: int, visited: set = None) -> bool:
        """
        Check if agent is connected to solid ground through constraint chain.
        Uses BFS to traverse the constraint graph.
        
        An agent is "grounded" if:
        1. They are NOT in a gap zone, OR
        2. They are connected via constraints to an agent who is grounded
        """
        if visited is None:
            visited = set()
        
        if agent_id in visited:
            return False  # Avoid cycles
        visited.add(agent_id)
        
        agent = self.agents[agent_id]
        
        # Base case: Agent is on solid ground (not in gap)
        if not self._is_in_gap(agent):
            return True
        
        # Recursive case: Check all connected agents through constraints
        for (id_a, id_b, _) in self.constraints:
            if id_a == agent_id:
                if self._is_connected_to_ground(id_b, visited):
                    return True
            elif id_b == agent_id:
                if self._is_connected_to_ground(id_a, visited):
                    return True
        
        return False  # Not connected to ground
    
    def _apply_physics(self):
        """Apply physics simulation with substepping (Physics V2)."""
        dt = self.time_step
        substeps = 4
        dt_sub = dt / substeps
        
        # 0. Initial Hash Update
        self._update_spatial_hash()
        
        # 1. Manage Grabbing (Create/Destroy Constraints)
        grab_dist = 4.0  # Max distance to grab
        if self.physics_profile == "arcade":
            grab_dist = 6.0  # Easier grabbing assist
            
        size = self.spatial_cell_size
        
        # Count incoming constraints to prevent tearing (Safety)
        incoming_counts = np.zeros(self.num_agents, dtype=np.int32)
        for (_, id_b, _) in self.constraints:
            incoming_counts[id_b] += 1
        
        for i, agent in enumerate(self.agents):
            if agent.is_grabbing:
                # If grabbing, try to form a constraint if none exists
                # Limit: hold max 2 agents? Or 1? Let's say 2 for stability
                if len(agent.active_constraints) < 2:
                    nearest = None
                    min_d = grab_dist
                    
                    cx = int(agent.x / size)
                    cy = int(agent.y / size)
                    
                    # Search grid 3x3
                    for ox in [-1, 0, 1]:
                        for oy in [-1, 0, 1]:
                            key = (cx + ox, cy + oy)
                            if key in self.spatial_hash:
                                for j in self.spatial_hash[key]:
                                    if i == j: continue
                                    if j in agent.active_constraints: continue
                                    
                                    # Limit incoming constraints (Safety)
                                    if incoming_counts[j] >= 3: continue
                                    
                                    # Fast bounding check
                                    other = self.agents[j]
                                    dx = other.x - agent.x
                                    dy = other.y - agent.y
                                    
                                    d = math.sqrt(dx*dx + dy*dy)
                                    if d < min_d:
                                        min_d = d
                                        nearest = j
                    
                    # Create constraint
                    if nearest is not None:
                        # Add to constraints list: (id_a, id_b, target_dist)
                        self.constraints.append([i, nearest, min_d]) # List so we can modify if needed (or tuple)
                        agent.active_constraints.append(nearest)
                        
                        # Add back-reference for stability? 
                        # For now, one-way ownership in active_constraints list, but bi-directional physics
            else:
                # Not grabbing: release all
                if agent.active_constraints:
                    # Remove my constraints
                    # Filter self.constraints where id_a == i and id_b in active_constraints
                    # Note: complex list modification while iterating is bad.
                    # Rebuild list is safer for small N
                    self.constraints = [c for c in self.constraints if not (c[0] == i and c[1] in agent.active_constraints)]
                    agent.active_constraints = []

        # 2. Physics Substepping
        for _ in range(substeps):
            for agent in self.agents:
                # TOP-DOWN MODE: No gravity (horizontal plane)
                # gravity_y is ignored in top-down for movement, only used for special effects
                # agent.vy -= self.gravity_y * dt_sub  # REMOVED for top-down
                
                # Apply light drag (top-down: surface resistance)
                # Note: Applied every substep, so keep LOW
                drag = 0.005  # Very light drag
                agent.vx *= (1 - drag)
                agent.vy *= (1 - drag)
                
                # Integrate Position (Symplectic Euler)
                agent.x += agent.vx * dt_sub
                agent.y += agent.vy * dt_sub
            
            # Rebuild Hash and Solve
            self._update_spatial_hash()
            self._solve_collisions()
            self._solve_constraints()
            
            # Bounds Check & Gap Handling (TOP-DOWN)
            for i, agent in enumerate(self.agents):
                agent.is_grounded = True  # In top-down, always "grounded"
                
                # TOP-DOWN Gap Logic: Chain-based survival
                if self._is_in_gap(agent):
                    # Agent is in gap zone - check if connected to ground
                    if not self._is_connected_to_ground(i):
                        # NOT connected to solid ground = DEATH!
                        # Respawn agent to spawn zone
                        spawn_zone = self.config.get("env_params", {}).get("spawn_zone", None)
                        if spawn_zone:
                            agent.x = np.random.uniform(spawn_zone.get("x1", 5), spawn_zone.get("x2", 20))
                            agent.y = np.random.uniform(spawn_zone.get("y1", 5), spawn_zone.get("y2", 50))
                        else:
                            agent.x = np.random.uniform(5, 20)
                            agent.y = np.random.uniform(5, self.world_height / 2)
                        
                        agent.vx = 0
                        agent.vy = 0
                        agent.energy = max(0.2, agent.energy - 0.3)  # Energy penalty
                        
                        # Release all constraints (chains break on death)
                        self.constraints = [c for c in self.constraints if c[0] != i and c[1] != i]
                        agent.active_constraints = []
                
                # World Bounds (Top-Down: All 4 walls)
                agent.x = np.clip(agent.x, 0, self.world_width)
                agent.y = np.clip(agent.y, 0, self.world_height)

            # Static Obstacles Collision
            for obj in self.special_objects:
                if obj.type == "obstacle":
                    for agent in self.agents:
                        dist = math.sqrt((agent.x - obj.x)**2 + (agent.y - obj.y)**2)
                        overlap = (agent.radius + obj.radius) - dist
                        if overlap > 0 and dist > 0:
                             nx = (agent.x - obj.x) / dist
                             ny = (agent.y - obj.y) / dist
                             # Move out
                             agent.x += nx * overlap
                             agent.y += ny * overlap
                             # Bounce (simplified)
                             dot = agent.vx * nx + agent.vy * ny
                             if dot < 0:
                                 agent.vx -= 1.2 * dot * nx
                                 agent.vy -= 1.2 * dot * ny
    
    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards for all agents using the compiled reward function."""
        # Imports passed to the function (must be before usage to avoid UnboundLocalError)
        import math
        import numpy as np
        
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            agent_dict = agent.to_dict()
            env_state = self._build_env_state_for_agent(i)
            
            try:
                # Direct call (Optimized)
                rewards[i] = self.reward_function(agent_dict, env_state, math, np)
            except Exception as e:
                # Log error occasionally
                if self.current_step % 100 == 0 and i == 0:
                    print(f"Reward calculation error: {e}")
                rewards[i] = 0.0
        
        return rewards
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate (e.g., all goals reached)."""
        # Default: check if any agent reached any goal
        goals = [obj for obj in self.special_objects if obj.type == "goal"]
        if not goals:
            return False
        
        goal_reach_distance = 2.0
        goals_reached = 0
        
        for goal in goals:
            for agent in self.agents:
                dist = math.sqrt((agent.x - goal.x)**2 + (agent.y - goal.y)**2)
                if dist < goal_reach_distance:
                    goals_reached += 1
                    break
        
        # Terminate if all goals are reached
        return goals_reached >= len(goals)
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """
        Render the environment as an RGB image.
        
        Args:
            mode: Rendering mode ("rgb_array" returns numpy array)
        
        Returns:
            RGB image as numpy array of shape (H, W, 3)
        """
        # Image dimensions
        img_width = int(self.world_width * self.render_scale)
        img_height = int(self.world_height * self.render_scale)
        
        # Create blank image (dark background)
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img[:, :] = [30, 30, 40]  # Dark blue-gray background
        
        # Draw gaps (different background color)
        for obj in self.special_objects:
            if obj.type == "gap":
                x1 = int(obj.x1 * self.render_scale)
                x2 = int(obj.x2 * self.render_scale)
                # Render as full vertical void
                y1 = 0
                y2 = img_height
                x1 = max(0, x1)
                x2 = min(img_width, x2)
                img[y1:y2, x1:x2] = [10, 10, 20]  # Dark void color
        
        # Draw obstacles
        for obj in self.special_objects:
            if obj.type == "obstacle":
                cx = int(obj.x * self.render_scale)
                cy = int((self.world_height - obj.y) * self.render_scale)  # Flip Y
                radius = int(obj.radius * self.render_scale)
                self._draw_circle(img, cx, cy, radius, [80, 80, 90])
        
        # Draw goals
        for obj in self.special_objects:
            if obj.type == "goal":
                cx = int(obj.x * self.render_scale)
                cy = int((self.world_height - obj.y) * self.render_scale)  # Flip Y
                self._draw_circle(img, cx, cy, 4, [50, 255, 100])  # Green
        
        # Draw Ghost Trails (Action Skip Visualization)
        if hasattr(self, 'render_history'):
             ghost_color = [60, 60, 80]  # Faint trace
             for snapshot in self.render_history[:-1]:  # Skip potential last duplicate
                 for i in range(len(snapshot)):
                     x, y = snapshot[i]
                     cx = int(x * self.render_scale)
                     cy = int((self.world_height - y) * self.render_scale)
                     self._draw_circle(img, cx, cy, 2, ghost_color)

        # Draw agents
        for agent in self.agents:
            cx = int(agent.x * self.render_scale)
            cy = int((self.world_height - agent.y) * self.render_scale)  # Flip Y
            color = [255, 150, 50] if agent.is_grabbing else [100, 150, 255]  # Orange if grabbing, blue otherwise
            self._draw_circle(img, cx, cy, 3, color)
        
        return img
    
    def _draw_circle(self, img: np.ndarray, cx: int, cy: int, radius: int, color: list):
        """Draw a filled circle on the image."""
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        img[mask] = color
    
    def close(self):
        """Clean up environment resources."""
        pass


# Factory function for creating environments with config
def make_env(config: dict):
    """Factory function for creating UniversalSwarmEnv instances."""
    def _init():
        return UniversalSwarmEnv(config)
    return _init
