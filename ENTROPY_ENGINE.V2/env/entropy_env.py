import functools
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from core.world import PhysicsWorld
from core.entities import Agent, Wall, Goal
from core.communication import StructuredMessage, Vocab, CommunicationState
try:
    from core.numba_utils import compute_sensors_batch
    USE_NUMBA = True
except ImportError:
    print("Numba not found, falling back to slow sensors.")
    USE_NUMBA = False

class EntropyEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "entropy_v2"}

    def __init__(self, render_mode=None, nr_agents=5):
        self.possible_agents = [f"agent_{i}" for i in range(nr_agents)]
        self.nr_agents = nr_agents
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.renderer = None 
        
        # Physics
        self.world = None
        self.agents_map = {} # agent_id -> Agent object
        self.goals_map = {} # agent_id -> Goal object
        
        # Spaces
        # Spaces
        # Action: [left, right] + [comm_vector(8)]
        # comm_vector: [priority, msg_type, target, role, p1, p2, p3, p4]
        # Motors: -1 to 1, Comm: -1 to 1 (will be parsed)
        self.action_spaces = {agent: spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32) for agent in self.possible_agents}
        
        # Observation:
        # Physics (36): Lidar(32) + Vel(2) + Rel_Goal(2)
        # Token (8): [sender, type, target, role, p1, p2, p3, p4]
        # Broadcast (10 * (N-1)): [rel_x, rel_y, sender, type, target, role, p1...p4] for each neighbor
        # Added rel_x, rel_y for Grounded Communication
        # Assuming N=5, neighbors=4 -> 40 floats
        # Total = 36 + 8 + 40 = 84
        
        lidar_rays = 32
        physics_dim = lidar_rays + 2 + 2
        comm_dim = 8 + (nr_agents - 1) * 10 # 8 -> 10 per neighbor
        obs_dim = physics_dim + comm_dim
        
        self.observation_spaces = {agent: spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=np.float32) for agent in self.possible_agents}
        
        self.comm_state = CommunicationState(nr_agents)
        
        # Token Fairness Tracking
        # Count recent token holds per agent. Decays over time.
        self.speaker_history = np.zeros(nr_agents, dtype=np.float32)
        
        # Curriculum Learning
        # Level 1 (Easy): Small, Close Goals
        # Level 2 (Medium): Standard, Random Goals
        # Level 3 (Hard): Large, Random Goals + Obstacles (Future)
        self.difficulty = 1
        
    def set_difficulty(self, level):
        self.difficulty = level
        # print(f"Env Difficulty Set to: {level}")
        
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.world = PhysicsWorld()
        
        # Build Map (Simple Box)
        # 800x600 arena
        # 800x600 arena
        # Build Map (Simple Box)
        # Randomize Arena Size (Domain Randomization) based on Difficulty
        rng = np.random.default_rng(seed)
        
        if self.difficulty == 1:
            # Easy: Smaller arena, Goals close
            width = int(rng.uniform(500, 600))
            height = int(rng.uniform(400, 500))
        elif self.difficulty == 2:
            # Medium: Standard
            width = int(rng.uniform(600, 800))
            height = int(rng.uniform(500, 600))
        else:
            # Hard: Large
            width = int(rng.uniform(800, 1000))
            height = int(rng.uniform(600, 800))
        
        walls = [
            ((0, 0), (width, 0)),
            ((width, 0), (width, height)),
            ((width, height), (0, height)),
            ((0, height), (0, 0))
        ]
        
        self.wall_segments = np.zeros((len(walls), 4), dtype=np.float32)
        for i, (p1, p2) in enumerate(walls):
            Wall(self.world, p1, p2)
            self.wall_segments[i] = [p1[0], p1[1], p2[0], p2[1]]
        
        self.agents_map = {}
        self.goals_map = {}
        
        observations = {}
        
        rng = np.random.default_rng(seed)
        
        # Keep track of agent positions to place goals near them if Easy
        agent_positions = {}
        
        for agent_id in self.agents:
            # Random start pos (ensure inside walls)
            x = rng.uniform(50, width - 50)
            y = rng.uniform(50, height - 50)
            agent = Agent(self.world, (x, y))
            self.agents_map[agent_id] = agent
            agent_positions[agent_id] = (x, y)
            
            # Goal Placement
            if self.difficulty == 1:
                # Easy: Goal within 200 units
                # Rejection sampling or polar offset
                while True:
                    angle = rng.uniform(0, 2*np.pi)
                    dist = rng.uniform(50, 200)
                    gx = x + np.cos(angle) * dist
                    gy = y + np.sin(angle) * dist
                    # Check bounds
                    if 50 < gx < width - 50 and 50 < gy < height - 50:
                        break
            else:
                # Medium/Hard: Random goal anywhere
                gx = rng.uniform(50, width - 50)
                gy = rng.uniform(50, height - 50)
                
            goal = Goal(self.world, (gx, gy))
            self.goals_map[agent_id] = goal
            
        # Initial step to populate sensors
        self.world.step()
        
        self._compute_all_sensors()
        
        for agent_id in self.agents:
            observations[agent_id] = self._get_obs(agent_id)
            
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _compute_all_sensors(self):
        if not USE_NUMBA:
            for agent in self.agents_map.values():
                agent.update_sensors()
            return

        # Prepare arrays
        # Ensure consistent order matching self.agents
        agents_list = [self.agents_map[aid] for aid in self.agents]
        n_agents = len(agents_list)
        
        if n_agents == 0:
            return

        agent_states = np.zeros((n_agents, 4), dtype=np.float32)
        
        for i, agent in enumerate(agents_list):
            # x, y, angle, radius
            p = agent.body.position
            agent_states[i] = [p.x, p.y, agent.body.angle, agent.radius]
            
        # Obstacles: x, y, radius
        # We reuse agent states for dynamic obstacles
        obs_circles = agent_states[:, [0, 1, 3]].copy() # (N, 3)
        
        # Call Numba
        # Assuming all agents have same lidar config
        first_agent = agents_list[0]
        rays = first_agent.lidar_rays
        rng = first_agent.lidar_range
        
        sensors_batch = compute_sensors_batch(agent_states, self.wall_segments, obs_circles, rays, rng)
        
        # Assign back
        for i, agent in enumerate(agents_list):
            agent.sensors = sensors_batch[i]

    def _compute_communication(self, actions):
        """
        Process the Dual-Channel Communication.
        1. Parse valid actions into StructuredMessages.
        2. Channel A: Determine Token Holder (Max Priority).
        3. Channel B: Store Broadcast Messages.
        """
        if not self.agents:
            return

        self.comm_state.reset()
        
        parsed_msgs = []
        priorities = []
        
        # 1. Parse Actions
        for i, agent_id in enumerate(self.agents):
            act = actions.get(agent_id, np.zeros(10))
            
            # Action Structure:
            # 0,1: Motors
            # 2: Priority (sigmoid-like behavior expected from net, but here raw)
            # 3: Msg Type (discrete index)
            # 4: Target Agent (discrete index)
            # 5: Role Claim (discrete index)
            # 6-9: Payload (4 floats)
            
            # --- PARSING ---
            # Priority: Action[2] is usually tanh (-1 to 1). Map to 0-1.
            prio_raw = act[2]
            prio = (prio_raw + 1.0) / 2.0  # Normalize to 0-1
            prio = np.clip(prio, 0.0, 1.0)
            
            # Msg Type: Mapped from Action[3] (-1 to 1) to (0 to 31)
            # Strategy: (act + 1) * 16 -> 0 to 32
            type_raw = act[3]
            msg_type_idx = int((type_raw + 1.0) * 16.0)
            msg_type_idx = max(0, min(31, msg_type_idx))
            
            # Target Agent: Map -1 to 1 -> -1 to N-1
            # -1 = All. 0..N-1 = Specific.
            # Let's say: < -0.8 = All (-1). Else: (val + 0.8) scaled to 0..N-1
            target_raw = act[4]
            target_agent = -1
            if target_raw > -0.8:
                # Scale -0.8..1.0 to 0..N-1
                norm = (target_raw + 0.8) / 1.8 
                target_agent = int(norm * self.nr_agents)
                target_agent = max(0, min(self.nr_agents - 1, target_agent))
            
            # Role Claim: Map -1 to 1 -> 0 to 7
            role_raw = act[5]
            role_claim = int((role_raw + 1.0) * 4.0) # 0 to 8
            role_claim = max(0, min(7, role_claim))
            
            # Payload: Just take raw values
            payload = act[6:10]
            
            # Construct Key Info
            parsed_msgs.append({
                "sender": i,
                "type": msg_type_idx,
                "target": target_agent,
                "role": role_claim,
                "payload": payload
            })
            priorities.append(prio)
            
            # Store in Agent Entity for rendering/debugging
            if agent_id in self.agents_map:
                # Store Full Comm Vector for visualization
                self.agents_map[agent_id].set_signal(act[2:])

        # 2. Channel A: Token Election
        # Who has max priority?
        
        # --- FAIRNESS PENALTY (Safeguard against Token Starvation) ---
        # adjusted_prio = raw_prio - penalty_coef * history
        # If history is high (spoke recently), priority is lowered.
        penalty_coef = 0.3
        
        # Convert list to array for calc
        priorities_arr = np.array(priorities)
        if len(priorities_arr) == self.nr_agents:
            # Apply penalty
            adjusted_priorities = priorities_arr - (penalty_coef * self.speaker_history)
        else:
            adjusted_priorities = priorities_arr # Should match size unless agents died
            
        # Tie-breaking: Lower ID wins (deterministic)
        if len(adjusted_priorities) > 0:
            max_prio = np.max(adjusted_priorities)
            # Find first agent with max_prio
            winner_idx = np.argmax(adjusted_priorities)
            
            # Create Token Message
            w_msg = parsed_msgs[winner_idx]
            self.comm_state.token_holder = winner_idx
            self.comm_state.token_message = StructuredMessage(
                sender_id=w_msg["sender"],
                msg_type=w_msg["type"],
                target_agent=int(w_msg["target"]),
                role_claim=int(w_msg["role"]),
                payload=w_msg["payload"]
            )
            self.comm_state.agent_priorities = adjusted_priorities
            
            # Update History logic
            # Winner gets +1.0
            self.speaker_history[winner_idx] += 1.0
            
        # Decay history for everyone
        # history = history * 0.95 (Leaky integrator)
        self.speaker_history *= 0.95
            
        # 3. Channel B: Broadcast
        # Everyone speaks on this channel
        for i, msg in enumerate(parsed_msgs):
            self.comm_state.broadcast_messages[i] = StructuredMessage(
                sender_id=msg["sender"],
                msg_type=msg["type"],
                target_agent=int(msg["target"]),
                role_claim=int(msg["role"]),
                payload=msg["payload"]
            )

    def step(self, actions):
        # Apply actions
        for agent_id, action in actions.items():
            if agent_id in self.agents_map:
                agent = self.agents_map[agent_id]
                agent.control(action[0], action[1])
                
        # Step Physics
        for _ in range(3):
            self.world.step()
            
        # Update sensors
        self._compute_all_sensors()
        
        # Compute Signals (New V3)
        self._compute_communication(actions)
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent_id in self.agents:
            agent = self.agents_map[agent_id]
            goal = self.goals_map[agent_id]
            
            # Calculate Reward
            dist = agent.body.position.get_distance(goal.body.position)
            reward = -dist / 1000.0
            
            # Reached Goal?
            term = False
            if dist < (agent.radius + goal.radius):
                reward += 10.0
                term = True
                agent.body.position = (np.random.uniform(50, 750), np.random.uniform(50, 550))
                goal.body.position = (np.random.uniform(50, 750), np.random.uniform(50, 550))
                agent.body.velocity = (0,0)
                
            observations[agent_id] = self._get_obs(agent_id)
            rewards[agent_id] = reward
            terminations[agent_id] = term
            truncations[agent_id] = False
            infos[agent_id] = {}
        
        # If we render
        if self.render_mode == "human":
            self._render_frame()
            
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_id):
        agent = self.agents_map[agent_id]
        goal = self.goals_map[agent_id]
        
        # 1. Physics (36)
        lidar = agent.sensors # 32
        vel = [agent.body.velocity.x / 100.0, agent.body.velocity.y / 100.0] # 2
        
        rel_pos = goal.body.position - agent.body.position
        local_goal = rel_pos.rotated(-agent.body.angle)
        goal_obs = [local_goal.x / 800.0, local_goal.y / 600.0] # 2
        
        physics_obs = np.concatenate([lidar, vel, goal_obs], dtype=np.float32)
        
        # 2. Token Channel (8)
        # [sender, type, target, role, p1, p2, p3, p4]
        # Normalize discrete values for neural net (e.g. sender/N, type/32)
        t_msg = self.comm_state.token_message
        token_vec = np.zeros(8, dtype=np.float32)
        token_vec[0] = t_msg.sender_id / self.nr_agents
        token_vec[1] = t_msg.msg_type / 32.0
        token_vec[2] = t_msg.target_agent / self.nr_agents
        token_vec[3] = t_msg.role_claim / 8.0
        token_vec[4:8] = t_msg.payload
        
        # 3. Broadcast Channel (32 for 4 neighbors)
        # We need to collect messages from ALL other agents.
        # Fixed order based on ID for V2 stability.
        broadcast_obs = []
        
        current_agent_idx = self.agents.index(agent_id)
        
        for i in range(self.nr_agents):
            if i == current_agent_idx:
                continue # Skip self dictation
            
            b_msg = self.comm_state.broadcast_messages[i]
            
            # --- GROUNDED COMMUNICATION ---
            # Calculate relative position of sender 'i' to receiver 'agent_id'
            sender = self.agents_map[self.agents[i]]
            rel_pos = sender.body.position - agent.body.position
            # Rotate to agent's local frame
            local_pos = rel_pos.rotated(-agent.body.angle)
            
            b_vec = np.zeros(10, dtype=np.float32)
            b_vec[0] = local_pos.x / 800.0 # Normalized Rel X
            b_vec[1] = local_pos.y / 600.0 # Normalized Rel Y
            b_vec[2] = b_msg.sender_id / self.nr_agents
            b_vec[3] = b_msg.msg_type / 32.0
            b_vec[4] = b_msg.target_agent / self.nr_agents
            b_vec[5] = b_msg.role_claim / 8.0
            b_vec[6:10] = b_msg.payload
            
            broadcast_obs.append(b_vec)
            
        broadcast_flat = np.concatenate(broadcast_obs, dtype=np.float32)
        
        return np.concatenate([physics_obs, token_vec, broadcast_flat], dtype=np.float32)

    def render(self):
        if self.renderer is None:
            from shared.rendering import Renderer
            self.renderer = Renderer(render_mode=self.render_mode)
            
        return self.renderer.render_frame(self.agents_map, self.goals_map, [], self.comm_state)

    def _render_frame(self):
        self.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
