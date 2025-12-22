import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

class SwarmVecEnv(VecEnv):
    """
    Treats a multi-agent Swarm Environment as a Vectorized Environment.
    Single Swarm instance.
    """
    
    def __init__(self, swarm_env):
        self.swarm = swarm_env
        self.num_agents = self.swarm.num_agents
        
        # Define observation space (single agent)
        single_obs_shape = self.swarm.observation_space.shape[1:] 
        observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=single_obs_shape, dtype=np.float32
        )
        
        # Define action space (handle discrete vs continuous)
        if self.swarm.action_space_type == "discrete":
            # MultiDiscrete: each agent has 6 actions (0-5)
            action_space = spaces.Discrete(6)
        else:
            # Continuous: Box action space
            single_action_shape = self.swarm.action_space.shape[1:]
            action_space = spaces.Box(
                low=-1.0, high=1.0, shape=single_action_shape, dtype=np.float32
            )
        
        # Initialize VecEnv
        super().__init__(self.num_agents, observation_space, action_space)
        
        self.metadata = self.swarm.metadata
        self.render_mode = self.swarm.render_mode
        self.last_dones = [False] * self.num_agents
        self.actions = None
        
    def reset(self):
        """Reset the swarm and return [num_agents, obs_dim]."""
        obs, info = self.swarm.reset()
        self.last_dones = [False] * self.num_agents
        return obs
    
    def step_async(self, actions):
        """Save actions for step_wait."""
        self.actions = actions
        
    def step_wait(self):
        """Step the physics and return results."""
        obs, rewards, terminated, truncated, info = self.swarm.step(self.actions)
        
        done = terminated or truncated
        dones = np.array([done] * self.num_agents, dtype=bool)
        
        infos = [info.copy() for _ in range(self.num_agents)]
        
        if done:
            terminal_obs = obs.copy()
            for i in range(self.num_agents):
                infos[i]["terminal_observation"] = terminal_obs[i]
            
            obs, _ = self.swarm.reset()
            self.last_dones = [False] * self.num_agents
        
        return obs, rewards, dones, infos

    def close(self):
        self.swarm.close()
        
    def render(self, mode="rgb_array"):
        return self.swarm.render(mode=mode)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_agents
    
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.swarm, attr_name)] * self.num_agents

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.swarm, attr_name, value)
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.swarm, method_name)
        result = method(*method_args, **method_kwargs)
        return [result] * self.num_agents

    def seed(self, seed: int = None):
        # Gym API seed
        return [None] * self.num_agents # Simplified

class ParallelSwarmVecEnv(VecEnv):
    """
    Manages multiple independent Swarm Environments in parallel.
    Presents them as ONE massive vector of agents.
    
    Total Environments exposed to SB3 = num_swarms * num_agents_per_swarm.
    """
    
    def __init__(self, make_env_fn, num_swarms):
        self.swarms = [make_env_fn() for _ in range(num_swarms)]
        self.num_swarms = num_swarms
        self.agents_per_swarm = self.swarms[0].num_agents
        self.total_agents = self.num_swarms * self.agents_per_swarm
        
        # Determine spaces from first swarm
        single_obs_shape = self.swarms[0].observation_space.shape[1:]
        observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=single_obs_shape, dtype=np.float32
        )
        
        single_action_shape = self.swarms[0].action_space.shape[1:]
        action_space = spaces.Box(
            low=-1.0, high=1.0, shape=single_action_shape, dtype=np.float32
        )
        
        super().__init__(self.total_agents, observation_space, action_space)
        self.metadata = self.swarms[0].metadata
        self.render_mode = self.swarms[0].render_mode
        self.actions = None
        
    def reset(self):
        obs_list = []
        for swarm in self.swarms:
            o, _ = swarm.reset()
            obs_list.append(o)
        return np.concatenate(obs_list, axis=0) # (Total Agents, Dim)
    
    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # Split actions for each swarm
        chunk_size = self.agents_per_swarm
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, swarm in enumerate(self.swarms):
            start = i * chunk_size
            end = start + chunk_size
            chunk_actions = self.actions[start:end]
            
            o, r, term, trunc, info = swarm.step(chunk_actions)
            
            d = term or trunc
            # Expand done/info
            
            if d:
                term_obs = o.copy()
                o, _ = swarm.reset()
                
                swarm_dones = [True] * chunk_size
                swarm_infos = []
                for k in range(chunk_size):
                    inf = info.copy()
                    inf["terminal_observation"] = term_obs[k]
                    swarm_infos.append(inf)
            else:
                swarm_dones = [False] * chunk_size
                swarm_infos = [info.copy() for _ in range(chunk_size)]
                
            observations.append(o)
            rewards.append(r)
            dones.extend(swarm_dones)
            infos.extend(swarm_infos)
            
        return np.concatenate(observations, axis=0), np.concatenate(rewards, axis=0), np.array(dones), infos

    def close(self):
        for swarm in self.swarms:
            swarm.close()
            
    def render(self, mode="rgb_array"):
        # Just render the FIRST swarm for visualization
        return self.swarms[0].render(mode=mode)
        
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.total_agents
    
    def get_attr(self, attr_name, indices=None):
         val = getattr(self.swarms[0], attr_name)
         return [val] * self.total_agents

    def set_attr(self, attr_name, value, indices=None):
        for swarm in self.swarms:
            setattr(swarm, attr_name, value)
            
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        outputs = []
        for swarm in self.swarms:
             method = getattr(swarm, method_name)
             res = method(*method_args, **method_kwargs)
             outputs.extend([res] * self.agents_per_swarm)
        return outputs

    def seed(self, seed: int = None):
        return [None] * self.total_agents
