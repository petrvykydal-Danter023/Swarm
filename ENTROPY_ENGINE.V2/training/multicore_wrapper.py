import multiprocessing as mp
import numpy as np
import sys
import os
from enum import Enum
from typing import List, Dict, Any

# Ensure root is in path for imports in child processes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.entropy_env import EntropyEnv
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

class Command(Enum):
    RESET = 1
    STEP = 2
    CLOSE = 3
    GET_ATTR = 4

def worker_process(remote, config):
    """
    Worker process running a single EntropyEnv instance.
    """
    try:
        env = EntropyEnv(**config)
        
        while True:
            cmd, data = remote.recv()
            
            if cmd == Command.RESET:
                obs, info = env.reset()
                remote.send((obs, info))
                
            elif cmd == Command.STEP:
                actions = data # Dict {agent_id: action}
                obs, rewards, term, trunc, info = env.step(actions)
                remote.send((obs, rewards, term, trunc, info))
                
            elif cmd == Command.GET_ATTR:
                attr_name = data
                if hasattr(env, attr_name):
                    val = getattr(env, attr_name)
                    remote.send((True, val))
                else:
                    remote.send((False, None))
                
            elif cmd == Command.CLOSE:
                env.close()
                remote.close()
                break
    except Exception as e:
        print(f"Worker Error: {e}")
        # We can't easily propagate error unless we wrap everything.
        # But failing hard is better than hanging.
        raise e




class AsyncVectorizedEntropyEnv(VecEnv):
    """
    Spawns N processes, each running an EntropyEnv with M agents.
    Exposes a flattened VecEnv interface for SB3 (Total agents = N * M).
    """
    def __init__(self, n_envs: int, agents_per_env: int, env_config: Dict[str, Any] = None):
        self.n_envs = n_envs # Number of processes
        self.agents_per_env = agents_per_env # Agents per process
        self.total_agents = n_envs * agents_per_env
        
        if env_config is None:
            env_config = {}
        env_config["nr_agents"] = agents_per_env
        env_config["render_mode"] = None # Workers don't render
        
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.processes = []
        
        for work_remote in self.work_remotes:
            p = mp.Process(target=worker_process, args=(work_remote, env_config))
            p.daemon = True # Process killed if main dies
            p.start()
            self.processes.append(p)
            
        # Define Spaces (Assuming all agents have same space)
        # We need to instantiate a dummy env to get spaces
        dummy_env = EntropyEnv(**env_config)
        obs_space = dummy_env.observation_space(dummy_env.possible_agents[0])
        act_space = dummy_env.action_space(dummy_env.possible_agents[0])
        dummy_env.close()
        
        super().__init__(self.total_agents, obs_space, act_space)
        
    def reset(self):
        for remote in self.remotes:
            remote.send((Command.RESET, None))
            
        # Collect results
        self.agent_map = [] # List mapping index -> (env_idx, agent_id)
        flat_obs = []
        
        for i, remote in enumerate(self.remotes):
            obs_dict, info = remote.recv()
            # obs_dict is {agent_0: [...], agent_1: [...]}
            # We need to maintain a consistent order
            sorted_keys = sorted(obs_dict.keys())
            for agent_id in sorted_keys:
                self.agent_map.append((i, agent_id))
                flat_obs.append(obs_dict[agent_id])
                
        return np.stack(flat_obs)

    def step_async(self, actions):
        # actions is a numpy array of shape (total_agents, action_dim)
        # We need to split this back to envs
        
        self.last_actions_split = [{} for _ in range(self.n_envs)]
        
        for global_idx, action in enumerate(actions):
            env_idx, agent_id = self.agent_map[global_idx]
            self.last_actions_split[env_idx][agent_id] = action
            
        for remote, act_dict in zip(self.remotes, self.last_actions_split):
            remote.send((Command.STEP, act_dict))

    def step_wait(self):
        flat_obs = []
        flat_rewards = []
        flat_dones = []
        flat_infos = []
        
        for i, remote in enumerate(self.remotes):
            # obs_dict, rewards, term, trunc, info
            obs, rew, term, trunc, info = remote.recv()
            
            # We iterate based on the know sorted keys for this env
            # (Assuming keys don't change dynamically mid-episode randomly, which is true for our env)
            sorted_keys = sorted(obs.keys())
            
            for agent_id in sorted_keys:
                flat_obs.append(obs[agent_id])
                flat_rewards.append(rew[agent_id])
                # Done if term or trunc
                is_done = term[agent_id] or trunc[agent_id]
                flat_dones.append(is_done)
                
                # SB3 expects 'terminal_observation' in info if done
                inf = info.get(agent_id, {})
                if is_done:
                    inf["terminal_observation"] = obs[agent_id]
                flat_infos.append(inf)
                
        return np.stack(flat_obs), np.array(flat_rewards), np.array(flat_dones), flat_infos

    def close(self):
        for remote in self.remotes:
            remote.send((Command.CLOSE, None))
        for p in self.processes:
            p.join()
            
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.total_agents

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Not supported in async wrapper")

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        # We replicate the attribute from each env 'agents_per_env' times
        
        for remote in self.remotes:
            remote.send((Command.GET_ATTR, attr_name))
            
        results = []
        for remote in self.remotes:
            found, val = remote.recv()
            if not found:
                 val = None
            
            # Replicate for all agents in this env
            for _ in range(self.agents_per_env):
                results.append(val)
                
        if indices is None:
            return results
        else:
            if isinstance(indices, int):
                target_indices = [indices]
            elif isinstance(indices, slice):
                target_indices = list(range(self.total_agents))[indices]
            else:
                target_indices = indices
                
            return [results[i] for i in target_indices]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Not supported in async wrapper")
