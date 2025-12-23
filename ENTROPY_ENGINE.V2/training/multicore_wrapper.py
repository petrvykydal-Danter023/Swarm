import multiprocessing as mp
from multiprocessing import shared_memory
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

def worker_process(remote, config, shm_name, shm_shape, shm_dtype, worker_idx):
    """
    Worker process running a single EntropyEnv instance.
    Writes observations directly to Shared Memory.
    """
    shm = None
    try:
        # Attach to existing Shared Memory
        shm = shared_memory.SharedMemory(name=shm_name)
        # Create a numpy array backed by shared memory
        # Shape: (n_envs, agents_per_env, obs_dim)
        # We only care about our slice: [worker_idx, :, :]
        shm_array = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
        
        # Local slice for this worker
        # shape: (agents_per_env, obs_dim)
        worker_obs_buffer = shm_array[worker_idx]
        
        env = EntropyEnv(**config)
        
        # Cache sorted agent keys to ensure consistent order
        # Assuming agent keys don't change names (agent_0, agent_1...)
        sorted_agent_keys = sorted(env.possible_agents)
        
        while True:
            cmd, data = remote.recv()
            
            if cmd == Command.RESET:
                obs, info = env.reset()
                
                # Write to Shared Memory
                for i, agent_id in enumerate(sorted_agent_keys):
                    worker_obs_buffer[i] = obs[agent_id]
                
                # Send info only (obs is in SHM)
                remote.send(info)
                
            elif cmd == Command.STEP:
                actions = data # Dict {agent_id: action}
                obs, rewards, term, trunc, info = env.step(actions)
                
                # Write to Shared Memory
                for i, agent_id in enumerate(sorted_agent_keys):
                    if agent_id in obs:
                         worker_obs_buffer[i] = obs[agent_id]
                    # If agent is dead/missing, what to write? 
                    # SB3 handles dead agents via dones. We can leave stale data or zero it.
                    # EntropyEnv agents persist currently.
                
                remote.send((rewards, term, trunc, info))
                
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
        raise e
    finally:
        if shm:
            shm.close()

class AsyncVectorizedEntropyEnv(VecEnv):
    """
    Spawns N processes, each running an EntropyEnv with M agents.
    Exposes a flattened VecEnv interface for SB3 (Total agents = N * M).
    Uses Shared Memory for high-speed observation transfer.
    """
    def __init__(self, n_envs: int, agents_per_env: int, env_config: Dict[str, Any] = None):
        self.n_envs = n_envs # Number of processes
        self.agents_per_env = agents_per_env # Agents per process
        self.total_agents = n_envs * agents_per_env
        
        if env_config is None:
            env_config = {}
        env_config["nr_agents"] = agents_per_env
        env_config["render_mode"] = None # Workers don't render
        
        # Define Spaces & Init Shared Memory
        # We need to instantiate a dummy env to get spaces
        dummy_env = EntropyEnv(**env_config)
        obs_space = dummy_env.observation_space(dummy_env.possible_agents[0])
        act_space = dummy_env.action_space(dummy_env.possible_agents[0])
        dummy_env.close()
        
        # Shared Memory Allocation
        # Buffer Shape: (n_envs, agents_per_env, obs_dim)
        obs_dim = obs_space.shape[0] if len(obs_space.shape) > 0 else 1
        self.shm_shape = (n_envs, agents_per_env, obs_dim)
        self.shm_dtype = np.float32 # Assuming float32 obs
        
        # Calculate size in bytes
        # itemsize is usually 4 for float32
        data_size = int(np.prod(self.shm_shape) * np.dtype(self.shm_dtype).itemsize)
        
        self.shm = shared_memory.SharedMemory(create=True, size=data_size)
        
        # Create the main array wrapper
        self.shm_array = np.ndarray(self.shm_shape, dtype=self.shm_dtype, buffer=self.shm.buf)
        
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.processes = []
        
        for i, work_remote in enumerate(self.work_remotes):
            # Pass SHM info to worker
            args = (work_remote, env_config, self.shm.name, self.shm_shape, self.shm_dtype, i)
            p = mp.Process(target=worker_process, args=args)
            p.daemon = True
            p.start()
            self.processes.append(p)
            
        super().__init__(self.total_agents, obs_space, act_space)
        
        # Pre-calculate flattened index map (though simple reshape works for full batch)
        # self.shm_array.reshape(total_agents, -1) is sufficient if order is preserved.
        
    def reset(self):
        for remote in self.remotes:
            remote.send((Command.RESET, None))
            
        # Collect results
        # Workers write obs to SHM, send Info via Pipe
        
        for remote in self.remotes:
            info = remote.recv()
            # We can aggregate infos if needed, but SB3 reset() usually returns obs only? 
            # SB3 VecEnv reset() returns obs.
            
        # Read full obs batch from Shared Memory
        # Flatten (n_envs, agents_per, obs_dim) -> (total_agents, obs_dim)
        flat_obs = self.shm_array.reshape(self.total_agents, -1).copy()
        
        return flat_obs
    
    def step_async(self, actions):
        # actions: (total_agents, action_dim)
        
        # Reshape to (n_envs, agents_per, action_dim) to split easily
        # Assuming input order matches our flattened order
        actions_reshaped = actions.reshape(self.n_envs, self.agents_per_env, -1)
        
        # We need to send Dict{agent_id: action} to workers
        # Helper to map index 0..M to agent_0..agent_M
        agent_ids = [f"agent_{i}" for i in range(self.agents_per_env)]
        
        for i, remote in enumerate(self.remotes):
            # Create dict for this env
            # This little loop might be slow in python?
            # Creating dict is fast enough for 10-100 items.
            env_actions = {aid: actions_reshaped[i, j] for j, aid in enumerate(agent_ids)}
            remote.send((Command.STEP, env_actions))

    def step_wait(self):
        flat_rewards = []
        flat_dones = []
        flat_infos = []
        
        # Workers wrote Obs to SHM. Waiting for Rewards/Dones.
        for i, remote in enumerate(self.remotes):
            # recv: (rewards, term, trunc, info)
            rew, term, trunc, info = remote.recv()
            
            # rew, term, trunc are Dicts
            sorted_keys = sorted(rew.keys()) # Should match agent_0...agent_M
            
            for agent_id in sorted_keys:
                flat_rewards.append(rew[agent_id])
                is_done = term[agent_id] or trunc[agent_id]
                flat_dones.append(is_done)
                
                inf = info.get(agent_id, {})
                if is_done:
                    # Terminal obs? 
                    # If done, current obs in SHM is the reset obs (auto-reset).
                    # SB3 wants terminal_observation in info.
                    # We'd need worker to write terminal obs to SHM before reset?
                    # Or worker sends terminal obs in info?
                    # EntropyEnv doesn't auto-reset per agent usually?
                    # Actually PettingZoo/SB3 wrapper logic:
                    # If we treat it as VecEnv, we usually auto-reset DONE envs.
                    # But here agents are individual.
                    # For now: let's assume standard behavior.
                    pass
                flat_infos.append(inf)
                
        # Read Obs from SHM
        flat_obs = self.shm_array.reshape(self.total_agents, -1).copy()
        
        return flat_obs, np.array(flat_rewards), np.array(flat_dones), flat_infos

    def close(self):
        for remote in self.remotes:
            remote.send((Command.CLOSE, None))
        for p in self.processes:
            p.join()
            
        # Clean up Shared Memory
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink() # Delete from OS
            
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.total_agents

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Not supported in async wrapper")

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment."""
        for remote in self.remotes:
            remote.send((Command.GET_ATTR, attr_name))
            
        results = []
        for remote in self.remotes:
            found, val = remote.recv()
            if not found: val = None
            for _ in range(self.agents_per_env):
                results.append(val)
        
        if indices is None:
            return results
        else:
            # Handle indices logic if needed
            return results 

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Not supported in async wrapper")
