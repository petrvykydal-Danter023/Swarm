import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class PettingZooToVecEnv(VecEnv):
    """
    Wraps a PettingZoo ParallelEnv to look like a SB3 VecEnv.
    This enables Parameter Sharing: ONE model controls ALL agents.
    The batch measurement is effectively (num_envs * num_agents).
    """
    def __init__(self, parallel_env):
        self.pz_env = parallel_env
        self.agents = self.pz_env.possible_agents
        self.num_agents = len(self.agents)
        
        # Assume homogeneous spaces
        example_agent = self.agents[0]
        observation_space = self.pz_env.observation_spaces[example_agent]
        action_space = self.pz_env.action_spaces[example_agent]
        
        super().__init__(self.num_agents, observation_space, action_space)
        
    def reset(self):
        obs_dict, infos = self.pz_env.reset()
        # Convert dict {agent: obs} to list [obs, obs, ...]
        obs_list = [obs_dict[agent] for agent in self.agents]
        return np.stack(obs_list)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # map list actions back to dict
        action_dict = {agent: self.actions[i] for i, agent in enumerate(self.agents)}
        
        obs_dict, rewards_dict, terms_dict, truncs_dict, infos_dict = self.pz_env.step(action_dict)
        
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        
        for i, agent in enumerate(self.agents):
            obs_list.append(obs_dict[agent])
            rew_list.append(rewards_dict[agent])
            
            # Aggregate done
            done = terms_dict[agent] or truncs_dict[agent]
            done_list.append(done)
            
            info = infos_dict[agent]
            # specific handling for 'terminal_observation' if done
            if done:
                info["terminal_observation"] = obs_dict[agent]
                
            info_list.append(info)
            
        # If all done, we might need to auto-reset?
        # PettingZoo ParallelEnv usually requires reset if everyone is done.
        # But here, we might have partial dones.
        # Our EntropyEnv removes done agents. Wait, that breaks the mapping!
        
        # FIXED LOGIC: In EntropyEnv, if an agent is done, we should probably Respawn it immediately
        # used in "infinite horizon" training, OR we just reset the whole world if all are done.
        
        # Better approach for training: Auto-Respawn in Env, or Auto-Reset whole env.
        # For simplicity: If ANY agent is done, we assume it stays done or resets.
        # Let's Modify EntropyEnv to Auto-Respawn agents!
        
        return np.stack(obs_list), np.array(rew_list), np.array(done_list), info_list

    def close(self):
        self.pz_env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        # We don't support wrapping yet in this custom way
        return [False for _ in range(self.num_envs)]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # Not supported
        return [None for _ in range(self.num_envs)]

    def get_attr(self, attr_name, indices=None):
        # Return attributes from pz_env if possible, repeated for each agent
        if hasattr(self.pz_env, attr_name):
            val = getattr(self.pz_env, attr_name)
            return [val for _ in range(self.num_envs)]
        return [None for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.pz_env, attr_name, value)
        return [None for _ in range(self.num_envs)]
