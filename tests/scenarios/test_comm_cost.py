
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestCommCost(unittest.TestCase):
    def test_communication_penalty(self):
        """
        Compare two agents:
        1. Silent Agent: Comm signal = 0.0.
        2. Chatty Agent: Comm signal = 1.0.
        
        The reward function explicitly penalizes |comm|.
        Expect Silent Agent Reward > Chatty Agent Reward.
        """
        
        # Reward code that penalizes Comm
        reward_code = """
reward = 0.0
if 'comm' in agent:
    comm_cost = abs(agent['comm']) * 0.1
    reward -= comm_cost
"""
        
        config = {
            "num_agents": 1,
            "action_type": "continuous", # Need continuous for comms usually
            "enable_communication": True,
            "reward_code": reward_code,
            "dt": 0.1
        }
        
        # --- Run 1: Silent Agent ---
        env_silent = TopDownSwarmEnv(config)
        env_silent.reset()
        silent_reward_total = 0
        
        for i in range(5):
            # [vx, vy, grab, comm]
            action = np.zeros((1, 4))
            action[0, 3] = 0.0 # Silent
            _, r, _, _, _ = env_silent.step(action)
            silent_reward_total += r[0]
            
        # --- Run 2: Chatty Agent ---
        env_chatty = TopDownSwarmEnv(config)
        env_chatty.reset()
        chatty_reward_total = 0
        
        for i in range(5):
            action = np.zeros((1, 4))
            action[0, 3] = 1.0 # Shout
            _, r, _, _, _ = env_chatty.step(action)
            chatty_reward_total += r[0]
            
        print(f"Silent Reward: {silent_reward_total}")
        print(f"Chatty Reward (should be negative): {chatty_reward_total}")
        
        self.assertGreater(silent_reward_total, chatty_reward_total, 
                          "Silent agent should have higher reward (less cost)")

if __name__ == '__main__':
    unittest.main()
