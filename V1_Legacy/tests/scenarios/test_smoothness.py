
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestSmoothness(unittest.TestCase):
    def test_jerk_penalty(self):
        """
        Compare two agents:
        1. Smooth Agent: Accelerates slowly.
        2. Jerky Agent: Switches direction every step.
        
        The reward function explicitly penalizes |action - last_action|.
        Expect Smooth Agent Total Reward > Jerky Agent Total Reward.
        """
        
        # Reward code that penalizes Jerk
        reward_code = """
reward = 0.0
# Penalize difference between current and last action
if 'last_action' in agent and 'action' in agent:
    diff = agent['action'] - agent['last_action']
    jerk = np.sum(np.abs(diff))
    reward -= jerk * 1.0
"""
        
        config = {
            "num_agents": 1,
            "action_type": "continuous",
            "reward_code": reward_code,
            "dt": 0.1
        }
        
        # --- Run 1: Smooth Agent ---
        env_smooth = TopDownSwarmEnv(config)
        env_smooth.reset()
        smooth_reward_total = 0
        
        # Ramp up: 0.1, 0.2, 0.3...
        for i in range(10):
            action = np.zeros((1, 3))
            action[0, 0] = i * 0.1 
            _, r, _, _, _ = env_smooth.step(action)
            smooth_reward_total += r[0]
            
        # --- Run 2: Jerky Agent ---
        env_jerky = TopDownSwarmEnv(config)
        env_jerky.reset()
        jerky_reward_total = 0
        
        # Vibrate: +1, -1, +1, -1...
        for i in range(10):
            action = np.zeros((1, 3))
            if i % 2 == 0:
                action[0, 0] = 1.0
            else:
                action[0, 0] = -1.0
            _, r, _, _, _ = env_jerky.step(action)
            jerky_reward_total += r[0]
            
        print(f"Smooth Reward: {smooth_reward_total}")
        print(f"Jerky Reward: {jerky_reward_total}")
        
        self.assertGreater(smooth_reward_total, jerky_reward_total, 
                          "Smooth agent should have higher reward (less negative) than jerky agent")

if __name__ == '__main__':
    unittest.main()
