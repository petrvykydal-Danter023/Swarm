
import unittest
import numpy as np
import sys
import os
import math

import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))) # for rich_runner

from topdown_engine import TopDownSwarmEnv
from rich_runner import run_episode

class TestStallProtection(unittest.TestCase):
    def test_stall_penalty_visual(self):
        """
        Visual test for Stall Protection using Rich Runner.
        """
        
        # Reward code that penalizes Stall
        reward_code = """
reward = 0.0
speed = math.sqrt(agent['vx']**2 + agent['vy']**2)
action_mag = 0.0
if 'action' in agent:
    action_mag = math.sqrt(agent['action'][0]**2 + agent['action'][1]**2)

# STALL PROTECTION
if action_mag > 0.8 and speed < 0.3: # Relaxed threshold
    reward -= 10.0 # BURNOUT DETECTED
"""
        
        config = {
            "num_agents": 1,
            "action_type": "continuous",
            "reward_code": reward_code,
            "dt": 0.1,
            "physics": {"friction": 0.0}
        }
        
        # --- Run 1: Moving Agent ---
        print("\n=== RUN 1: FREE MOVING AGENT ===")
        env_moving = TopDownSwarmEnv(config)
        env_moving.reset()
        
        def policy_accelerate(step):
            action = np.zeros((1, 3))
            action[0, 0] = 1.0
            return action
            
        moving_score = run_episode(env_moving, policy_accelerate, max_steps=10, title="Free Moving Agent")
        
        # --- Run 2: Stalled Agent ---
        print("\n=== RUN 2: STALLED AGENT ===")
        config_stalled = config.copy()
        config_stalled["special_objects"] = [{"type": "obstacle", "x": 50, "y": 50, "radius": 5}]
        config_stalled["physics"] = {"friction": 5.0} # High friction
        
        env_stalled = TopDownSwarmEnv(config_stalled)
        env_stalled.reset()
        env_stalled.agents[0].x = 44.5 # Blocked pos
        env_stalled.agents[0].y = 50.0
        
        def policy_push_wall(step):
            # Force velocity to 0 to simulate stall for the test purpose logic
            # Note: This hack is specific because we want to test REWARD logic when speed IS low.
            env_stalled.agents[0].vx = 0.0 
            
            action = np.zeros((1, 3))
            action[0, 0] = 1.0
            return action
            
        stalled_score = run_episode(env_stalled, policy_push_wall, max_steps=10, title="Stalled Agent (Pushing Wall)")
        
        self.assertGreater(moving_score, stalled_score)

if __name__ == '__main__':
    unittest.main()

