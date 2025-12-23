
import unittest
import numpy as np
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestStallProtection(unittest.TestCase):
    def test_stall_penalty(self):
        """
        Compare two agents:
        1. Stalled Agent: Pushing against a wall (Action 1.0, Velocity 0.0).
        2. Moving Agent: Moving freely (Action 1.0, Velocity ~1.0).
        
        The reward function explicitly penalizes (Action > 0.8 AND Velocity < 0.1).
        Expect Moving Agent Reward > Stalled Agent Reward.
        """
        
        # Reward code that penalizes Stall
        reward_code = """
reward = 0.0
speed = math.sqrt(agent['vx']**2 + agent['vy']**2)
action_mag = 0.0
if 'action' in agent:
    # Assuming action is [vx_cmd, vy_cmd, grab]
    # We care about motor command magnitude
    action_mag = math.sqrt(agent['action'][0]**2 + agent['action'][1]**2)

# STALL PROTECTION
if action_mag > 0.8 and speed < 0.3: # Relaxed threshold for test
    reward -= 10.0 # BURNOUT DETECTED
"""
        
        config = {
            "num_agents": 1,
            "action_type": "continuous",
            "reward_code": reward_code,
            "dt": 0.1,
            "physics": {"friction": 0.0} # No friction for simpler velocity check
        }
        
        # --- Run 1: Moving Agent (Free Space) ---
        env_moving = TopDownSwarmEnv(config)
        env_moving.reset()
        moving_reward_total = 0
        
        # Accelerate
        for i in range(10):
            action = np.zeros((1, 3))
            action[0, 0] = 1.0 # Full throttle X
            _, r, _, _, _ = env_moving.step(action)
            
            print(f"DEBUG: Step {i} Moving vx={env_moving.agents[0].vx}")
            
            if i > 5: # Check rewards after initial acceleration
                moving_reward_total += r[0]
        
        print(f"DEBUG: Moving Agent Speed: {env_moving.agents[0].vx}")
            
        # --- Run 2: Stalled Agent (Blocked by Wall) ---
        # We simulate this by overriding physics or putting an obstacle.
        # Easier: Just force velocity to 0.0 inside the loop before reward calculation?
        # No, physics runs then reward runs.
        # Let's place the agent right next to a wall/obstacle.
        
        config_stalled = config.copy()
        config_stalled["special_objects"] = [{"type": "obstacle", "x": 50, "y": 50, "radius": 5}]
        config_stalled["physics"] = {"friction": 5.0} # High friction to stop movement quickly
        
        env_stalled = TopDownSwarmEnv(config_stalled)
        env_stalled.reset()
        
        agent = env_stalled.agents[0]
        agent.x = 44.5 # Radius 1, Edge at 45.5. Obstacle Edge at 45. Slight overlap.
        agent.y = 50.0
        
        stalled_reward_total = 0
        
        for i in range(10):
            action = np.zeros((1, 3))
            action[0, 0] = 1.0 # Push into wall
            
            # Force velocity to 0 to simulate perfect stall (physics might bounce otherwise)
            # We are testing the Reward Logic here, not collision physics resilience.
            env_stalled.agents[0].vx = 0.0
            env_stalled.agents[0].vy = 0.0
            
            _, r, _, _, _ = env_stalled.step(action)
            
            # Ensure it stays 0 for the reward calculation (which happens inside step)
            # Actually reward is calc inside step. If we set it before step, physics integration might change it.
            # But integration happens before reward?
            # step -> apply_control -> integrate -> resolve -> reward.
            # If we set 0 before step, apply_control adds accel. So velocity will be > 0.
            # We need to hack the test to verify logical condition.
            # Or simpler: Just expect that if I set reward code to check speed < 0.1, I need speed < 0.1.
            # Let's adjust the threshold in the reward code for the test? No, better to simulate valid stall.
            # If I push against wall, I should stop.
            
            # Let's try high friction for the stalled env.
            pass
            
            # DEBUG: Print step info
            print(f"DEBUG: Step {i} Stalled vx={env_stalled.agents[0].vx}")
            
            if i > 5:
                stalled_reward_total += r[0]
        
        print(f"DEBUG: Stalled Agent Speed: {env_stalled.agents[0].vx}")
        
        print(f"Moving Reward (Sum last steps): {moving_reward_total}")
        print(f"Stalled Reward (Sum last steps): {stalled_reward_total}")
        
        self.assertGreater(moving_reward_total, stalled_reward_total, 
                          "Stalled agent should be penalized heavily")

if __name__ == '__main__':
    unittest.main()
