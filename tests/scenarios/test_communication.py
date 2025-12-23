
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestCommunication(unittest.TestCase):
    def test_comm_range(self):
        config = {
            "num_agents": 3,
            "world_width": 200,
            "world_height": 200,
            "enable_communication": True,
            "comm_range": 50.0,
            "sensors": ["neighbor_signals"] # only signals to simplify index
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        # Setup positions
        # Agent 0: (50, 50) - Listener
        # Agent 1: (50+40, 50) = (90, 50) -> Dist 40 (In Range)
        # Agent 2: (50+60, 50) = (110, 50) -> Dist 60 (Out of Range)
        
        env.agents[0].x = 50.0; env.agents[0].y = 50.0
        env.agents[1].x = 90.0; env.agents[1].y = 50.0
        env.agents[2].x = 110.0; env.agents[2].y = 50.0
        
        # Agents broadcast
        # Act 0: Silent
        # Act 1: Signal 1.0
        # Act 2: Signal 0.5
        actions = np.zeros((3, 4))
        actions[1, 3] = 1.0
        actions[2, 3] = 0.5
        
        # Step
        obs, _, _, _, _ = env.step(actions)
        
        # Check Agent 0 obs
        # neighbor_signals size is 3 (max neighbors).
        # Should see Agent 1 (Dist 40).
        # Should NOT see Agent 2 (Dist 60 > 50).
        
        # The sensor sorts neighbors by distance.
        # Neighbors of 0:
        # - Agent 1 (Dist 40) -> Index 0 in sensor
        # - Agent 2 (Dist 60) -> Index 1 in sensor
        
        signals = obs[0]
        # signals[0] is nearest neighbor (Agent 1) -> Should be 1.0
        self.assertAlmostEqual(signals[0], 1.0, msg="Agent 0 should hear Agent 1 (In Range)")
        
        # signals[1] is 2nd nearest (Agent 2) -> Should be 0.0 (Out of Range)
        self.assertAlmostEqual(signals[1], 0.0, msg="Agent 0 should NOT hear Agent 2 (Out of Range)")

    def test_packet_loss(self):
        config = {
            "num_agents": 2,
            "enable_communication": True,
            "packet_loss_prob": 1.0, # 100% loss
            "sensors": ["neighbor_signals"]
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        # Close together
        env.agents[0].x = 50.0
        env.agents[1].x = 51.0 
        
        # Agent 1 broadcasts
        actions = np.zeros((2, 4))
        actions[1, 3] = 1.0
        
        obs, _, _, _, _ = env.step(actions)
        
        signals = obs[0]
        # Should be 0 despite being close and broadcasting
        self.assertEqual(signals[0], 0.0, "Should lose packet due to 100% loss probability")
        
    def test_no_packet_loss(self):
        config = {
            "num_agents": 2,
            "enable_communication": True,
            "packet_loss_prob": 0.0, # 0% loss
            "sensors": ["neighbor_signals"]
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        env.agents[0].x = 50.0
        env.agents[1].x = 51.0
        
        actions = np.zeros((2, 4))
        actions[1, 3] = 0.8
        
        obs, _, _, _, _ = env.step(actions)
        
        self.assertEqual(obs[0, 0], 0.8, "Should receive packet perfectly with 0% loss")

if __name__ == '__main__':
    unittest.main()
