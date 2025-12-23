
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestRadarOcclusion(unittest.TestCase):
    def test_occlusion(self):
        """
        Test that a closer object blocks the ray to a farther object.
        """
        config = {
            "num_agents": 1,
            # Obstacle 1 (Near): dist 10
            # Obstacle 2 (Far): dist 20
            # Both on x-axis (angle 0)
            "special_objects": [
                {"type": "obstacle", "x": 60, "y": 50, "radius": 5}, # Dist 10 from (50,50), hits at 60-5=55. Dist=5
                {"type": "obstacle", "x": 70, "y": 50, "radius": 5}  # Dist 20 from (50,50)
            ],
            "sensors": ["obstacle_radar"],
            "comm_range": 1000.0
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        agent = env.agents[0]
        agent.x, agent.y = 50.0, 50.0
        
        # Rays are at [0, 45, 90, ...]. 
        # Ray 0 is angle 0 (Right).
        
        # Calculate expected hit
        # Obstacle 1 center at 60. Radius 5. Surface at 55.
        # Agent at 50.
        # Distance to surface = 5.0.
        # Max Range = 30.0.
        # Expected Radar Value = 5.0 / 30.0 = 0.1666...
        
        # Obstacle 2 is strictly behind Obstacle 1.
        # If Occlusion works, we see distance 5.0.
        # If Occlusion fails (see through), we might see Obstacle 2? Or if logic was "min(dists)", we see 5.0 anyway.
        # Wait, if logic is "min(dists)", that IS occlusion logic for 1D ray. 
        # The key difference from old radar: old radar checked sectors. Two small objects in same sector at different dists...
        # In Raycasting, if Object 2 was slightly offset but still in cone, Ray might miss it if point-thin.
        
        obs = env._get_obs()
        radar = obs[0, 0:8]
        
        # Ray 0 (Right)
        print(f"DEBUG: Radar[0] = {radar[0]}")
        
        # dist to surface = 5
        expected_val = 5.0 / 30.0
        self.assertAlmostEqual(radar[0], expected_val, places=4, msg="Should hit the First obstacle surface")
        
        # To strictly prove occlusion logic vs 'sector' logic:
        # Move Obstacle 1 slightly UP so it doesn't block Ray 0 perfectly?
        # No, Raycasting is precise. If I move Obstacle 1 up by 6 (radius 5), Ray 0 misses it.
        # Obstacle 2 at y=50. 
        # Test: Move Obs 1 out of way.
        
        env.obstacles[0].y = 1000.0 # Teleport away
        obs = env._get_obs()
        radar = obs[0, 0:8]
        # Now should hit Obs 2. Wall at 70-5=65. Dist = 15.
        expected_val_2 = 15.0 / 30.0
        print(f"DEBUG: Radar[0] after clear = {radar[0]}")
        self.assertAlmostEqual(radar[0], expected_val_2, places=4, msg="Should hit the Second obstacle when First is gone")

        # Test Miss
        env.obstacles[1].y = 1000.0
        obs = env._get_obs()
        radar = obs[0, 0:8]
        self.assertAlmostEqual(radar[0], 1.0, places=4, msg="Should miss everything (Max Range)")

if __name__ == '__main__':
    unittest.main()
