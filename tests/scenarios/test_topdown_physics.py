
import unittest
import numpy as np
import sys
import os
import math

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestTopDownPhysics(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_agents": 1,
            "world_width": 100,
            "world_height": 100,
            "enable_communication": True,
            "action_type": "continuous",
            "sensors": ["velocity", "grabbing_state", "energy"],
            "friction": 0.0 # Disable friction for easier math
        }
        self.env = TopDownSwarmEnv(self.config)
        self.env.reset()

    def test_energy_sensor_presence(self):
        obs, _ = self.env.reset()
        # Obs dim: velocity(2) + grabbing(1) + energy(1) = 4
        self.assertEqual(obs.shape[1], 4)
        # Energy should be at index 3 (last)
        self.assertEqual(obs[0, 3], 1.0) 

    def test_movement_cost(self):
        agent = self.env.agents[0]
        agent.energy = 1.0
        agent.vx = 10.0 # Speed 10
        agent.vy = 0.0
        
        # Action: Coasting (0 force)
        actions = np.array([[0.0, 0.0, 0.0, 0.0]])
        
        self.env.step(actions)
        
        # Drain = speed(10) * 0.005 = 0.05
        # Regen = 0.001
        # New energy = 1.0 - 0.05 + 0.001 = 0.951
        self.assertAlmostEqual(agent.energy, 0.951, places=3)

    def test_grab_cost(self):
        agent = self.env.agents[0]
        agent.energy = 1.0
        
        # Action: Grab (idx 2 > 0.5)
        actions = np.array([[0.0, 0.0, 1.0, 0.0]])
        
        self.env.step(actions)
        
        # Drain: 0.002 (grab)
        # Regen: 0.001
        # Net: -0.001
        self.assertAlmostEqual(agent.energy, 0.999, places=4)

    def test_signal_cost(self):
        agent = self.env.agents[0]
        agent.energy = 1.0
        
        # Action: Signal 1.0 (idx 3) + Grab seems tricky if I want net loss higher than regen?
        # Let's just do Signal 1.0.
        # Cost = 0.001 * 1.0 = 0.001. Regen = 0.001. Net 0.
        # Expected: 1.0
        
        actions = np.array([[0.0, 0.0, 0.0, 1.0]])
        self.env.step(actions)
        self.assertAlmostEqual(agent.energy, 1.0, places=4)
        
        # Now Signal + Grab
        # Actions: grab=1.0, signal=1.0
        actions = np.array([[0.0, 0.0, 1.0, 1.0]])
        
        # Reset energy to 1.0 just in case
        agent.energy = 1.0
        self.env.step(actions)
        
        # Drain: 0.002 (grab) + 0.001 (signal) = 0.003
        # Regen: 0.001
        # Net: -0.002
        self.assertAlmostEqual(agent.energy, 0.998, places=4)

    def test_low_battery_penalty(self):
        agent = self.env.agents[0]
        agent.energy = 0.01 # Low battery
        agent.vx = 0
        agent.vy = 0
        
        # Accelerate hard [1.0, 0, 0, 0]
        actions = np.array([[1.0, 0.0, 0.0, 0.0]])
        
        # Logic:
        # ax = 1.0
        # Energy < 0.05 => ax *= 0.2 => 0.2
        # vx += ax(0.2) * speed(1.0) * dt(0.1) = 0.02
        # Friction is 0.0 now!
        # final vx = 0.02
        
        self.env.step(actions)
        
        self.assertAlmostEqual(agent.vx, 0.02, places=4)

if __name__ == '__main__':
    unittest.main()
