
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestGrabbingPhysics(unittest.TestCase):
    def test_elastic_collision_no_grab(self):
        """Test standard elastic collision (bouncing) when NOT grabbing."""
        config = {
            "num_agents": 1,
            "special_objects": [{"type": "payload", "x": 60, "y": 50, "radius": 5, "mass": 10.0}],
            "friction": 0.0, # zero friction for easier momentum calc
            "payload_friction": 0.0
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        agent = env.agents[0]
        payload = env.payloads[0]
        
        # Setup: Agent moves right towards Payload
        agent.x, agent.y = 40.0, 50.0
        agent.vx = 2.0
        agent.mass = 1.0 # default
        agent.is_grabbing = False
        
        payload.x, payload.y = 60.0, 50.0
        payload.vx = 0.0
        
        # Run until collision (dist < 2+5=7). current dist 20. approach speed 2. 10 steps.
        for _ in range(15):
            env.step(np.zeros((1, 4)))
            
        # Agent should have bounced back (vx < 0) because payload is heavier (1.0 vs 10.0)
        # Elastic collision 1D:
        # v1' = (m1-m2)/(m1+m2)*v1 = (1-10)/11 * 2 = -9/11 * 2 = -1.63
        # v2' = 2m1/(m1+m2)*v1 = 2/11 * 2 = 0.36
        
        self.assertTrue(agent.vx < 0, f"Agent should bounce back. vx={agent.vx}")
        self.assertTrue(payload.vx > 0, f"Payload should move forward. vx={payload.vx}")

    def test_sticky_collision_with_grab(self):
        """Test inelastic collision (sticking) when grabbing."""
        config = {
            "num_agents": 1,
            "special_objects": [{"type": "payload", "x": 60, "y": 50, "radius": 5, "mass": 10.0}],
            "friction": 0.0,
            "payload_friction": 0.0
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        agent = env.agents[0]
        payload = env.payloads[0]
        
        # Setup
        agent.x, agent.y = 40.0, 50.0
        agent.vx = 2.0
        agent.is_grabbing = True # ACTIVE GRAB
        
        payload.x, payload.y = 60.0, 50.0
        
        # Run until collision
        for _ in range(15):
             # Keep sending grab action!
             actions = np.zeros((1, 4))
             actions[0, 2] = 1.0 # Grab button
             env.step(actions)
             
        # Inelastic collision (Stick):
        # v_final = (m1*v1 + m2*v2) / (m1+m2) = (1*2 + 10*0) / 11 = 2/11 = 0.1818
        
        print(f"DEBUG: Sticky Agent vx={agent.vx}, Payload vx={payload.vx}")
        
        self.assertAlmostEqual(agent.vx, payload.vx, places=3, msg="Agent and Payload should have same velocity")
        self.assertTrue(agent.vx > 0, "Velocity should be positive")
        self.assertAlmostEqual(agent.vx, 2.0/11.0, places=2, msg="Velocity should match inelastic collision formula")

if __name__ == '__main__':
    unittest.main()
