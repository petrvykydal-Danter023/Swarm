
import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

class TestMotorLag(unittest.TestCase):
    def test_latency_delay(self):
        """
        Test that actions are delayed by exactly 'motor_lag' steps.
        motor_lag = 1.
        Step 1: Send MOVE. Executed: REST.
        Step 2: Send MOVE. Executed: MOVE.
        """
        config = {
            "num_agents": 1,
            "motor_lag": 1,
            "friction": 0.0, # zero friction to see velocity limit clearly
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        
        agent = env.agents[0]
        # Ensure start at rest
        agent.vx, agent.vy = 0.0, 0.0
        
        # Action: Move Right Full Speed
        action_move = np.zeros((1, 4))
        action_move[0, 0] = 1.0 
        
        # Step 1: Send Move. 
        # Due to lag 1, the applied action should be the initial ZERO vector.
        env.step(action_move)
        
        print(f"DEBUG: Step 1 vx={agent.vx}")
        self.assertAlmostEqual(agent.vx, 0.0, places=4, msg="Step 1: Velocity should be 0 due to lag")
        
        # Step 2: Send Move again (or anything).
        # Now the action from Step 1 should be applied.
        env.step(action_move)
        
        print(f"DEBUG: Step 2 vx={agent.vx}")
        self.assertGreater(agent.vx, 0.0, "Step 2: Velocity should increase as Action 1 is applied")
        
    def test_latency_delay_longer(self):
        """
        Test with larger lag. motor_lag = 3.
        Steps 1, 2, 3: Send MOVE. Result: REST.
        Step 4: Result: MOVE.
        """
        config = {
            "num_agents": 1,
            "motor_lag": 3,
            "friction": 0.0,
        }
        env = TopDownSwarmEnv(config)
        env.reset()
        agent = env.agents[0]
        agent.vx = 0.0
        
        action_move = np.zeros((1, 4))
        action_move[0, 0] = 1.0 
        
        # Steps 1, 2, 3 should see no movement
        for k in range(3):
            env.step(action_move)
            self.assertAlmostEqual(agent.vx, 0.0, places=4, msg=f"Step {k+1}: Should be 0 lag")
            
        # Step 4: Action from Step 1 applies
        env.step(action_move)
        self.assertGreater(agent.vx, 0.0, "Step 4: Should move")

if __name__ == '__main__':
    unittest.main()
