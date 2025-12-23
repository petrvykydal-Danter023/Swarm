import sys
import os
import numpy as np
import imageio

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from topdown_engine import TopDownSwarmEnv

def test_v4_basic():
    print("Testing Entropy Engine V4...")
    
    config = {
        "world_width": 100,
        "world_height": 100,
        "num_agents": 5,
        "dt": 0.1,
        "friction": 0.1,
        "action_type": "continuous",
        "spawn_zone": {"x": 40, "y": 40, "w": 20, "h": 20},
        "special_objects": [
            {"type": "obstacle", "x": 50, "y": 80, "radius": 5},
            {"type": "goal", "x": 90, "y": 50, "radius": 2}
        ]
    }
    
    env = TopDownSwarmEnv(config)
    obs, info = env.reset(seed=42)
    
    print(f"Env Initialized. Agents: {len(env.agents)}")
    print(f"Agent 0 Pos: ({env.agents[0].x:.2f}, {env.agents[0].y:.2f})")
    
    frames = []
    
    # Run loop
    for i in range(50):
        # Action: Move randomly but biased to Right (+X)
        actions = np.zeros((5, 3)) 
        actions[:, 0] = 0.5 + np.random.uniform(-0.1, 0.1, 5) # VX = 0.5
        actions[:, 1] = np.random.uniform(-0.5, 0.5, 5)     # VY = Random
        
        obs, reward, term, trunc, info = env.step(actions)
        
        if i % 10 == 0:
            print(f"Step {i}: Agent 0 Pos: ({env.agents[0].x:.2f}, {env.agents[0].y:.2f})")
        
        frames.append(env.render())
        
    print("Saving v4_test.gif...")
    imageio.mimsave("videos/v4_test.gif", frames, duration=0.1)
    print("Done!")

if __name__ == "__main__":
    test_v4_basic()
