import sys
import os
import numpy as np
import imageio
import math

# Add root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from topdown_engine import TopDownSwarmEnv

def test_v4_transport():
    print("Testing Entropy Engine V4 - Cooperative Transport...")
    
    config = {
        "world_width": 100,
        "world_height": 100,
        "num_agents": 5,
        "dt": 0.1,
        "friction": 0.2,
        "payload_friction": 0.05, # Payload slides easier
        "action_type": "continuous",
        "spawn_zone": {"x": 20, "y": 45, "w": 10, "h": 10}, # Spawn left of payload
        "special_objects": [
            {"type": "payload", "x": 50, "y": 50, "radius": 5, "mass": 10.0}, # Central payload
            {"type": "goal", "x": 90, "y": 50, "radius": 2}
        ]
    }
    
    env = TopDownSwarmEnv(config)
    env.reset(seed=42)
    
    print(f"Env Initialized.")
    payload = env.payloads[0]
    print(f"Payload Start: ({payload.x:.2f}, {payload.y:.2f})")
    
    frames = []
    
    # Run loop (100 steps)
    for i in range(100):
        # Action: Move Towards Payload Center (push it right)
        actions = np.zeros((5, 3))
        
        # Simple Logic: each agent targets payload position
        # Actually, we want them to push it +X.
        # So they should target Payload X + Offset? No, just push Right.
        
        for idx, agent in enumerate(env.agents):
            dx = payload.x - agent.x
            dy = payload.y - agent.y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # If behind payload (x < payload.x), push hard right
            # If above/below, adjust Y to match payload Y
            
            actions[idx, 0] = 1.0 # Full speed right
            # Steering to center
            if agent.y > payload.y + 2: actions[idx, 1] = -0.5
            elif agent.y < payload.y - 2: actions[idx, 1] = 0.5
        
        env.step(actions)
        
        if i % 20 == 0:
            print(f"Step {i}: Payload X: {payload.x:.2f}")
        
        frames.append(env.render())
        
    print(f"Final Payload X: {payload.x:.2f}")
    if payload.x > 55:
        print("SUCCESS: Payload moved significantly!")
    else:
        print("FAILURE: Payload barely moved.")
        
    print("Saving v4_transport.gif...")
    imageio.mimsave("videos/v4_transport.gif", frames, duration=0.1)
    print("Done!")

if __name__ == "__main__":
    test_v4_transport()
