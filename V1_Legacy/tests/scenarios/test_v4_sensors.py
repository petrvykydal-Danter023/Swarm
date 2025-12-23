import sys
import os
import numpy as np

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from topdown_engine import TopDownSwarmEnv

def test_sensors():
    print("Testing V4 Sensors & Comms...")
    config = {
        "sensors": ["position", "velocity", "neighbor_vectors", "obstacle_radar", "neighbor_signals"],
        "enable_communication": True,
        "obs_noise_std": 0.05,
        "num_agents": 5,
        "action_type": "continuous",
        "spawn_zone": {"x": 50, "y": 50, "w": 10, "h": 10}
    }
    env = TopDownSwarmEnv(config)
    obs, _ = env.reset(seed=42)
    
    print(f"Obs Dim: {env.obs_dim}") # Expected 21
    print(f"Action Space: {env.action_space.shape}") # Expected (5, 4)
    
    # Check Obs Variance (Noise)
    var = np.var(obs)
    print(f"Obs Variance (Init): {var:.4f}")
    if var > 0.001:
        print("SUCCESS: Noise detected.")
    else:
        print("FAILURE: No noise detected (or agents stacked exactly).")
        
    # Check Comms
    # Step 1: Send signals
    actions = np.zeros((5, 4))
    # Agent 0 sends 1.0, Agent 1 sends -1.0, etc.
    actions[:, 3] = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
    
    obs, _, _, _, _ = env.step(actions)
    
    # Check if Agent 0 perceives Agent 1's signal (-1.0) or others
    # neighbor_signals are last 3 dims (indices 18, 19, 20)
    # Since all agents spawned close, they should see each other.
    
    # We need to find who is neighbor to Agent 0
    # In spawn zone 50,50 w=10, h=10.
    # We can inspect internal state to verify.
    
    print("Agent Signals Sent:", env.comm_signals)
    print("Agent 0 Received Signals:", obs[0, -3:])
    
    if np.any(np.abs(obs[0, -3:]) > 0.1):
         print("SUCCESS: Signals received.")
    else:
         print("FAILURE: No signals received.")

    print("Done")

if __name__ == "__main__":
    test_sensors()
