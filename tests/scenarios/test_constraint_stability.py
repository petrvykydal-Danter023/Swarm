import sys
import os
import numpy as np
import math
sys.path.append(os.getcwd())
from universal_env import UniversalSwarmEnv

def run_test(profile, iterations, label):
    config = {
        "env_params": {
            "world_width": 100,
            "world_height": 100,
            "num_agents": 5,
            "physics_profile": profile,
            "physics": {
                "constraint_iterations": iterations,
                "gravity_y": 0.5,
                "agent_mass": 1.0
            },
            "spawn_zone": {"x1": 50, "x2": 50, "y1": 50, "y2": 50}, # Ignored by manual setup
            "special_objects": []
        },
        "reward_code": "reward=0",
        "training_params": {"max_episode_steps": 200},
        "observation_type": "spatial",
        "action_space_type": "continuous"
    }

    print(f"\n--- Running {label} ---")
    try:
        env = UniversalSwarmEnv(config)
        env.reset()
        
        # Manually form a chain
        # Agent 0 at (50, 90) (Top - Anchor)
        # Agent 1..4 hanging below
        target_dist = 5.0
        
        for i, agent in enumerate(env.agents):
            agent.x = 50.0
            agent.y = 90.0 - i * target_dist
            agent.vx = 0.0
            agent.vy = 0.0
            agent.is_grounded = False
            
            if i > 0:
                # Grab the one above (i-1)
                # Form constraint manually
                env.constraints.append([i, i-1, target_dist])
                agent.active_constraints.append(i-1)
                agent.is_grabbing = True # CRITICAL: Prevent auto-release
        
        # Run simulation
        stretches = []
        for step in range(100):
            # Anchor Agent 0
            env.agents[0].x = 50.0
            env.agents[0].y = 90.0
            env.agents[0].vx = 0
            env.agents[0].vy = 0
            
            # Step with zero actions (gravity only)
            env.step(np.zeros((5, 2)))
            
            # Measure chain length (0 to 4)
            last = env.agents[-1]
            first = env.agents[0]
            dist = math.sqrt((last.x - first.x)**2 + (last.y - first.y)**2)
            stretches.append(dist)
        
        avg_len = np.mean(stretches[-20:])
        ideal = target_dist * 4 # 20.0
        ratio = avg_len / ideal
        print(f"[{label}] Result: Length={avg_len:.2f} (Ideal {ideal}), Stretch Ratio={ratio:.3f}")
        return ratio
        
    except Exception as e:
        print(f"[{label}] Failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    print("Testing Constraint Stability...")
    # 1. Baseline: Realistic, 1 iteration (Default/Old)
    r1 = run_test("realistic", 1, "Realistic (Iter=1)")
    
    # 2. Improved: Realistic, 5 iterations
    r5 = run_test("realistic", 5, "Realistic (Iter=5)")
    
    # 3. Arcade: Arcade, 1 iteration
    a1 = run_test("arcade", 1, "Arcade (Iter=1)")

    print("\nSummary:")
    print(f"Stretch Reduction (5 vs 1): {((r1 - r5)/r1)*100:.1f}% improvement")
