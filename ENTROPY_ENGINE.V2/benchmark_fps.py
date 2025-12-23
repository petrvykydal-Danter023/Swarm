
import time
import numpy as np
from env.entropy_env import EntropyEnv
import os

def run_benchmark(n_steps=500, n_agents=100):
    print(f"Benchmarking with {n_agents} agents for {n_steps} steps...")
    
    # Initialize Env
    env = EntropyEnv(render_mode=None, nr_agents=n_agents)
    env.reset()
    
    # Random actions
    # Pre-generate to avoid sampling overhead in loop
    # actions = {f"agent_{i}": env.action_space(f"agent_{i}").sample() for i in range(n_agents)}
    
    start_time = time.time()
    
    for i in range(n_steps):
        # reuse same actions to minimize overhead
        actions = {f"agent_{j}": np.array([1.0, 1.0]) for j in range(n_agents)} 
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        if i % 100 == 0:
            print(f"Step {i}/{n_steps}")
            
    end_time = time.time()
    duration = end_time - start_time
    fps = n_steps / duration
    
    print(f"Done! {n_steps} steps in {duration:.4f}s")
    print(f"FPS (Steps/sec): {fps:.2f}")
    return fps

if __name__ == "__main__":
    # Check if we are using Numba
    from env.entropy_env import USE_NUMBA
    print(f"USE_NUMBA: {USE_NUMBA}")
    
    run_benchmark()
