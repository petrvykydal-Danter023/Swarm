
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import timeit
import numpy as np
import gymnasium as gym
from topdown_engine import TopDownSwarmEnv

def benchmark(num_agents=50, steps=1000, vectorized=True):
    print(f"--- Benchmarking: {num_agents} Agents, {steps} Steps ---")
    
    config = {
        "world_width": 100,
        "world_height": 100,
        "num_agents": num_agents,
        "dt": 0.1,
        "sensors": ["position", "velocity", "goal_vector", "neighbor_vectors", "obstacle_radar"],
        "agent_mode": "nav",
        "action_type": "continuous",
        "reward_code": "goal = env_state['goals'][0]\ndx = goal['x'] - agent['x']\ndy = goal['y'] - agent['y']\ndist = math.sqrt(dx**2 + dy**2)\nreward = -dist",
        "special_objects": [{"type": "goal", "x": 50, "y": 50}]
    }
    
    env = TopDownSwarmEnv(config)
    obs, _ = env.reset()
    
    start_time = time.time()
    
    total_rewards = 0
    for i in range(steps):
        # Random actions (N, 3)
        actions = np.random.uniform(-1, 1, (num_agents, 3)).astype(np.float32)
        obs, rewards, term, trunc, info = env.step(actions)
        total_rewards += np.sum(rewards)
        
    end_time = time.time()
    duration = end_time - start_time
    fps = (steps * num_agents) / duration # Agent-Steps per second? Or just Steps/sec?
    # Usually we measure Env Steps per second
    env_fps = steps / duration
    
    print(f"Total Time: {duration:.4f}s")
    print(f"Env FPS: {env_fps:.2f}")
    print(f"Agent Steps/Sec: {env_fps * num_agents:.2f}")
    print(f"Mean Reward: {total_rewards / (steps * num_agents):.4f}")
    return env_fps

if __name__ == "__main__":
    # fast warmup
    benchmark(10, 100)
    
    fps_50 = benchmark(50, 1000)
    fps_100 = benchmark(100, 1000)
    fps_500 = benchmark(500, 100) # Stress Test
