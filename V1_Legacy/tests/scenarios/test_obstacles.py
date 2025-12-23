"""Test with obstacles - more challenging configuration."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "obstacle_test",
    "description": "Agents navigate around obstacles to reach goal",
    "observation_type": "spatial",
    "action_space_type": "continuous",
    "env_params": {
        "world_width": 80,
        "world_height": 80,
        "num_agents": 8,
        "physics": {
            "gravity_y": 0.0,
            "friction": 0.15,
            "time_step": 0.1
        },
        "special_objects": [
            {"type": "goal", "x": 70, "y": 70},
            {"type": "obstacle", "x": 40, "y": 40, "radius": 8},
            {"type": "obstacle", "x": 25, "y": 55, "radius": 5},
            {"type": "obstacle", "x": 55, "y": 25, "radius": 6},
            {"type": "obstacle", "x": 60, "y": 60, "radius": 4}
        ]
    },
    "reward_code": """
# Navigate to goal while avoiding obstacles
goal = env_state['goals'][0] if env_state['goals'] else None
if goal:
    dx = goal['x'] - agent['x']
    dy = goal['y'] - agent['y']
    distance = math.sqrt(dx*dx + dy*dy)
    
    # Distance-based reward
    reward = -distance / 50.0
    
    # Big bonus for reaching goal
    if distance < 5:
        reward = 20.0
    
    # Reward for movement (anti-stuck)
    speed = math.sqrt(agent['vx']**2 + agent['vy']**2)
    if speed > 1.0:
        reward += 0.1
    
    # Penalty for being too close to obstacles
    for obs in env_state['obstacles']:
        obs_dist = math.sqrt((obs['x']-agent['x'])**2 + (obs['y']-agent['y'])**2)
        if obs_dist < obs['radius'] + 3:
            reward -= 1.0
""",
    "training_params": {
        "total_timesteps": 50000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 300
    }
}

print("=" * 50)
print("OBSTACLE TEST - Harder Configuration")
print("=" * 50)
print(f"World: {config['env_params']['world_width']}x{config['env_params']['world_height']}")
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Obstacles: {len([o for o in config['env_params']['special_objects'] if o['type'] == 'obstacle'])}")
print(f"Timesteps: {config['training_params']['total_timesteps']}")
print("=" * 50)

result = train_task(config)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Status: {result['status']}")
print(f"Video: {result.get('video_path', 'N/A')}")
if result.get('metrics'):
    m = result['metrics']
    print(f"Mean training reward: {m.get('mean_reward', 'N/A'):.2f}")
    print(f"Eval reward: {m.get('eval_reward', 'N/A'):.2f}")
    print(f"Eval steps: {m.get('eval_steps', 'N/A')}")
    print(f"Episodes: {m.get('episodes', 'N/A')}")
if result.get('error'):
    print(f"Error: {result['error']}")
