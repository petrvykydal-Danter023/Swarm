"""Quick smoke test for the training system."""

import sys
import os
sys.path.append(os.getcwd())

from models.rl_trainer import train_task

config = {
    "task_name": "smoke_test",
    "observation_type": "spatial",
    "action_space_type": "continuous",
    "env_params": {
        "world_width": 50,
        "world_height": 50,
        "num_agents": 3,
        "action_repeat": 4,
        "physics": {},
        "special_objects": [{"type": "goal", "x": 45, "y": 45}]
    },
    "reward_code": """
goal = env_state['goals'][0] if env_state['goals'] else None
if goal:
    dx = goal['x'] - agent['x']
    dy = goal['y'] - agent['y']
    reward = -math.sqrt(dx*dx + dy*dy) / 100.0
""",
    "training_params": {
        "total_timesteps": 500,
        "max_episode_steps": 100
    }
}

result = train_task(config)
print(f"Status: {result['status']}")
print(f"Video: {result.get('video_path', 'N/A')}")
if result.get('metrics'):
    print(f"Metrics: {result['metrics']}")
if result.get('error'):
    print(f"Error: {result['error']}")
