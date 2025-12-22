"""Bridge building test - agents form a chain across a gap with gravity."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "bridge_test",
    "description": "Agents form a human bridge across a gap",
    "observation_type": "spatial",
    "action_space_type": "discrete",  # discrete with grab action
    "env_params": {
        "world_width": 100,
        "world_height": 50,
        "num_agents": 12,
        "physics": {
            "gravity_y": 1.5,  # Gravity pulling down
            "friction": 0.25,
            "time_step": 0.1
        },
        "special_objects": [
            {"type": "gap", "x1": 40, "x2": 60, "y": 25},
            {"type": "goal", "x": 90, "y": 25}
        ]
    },
    "reward_code": """
# Bridge building reward - form chain across gap
reward = 0.0

# Find gap
gap = None
for obj in env_state.get('gaps', []):
    gap = obj
    break

if gap:
    gap_x1, gap_x2 = gap['x1'], gap['x2']
    gap_y = gap['y']
    gap_center = (gap_x1 + gap_x2) / 2
    
    # Check if agent is in gap zone
    in_gap = gap_x1 <= agent['x'] <= gap_x2
    near_gap = gap_x1 - 10 <= agent['x'] <= gap_x2 + 10
    
    # Reward for being near gap level (not falling)
    height_diff = abs(agent['y'] - gap_y)
    if height_diff < 3:
        reward += 1.0
    elif height_diff < 10:
        reward += 0.3
    
    # Big penalty for falling (below gap)
    if agent['y'] < gap_y - 15:
        reward = -10.0
    
    # Reward for grabbing in gap zone
    if in_gap and agent['is_grabbing']:
        reward += 3.0
        
        # Extra reward for being near other grabbing agents (chain)
        for neighbor in env_state['neighbors']:
            n_in_gap = gap_x1 <= neighbor['x'] <= gap_x2
            if n_in_gap and neighbor['is_grabbing']:
                dist = math.sqrt((agent['x']-neighbor['x'])**2 + (agent['y']-neighbor['y'])**2)
                if dist < 8:
                    reward += 2.0
    
    # Encourage movement toward gap if not there
    if not near_gap:
        dist_to_gap = min(abs(agent['x'] - gap_x1), abs(agent['x'] - gap_x2))
        reward -= dist_to_gap / 50.0
else:
    # No gap, just move toward goal
    if env_state['goals']:
        goal = env_state['goals'][0]
        dist = math.sqrt((goal['x']-agent['x'])**2 + (goal['y']-agent['y'])**2)
        reward = -dist / 100.0
""",
    "training_params": {
        "total_timesteps": 30000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 400
    }
}

print("=" * 50)
print("BRIDGE BUILDING TEST")
print("=" * 50)
print(f"World: {config['env_params']['world_width']}x{config['env_params']['world_height']}")
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Gravity: {config['env_params']['physics']['gravity_y']}")
print(f"Gap: x={config['env_params']['special_objects'][0]['x1']}-{config['env_params']['special_objects'][0]['x2']}")
print(f"Action type: {config['action_space_type']} (with grab)")
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
