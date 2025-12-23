"""
Bridge building with COOPERATIVE reward function.
Fixed version with proper variable scoping.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "bridge_coop",
    "description": "Cooperative bridge building - reward based on team success",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 100,
        "world_height": 50,
        "num_agents": 12,
        "physics": {
            "gravity_y": 1.5,
            "friction": 0.25,
            "time_step": 0.1
        },
        "special_objects": [
            {"type": "gap", "x1": 40, "x2": 60, "y": 25},
            {"type": "goal", "x": 90, "y": 25}
        ]
    },
    "reward_code": """
# === COOPERATIVE BRIDGE REWARD ===
reward = 0.0
gap_x1, gap_x2, gap_y = 40, 60, 25  # hardcoded from config for simplicity

all_agents = env_state['all_agents']

# Count agents in bridge zone (above gap)
agents_in_zone = [a for a in all_agents if gap_x1 - 5 <= a['x'] <= gap_x2 + 5 and a['y'] >= gap_y - 5]
num_in_zone = len(agents_in_zone)

# Count grabbing agents
grabbing_agents = [a for a in agents_in_zone if a['is_grabbing']]
num_grabbing = len(grabbing_agents)

# Check chain connectivity
chain_connected = False
chain_spans = False
if len(grabbing_agents) >= 2:
    sorted_agents = sorted(grabbing_agents, key=lambda a: a['x'])
    max_gap = 0
    for i in range(len(sorted_agents) - 1):
        dx = sorted_agents[i+1]['x'] - sorted_agents[i]['x']
        if dx > max_gap:
            max_gap = dx
    chain_connected = max_gap < 8
    chain_spans = sorted_agents[0]['x'] <= gap_x1 + 5 and sorted_agents[-1]['x'] >= gap_x2 - 5

# Count fallen agents
fallen = sum(1 for a in all_agents if a['y'] < gap_y - 15)

# === SHARED TEAM REWARD ===
reward = num_in_zone * 0.5
reward += num_grabbing * 1.0
if chain_connected:
    reward += 5.0
if chain_connected and chain_spans:
    reward += 20.0
reward -= fallen * 3.0

# Small personal incentives
in_zone = gap_x1 - 10 <= agent['x'] <= gap_x2 + 10
if not in_zone:
    dist_to_gap = min(abs(agent['x'] - gap_x1), abs(agent['x'] - gap_x2))
    reward -= dist_to_gap / 100.0
if agent['y'] < gap_y - 15:
    reward -= 5.0
""",
    "training_params": {
        "total_timesteps": 50000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 400
    }
}

print("=" * 60)
print("🌉 COOPERATIVE BRIDGE (FIXED)")
print("=" * 60)
print(f"Agents: {config['env_params']['num_agents']}, Timesteps: {config['training_params']['total_timesteps']}")
print("=" * 60)

result = train_task(config)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Status: {result['status']}")
print(f"Video: {result.get('video_path', 'N/A')}")
if result.get('metrics'):
    m = result['metrics']
    print(f"Mean training reward: {m.get('mean_reward', 'N/A'):.2f}")
    print(f"Eval reward: {m.get('eval_reward', 'N/A'):.2f}")
    print(f"Eval steps: {m.get('eval_steps', 'N/A')}")
if result.get('error'):
    print(f"Error: {result['error']}")
