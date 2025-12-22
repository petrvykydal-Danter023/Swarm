"""
Bridge building V2 - Improved cooperative reward.

Changes:
- Goal is ON THE OTHER SIDE of the gap (must cross to reach it)
- Stronger cooperation incentives
- Better physics (lower gravity, more friction)
- Longer training
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "bridge_v2",
    "description": "Cooperative bridge - agents must cross gap to reach goal",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 100,
        "world_height": 50,
        "num_agents": 15,  # More agents for better chain
        "physics": {
            "gravity_y": 0.8,   # Lower gravity - easier to stay up
            "friction": 0.35,   # More friction - less sliding
            "time_step": 0.1
        },
        "special_objects": [
            # Gap in the middle
            {"type": "gap", "x1": 40, "x2": 60, "y": 25},
            # Goal is PAST the gap - forces crossing
            {"type": "goal", "x": 85, "y": 25}
        ]
    },
    "reward_code": """
# === BRIDGE V2 - MUST CROSS GAP TO WIN ===
reward = 0.0
gap_x1, gap_x2, gap_y = 40, 60, 25
goal_x, goal_y = 85, 25

all_agents = env_state['all_agents']

# === COUNT KEY METRICS ===
# Agents on left side (need to cross)
left_side = [a for a in all_agents if a['x'] < gap_x1 and a['y'] >= 0]

# Agents in bridge zone (forming structure)  
in_gap = [a for a in all_agents if gap_x1 <= a['x'] <= gap_x2 and a['y'] >= gap_y - 3]

# Agents on right side (successfully crossed!)
right_side = [a for a in all_agents if a['x'] > gap_x2 and a['y'] >= gap_y - 5]

# Fallen agents
fallen = [a for a in all_agents if a['y'] < gap_y - 15]

# Grabbing agents (forming chain)
grabbing_in_gap = [a for a in in_gap if a['is_grabbing']]

# Check connected chain
chain_exists = False
if len(grabbing_in_gap) >= 3:
    sorted_g = sorted(grabbing_in_gap, key=lambda a: a['x'])
    max_dist = max((sorted_g[i+1]['x'] - sorted_g[i]['x']) for i in range(len(sorted_g)-1))
    chain_exists = max_dist < 10  # agents within 10 units form chain

# Chain spans gap?
chain_spans = False
if chain_exists and len(grabbing_in_gap) >= 4:
    sorted_g = sorted(grabbing_in_gap, key=lambda a: a['x'])
    chain_spans = sorted_g[0]['x'] <= gap_x1 + 8 and sorted_g[-1]['x'] >= gap_x2 - 8

# === GLOBAL TEAM REWARD (same for all!) ===

# Goal: Get agents to the OTHER SIDE
reward = len(right_side) * 10.0  # BIG reward for crossing

# Bonus for agents in bridge zone
reward += len(in_gap) * 2.0

# Bonus for grabbing
reward += len(grabbing_in_gap) * 3.0

# Chain formation bonus
if chain_exists:
    reward += 15.0

# Bridge spanning bonus (key for crossing!)
if chain_spans:
    reward += 30.0

# Penalty for fallen
reward -= len(fallen) * 5.0

# === SMALL INDIVIDUAL GUIDE ===
# Encourage movement toward gap or goal
if agent['x'] < gap_x1 - 15:
    # Too far left - move toward gap
    reward -= (gap_x1 - agent['x']) / 50.0
elif agent['x'] > gap_x2 + 5:
    # On right side - move toward goal
    dx = goal_x - agent['x']
    reward += max(0, 5.0 - abs(dx)/10)

# Personal falling penalty
if agent['y'] < gap_y - 15:
    reward -= 10.0
""",
    "training_params": {
        "total_timesteps": 100000,  # Longer training
        "learning_rate": 0.0003,
        "gamma": 0.995,  # Higher gamma for long-term planning
        "batch_size": 128,
        "n_envs": 8,  # More parallel envs
        "max_episode_steps": 500
    }
}

print("=" * 60)
print("🌉 BRIDGE V2 - MUST CROSS GAP TO WIN")
print("=" * 60)
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Gap: x={config['env_params']['special_objects'][0]['x1']}-{config['env_params']['special_objects'][0]['x2']}")
print(f"Goal: x={config['env_params']['special_objects'][1]['x']} (past the gap!)")
print(f"Gravity: {config['env_params']['physics']['gravity_y']} (lower)")
print(f"Timesteps: {config['training_params']['total_timesteps']}")
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
