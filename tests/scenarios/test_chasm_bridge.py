"""
CHASM BRIDGE TEST - Agents must form human bridge to cross

Layout:
  RIGHT: All agents spawn here
  MIDDLE: Vertical chasm (full height gap)
  LEFT: Goal/reward

Agents must cooperate to form a bridge across the chasm!
Uses GPU if available.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "chasm_platformer",
    "description": "Platformer Bridge: Wide chasm requires cooperation (Real Physics)",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 100,
        "world_height": 60,
        "num_agents": 20,  # More agents for a longer bridge!
        "physics": {
            "gravity_y": 0.5,
            "friction": 0.4,
            "time_step": 0.1
        },
        # Spawn zone: RIGHT side
        "spawn_zone": {
            "x1": 70, "x2": 95,
            "y1": 0, "y2": 50
        },
        # WIDE Chasm (25 units) - Cannot be jumped solo! based on physics calc
        "special_objects": [
            {"type": "gap", "x1": 35, "x2": 60, "y": 0},
            {"type": "goal", "x": 10, "y": 5}
        ],
        "sensors": [
            "position",
            "velocity", 
            "goal_vector",
            "goal_distance",
            "neighbor_count",
            "neighbor_vectors",
            "wall_distance",
            "grabbing_state"
        ]
    },
    "reward_code": """
# CHASM BRIDGE REWARD (PLATFORMER PHYSICS)
reward = 0.0
chasm_x1, chasm_x2 = 35, 60
goal_x = 10

all_agents = env_state['all_agents']

# === GLOBAL METRICS ===
# Agents on LEFT side (crossed successfully!)
left_side = [a for a in all_agents if a['x'] < chasm_x1 and a['y'] > -5]
num_crossed = len(left_side)

# Agents in chasm zone (forming bridge)
in_chasm = [a for a in all_agents if chasm_x1 <= a['x'] <= chasm_x2 and a['y'] > -10]
num_in_bridge = len(in_chasm)

# Fallen agents (died in pit)
fallen = [a for a in all_agents if a['y'] < -10]
num_fallen = len(fallen)

# Grabbing agents in chasm
grabbing_in_chasm = [a for a in in_chasm if a['is_grabbing']]
num_grabbing = len(grabbing_in_chasm)

# === REWARDS ===
# Massive reward for crossing
reward = num_crossed * 25.0

# Reward for maintaining position over chasm (bridge)
reward += num_in_bridge * 3.0

# Major penalty for dying
reward -= num_fallen * 2.0

# Reward for grabbing in dangerous zone
reward += num_grabbing * 5.0

# === INDIVIDUAL GUIDANCE ===
if agent['y'] < -10:
    reward -= 20.0  # You are dead
else:
    # Encourage movement toward chasm if on right
    if agent['x'] > chasm_x2 + 2:
        reward -= (agent['x'] - chasm_x2) / 40.0
    
    # Tiny reward for moving left
    reward -= (agent['x'] - goal_x) / 100.0
""",
    "training_params": {
        "total_timesteps": 200000,
        "learning_rate": 0.0003,
        "gamma": 0.995,
        "batch_size": 256,
        "n_envs": 16,
        "max_episode_steps": 600,
        "device": "auto",  
        "save_model_path": "saved_models/chasm_platformer.zip"
    }
}

print("=" * 60)
print("🌉 CHASM BRIDGE TEST")
print("=" * 60)
print("Layout: [GOAL] <--- [CHASM] <--- [AGENTS START]")
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Spawn zone: x={config['env_params']['spawn_zone']['x1']}-{config['env_params']['spawn_zone']['x2']}")
print(f"Chasm: x={config['env_params']['special_objects'][0]['x1']}-{config['env_params']['special_objects'][0]['x2']}")
print(f"Goal: x={config['env_params']['special_objects'][1]['x']}")
print(f"Timesteps: {config['training_params']['total_timesteps']}")
print("=" * 60)

result = train_task(config)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Status: {result['status']}")
print(f"Video: {result.get('video_path', 'N/A')}")
print(f"Model: {result.get('model_path', 'N/A')}")
if result.get('metrics'):
    m = result['metrics']
    print(f"Mean training reward: {m.get('mean_reward', 'N/A'):.2f}")
    print(f"Eval reward: {m.get('eval_reward', 'N/A'):.2f}")
    print(f"Eval steps: {m.get('eval_steps', 'N/A')}")
if result.get('error'):
    print(f"Error: {result['error']}")
