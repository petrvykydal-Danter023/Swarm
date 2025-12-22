"""
Transfer Learning Demo - Train on Bridge, Fine-tune for Logistics

Stage 1: Train agents on bridge building task (basic cooperation)
Stage 2: Load pretrained model, fine-tune for warehouse logistics

This demonstrates model reuse and transfer learning.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

# ============================================
# STAGE 1: Train on Bridge Task
# ============================================
print("=" * 60)
print("🌉 STAGE 1: Base Training on Bridge")
print("=" * 60)

bridge_config = {
    "task_name": "bridge_base",
    "description": "Base training on cooperative bridge building",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 80,
        "world_height": 40,
        "num_agents": 10,
        "physics": {
            "gravity_y": 0.5,
            "friction": 0.3,
            "time_step": 0.1
        },
        "special_objects": [
            {"type": "gap", "x1": 35, "x2": 45, "y": 20},
            {"type": "goal", "x": 70, "y": 20}
        ]
    },
    "reward_code": """
reward = 0.0
gap_x1, gap_x2, gap_y = 35, 45, 20
goal_x = 70

all_agents = env_state['all_agents']

# Team metrics
right_side = [a for a in all_agents if a['x'] > gap_x2 and a['y'] >= gap_y - 5]
in_gap = [a for a in all_agents if gap_x1 <= a['x'] <= gap_x2 and a['y'] >= gap_y - 3]
grabbing = [a for a in in_gap if a['is_grabbing']]
fallen = [a for a in all_agents if a['y'] < gap_y - 10]

# Global reward
reward = len(right_side) * 8.0
reward += len(in_gap) * 2.0
reward += len(grabbing) * 2.0
reward -= len(fallen) * 4.0

# Personal guidance
if agent['x'] < gap_x1:
    reward -= (gap_x1 - agent['x']) / 40.0
if agent['y'] < gap_y - 10:
    reward -= 5.0
""",
    "training_params": {
        "total_timesteps": 30000,  # Shorter for demo
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 300,
        # SAVE the model!
        "save_model_path": "saved_models/bridge_base.zip"
    }
}

result1 = train_task(bridge_config)
print(f"\nStage 1 completed: {result1['status']}")
print(f"Model saved: {result1.get('model_path')}")
print(f"Mean reward: {result1['metrics']['mean_reward']:.2f}")

# ============================================
# STAGE 2: Fine-tune for Logistics
# ============================================
print("\n" + "=" * 60)
print("📦 STAGE 2: Fine-tune for Warehouse Logistics")
print("=" * 60)

logistics_config = {
    "task_name": "logistics_finetune",
    "description": "Fine-tuned agents for warehouse logistics",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 80,
        "world_height": 40,
        # Same agent count for compatible network!
        "num_agents": 10,
        "physics": {
            "gravity_y": 0.0,  # No gravity in warehouse
            "friction": 0.4,
            "time_step": 0.1
        },
        "special_objects": [
            # Multiple goals (packages to deliver)
            {"type": "goal", "x": 10, "y": 35},
            {"type": "goal", "x": 70, "y": 35},
            {"type": "goal", "x": 10, "y": 5},
            {"type": "goal", "x": 70, "y": 5},
            # Obstacles (shelves)
            {"type": "obstacle", "x": 40, "y": 20, "radius": 5}
        ]
    },
    "reward_code": """
reward = 0.0

all_agents = env_state['all_agents']
goals = env_state['goals']

# Count agents near each goal (coverage)
coverage = 0
for goal in goals:
    gx, gy = goal['x'], goal['y']
    near_goal = [a for a in all_agents if abs(a['x']-gx) < 8 and abs(a['y']-gy) < 8]
    if len(near_goal) >= 1:
        coverage += 1

# Team reward for coverage
reward = coverage * 5.0

# Bonus if all goals covered
if coverage >= len(goals):
    reward += 20.0

# Personal: distance to nearest uncovered goal
agent_x, agent_y = agent['x'], agent['y']
min_dist = 999
for goal in goals:
    gx, gy = goal['x'], goal['y']
    dist = math.sqrt((gx-agent_x)**2 + (gy-agent_y)**2)
    if dist < min_dist:
        min_dist = dist

reward -= min_dist / 50.0

# Penalty for collision with obstacles
for obs in env_state['obstacles']:
    dist = math.sqrt((obs['x']-agent_x)**2 + (obs['y']-agent_y)**2)
    if dist < obs['radius'] + 2:
        reward -= 3.0
""",
    "training_params": {
        "total_timesteps": 20000,  # Shorter - already pretrained!
        "learning_rate": 0.0001,   # Lower LR for fine-tuning
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 300,
        # LOAD pretrained, save new model
        "load_pretrained_path": "saved_models/bridge_base.zip",
        "save_model_path": "saved_models/logistics_finetuned.zip"
    }
}

result2 = train_task(logistics_config)
print(f"\nStage 2 completed: {result2['status']}")
print(f"Model saved: {result2.get('model_path')}")
print(f"Mean reward: {result2['metrics']['mean_reward']:.2f}")

# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("📊 TRANSFER LEARNING SUMMARY")
print("=" * 60)
print(f"Stage 1 (Bridge from scratch):")
print(f"  - Timesteps: 30k, Mean reward: {result1['metrics']['mean_reward']:.2f}")
print(f"  - Video: {result1['video_path']}")
print(f"\nStage 2 (Logistics fine-tuned):")
print(f"  - Timesteps: 20k (33% less!), Mean reward: {result2['metrics']['mean_reward']:.2f}")
print(f"  - Video: {result2['video_path']}")
print(f"\nSaved models:")
print(f"  - saved_models/bridge_base.zip")
print(f"  - saved_models/logistics_finetuned.zip")
