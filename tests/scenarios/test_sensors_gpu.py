"""
Test: Configurable Sensors + GPU Support

Demonstrates:
1. Custom sensor configuration for rich observations
2. GPU acceleration for faster training
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "sensors_test",
    "description": "Testing configurable sensors with GPU",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 80,
        "world_height": 60,
        "num_agents": 12,
        "physics": {
            "gravity_y": 0.0,
            "friction": 0.3,
            "time_step": 0.1
        },
        "special_objects": [
            {"type": "goal", "x": 70, "y": 30},
            {"type": "obstacle", "x": 40, "y": 30, "radius": 8},
            {"type": "obstacle", "x": 25, "y": 45, "radius": 5},
        ],
        # ===============================================
        # CONFIGURABLE SENSORS - pick what you need!
        # ===============================================
        "sensors": [
            "position",         # 2 dims - where am I?
            "velocity",         # 2 dims - how fast am I moving?
            "goal_vector",      # 2 dims - direction to goal
            "goal_distance",    # 1 dim - how far is goal?
            "obstacle_radar",   # 8 dims - obstacles in 8 directions
            "neighbor_count",   # 1 dim - how many agents nearby?
            "neighbor_vectors", # 6 dims - vectors to 3 nearest agents
            "wall_distance",    # 4 dims - distance to 4 walls
            "grabbing_state",   # 1 dim - am I grabbing?
            "time_remaining",   # 1 dim - how much time left?
        ]
        # Total: 2+2+2+1+8+1+6+4+1+1 = 28 dimensional observation!
    },
    "reward_code": """
reward = 0.0

# Goal-seeking reward
goals = env_state['goals']
if goals:
    goal = goals[0]
    dist = math.sqrt((goal['x']-agent['x'])**2 + (goal['y']-agent['y'])**2)
    reward = -dist / 50.0
    
    # Bonus for reaching goal
    if dist < 5:
        reward += 20.0

# Penalty for hitting obstacles
for obs in env_state['obstacles']:
    d = math.sqrt((obs['x']-agent['x'])**2 + (obs['y']-agent['y'])**2)
    if d < obs['radius'] + 2:
        reward -= 5.0

# Team spread bonus (avoid clustering)
neighbors = env_state['neighbors']
if len(neighbors) > 3:
    reward -= 0.5  # too crowded
""",
    "training_params": {
        "total_timesteps": 30000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_envs": 4,
        "max_episode_steps": 400,
        
        # ===============================================
        # GPU SUPPORT
        # ===============================================
        "device": "auto",  # "auto", "cuda", or "cpu"
    }
}

print("=" * 60)
print("🔬 CONFIGURABLE SENSORS + GPU TEST")
print("=" * 60)

# Show sensor config
sensors = config['env_params']['sensors']
obs_dim = sum({
    "position": 2, "velocity": 2, "goal_vector": 2, "goal_distance": 1,
    "obstacle_radar": 8, "neighbor_count": 1, "neighbor_vectors": 6,
    "neighbor_density": 4, "wall_distance": 4, "grabbing_state": 1, "time_remaining": 1
}.get(s, 0) for s in sensors)

print(f"Sensors: {sensors}")
print(f"Total observation dimension: {obs_dim}")
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Device: {config['training_params']['device']}")
print("=" * 60)

result = train_task(config)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Status: {result['status']}")
print(f"Video: {result.get('video_path', 'N/A')}")
if result.get('metrics'):
    m = result['metrics']
    print(f"Mean reward: {m.get('mean_reward', 'N/A'):.2f}")
    print(f"Eval reward: {m.get('eval_reward', 'N/A'):.2f}")
    print(f"Eval steps: {m.get('eval_steps', 'N/A')}")
