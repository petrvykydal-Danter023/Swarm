"""Bridge building test - agents form a chain across a gap with gravity."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from models.rl_trainer import train_task

config = {
    "task_name": "topdown_move_test",
    "description": "Top-down navigation test - no gaps, just obstacles and goal",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 100,
        "world_height": 50,
        "num_agents": 5,  # Fewer agents for clearer visualization
        "physics": {
            "gravity_y": 0.0,  # TOP-DOWN: No gravity
            "friction": 0.1,   # Low friction for easy movement
            "time_step": 0.1
        },
        "spawn_zone": {
            "x1": 5, "x2": 20,
            "y1": 5, "y2": 45
        },
        "enable_communication": True,
        "sensor_noise_std": 0.05,
        "packet_loss_prob": 0.1,
        "comm_range": 60.0,
        "special_objects": [
            # NO GAP - Pure movement test
            {"type": "obstacle", "x": 40, "y": 25, "radius": 5},
            {"type": "obstacle", "x": 60, "y": 15, "radius": 4},
            {"type": "obstacle", "x": 60, "y": 35, "radius": 4},
            {"type": "goal", "x": 90, "y": 25}
        ],
        "sensors": [
            "position", "velocity", "goal_vector", "obstacle_radar",
            "neighbor_vectors", "neighbor_signals", "grabbing_state", "energy"
        ]
    },
    "reward_code": """
# Pure Navigation Reward
reward = 0.0
agent_idx = env_state['agent_idx']
agent = env_state['agents'][agent_idx]

# 1. Distance to Goal (Dense Reward)
if env_state['goals']:
    goal = env_state['goals'][0]
    dist = math.sqrt((goal['x']-agent['x'])**2 + (goal['y']-agent['y'])**2)
    
    # Normalized distance reward (0 to 1)
    max_dist = 100.0
    reward += (1.0 - min(dist, max_dist) / max_dist) * 0.1
    
    # Goal Reached Bonus
    if dist < 3.0:
        reward += 10.0

# 2. Movement Incentive (Small bonus for velocity towards goal)
# v_x towards goal (goal at x=90)
if agent['vx'] > 0.1:
    reward += 0.01

# 3. Energy Penalty (Tiny)
reward -= 0.001
""",
    "training_params": {
        "total_timesteps": 60000,  # Quick regen
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 64,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "max_episode_steps": 500
    }
}

print("=" * 50)
print("TOP-DOWN NAVIGATION TEST")
print("=" * 50)
print(f"World: {config['env_params']['world_width']}x{config['env_params']['world_height']}")
print(f"Agents: {config['env_params']['num_agents']}")
print(f"Gravity: {config['env_params']['physics']['gravity_y']}")
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
