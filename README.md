# POLYMORPH Engine

**Universal Multi-Agent Simulation Engine for Reinforcement Learning**

A modular Python engine that enables training swarm AI agents for diverse tasks through JSON configuration. Supports spatial (2D physics) and grid-based environments with customizable reward functions.

## Features

- **Universal Environment**: Single `UniversalSwarmEnv` class handles multiple task types
- **Flexible Observations**: Spatial (2D physics) or grid-based (2D matrix)
- **Flexible Actions**: Continuous (2D vectors) or discrete (directional + grab)
- **Dynamic Rewards**: Python code as string, executed at runtime
- **Swarm AI**: Multiple agents share single policy network
- **Visualization**: Automatic GIF generation of trained behavior
- **REST API**: FastAPI server for external integration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start the API Server

```bash
cd c:\Agency\ViceCoding\Entropy_Engine
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 2. Send Training Request

```bash
curl -X POST http://localhost:8000/train_task \
  -H "Content-Type: application/json" \
  -d @examples/forest_search.json
```

### 3. View Results

The response includes:
- `video_path`: Path to generated GIF
- `metrics`: Training statistics

## Configuration Format

```json
{
  "task_name": "my_task",
  "description": "Task description",
  "observation_type": "spatial",
  "action_space_type": "continuous",
  "env_params": {
    "world_width": 100,
    "world_height": 100,
    "num_agents": 10,
    "physics": {
      "gravity_y": 0.0,
      "friction": 0.1,
      "time_step": 0.1
    },
    "special_objects": [
      {"type": "goal", "x": 80, "y": 80},
      {"type": "obstacle", "x": 50, "y": 50, "radius": 5}
    ]
  },
  "reward_code": "reward = -distance_to_goal",
  "training_params": {
    "algo": "PPO",
    "total_timesteps": 10000,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 64
  }
}
```

## Example Scenarios

1. **Forest Search** (`examples/forest_search.json`): Agents search for lost persons
2. **Shift Optimization** (`examples/shift_optimization.json`): Grid-based shift scheduling
3. **Bridge Building** (`examples/bridge_building.json`): Agents form chain across gap

## API Endpoints

- `GET /`: Health check
- `POST /train_task`: Start training (body = config JSON)
- `GET /video/{filename}`: Retrieve generated GIF

## Architecture

```
Manager (AI) → n8n → POST /train_task → rl_trainer → UniversalSwarmEnv → GIF
```

## License

MIT
