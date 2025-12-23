"""
POLYMORPH Engine - FastAPI Server

REST API for training swarm AI agents. Accepts JSON configuration
and returns training results with visualization GIFs.
"""

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rl_trainer import train_task


# Pydantic models for request/response validation

class PhysicsParams(BaseModel):
    """Physics simulation parameters."""
    gravity_y: float = Field(default=0.0, description="Gravity force in Y direction")
    friction: float = Field(default=0.1, ge=0.0, le=1.0, description="Friction coefficient")
    time_step: float = Field(default=0.1, gt=0.0, description="Physics time step")


class SpecialObject(BaseModel):
    """Special object in the environment."""
    type: Literal["goal", "obstacle", "gap"] = Field(description="Object type")
    x: Optional[float] = Field(default=0.0, description="X position")
    y: Optional[float] = Field(default=0.0, description="Y position")
    x1: Optional[float] = Field(default=0.0, description="Gap start X")
    x2: Optional[float] = Field(default=0.0, description="Gap end X")
    radius: Optional[float] = Field(default=1.0, description="Obstacle radius")


class EnvParams(BaseModel):
    """Environment configuration parameters."""
    world_width: int = Field(default=100, gt=0, description="World width in units")
    world_height: int = Field(default=100, gt=0, description="World height in units")
    num_agents: int = Field(default=10, gt=0, le=100, description="Number of swarm agents")
    physics: PhysicsParams = Field(default_factory=PhysicsParams)
    special_objects: list[SpecialObject] = Field(default_factory=list)


class TrainingParams(BaseModel):
    """Training hyperparameters."""
    algo: Literal["PPO"] = Field(default="PPO", description="RL algorithm (currently only PPO)")
    total_timesteps: int = Field(default=10000, gt=0, description="Total training timesteps")
    learning_rate: float = Field(default=3e-4, gt=0, description="Learning rate")
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")
    batch_size: int = Field(default=64, gt=0, description="Batch size for training")
    n_envs: int = Field(default=4, gt=0, le=16, description="Number of parallel environments")
    max_episode_steps: int = Field(default=500, gt=0, description="Max steps per episode")


class TrainTaskRequest(BaseModel):
    """Request body for /train_task endpoint."""
    task_name: str = Field(description="Name of the training task")
    description: str = Field(default="", description="Task description")
    observation_type: Literal["spatial", "grid"] = Field(
        default="spatial",
        description="Observation type: 'spatial' for 2D physics, 'grid' for 2D matrix"
    )
    action_space_type: Literal["continuous", "discrete"] = Field(
        default="continuous",
        description="Action type: 'continuous' for 2D vectors, 'discrete' for directional"
    )
    env_params: EnvParams = Field(default_factory=EnvParams)
    reward_code: str = Field(
        description="Python code for reward calculation. Has access to 'agent', 'env_state', 'math', and must set 'reward' variable."
    )
    training_params: TrainingParams = Field(default_factory=TrainingParams)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task_name": "forest_search",
                    "description": "Search for lost persons in a forest",
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
                    "reward_code": "goal = env_state['goals'][0]\ndist = math.sqrt((goal['x']-agent['x'])**2 + (goal['y']-agent['y'])**2)\nreward = -dist/100 + (10 if dist < 5 else 0)",
                    "training_params": {
                        "algo": "PPO",
                        "total_timesteps": 10000,
                        "learning_rate": 0.0003,
                        "gamma": 0.99,
                        "batch_size": 64
                    }
                }
            ]
        }
    }


class TrainingMetrics(BaseModel):
    """Training metrics returned after training."""
    mean_reward: float = Field(description="Mean reward across training episodes")
    eval_reward: float = Field(default=0.0, description="Total reward in evaluation episode")
    eval_steps: int = Field(default=0, description="Number of steps in evaluation episode")
    episodes: int = Field(description="Number of training episodes completed")
    total_timesteps: int = Field(description="Total timesteps trained")


class TrainTaskResponse(BaseModel):
    """Response from /train_task endpoint."""
    status: Literal["done", "error"] = Field(description="Training status")
    video_path: Optional[str] = Field(default=None, description="Path to generated GIF video")
    metrics: Optional[TrainingMetrics] = Field(default=None, description="Training metrics")
    error: Optional[str] = Field(default=None, description="Error message if status is 'error'")


# FastAPI application
app = FastAPI(
    title="POLYMORPH Engine API",
    description="Universal Multi-Agent Simulation Engine for Reinforcement Learning",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "POLYMORPH Engine",
        "version": "0.1.0",
        "status": "running"
    }


@app.post("/train_task", response_model=TrainTaskResponse)
async def train_task_endpoint(request: TrainTaskRequest) -> TrainTaskResponse:
    """
    Start training a swarm AI task.
    
    Accepts a JSON configuration with:
    - Task name and description
    - Environment parameters (world size, agents, objects)
    - Observation and action types
    - Reward function as Python code
    - Training hyperparameters
    
    Returns training metrics and path to visualization GIF.
    """
    try:
        # Convert Pydantic models to dict for train_task
        config = request.model_dump()
        
        # Run training
        result = train_task(config)
        
        if result["status"] == "error":
            return TrainTaskResponse(
                status="error",
                error=result.get("error", "Unknown error"),
                video_path=None,
                metrics=None
            )
        
        return TrainTaskResponse(
            status="done",
            video_path=result.get("video_path"),
            metrics=TrainingMetrics(**result.get("metrics", {}))
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/video/{filename}")
async def get_video(filename: str):
    """Serve a generated video file."""
    video_path = Path(__file__).parent.parent / "videos" / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path=str(video_path),
        media_type="image/gif",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
