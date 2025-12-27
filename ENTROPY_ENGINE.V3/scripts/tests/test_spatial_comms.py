import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

"""
Verification Script for Advanced Communication (Spatial + Recurrent)
"""
import jax
from entropy.config import ExperimentConfig, SimConfig, AgentConfig, CommConfig, PPOConfig, HogConfig, RenderConfig
from train_master import run_experiment

def test_spatial_comms():
    cfg = ExperimentConfig(
        name="spatial_comms_verify",
        sim=SimConfig(num_envs=4, max_steps=50, arena_width=1000.0, arena_height=1000.0),
        agent=AgentConfig(
            num_agents=10,
            use_communication=True,
            comm=CommConfig(
                mode="spatial",
                msg_dim=8,
                max_neighbors=3,
                gating_threshold=0.0, # Make them talk
                spam_penalty=0.0
            ) 
        ),
        ppo=PPOConfig(actor_updates=2, critic_updates=2),
        hog=HogConfig(enabled=False),
        render=RenderConfig(enabled=False),
        total_epochs=2 # Short run
    )
    
    print("🧪 Running Spatial Comms Verification...")
    run_experiment(cfg)
    print("✅ Verification Complete!")

if __name__ == "__main__":
    test_spatial_comms()
