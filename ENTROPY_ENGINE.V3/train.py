"""
Entropy Engine V3 - Main Training Script
Entry point for training experiments using Hydra configuration.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import wandb
import os
from entropy.config import register_configs, Config
from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.training.ppo import make_train, PPOConfig
from entropy.brain.manager import BrainManager, BrainMeta
from entropy.training.logging import setup_wandb, WandBCallback

# Register structured configs
register_configs()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training entry point.
    """
    print(f"=== Entropy Engine V3 Training ===")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Agents: {cfg.env.num_agents}")
    print(f"Device: {jax.devices()[0]}")
    
    # 1. Setup WandB
    if cfg.wandb.enabled:
        setup_wandb(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags
        )
    
    # 2. Prepare Config for PPO
    # Map Hydra config to PPOConfig dataclass
    ppo_cfg = PPOConfig(
        num_agents=cfg.env.num_agents,
        arena_width=float(cfg.env.arena_width),
        arena_height=float(cfg.env.arena_height),
        max_steps=cfg.env.max_steps,
        context_dim=cfg.env.communication.context_dim,
        
        # Training params
        learning_rate=cfg.agent.ppo.learning_rate,
        num_steps=cfg.agent.ppo.num_steps,
        num_minibatches=cfg.agent.ppo.num_minibatches,
        update_epochs=cfg.agent.ppo.update_epochs,
        gamma=cfg.agent.ppo.gamma,
        gae_lambda=cfg.agent.ppo.gae_lambda,
        clip_eps=cfg.agent.ppo.clip_coef,
        ent_coef=cfg.agent.ppo.ent_coef,
        vf_coef=cfg.agent.ppo.vf_coef,
        seed=cfg.seed
    )
    
    # 3. Compile Training Loop
    print("Compiling training loop...")
    train_fn = make_train(ppo_cfg)
    
    # 4. Run Training
    print("Starting training...")
    rng = jax.random.PRNGKey(cfg.seed)
    
    # Run the JAX scan loop
    # Note: Our make_train returns a simple function that runs valid N iterations
    # In a real heavy loop, we might want to run it step-by-step or in chunks
    # to allow for logging/checkpointing in Python land.
    # The current ppo.py `make_train` compiles the WHOLE loop (num_iterations).
    # This is great for speed but prevents intermediate python logging unless using host_callback.
    
    # For now, let's run it.
    out = train_fn(rng)
    
    # Unpack result (train_state, metrics)
    train_state, metrics = out
    print("Training finished!")
    
    # 5. Save Model via BrainManager
    manager = BrainManager()
    meta = BrainMeta(
        name=cfg.experiment_name,
        version=0, # Auto
        brain_type="ppo",
        input_dim=0, # TODO: Get from env
        output_dim=2,
        hidden_dim=cfg.agent.network.hidden_dim,
        created_at="", # Auto
        training_steps=cfg.training.num_iterations * cfg.agent.ppo.num_steps,
        notes=f"Hydra run: {cfg.experiment_name}"
    )
    
    # Save params
    # We need to extract params from TrainState.
    # train_state.params is the FrozenDict
    full_name = manager.save(cfg.experiment_name, train_state.params, meta)
    
    # Log artifact
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
