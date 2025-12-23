import os
import numpy as np
import imageio
from stable_baselines3.common.callbacks import BaseCallback
from env.entropy_env import EntropyEnv
from shared.logger import RichLogger

class GifRecorderCallback(BaseCallback):
    def __init__(self, eval_env: EntropyEnv, save_path: str, name_prefix: str = "training", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.best_reward = -float('inf')
        self.frames_start = []
        self.frames_end = []
        self.recorded_start = False
        
        os.makedirs(save_path, exist_ok=True)
        
    def _on_training_start(self):
        # Record initial behavior
        print("Recording Untrained Agent Behavior...")
        self.frames_start = self.record_episode()
        self.recorded_start = True
        self._save_gif(self.frames_start, f"{self.name_prefix}_start.gif")

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self):
        # Record final behavior
        print("Recording Trained Agent Behavior...")
        self.frames_end = self.record_episode()
        self._save_gif(self.frames_end, f"{self.name_prefix}_end.gif")
        
        # Combine if possible
        if self.recorded_start:
            self.create_comparison_gif()

    def record_episode(self, max_steps=500):
        frames = []
        obs, _ = self.eval_env.reset()
        
        for _ in range(max_steps):
            frame = self.eval_env.render()
            if frame is not None:
                frames.append(frame)
            
            # Get actions for all agents
            actions = {}
            
            # Query model for each agent
            # This is slow but fine for recording
            if isinstance(obs, dict):
                 for agent_id, agent_obs in obs.items():
                    action, _states = self.model.predict(agent_obs, deterministic=True)
                    actions[agent_id] = action
            else:
                # Should not happen in ParallelEnv but handling just in case
                pass
                
            obs, rewards, terminations, truncations, infos = self.eval_env.step(actions)
            
            if not self.eval_env.agents:
                break
            
        return frames

    def _save_gif(self, frames, filename):
        path = os.path.join(self.save_path, filename)
        imageio.mimsave(path, frames, fps=30)
        print(f"Saved GIF: {path}")
        
        # Log to WandB if active
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"video/{self.name_prefix}_{filename}": wandb.Video(path, fps=30, format="gif")})
        except ImportError:
            pass
        except Exception as e:
            print(f"WandB Video Log failed: {e}")

    def create_comparison_gif(self):
        print("Generating Comparison GIF...")
        try:
            import moviepy.editor as mpy
            
            path_start = os.path.join(self.save_path, f"{self.name_prefix}_start.gif")
            path_end = os.path.join(self.save_path, f"{self.name_prefix}_end.gif")
            
            clip_start = mpy.VideoFileClip(path_start)
            clip_end = mpy.VideoFileClip(path_end)
            
            final_clip = mpy.clips_array([[clip_start, clip_end]])
            
            output_path = os.path.join(self.save_path, f"{self.name_prefix}_comparison.gif")
            final_clip.write_gif(output_path, fps=30, verbose=False, logger=None)
            print(f"Saved Comparison GIF: {output_path}")
            
            # Log to WandB
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({"video/comparison": wandb.Video(output_path, fps=30, format="gif")})
            except ImportError:
                pass
            
        except Exception as e:
            print(f"Failed to create comparison GIF using moviepy: {e}")
            try:
                combined_frames = []
                min_len = min(len(self.frames_start), len(self.frames_end))
                for i in range(min_len):
                    f1 = self.frames_start[i]
                    f2 = self.frames_end[i]
                    combined = np.concatenate((f1, f2), axis=1)
                    combined_frames.append(combined)
                path = os.path.join(self.save_path, f"{self.name_prefix}_comparison_fallback.gif")
                imageio.mimsave(path, combined_frames, fps=30)
                
                # Log to WandB
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"video/comparison_fallback": wandb.Video(path, fps=30, format="gif")})
                except ImportError:
                    pass
                    
            except Exception as e2:
                 print(f"Fallback also failed: {e2}")

class RichLoggerCallback(BaseCallback):
    def __init__(self, logger: RichLogger, verbose=0):
        super().__init__(verbose)
        self.rich_logger = logger
        self.last_time = 0
        
    def _on_training_start(self):
        # Capture starting timesteps for offset calculation
        self.rich_logger.start_offset = self.model.num_timesteps
        self.rich_logger.start()
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 100 == 0:
            # Extract metrics from logger
            fps = 0
            mean_reward = 0.0
            loss = 0.0
            explained_var = 0.0
            
            # SB3 stores metrics in self.logger.name_to_value
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = getattr(self.model.logger, 'name_to_value', {})
                fps = int(name_to_value.get('time/fps', 0))
                loss = name_to_value.get('train/loss', 0.0)
                explained_var = name_to_value.get('train/explained_variance', 0.0)
                mean_reward = name_to_value.get('rollout/ep_rew_mean', 0.0) or 0.0
            
            self.rich_logger.update(
                self.num_timesteps, 
                fps, 
                mean_reward,
                loss=loss,
                explained_var=explained_var
            )
        return True
    
    def _on_training_end(self):
        self.rich_logger.finish()


class RollingCheckpointCallback(BaseCallback):
    """
    Saves a single checkpoint that gets overwritten each time.
    Checkpoint is deleted after training completes successfully.
    """
    
    def __init__(self, save_dir: str, save_freq: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.checkpoint_path = os.path.join(save_dir, "checkpoint.zip")
        self.state_path = os.path.join(save_dir, "checkpoint_state.json")
        os.makedirs(save_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Save model checkpoint (overwrites previous)
            self.model.save(self.checkpoint_path)
            
            # Save state info
            import json
            state = {
                "timesteps": self.num_timesteps,
                "time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(self.state_path, 'w') as f:
                json.dump(state, f)
                
            if self.verbose > 0:
                print(f"Checkpoint saved at {self.num_timesteps} steps")
        return True
    
    def _on_training_end(self):
        # Training completed successfully - remove checkpoint
        self.cleanup()
        
    def cleanup(self):
        """Remove checkpoint files."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
        if os.path.exists(self.state_path):
            os.remove(self.state_path)
    
    @staticmethod
    def load_checkpoint(save_dir: str):
        """Load checkpoint if exists. Returns (model_path, timesteps) or (None, 0)."""
        import json
        checkpoint_path = os.path.join(save_dir, "checkpoint.zip")
        state_path = os.path.join(save_dir, "checkpoint_state.json")
        
        if os.path.exists(checkpoint_path) and os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            return checkpoint_path, state.get("timesteps", 0)
        return None, 0
