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
        self.rich_logger.start()
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 100 == 0:
            fps = int(self.locals.get("fps", 0)) if self.locals.get("fps") else 0
            # Try to get reward info if available
            # info = self.locals.get("infos")
            self.rich_logger.update(self.num_timesteps, fps, 0.0)
        return True
    
    def _on_training_end(self):
        self.rich_logger.finish()
