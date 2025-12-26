
"""
Entropy Engine V3 - Recorder
Orchestrates episode rollout and video saving.
"""
import jax
import numpy as np
import os
import time
import imageio
from typing import Any, Callable, Optional, Dict
from entropy.render.cpu_renderer import CPURenderer
from entropy.render.schema import RenderFrame

# Try to import cv2 for video writing, fallback to imageio
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class Recorder:
    def __init__(self, output_dir: str, fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.renderer = CPURenderer(width=800, height=600)
        os.makedirs(output_dir, exist_ok=True)
        
    def record_episode(
        self, 
        env_wrapper: Any, 
        actor_state: Any, 
        params: Any, 
        filename: str,
        max_steps: int = 400
    ) -> str:
        """
        Runs one episode, renders it, and saves directly to disk.
        Returns the path to the saved file.
        """
        frames = []
        
        # JIT inference function if not passed
        # Currently assuming actor_state has apply_fn or params structure
        # ... logic similar to viz script ...
        
        # Setup Env
        rng = jax.random.PRNGKey(int(time.time()))
        rng, reset_rng = jax.random.split(rng)
        state, obs = env_wrapper.reset(reset_rng)
        
        # Rollout
        for t in range(max_steps):
            # Inference
            # Assuming MAPPO/Actor style: apply_fn(params, obs)
            # Or direct callable if simple model
            if hasattr(actor_state, 'apply_fn'):
                mean, _ = actor_state.apply_fn(params, obs)
            else:
                # Fallback for simple models
                mean, _ = actor_state.apply(params, obs)
                
            actions = mean # Deterministic for viz
            
            # Step
            rng, step_rng = jax.random.split(rng)
            next_state, next_obs, rewards, dones, info = env_wrapper.step(state, actions, step_rng)
            
            # Construct Frame
            # Create RenderFrame object manually
            rframe = RenderFrame(
                timestep=t,
                agent_positions=np.array(state.agent_positions),
                agent_angles=np.array(state.agent_angles),
                agent_colors=None,
                agent_messages=np.array(state.agent_messages),
                agent_radii=np.full(len(state.agent_positions), 15.0),
                goal_positions=np.array(state.goal_positions),
                object_positions=np.zeros((0, 2)),
                object_types=np.zeros((0,)),
                wall_segments=np.array([
                    [[0, 0], [800, 0]],
                    [[800, 0], [800, 600]],
                    [[800, 600], [0, 600]],
                    [[0, 600], [0, 0]]
                ]), # Default walls
                rewards=np.array(rewards),
                # Pheromone Visualization
                pheromone_positions=np.array(state.pheromone_positions) if hasattr(state, 'pheromone_positions') else None,
                pheromone_ttls=np.array(state.pheromone_ttls) if hasattr(state, 'pheromone_ttls') else None,
                pheromone_valid=np.array(state.pheromone_valid) if hasattr(state, 'pheromone_valid') else None,
                pheromone_max_ttl=100.0,
                pheromone_radius=50.0,
                # Hierarchy Visualization
                agent_squad_ids=np.array(state.agent_squad_ids) if hasattr(state, 'agent_squad_ids') else None,
                agent_is_leader=np.array(state.agent_is_leader) if hasattr(state, 'agent_is_leader') else None,
                
                # Safety & Intent Visualization
                safety_enabled=env_wrapper.safety_cfg.enabled if hasattr(env_wrapper, 'safety_cfg') and env_wrapper.safety_cfg else False,
                safety_radius=env_wrapper.safety_cfg.safety_radius if hasattr(env_wrapper, 'safety_cfg') and env_wrapper.safety_cfg else 30.0,
                safety_repulsion_radius=env_wrapper.safety_cfg.repulsion_radius if hasattr(env_wrapper, 'safety_cfg') and env_wrapper.safety_cfg else 25.0,
                geofence_zones=None, # Populate if zones exist in world or config
                
                intent_enabled=env_wrapper.intent_cfg.enabled if hasattr(env_wrapper, 'intent_cfg') and env_wrapper.intent_cfg else False,
                # Would need to capture raw intents from somewhere if we want to viz them here, 
                # but actions passed to step() are already motor actions in Direct mode?
                # Actually, in Intent mode, actions ARE intents. So we can visualize them.
                agent_intents=np.array(actions) if (hasattr(env_wrapper, 'intent_cfg') and env_wrapper.intent_cfg and env_wrapper.intent_cfg.enabled) else None
            )
            
            # Render to Image
            img = self.renderer.render(rframe)
            frames.append(img)
            
            if np.all(dones):
                break
                
            state = next_state
            obs = next_obs
            
        # Save Video
        save_path = os.path.join(self.output_dir, filename)
        
        # Prefer GIF for user convenience or full video? 
        # GIF is standard for "V2 style".
        # Let's save GIF using imageio
        print(f"💾 Saving recording to {save_path}...")
        
        # Convert BGR (OpenCV) to RGB (ImageIO)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames] if HAS_CV2 else frames
        
        imageio.mimsave(save_path, rgb_frames, fps=self.fps)
        return save_path

