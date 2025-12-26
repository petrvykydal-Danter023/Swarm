
import jax
import jax.numpy as jnp
import numpy as np
import time
import sys
import os
import subprocess
import pickle

# Ensure imports
sys.path.insert(0, os.getcwd())
# jax.config.update("jax_disable_jit", True) # ENABLE JIT FOR SPEED

from entropy.training.checkpoint import load_checkpoint
from entropy.training.env_wrapper import EntropyGymWrapper
from entropy.render.server import RenderServer
from entropy.render.schema import RenderFrame

AGENTS_COLOR = np.array([0, 0, 1]) # Blue

def run_viz():
    print("🎥 Starting Visualization of Certified Master Model...")
    
    # 1. Launch Viewer in Subprocess
    print("🖥️ Launching Entropy Viewer Window...")
    # ... (rest is same)
    viewer_process = subprocess.Popen([sys.executable, "-m", "entropy.render.viewer"])
    time.sleep(2.0) # Wait for viewer to start
    
    server = RenderServer(port=5555)

    # 2. Logic to Load Checkpoint
    checkpoint_dir = "outputs/checkpoints"
    # Find latest pkl
    try:
        files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl") and "best" not in f]
        if not files:
            # Fallback to best if exists
            if os.path.exists(os.path.join(checkpoint_dir, "best.pkl")):
                files = ["best.pkl"]
            else:
                print("❌ No checkpoints found in outputs/checkpoints/")
                return
        
        # Sort by Name (step number usually) or mod time
        latest_file = sorted(files)[-1]
        ckpt_path = os.path.join(checkpoint_dir, latest_file)
        
        print(f"📂 Loading model from: {ckpt_path}")
        checkpoint = load_checkpoint(ckpt_path)
        params = checkpoint["params"]
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        try:
             viewer_process.terminate()
        except:
             pass
        return

    # 3. Setup Env
    class EnvConfig:
        class Env:
            num_agents = 5
            arena_width = 800.0
            arena_height = 600.0
            max_steps = 1000 # Long episode for viz
        class Model:
            context_dim = 64
        env = Env()
        model = Model()
        
    print("🌍 Initializing Environment...")
    env = EntropyGymWrapper(EnvConfig())
    
    # Init Network (Try MAPPOActor)
    from entropy.training.mappo import MAPPOActor
    actor = MAPPOActor(action_dim=2) 
    
    # JIT Compile Critical Functions
    print("⚡ JIT Compiling Simulation Step...")
    
    @jax.jit
    def jit_act(params, obs):
        return actor.apply(params, obs)[0]
        
    @jax.jit
    def jit_step(state, actions, rng):
        return env.step(state, actions, rng)

    rng = jax.random.PRNGKey(0)
    
    # Warmup JIT
    r_warm, r_reset = jax.random.split(rng)
    s_warm, o_warm = env.reset(r_reset)
    _ = jit_act(params, o_warm)
    _ = jit_step(s_warm, jnp.zeros((5, 2)), r_warm)
    print("⚡ JIT Warmup Complete.")
    
    print("🎬 Running Simulation loop...")
    try:
        for ep in range(5):
            print(f"  Episode {ep+1}/5")
            rng, reset_rng = jax.random.split(rng)
            state, obs = env.reset(reset_rng)
            done = False
            step = 0
            
            t0 = time.time()
            
            while not done:
                # 4. Inference
                # Use JIT function
                try:
                    mean = jit_act(params, obs)
                except:
                     # Fallback if structure mismatches (ActorCritic)
                     from entropy.training.network import ActorCritic
                     net = ActorCritic(action_dim=2)
                     @jax.jit
                     def jit_act_ac(p, o): return net.apply(p, o)[0]
                     mean = jit_act_ac(params, obs)
                
                actions = mean
                
                # 5. Physics Step
                rng, step_rng = jax.random.split(rng)
                state, obs, reward, d, info = jit_step(state, actions, step_rng)
                done = jnp.all(d)
                
                # 6. Render Frame
                # Convert JAX to Numpy
                positions = np.array(state.agent_positions)
                angles = np.array(state.agent_angles)
                goals = np.array(state.goal_positions)
                messages = np.array(state.agent_messages)
                
                # Construct Frame
                frame = RenderFrame(
                    timestep=step,
                    agent_positions=positions,
                    agent_angles=angles,
                    agent_colors=None, # Use default or compute from rewards
                    agent_messages=messages,
                    agent_radii=np.full(len(positions), 15.0),
                    goal_positions=goals,
                    object_positions=np.zeros((0, 2)),
                    object_types=np.zeros((0,)),
                    wall_segments=np.array([
                        [[0, 0], [800, 0]],
                        [[800, 0], [800, 600]],
                        [[800, 600], [0, 600]],
                        [[0, 600], [0, 0]]
                    ]),
                    rewards=np.array(reward),
                    fps=1.0/(time.time() - t0 + 1e-6)
                )
                
                t0 = time.time()
                server.publish_frame(frame)
                
                time.sleep(0.05) # ~20 FPS cap
                step += 1
                
            print(f"  ✅ Episode {ep+1} finished.")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("🛑 User stopped visualization.")
    finally:
        print("Killing Viewer...")
        viewer_process.terminate()
        server.close()

if __name__ == "__main__":
    run_viz()
