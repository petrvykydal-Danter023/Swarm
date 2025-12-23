import os
import numpy as np
from sb3_contrib import RecurrentPPO
from training.multicore_wrapper import AsyncVectorizedEntropyEnv

# Define Root Path
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_v3_model():
    print("🧪 Verifying V3 Model...")
    
    # Path
    model_path = os.path.join(V2_ROOT, "models", "entropy_v3_0.zip")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    try:
        # Init Env (VecEnv)
        # 1 env, 5 agents -> Total 5 agents
        print("  Initializing Env...", flush=True)
        env = AsyncVectorizedEntropyEnv(n_envs=1, agents_per_env=5)
        
        # Load Model
        print("  Loading Model...", flush=True)
        model = RecurrentPPO.load(model_path, env=env)
        print("  ✅ Model Loaded Successfully", flush=True)
        
        # Run loop
        print("  Running Simulation...", flush=True)
        obs = env.reset() # Returns numpy array (total_agents, obs_dim)
        lstm_states = None
        episode_starts = np.ones((env.total_agents,), dtype=bool)
        
        for i in range(50):
            # RecurrentPPO predict takes array of obs
            actions, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts,
                deterministic=True
            )
            
            # Step VecEnv
            obs, rewards, dones, infos = env.step(actions)
            
            episode_starts = dones
            
            if i % 10 == 0:
                print(f"    Step {i}: Loop OK", flush=True)
                
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("✅ Test Complete")

if __name__ == "__main__":
    test_v3_model()
