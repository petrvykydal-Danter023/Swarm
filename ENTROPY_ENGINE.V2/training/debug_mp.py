import sys
import os
import multiprocessing as mp
import time

# Add root
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

from training.multicore_wrapper import AsyncVectorizedEntropyEnv

def main():
    print("Testing Async Env...")
    try:
        env = AsyncVectorizedEntropyEnv(n_envs=2, agents_per_env=2)
        print("Env created.")
        obs = env.reset()
        print(f"Reset done. Obs shape: {obs.shape}")
        
        # Step
        actions = [env.action_space.sample() for _ in range(4)] # 2 * 2 = 4 agents
        print("Stepping...")
        env.step_async(actions)
        obs, rews, dones, infos = env.step_wait()
        print("Step done.")
        
        env.close()
        print("Env closed.")
    except Exception as e:
        print("MAIN EXCEPTION:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mp.freeze_support()
    main()
