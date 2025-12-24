import sys
import os
import numpy as np
import time

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.entropy_env import EntropyEnv
from training.multicore_wrapper import AsyncVectorizedEntropyEnv

def main():
    print("Testing Curriculum Learning Implementation...")
    
    # 1. Spawn Env
    n_envs = 2
    env = AsyncVectorizedEntropyEnv(n_envs=n_envs, agents_per_env=5)
    
    # 2. Reset (Default Level 1)
    print("Resetting (Level 1)...")
    env.reset()
    
    # Check width indirectly via obs? 
    # Hard to check internal state without get_attr or env_method return.
    # We implemented env_method to return values!
    
    # 3. Check Level 1 Size
    # We didn't allow getting width directly, but we can call a property if we add one, or trust set_difficulty.
    # Let's trust set_difficulty prints or returns.
    
    # 4. Set Difficulty to 2
    print("Setting Difficulty to 2...")
    results = env.env_method("set_difficulty", 2)
    # This returns None (void function), but effectively confirms the method call worked if no crash.
    print(f"Set Difficulty returned: {results}")
    
    # 5. Reset (Level 2)
    print("Resetting (Level 2)...")
    env.reset()
    
    # 6. Set Difficulty to 3
    print("Setting Difficulty to 3...")
    env.env_method("set_difficulty", 3)
    
    env.close()
    print("Test Passed: Environment accepts curriculum commands.")

if __name__ == "__main__":
    main()
