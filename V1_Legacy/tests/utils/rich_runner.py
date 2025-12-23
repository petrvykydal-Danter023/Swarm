
import numpy as np
import time

def run_episode(env, policy_fn, max_steps=20, verbose=True, title="Test Run"):
    """
    Runs an episode and prints a rich ASCII table of the results.
    
    Args:
        env: The Gym environment.
        policy_fn: A function (step_idx) -> action.
        max_steps: How many steps to run.
        verbose: If True, prints the table.
        title: Title of the run.
    """
    print(f"\n{'='*60}")
    print(f" >>> {title}")
    print(f"{'='*60}")
    print(f"{'STEP':^4} | {'SPEED':^8} | {'ACTION':^20} | {'REWARD':^8} | {'NOTE'}")
    print("-" * 60)
    
    total_reward = 0
    obs, _ = env.reset()
    
    history = []
    
    for step in range(max_steps):
        actions = policy_fn(step)
        
        # Step with verbose=False because we handle printing here for the table
        # But wait, env.step signature might not allow verbose if we didn't update the base class?
        # We updated TopDownSwarmEnv.
        
        # We need to capture the state AFTER step for velocity, but BEFORE step for action?
        # Action is applied. Velocity results from integration.
        
        # To get breakdown, we might need to modify engine to return info?
        # For now, just printing what we have.
        
        try:
            # Try passing verbose=False to suppress internal engine prints if we want our own table
            obs, rewards, term, trunc, info = env.step(actions, verbose=False) 
        except TypeError:
            # Fallback if I messed up the signature update or inheritance
            obs, rewards, term, trunc, info = env.step(actions)
            
        agent = env.agents[0]
        speed = np.sqrt(agent.vx**2 + agent.vy**2)
        rew = rewards[0]
        total_reward += rew
        
        # Format action
        # Assuming single agent
        act_arr = actions[0]
        act_str = f"[{act_arr[0]:.1f}, {act_arr[1]:.1f}]"
        if len(act_arr) > 3:
             if act_arr[3] > 0.5: act_str += " 📡"
        
        # Note generation
        note = ""
        if rew < -5.0: note = "⚠️ PENALTY"
        if rew < -30.0: note = "🔥 CRITICAL"
        
        print(f"{step+1:^4} | {speed:^8.3f} | {act_str:^20} | {rew:^8.1f} | {note}")
        
        history.append({"speed": speed, "reward": rew})
        
        if term or trunc:
            print(f">>> Terminated at step {step}")
            break
            
    print("-" * 60)
    print(f"TOTAL REWARD: {total_reward:.2f}")
    print("="*60 + "\n")
    return total_reward
