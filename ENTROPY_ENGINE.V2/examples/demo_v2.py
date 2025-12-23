import sys
import os
import time
import numpy as np

# Add the module root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.entropy_env import EntropyEnv

def main():
    print("Initializing Environment...")
    env = EntropyEnv(render_mode="human", nr_agents=5)
    
    observations, infos = env.reset()
    
    print("Starting Loop...")
    running = True
    while running:
        # Random Actions
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Override to drive circles to test physics
        # for agent in env.agents:
        #     actions[agent] = np.array([1.0, 0.5], dtype=np.float32)
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render
        env.render()
        
        # Slow down for human eyes
        time.sleep(0.05)
        
        # Simple exit cond
        # In human mode Pygame handles window close but wrapped in Env it might not expose quit event easily
        # check pygame events via env
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
    env.close()

if __name__ == "__main__":
    main()
