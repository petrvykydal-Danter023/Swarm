import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
"""
Manual Verification for Phase 4: Rendering
Run this script to start a fake simulation server.
Then, in another terminal, run: `python -m entropy.render.viewer`
"""
import time
import numpy as np
import random
from entropy.render.server import RenderServer
from entropy.render.schema import RenderFrame

def run_fake_simulation():
    server = RenderServer(port=5555)
    
    num_agents = 50
    radius = 300
    center = np.array([400, 300])
    
    # Initialize state
    angles = np.linspace(0, 2*np.pi, num_agents)
    colors = np.random.rand(num_agents, 3)
    
    timestep = 0
    try:
        print("🚀 Fake Simulation started. Press Ctrl+C to stop.")
        print("Now run 'python -m entropy.render.viewer' in another terminal!")
        
        while True:
            # Update physics (Orbiting)
            angles += 0.02
            
            # Positions
            x = center[0] + np.cos(angles) * radius
            y = center[1] + np.sin(angles) * radius
            positions = np.stack([x, y], axis=1)
            
            # Fake messages for Auras
            # Randomly switch tokens
            messages = np.zeros((num_agents, 36))
            if timestep % 10 == 0:
                active_agents = np.random.choice(num_agents, 5)
                for i in active_agents:
                    token = random.choice([1, 2, 3, 7]) # HELP, DANGER, CARRYING, TARGET
                    messages[i, token] = 1.0 # One-hot-ish
                    messages[i, 34] = random.random() # Urgency
            
            # Frame
            frame = RenderFrame(
                timestep=timestep,
                agent_positions=positions,
                agent_angles=angles + np.pi/2, # Face tangent
                agent_colors=colors,
                agent_messages=messages,
                agent_radii=np.full(num_agents, 10.0),
                agent_velocities=np.stack([np.sin(angles), np.cos(angles)], axis=1) * 20,  # Tangent velocity
                lidar_readings=np.random.rand(num_agents, 32),  # Random lidar for debug viz
                goal_positions=np.array([[400, 300]]), # Center goal
                object_positions=np.zeros((0, 2)),
                object_types=np.zeros(0),
                wall_segments=np.array([
                    [50, 50, 750, 50],
                    [750, 50, 750, 550],
                    [750, 550, 50, 550],
                    [50, 550, 50, 50]
                ]),
                rewards=np.random.randn(num_agents),  # Random rewards for testing
                fps=60.0
            )
            
            server.publish_frame(frame)
            timestep += 1
            time.sleep(1/60.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")
        server.close()

if __name__ == "__main__":
    run_fake_simulation()
