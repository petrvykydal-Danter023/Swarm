"""
Side-by-side comparison of two models with LIVE STATS.
Usage: python enjoy_compare.py
"""
import sys
import os

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

import pygame
import numpy as np
from env.entropy_env import EntropyEnv
from sb3_contrib import RecurrentPPO

def main():
    pygame.init()
    
    # Two windows side by side
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    
    # Create two environments
    env1 = EntropyEnv(nr_agents=5, render_mode="rgb_array")
    env2 = EntropyEnv(nr_agents=5, render_mode="rgb_array")
    
    # Load Models
    model1_path = os.path.join(V2_ROOT, "models", "ppo_multicore_entropy_v2_xuoj0qxz.zip")
    model2_path = os.path.join(V2_ROOT, "models", "entropy_v2_1.zip")
    
    print(f"Model 1 (EARLY): {model1_path}")
    print(f"Model 2 (LATEST): {model2_path}")
    
    model1 = RecurrentPPO.load(model1_path)
    model2 = RecurrentPPO.load(model2_path)
    
    # Create combined window (extra height for stats)
    screen = pygame.display.set_mode((WINDOW_WIDTH * 2 + 20, WINDOW_HEIGHT + 80))
    pygame.display.set_caption("Model Comparison: EARLY (Left) vs LATEST (Right)")
    clock = pygame.time.Clock()
    
    # Reset
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    
    # LSTM states
    lstm_states1 = None
    lstm_states2 = None
    episode_starts1 = np.ones(5, dtype=bool)
    episode_starts2 = np.ones(5, dtype=bool)
    
    # STATS TRACKING
    goals_model1 = 0
    goals_model2 = 0
    total_reward1 = 0.0
    total_reward2 = 0.0
    steps = 0
    
    font = pygame.font.SysFont("Arial", 24)
    font_big = pygame.font.SysFont("Arial", 32, bold=True)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Model 1 actions
        actions1 = {}
        for i, agent_id in enumerate(env1.agents):
            obs = obs1[agent_id]
            action, lstm_states1 = model1.predict(obs, state=lstm_states1, episode_start=episode_starts1[i:i+1], deterministic=True)
            actions1[agent_id] = action
            episode_starts1[i] = False
            
        # Model 2 actions
        actions2 = {}
        for i, agent_id in enumerate(env2.agents):
            obs = obs2[agent_id]
            action, lstm_states2 = model2.predict(obs, state=lstm_states2, episode_start=episode_starts2[i:i+1], deterministic=True)
            actions2[agent_id] = action
            episode_starts2[i] = False
            
        # Step
        obs1, rewards1, _, _, _ = env1.step(actions1)
        obs2, rewards2, _, _, _ = env2.step(actions2)
        
        steps += 1
        
        # Count rewards (positive reward = goal reached)
        for agent_id, reward in rewards1.items():
            total_reward1 += reward
            if reward > 5.0:  # Goal bonus threshold
                goals_model1 += 1
                
        for agent_id, reward in rewards2.items():
            total_reward2 += reward
            if reward > 5.0:  # Goal bonus threshold
                goals_model2 += 1
        
        # Render
        frame1 = env1.render()
        frame2 = env2.render()
        
        # Convert to Pygame surfaces
        surf1 = pygame.surfarray.make_surface(np.transpose(frame1, (1, 0, 2)))
        surf2 = pygame.surfarray.make_surface(np.transpose(frame2, (1, 0, 2)))
        
        # Draw
        screen.fill((30, 30, 30))
        screen.blit(surf1, (0, 0))
        screen.blit(surf2, (WINDOW_WIDTH + 20, 0))
        
        # Labels
        label1 = font.render("EARLY (2M steps)", True, (255, 100, 100))
        label2 = font.render("LATEST (6.4M steps)", True, (100, 255, 100))
        screen.blit(label1, (10, 10))
        screen.blit(label2, (WINDOW_WIDTH + 30, 10))
        
        # Stats Panel (Bottom)
        stats_y = WINDOW_HEIGHT + 10
        
        # Model 1 Stats
        score1_text = font_big.render(f"Goals: {goals_model1}", True, (255, 150, 150))
        reward1_text = font.render(f"Total Reward: {total_reward1:.1f}", True, (200, 200, 200))
        screen.blit(score1_text, (50, stats_y))
        screen.blit(reward1_text, (50, stats_y + 35))
        
        # Model 2 Stats
        score2_text = font_big.render(f"Goals: {goals_model2}", True, (150, 255, 150))
        reward2_text = font.render(f"Total Reward: {total_reward2:.1f}", True, (200, 200, 200))
        screen.blit(score2_text, (WINDOW_WIDTH + 70, stats_y))
        screen.blit(reward2_text, (WINDOW_WIDTH + 70, stats_y + 35))
        
        # Comparison in center
        diff = goals_model2 - goals_model1
        if diff > 0:
            diff_color = (100, 255, 100)
            diff_text = f"LATEST +{diff}"
        elif diff < 0:
            diff_color = (255, 100, 100)
            diff_text = f"EARLY +{-diff}"
        else:
            diff_color = (200, 200, 200)
            diff_text = "TIE"
            
        diff_render = font_big.render(diff_text, True, diff_color)
        screen.blit(diff_render, (WINDOW_WIDTH - 30, stats_y + 15))
        
        # Step counter
        steps_text = font.render(f"Steps: {steps}", True, (150, 150, 150))
        screen.blit(steps_text, (WINDOW_WIDTH * 2 - 100, stats_y + 40))
        
        pygame.display.flip()
        clock.tick(30)
        
    env1.close()
    env2.close()
    pygame.quit()

if __name__ == "__main__":
    main()
