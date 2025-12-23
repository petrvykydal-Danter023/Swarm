import pygame
import numpy as np
import pymunk
from typing import Dict, List, Optional
from core.entities import Agent, Wall, Goal

class Renderer:
    def __init__(self, width: int = 800, height: int = 600, title: str = "Entropy Engine V2", render_mode: Optional[str] = "human"):
        self.width = width
        self.height = height
        self.title = title
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # Colors
        self.COLOR_BG = (250, 250, 250)
        self.COLOR_WALL = (50, 50, 50)
        self.COLOR_AGENT = (41, 128, 185) # Blue
        self.COLOR_AGENT_DIR = (20, 20, 20)
        self.COLOR_GOAL = (39, 174, 96) # Green
        self.COLOR_LIDAR = (200, 200, 200)
        
    def init_window(self):
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption(self.title)
            else:
                # Offscreen rendering
                pygame.init()
                self.screen = pygame.Surface((self.width, self.height))
                
            self.clock = pygame.time.Clock()

    def render_frame(self, agents: Dict[str, Agent], goals: Dict[str, Goal], walls: List[Wall]) -> np.ndarray:
        self.init_window()
        
        self.screen.fill(self.COLOR_BG)
        
        # Draw Walls
        # We assume walls are simple segments for now or we iterate all shapes if we had them
        # Here we just draw a border if walls are not strictly tracked as a list in env (Env init creates them)
        # But let's assume we pass walls from Env
        
        # Draw Border (hardcoded for now as Env creates 4 walls)
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.width, self.height), 5)
        
        # Draw Goals
        for goal in goals.values():
            pos = (int(goal.body.position.x), int(goal.body.position.y))
            pygame.draw.circle(self.screen, self.COLOR_GOAL, pos, int(goal.radius))
            # Draw outline
            pygame.draw.circle(self.screen, (30, 130, 70), pos, int(goal.radius), 2)

        # Draw Agents
        for agent in agents.values():
            pos = (int(agent.position.x), int(agent.position.y))
            
            # Draw Lidar Rays (Optional - can be noisy)
            # self._draw_lidar(agent)
            
            # Body
            pygame.draw.circle(self.screen, self.COLOR_AGENT, pos, int(agent.radius))
            
            # Direction Indicator
            end_pos = agent.position + pymunk.Vec2d(agent.radius, 0).rotated(agent.angle)
            pygame.draw.line(self.screen, self.COLOR_AGENT_DIR, pos, (int(end_pos.x), int(end_pos.y)), 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
            return None
        else:
            # Return RGB Array
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def _draw_lidar(self, agent: Agent):
        # Re-calculate lidar rays for visuals
        # This duplicates logic but effectively Visualizing what agent "sees" if we just draw lines
        pass

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
