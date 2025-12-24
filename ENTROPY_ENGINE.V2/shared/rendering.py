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

    def render_frame(self, agents: Dict[str, Agent], goals: Dict[str, Goal], walls: List[Wall], comm_state=None) -> np.ndarray:
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

        # Signal Overlay (Transparent)
        signal_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw Agents
        for agent in agents.values():
            pos = (int(agent.position.x), int(agent.position.y))
            
            # Draw Communication Signals (Waves)
            # Ch1: Cyan, Ch2: Magenta
            s1, s2 = agent.current_signal
            
            if s1 > 0.05:
                radius = int(agent.radius * 2 + (s1 * 30))
                alpha = int(min(255, s1 * 150))
                pygame.draw.circle(signal_surf, (0, 255, 255, alpha), pos, radius)
                # Outer ring
                pygame.draw.circle(signal_surf, (0, 255, 255, alpha), pos, radius, 2)
                
            if s2 > 0.05:
                # If both signals active, draw second one slightly larger/smaller or blended
                radius = int(agent.radius * 2.5 + (s2 * 30))
                alpha = int(min(255, s2 * 150))
                pygame.draw.circle(signal_surf, (255, 0, 255, alpha), pos, radius)
                pygame.draw.circle(signal_surf, (255, 0, 255, alpha), pos, radius, 2)

            # Draw Lidar Rays (Optional - can be noisy)
            # self._draw_lidar(agent)
            
        # Blit signals
        self.screen.blit(signal_surf, (0,0))
        
        # Draw Agents Body (after signals so agent is on top)
        for agent in agents.values():
            pos = (int(agent.position.x), int(agent.position.y))
            
            # Draw Lidar Rays (Optional - can be noisy)
            # self._draw_lidar(agent)
            
            # Body
            pygame.draw.circle(self.screen, self.COLOR_AGENT, pos, int(agent.radius))
            
            # Direction Indicator
            end_pos = agent.position + pymunk.Vec2d(agent.radius, 0).rotated(agent.angle)
            pygame.draw.line(self.screen, self.COLOR_AGENT_DIR, pos, (int(end_pos.x), int(end_pos.y)), 2)

        # Draw Communication (Aura & Token)
        if comm_state:
            self._draw_communication(agents, comm_state)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
            return None
        else:
            # Return RGB Array
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def _draw_communication(self, agents: Dict[str, Agent], comm_state):
        # 1. Broadcast Aura
        aura_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for i, msg in enumerate(comm_state.broadcast_messages):
            if msg.msg_type != 0: # Not SILENCE
                agent_key = f"agent_{i}"
                if agent_key in agents:
                    agent = agents[agent_key]
                    pos = (int(agent.position.x), int(agent.position.y))
                    
                    # Draw Aura
                    hue = (msg.msg_type * 30) % 360
                    color = pygame.Color(0)
                    color.hsla = (hue, 50, 50, 30) # 30% alpha
                    pygame.draw.circle(aura_surface, color, pos, 40)
                    
        self.screen.blit(aura_surface, (0,0))
        
        # 2. Token Indicator
        if comm_state.token_holder != -1:
            holder_idx = comm_state.token_holder
            holder_key = f"agent_{holder_idx}"
            
            if holder_key in agents:
                agent = agents[holder_key]
                pos = (int(agent.position.x), int(agent.position.y))
                
                # Draw Star/Icon for Token Holder
                pygame.draw.circle(self.screen, (255, 215, 0), pos, 8) # Gold center
                
                # Draw Line to Target
                target_idx = comm_state.token_message.target_agent
                if target_idx != -1:
                    target_key = f"agent_{target_idx}"
                    if target_key in agents:
                        target = agents[target_key]
                        t_pos = (int(target.position.x), int(target.position.y))
                        pygame.draw.line(self.screen, (255, 215, 0), pos, t_pos, 3)

    def _draw_lidar(self, agent: Agent):
        pass

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
