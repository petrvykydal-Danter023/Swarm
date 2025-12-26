
"""
Entropy Engine V3 - CPU Renderer
Headless rendering for generating training validation videos (GIFs).
Uses OpenCV/NumPy for fast rasterization without OpenGL/Window.
"""
import numpy as np
import cv2
from typing import Optional, List, Tuple
from entropy.render.schema import RenderFrame

# Colors (BGR for OpenCV)
# Colors (BGR for OpenCV)
COLOR_BG = (245, 245, 250)   # Very Light Blue-Grey / White
COLOR_AGENT = (200, 100, 0)  # Dark Blue (OpenCV uses BGR) -> 0, 100, 200 actually
COLOR_GOAL = (80, 180, 80)   # Soft Green
COLOR_WALL = (50, 50, 50)    # Dark Grey
COLOR_TEXT = (50, 50, 50)    # Dark Text

class CPURenderer:
    """
    Renders simulation state with a Clean/Vector aesthetic.
    """
    def __init__(self, width: int = 800, height: int = 600, scale: float = 1.0):
        self.width = int(width * scale)
        self.height = int(height * scale)
        self.scale = scale
        # Base canvas
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    def render(self, frame: RenderFrame) -> np.ndarray:
        """Draws the frame onto a new image."""
        # Reset canvas
        img = np.full((self.height, self.width, 3), COLOR_BG, dtype=np.uint8)
        
        # 1. Scaling Helper
        def to_pix(pos):
            return (int(pos[0] * self.scale), int(self.height - pos[1] * self.scale))

        # 2. Draw Walls
        if frame.wall_segments is not None:
             for i in range(len(frame.wall_segments)):
                 w = frame.wall_segments[i]
                 if len(w.shape) == 1 and w.shape[0] == 4:
                     p1 = to_pix((w[0], w[1]))
                     p2 = to_pix((w[2], w[3]))
                     cv2.line(img, p1, p2, COLOR_WALL, 3)
                 elif len(w.shape) == 2:
                     pts = np.array([to_pix(p) for p in w], np.int32)
                     cv2.polylines(img, [pts], True, COLOR_WALL, 3)

        # 2.5 Draw Pheromones (Stigmergy Markers)
        if frame.pheromone_positions is not None and frame.pheromone_valid is not None:
            overlay_phero = img.copy()
            
            for i in range(len(frame.pheromone_positions)):
                if frame.pheromone_valid[i]:
                    pos = frame.pheromone_positions[i]
                    center = to_pix(pos)
                    
                    # Fading based on TTL
                    if frame.pheromone_ttls is not None:
                        ttl_ratio = float(frame.pheromone_ttls[i]) / max(frame.pheromone_max_ttl, 1.0)
                    else:
                        ttl_ratio = 1.0
                    
                    # Cyan color with intensity based on TTL
                    intensity = int(255 * ttl_ratio)
                    color = (intensity, intensity, 0)  # Cyan in BGR (B=intensity, G=intensity, R=0)
                    
                    # Draw filled circle for detection radius
                    radius = int(frame.pheromone_radius * self.scale * 0.5)  # Half radius for visual clarity
                    cv2.circle(overlay_phero, center, radius, color, -1)
                    
                    # Border ring
                    cv2.circle(overlay_phero, center, radius, (180, 180, 0), 1)
                    
                    # Small dot at center
                    cv2.circle(overlay_phero, center, int(3 * self.scale), (255, 255, 0), -1)
            
            # Apply transparency for pheromones
            alpha_phero = 0.4
            img = cv2.addWeighted(overlay_phero, alpha_phero, img, 1 - alpha_phero, 0)

        # 3. Draw Ranges & Connections (Underlay)
        if frame.agent_positions is not None and frame.goal_positions is not None:
            n = len(frame.agent_positions)
            n_g = len(frame.goal_positions)
            
            # Create overlay for transparency
            overlay = img.copy()
            
            for i in range(n):
                pos = frame.agent_positions[i]
                center = to_pix(pos)
                
                # A) Sensor Range (Light Blue Aura)
                cv2.circle(overlay, center, int(200 * self.scale), (230, 210, 180), -1) 
                
                # B) Connection Line to Goal (Yellow)
                if i < n_g:
                    goal_pos = frame.goal_positions[i]
                    goal_center = to_pix(goal_pos)
                    cv2.line(overlay, center, goal_center, (0, 200, 255), 2) # Yellow
            
            # Apply transparency
            alpha = 0.3
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            # C) Communication Beams (Dynamic)
            # If agent has message > threshold (assuming index 0 in message is silence/activity check)
            # Or if vocab size is handled. Let's assume non-zero message vector implies talking.
            if frame.agent_messages is not None:
                for i in range(n):
                    msg = frame.agent_messages[i]
                    # Simple heuristic: if sum(abs(msg)) > 0.1, we are talking
                    if np.sum(np.abs(msg)) > 0.1:
                        # Find nearest neighbor to "talk to"
                        nearest_idx = -1
                        min_dist = float('inf')
                        
                        for j in range(n):
                            if i == j: continue
                            dist = np.linalg.norm(frame.agent_positions[i] - frame.agent_positions[j])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_idx = j
                        
                        # Draw Beam
                        if nearest_idx != -1 and min_dist < 300: # Only if within reasonable range
                            p1 = to_pix(frame.agent_positions[i])
                            p2 = to_pix(frame.agent_positions[nearest_idx])
                            
                            # Purple/Magenta Beam
                            cv2.line(img, p1, p2, (255, 0, 255), 2)
                            # Small "spark" at target
                            cv2.circle(img, p2, int(5 * self.scale), (255, 0, 255), -1)

        # 3.5 Draw Safety & Intent Overlays
        # A) Safety Radius
        if hasattr(frame, 'safety_enabled') and frame.safety_enabled and frame.agent_positions is not None:
             overlay_safety = img.copy()
             for i, pos in enumerate(frame.agent_positions):
                 center = to_pix(pos)
                 
                 # Safety Radius (Dashed or Solid Yellow Ring)
                 r_safe = int(frame.safety_radius * self.scale)
                 cv2.circle(overlay_safety, center, r_safe, (0, 200, 255), 1)
                 
                 # Repulsion Radius (Red Ring)
                 r_rep = int(frame.safety_repulsion_radius * self.scale)
                 cv2.circle(overlay_safety, center, r_rep, (0, 0, 255), 1)

             # Blend
             alpha_safe = 0.5
             img = cv2.addWeighted(overlay_safety, alpha_safe, img, 1 - alpha_safe, 0)

        # B) Intent Vectors
        if hasattr(frame, 'intent_enabled') and frame.intent_enabled and frame.agent_intents is not None:
             for i, pos in enumerate(frame.agent_positions):
                 center = to_pix(pos)
                 intent = frame.agent_intents[i]
                 
                 # Intent Vector Logic based on intent type
                 # [0]: Type (<0 Vel, >0 Target)
                 # [1]: P1 (V or RelX)
                 # [2]: P2 (Omega or RelY)
                 
                 intent_type = intent[0]
                 p1 = intent[1]
                 p2 = intent[2]
                 
                 if intent_type > 0: # Target Mode
                     # Draw Target Vector (Green Arrow)
                     # p1=RelX, p2=RelY. Relative to agent orientation?
                     # Step logic assumes relative to agent BODY.
                     # We need agent angle to project to world space.
                     angle = frame.agent_angles[i]
                     cos_a = np.cos(angle)
                     sin_a = np.sin(angle)
                     
                     # Rotate rel vector (p1, p2) by agent angle
                     # x_world = x_rel * cos - y_rel * sin
                     # y_world = x_rel * sin + y_rel * cos
                     vec_x = p1 * cos_a - p2 * sin_a
                     vec_y = p1 * sin_a + p2 * cos_a
                     
                     # Scale for visibility? (Intents might be large)
                     # Just draw target point
                     target_pos = (pos[0] + vec_x, pos[1] + vec_y)
                     target_pix = to_pix(target_pos)
                     
                     cv2.arrowedLine(img, center, target_pix, (0, 200, 0), 2, tipLength=0.2)
                     cv2.circle(img, target_pix, int(4 * self.scale), (0, 200, 0), -1)
                     
                 else: # Velocity Mode
                     # Draw Velocity Vector (Blue Arrow)
                     # p1=V, p2=Omega
                     v = p1 * 50.0 # Scale for viz
                     
                     end_pos = (
                         pos[0] + np.cos(frame.agent_angles[i]) * v,
                         pos[1] + np.sin(frame.agent_angles[i]) * v
                     )
                     end_pix = to_pix(end_pos)
                     cv2.arrowedLine(img, center, end_pix, (255, 0, 0), 2, tipLength=0.2)

        # Helper for unique colors/IDs
        def get_color(idx, total):
            # Gentle variation
            return COLOR_AGENT

        # 4. Draw Goals
        if frame.goal_positions is not None:
            n_goals = len(frame.goal_positions)
            for i in range(n_goals):
                pos = frame.goal_positions[i]
                center = to_pix(pos)
                
                # Goal Body
                cv2.circle(img, center, int(10 * self.scale), COLOR_GOAL, -1)
                cv2.circle(img, center, int(12 * self.scale), (255, 255, 255), 2)
                
                # ID (Subtle)
                if self.scale > 0.5:
                    cv2.putText(img, str(i), (center[0]-4, center[1]+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # 5. Draw Agents
        if frame.agent_positions is not None:
            n = len(frame.agent_positions)
            
            # Squad color palette (BGR)
            squad_colors = [
                (200, 100, 0),    # Blue (default)
                (0, 180, 0),      # Green
                (0, 0, 200),      # Red
                (200, 0, 200),    # Magenta
                (0, 200, 200),    # Yellow
                (200, 200, 0),    # Cyan
                (100, 50, 150),   # Purple
                (50, 150, 100),   # Teal
            ]
            
            for i in range(n):
                pos = frame.agent_positions[i]
                angle = frame.agent_angles[i]
                center = to_pix(pos)
                radius = int(12 * self.scale)
                
                # Determine color based on squad (if available)
                if frame.agent_squad_ids is not None:
                    squad_id = int(frame.agent_squad_ids[i])
                    color = squad_colors[squad_id % len(squad_colors)]
                else:
                    color = COLOR_AGENT
                
                # Agent Body
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(img, center, radius, (255,255,255), 2) 
                
                # Leader Star Marker
                if frame.agent_is_leader is not None and frame.agent_is_leader[i]:
                    # Draw golden star above agent
                    star_center = (center[0], center[1] - int(20 * self.scale))
                    cv2.drawMarker(img, star_center, (0, 215, 255),  # Gold in BGR
                                  cv2.MARKER_STAR, int(15 * self.scale), 2)
                
                # Direction Indicator (Dark Line)
                end_pos = (
                    pos[0] + np.cos(angle) * 12, # Inside radius
                    pos[1] + np.sin(angle) * 12
                )
                end = to_pix(end_pos)
                cv2.line(img, center, end, (50, 50, 50), 2)
                
                # ID
                if self.scale > 0.5:
                    cv2.putText(img, str(i), (center[0]-5, center[1]+4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                
        # 5. Overlay Text
        text = f"Step: {frame.timestep} | FPS: {getattr(frame, 'fps', 'N/A')}" # FPS placeholder
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 1)
        
        return img
