"""
Entropy Engine V3 - Vispy Viewer
Real-time visualization of the simulation state.
Complete implementation of 04_RENDERING.txt blueprint.
"""
from vispy import app, scene
from vispy.scene import visuals
import numpy as np
import time
from typing import Optional
from .client import RenderClient
from .schema import RenderFrame

# === COLOR PALETTE ===
AGENT_DEFAULT_COLOR = (0, 0, 1)    # Blue
GOAL_COLOR = (0, 1, 0, 0.5)        # Green semi-transparent
WALL_COLOR = (0.5, 0.5, 0.5, 1)    # Gray
COMM_LINE_COLOR = (0.2, 0.8, 1.0, 0.5)  # Cyan
LIDAR_COLOR = (1.0, 0.5, 0.0, 0.3)      # Orange

AURA_COLORS = {
    0: (0.5, 0.5, 0.5, 0.1),   # SILENCE
    1: (1.0, 1.0, 0.0, 0.6),   # HELP
    2: (1.0, 0.0, 0.0, 0.7),   # DANGER
    3: (0.0, 0.5, 1.0, 0.6),   # CARRYING
    7: (0.0, 1.0, 0.2, 0.7),   # FOUND_TARGET
}

TOKEN_NAMES = {
    0: "",          # SILENCE (don't display)
    1: "HELP",
    2: "DANGER",
    3: "CARRY",
    4: "DROP",
    5: "FOLLOW",
    6: "AVOID",
    7: "TARGET",
    8: "CONFIRM",
    9: "DENY",
}

class EntropyViewer:
    """
    Main visualization window using Vispy.
    Includes: Agents, Goals, Walls, Auras, Communication Lines, Debug Overlay.
    """
    def __init__(
        self, 
        port: int = 5555, 
        width=1024, 
        height=768, 
        show_debug=True,
        show_velocities=True,
        show_lidars=False,  # Expensive, off by default
        show_rewards=True
    ):
        self.client = RenderClient(server_address=f"tcp://localhost:{port}")
        self.show_debug = show_debug
        self.show_velocities = show_velocities
        self.show_lidars = show_lidars
        self.show_rewards = show_rewards
        
        # Setup Canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(width, height),
            title='Entropy Engine V3',
            show=True
        )
        
        # Setup Camera
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(-100, 900), y=(-100, 700))
        
        # === VISUALS (in draw order, back to front) ===
        
        # 1. Lidar Rays (background)
        self.lidar_visual = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            color=LIDAR_COLOR,
            width=1
        )
        
        # 2. Communication Auras
        self.auras_visual = scene.visuals.Markers(
            parent=self.view.scene,
            edge_width=0,
            symbol='disc'
        )
        
        # 3. Communication Lines
        self.comm_lines_visual = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            color=COMM_LINE_COLOR,
            width=1.5
        )
        
        # 4. Walls
        self.walls_visual = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            color=WALL_COLOR,
            width=3
        )
        
        # 5. Goals
        self.goals_visual = scene.visuals.Markers(
            parent=self.view.scene,
            edge_color='white',
            symbol='star'
        )
        
        # 6. Agents
        self.agents_visual = scene.visuals.Markers(
            parent=self.view.scene,
            edge_color='white',
            symbol='disc'
        )
        
        # 7. Agent Direction Indicators
        self.direction_visual = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            color='white',
            width=1.5
        )
        
        # 8. Velocity Arrows
        self.velocity_visual = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            color=(0, 1, 1, 0.8),
            width=2
        )
        
        # 9. Debug Text labels
        self.debug_texts = []
        
        # Timer for updates
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.last_frame_time = time.time()
        self.fps_counter = 0

    def on_timer(self, event):
        """Main loop callback."""
        frame = self.client.receive_frame()
        for _ in range(5):
             next_frame = self.client.receive_frame(timeout_ms=0)
             if next_frame:
                 frame = next_frame
             else:
                 break
                 
        if frame:
            self.update_scene(frame)

    def update_scene(self, frame: RenderFrame):
        """Update all visuals from frame data."""
        n = len(frame.agent_positions) if frame.agent_positions is not None else 0
        
        if n > 0:
            # === AGENTS ===
            # Color by rewards if enabled
            if self.show_rewards and frame.rewards is not None:
                colors = self._rewards_to_colors(frame.rewards)
            elif frame.agent_colors is not None:
                colors = frame.agent_colors
            else:
                colors = np.tile(AGENT_DEFAULT_COLOR, (n, 1))
                
            self.agents_visual.set_data(
                pos=frame.agent_positions,
                face_color=colors,
                size=20
            )
            
            # === DIRECTION INDICATORS ===
            radii = frame.agent_radii if frame.agent_radii is not None else np.full(n, 10)
            angles = frame.agent_angles
            tips = frame.agent_positions + np.stack([
                np.cos(angles) * radii,
                np.sin(angles) * radii
            ], axis=1)
            
            line_points = np.empty((n * 2, 2))
            line_points[0::2] = frame.agent_positions
            line_points[1::2] = tips
            self.direction_visual.set_data(pos=line_points)
            
            # === VELOCITY ARROWS ===
            if self.show_velocities:
                self._update_velocity_arrows(frame)
            
            # === AURAS ===
            self._update_auras(frame)
            
            # === COMMUNICATION LINES ===
            self._update_comm_lines(frame)
            
            # === LIDAR RAYS ===
            if self.show_lidars:
                self._update_lidars(frame)
            
            # === DEBUG OVERLAY ===
            if self.show_debug:
                self._update_debug_overlay(frame)
        
        # === GOALS ===
        if frame.goal_positions is not None and len(frame.goal_positions) > 0:
             self.goals_visual.set_data(
                pos=frame.goal_positions,
                face_color=GOAL_COLOR,
                size=30
            )

        # === WALLS ===
        if frame.wall_segments is not None and len(frame.wall_segments) > 0:
             walls_flat = frame.wall_segments.reshape(-1, 2, 2)
             wall_points = walls_flat.reshape(-1, 2)
             self.walls_visual.set_data(pos=wall_points)

        self.canvas.title = f"Entropy Engine V3 | Step: {frame.timestep} | FPS: {frame.fps:.1f}"
        self.canvas.update()

    def _rewards_to_colors(self, rewards: np.ndarray) -> np.ndarray:
        """Map rewards to colors (red=bad, green=good)."""
        n = len(rewards)
        # Normalize rewards to [0, 1] range
        r_min, r_max = rewards.min(), rewards.max()
        if r_max - r_min > 1e-6:
            normalized = (rewards - r_min) / (r_max - r_min)
        else:
            normalized = np.full(n, 0.5)
        
        colors = np.zeros((n, 3))
        colors[:, 0] = 1.0 - normalized  # Red channel
        colors[:, 1] = normalized        # Green channel
        colors[:, 2] = 0.2               # Blue constant
        return colors

    def _update_velocity_arrows(self, frame: RenderFrame):
        """Draw velocity vectors as arrows."""
        if frame.agent_velocities is None:
            self.velocity_visual.set_data(pos=np.zeros((0, 2)))
            return
            
        n = len(frame.agent_positions)
        scale = 0.5  # Scale factor for visibility
        
        ends = frame.agent_positions + frame.agent_velocities * scale
        
        lines = np.empty((n * 2, 2))
        self.velocity_visual.set_data(pos=lines)

    def _update_crates(self, positions: np.ndarray, frame):
        """Draw crates as squares."""
        if positions is None or positions.shape[0] == 0:
            if hasattr(self, 'crate_visual'):
                self.crate_visual.visible = False
            return
            
        if not hasattr(self, 'crate_visual'):
            self.crate_visual = scene.visuals.Markers(
                edge_color='black', face_color=(0.6, 0.4, 0.2, 1), size=20, symbol='square', parent=self.view.scene
            )
            
        self.crate_visual.set_data(pos=positions)
        self.crate_visual.visible = True
        
    def _update_zones(self, zone_bounds: np.ndarray, zone_types: np.ndarray):
        """Draw zones as rectangles."""
        if zone_bounds is None or zone_bounds.shape[0] == 0:
            if hasattr(self, 'zone_visuals'):
                for v in self.zone_visuals: v.parent = None
                self.zone_visuals = []
            return
            
        # Recreate visuals only if count changes (optimization: check len)
        # For simplicity, clear and redraw if needed, or update existing
        if not hasattr(self, 'zone_visuals'):
            self.zone_visuals = []
            
        # Ensure we have enough visuals
        while len(self.zone_visuals) < len(zone_bounds):
            rect = scene.visuals.Rectangle(center=(0,0), width=1, height=1, color=(0,1,0,0.2), parent=self.view.scene)
            self.zone_visuals.append(rect)
            
        # Update
        for i, (visual, bounds, ztype) in enumerate(zip(self.zone_visuals, zone_bounds, zone_types)):
            # bounds: min_x, min_y, max_x, max_y
            min_x, min_y, max_x, max_y = bounds
            w = max_x - min_x
            h = max_y - min_y
            cx = min_x + w/2
            cy = min_y + h/2
            
            # Color by type
            # 0=DropOff (Green), 1=Charging (Blue), 2=Hazard (Red)
            color = (0, 1, 0, 0.2) if ztype == 0 else (0, 0, 1, 0.2) if ztype == 1 else (1, 0, 0, 0.2)
            
            visual.center = (cx, cy, 0)
            visual.width = w
            visual.height = h
            visual.color = color
            visual.visible = True
            
        # Hide unused
        for i in range(len(zone_bounds), len(self.zone_visuals)):
            self.zone_visuals[i].visible = False

    def _update_lidars(self, frame: RenderFrame):
        """Draw lidar rays."""
        if frame.lidar_readings is None:
            self.lidar_visual.set_data(pos=np.zeros((0, 2)))
            return
            
        n = len(frame.agent_positions)
        num_rays = frame.lidar_readings.shape[1]
        max_range = 100.0  # Assumed max lidar range
        
        lines = []
        for i in range(n):
            pos = frame.agent_positions[i]
            base_angle = frame.agent_angles[i]
            
            # Rays spread around facing direction
            for r in range(num_rays):
                ray_angle = base_angle + (r - num_rays / 2) * (np.pi / num_rays)
                distance = frame.lidar_readings[i, r] * max_range
                
                end = pos + np.array([np.cos(ray_angle), np.sin(ray_angle)]) * distance
                lines.append(pos)
                lines.append(end)
        
        if lines:
            self.lidar_visual.set_data(pos=np.array(lines))

    def _update_auras(self, frame: RenderFrame):
        """Visualize communication as colored auras."""
        if frame.agent_messages is None:
            return
            
        n = len(frame.agent_positions)
        tokens = np.argmax(frame.agent_messages[:, :32], axis=1)
        
        colors = np.zeros((n, 4))
        sizes = np.zeros(n)
        
        for i in range(n):
            token = tokens[i]
            col = AURA_COLORS.get(token, (0.8, 0.8, 0.8, 0.2))
            colors[i] = col
            
            urgency = 0.5
            if frame.agent_messages.shape[1] >= 35:
                 urgency = frame.agent_messages[i, 34]
            
            pulse = 1.0 + 0.1 * np.sin(time.time() * 5 + i)
            
            if token == 0:
                 sizes[i] = 0
            else:
                 sizes[i] = (30 + urgency * 20) * pulse

        self.auras_visual.set_data(
             pos=frame.agent_positions,
             face_color=colors,
             size=sizes
        )

    def _update_comm_lines(self, frame: RenderFrame):
        """Draw lines between agents that are communicating."""
        if frame.agent_messages is None:
            self.comm_lines_visual.set_data(pos=np.zeros((0, 2)))
            return
            
        n = len(frame.agent_positions)
        lines = []
        colors = []
        
        for i in range(n):
            token = np.argmax(frame.agent_messages[i, :32])
            if token == 0:
                continue
                
            for j in range(i + 1, n):  # Avoid duplicates
                token_j = np.argmax(frame.agent_messages[j, :32])
                if token_j == 0:
                    continue
                    
                dist = np.linalg.norm(frame.agent_positions[i] - frame.agent_positions[j])
                if dist < 100:
                    lines.append(frame.agent_positions[i])
                    lines.append(frame.agent_positions[j])
                    alpha = 1.0 - (dist / 100)
                    colors.append((0.2, 0.8, 1.0, alpha))
                    colors.append((0.2, 0.8, 1.0, alpha))
        
        if lines:
            self.comm_lines_visual.set_data(pos=np.array(lines), color=np.array(colors))
        else:
            self.comm_lines_visual.set_data(pos=np.zeros((0, 2)))

    def _update_debug_overlay(self, frame: RenderFrame):
        """Update debug text: Agent IDs and current Token names."""
        n = len(frame.agent_positions)
        
        # Remove old texts
        for text in self.debug_texts:
            text.parent = None
        self.debug_texts.clear()
        
        tokens = np.argmax(frame.agent_messages[:, :32], axis=1) if frame.agent_messages is not None else np.zeros(n)
        
        for i in range(n):
            pos = frame.agent_positions[i]
            
            # Agent ID
            id_text = scene.visuals.Text(
                f"#{i}",
                pos=(pos[0], pos[1] + 15),
                color='white',
                font_size=8,
                anchor_x='center',
                anchor_y='bottom',
                parent=self.view.scene
            )
            self.debug_texts.append(id_text)
            
            # Token name
            token = int(tokens[i])
            token_name = TOKEN_NAMES.get(token, f"T{token}")
            if token_name:
                token_text = scene.visuals.Text(
                    token_name,
                    pos=(pos[0], pos[1] - 18),
                    color=AURA_COLORS.get(token, (1, 1, 1, 1))[:3],
                    font_size=7,
                    anchor_x='center',
                    anchor_y='top',
                    parent=self.view.scene
                )
                self.debug_texts.append(token_text)
            
            # Reward value (if available)
            if self.show_rewards and frame.rewards is not None:
                reward_text = scene.visuals.Text(
                    f"{frame.rewards[i]:.2f}",
                    pos=(pos[0] + 15, pos[1]),
                    color='yellow',
                    font_size=6,
                    anchor_x='left',
                    anchor_y='center',
                    parent=self.view.scene
                )
                self.debug_texts.append(reward_text)

    def run(self):
        app.run()

if __name__ == '__main__':
    viewer = EntropyViewer()
    viewer.run()
