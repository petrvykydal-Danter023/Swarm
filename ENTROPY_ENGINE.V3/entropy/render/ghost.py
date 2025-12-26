"""
Entropy Engine V3 - Ghost Renderer
Visualizes expert trajectory vs AI trajectory for debugging Hand of God.
"""
import numpy as np
from vispy import scene
from typing import Optional

class GhostRenderer:
    """
    Renders 'ghost' visualization showing divergence between AI actions and Expert actions.
    Used for debugging Hand of God module.
    
    Usage:
        ghost = GhostRenderer(viewer.view)
        # In update loop:
        ghost.update(frame, ai_actions, expert_actions)
    """
    def __init__(self, view: scene.ViewBox, opacity: float = 0.5):
        self.view = view
        self.opacity = opacity
        
        # Ghost Markers (where expert would move agents)
        self.ghost_visual = scene.visuals.Markers(
            parent=self.view.scene,
            edge_color=(0, 1, 0, 0.5),
            symbol='diamond'
        )
        
        # Divergence Lines (AI position -> Expert position)
        self.divergence_lines = scene.visuals.Line(
            parent=self.view.scene,
            connect='segments',
            width=2
        )
        
        self.enabled = True
        
    def set_enabled(self, enabled: bool):
        """Toggle ghost visualization."""
        self.enabled = enabled
        if not enabled:
            self.ghost_visual.set_data(pos=np.zeros((0, 2)))
            self.divergence_lines.set_data(pos=np.zeros((0, 2)))
    
    def update(
        self,
        current_positions: np.ndarray,
        ai_actions: np.ndarray,
        expert_actions: np.ndarray,
        dt: float = 0.1
    ):
        """
        Update ghost visualization.
        
        Args:
            current_positions: [N, 2] current agent positions
            ai_actions: [N, 2] actions from AI policy
            expert_actions: [N, 2] actions from expert (Hand of God)
            dt: time step for position prediction
        """
        if not self.enabled:
            return
            
        n = len(current_positions)
        
        # Predict next positions (simplified: pos + action * scale)
        # In reality, use differential drive model
        scale = 50.0 * dt
        ai_next = current_positions + ai_actions * scale
        expert_next = current_positions + expert_actions * scale
        
        # Draw ghost markers at expert positions
        self.ghost_visual.set_data(
            pos=expert_next,
            face_color=(0, 1, 0, self.opacity),
            size=15
        )
        
        # Draw divergence lines
        lines = []
        colors = []
        
        for i in range(n):
            dist = np.linalg.norm(ai_next[i] - expert_next[i])
            if dist > 2:  # Threshold for showing divergence
                lines.append(ai_next[i])
                lines.append(expert_next[i])
                
                # Color: Red intensity based on divergence magnitude
                intensity = min(dist / 30.0, 1.0)
                colors.append((1, 1 - intensity, 0, 0.8))
                colors.append((1, 1 - intensity, 0, 0.8))
        
        if lines:
            self.divergence_lines.set_data(
                pos=np.array(lines),
                color=np.array(colors)
            )
        else:
            self.divergence_lines.set_data(pos=np.zeros((0, 2)))
    
    def compute_divergence_metrics(
        self,
        ai_actions: np.ndarray,
        expert_actions: np.ndarray
    ) -> dict:
        """
        Compute metrics comparing AI and Expert actions.
        
        Returns:
            dict with:
                - mean_divergence: Average action difference
                - max_divergence: Maximum action difference
                - agreement_rate: Fraction of agents with similar actions
        """
        diff = np.linalg.norm(ai_actions - expert_actions, axis=1)
        
        return {
            "mean_divergence": float(np.mean(diff)),
            "max_divergence": float(np.max(diff)),
            "agreement_rate": float(np.mean(diff < 0.1))  # Threshold for "agreement"
        }
