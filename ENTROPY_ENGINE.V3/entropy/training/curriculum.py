"""
Entropy Engine V3 - Curriculum Learning
Progressive difficulty scheduling for training.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class CurriculumStage:
    """A single stage in the curriculum."""
    threshold: float      # Reward threshold to advance
    difficulty: int       # Difficulty level (1 = easy, higher = harder)
    config_overrides: Dict[str, Any] = field(default_factory=dict)  # Env config changes

class CurriculumScheduler:
    """
    Progressively increases training difficulty based on agent performance.
    
    Usage:
        stages = [
            CurriculumStage(threshold=0.3, difficulty=1),  # Easy
            CurriculumStage(threshold=0.5, difficulty=2),  # Medium
            CurriculumStage(threshold=0.7, difficulty=3),  # Hard
        ]
        curriculum = CurriculumScheduler(stages)
        
        for episode in range(total_episodes):
            difficulty = curriculum.update(last_reward)
            env.set_difficulty(difficulty)
    """
    def __init__(
        self, 
        stages: List[CurriculumStage], 
        history_size: int = 100,
        promotion_patience: int = 10
    ):
        self.stages = stages
        self.current_stage = 0
        self.reward_history = deque(maxlen=history_size)
        self.promotion_patience = promotion_patience
        self.steps_above_threshold = 0
        
    def update(self, episode_reward: float) -> int:
        """
        Update reward history and check for stage promotion.
        
        Args:
            episode_reward: Mean reward from last episode
            
        Returns:
            Current difficulty level
        """
        self.reward_history.append(episode_reward)
        
        if len(self.reward_history) < 10:
            return self.get_difficulty()
            
        avg_reward = np.mean(self.reward_history)
        
        # Check for promotion
        if self.current_stage < len(self.stages) - 1:
            threshold = self.stages[self.current_stage].threshold
            if avg_reward > threshold:
                self.steps_above_threshold += 1
                if self.steps_above_threshold >= self.promotion_patience:
                    self._promote()
            else:
                self.steps_above_threshold = 0
        
        return self.get_difficulty()
    
    def _promote(self):
        """Advance to next curriculum stage."""
        self.current_stage += 1
        self.steps_above_threshold = 0
        self.reward_history.clear()
        print(f"[Curriculum] Promoted to Stage {self.current_stage + 1}: "
              f"difficulty={self.stages[self.current_stage].difficulty}")
    
    def get_difficulty(self) -> int:
        """Get current difficulty level."""
        return self.stages[self.current_stage].difficulty
    
    def get_config_overrides(self) -> Dict[str, Any]:
        """Get environment config overrides for current stage."""
        return self.stages[self.current_stage].config_overrides
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current curriculum progress for logging."""
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
        return {
            "curriculum/stage": self.current_stage + 1,
            "curriculum/difficulty": self.get_difficulty(),
            "curriculum/avg_reward": avg_reward,
            "curriculum/steps_above_threshold": self.steps_above_threshold,
            "curriculum/threshold": self.stages[self.current_stage].threshold,
        }


# Default curriculum for Entropy Engine
DEFAULT_CURRICULUM = [
    CurriculumStage(
        threshold=0.3, 
        difficulty=1,
        config_overrides={"num_obstacles": 0, "comm_noise": 0.0}
    ),
    CurriculumStage(
        threshold=0.5, 
        difficulty=2,
        config_overrides={"num_obstacles": 3, "comm_noise": 0.1}
    ),
    CurriculumStage(
        threshold=0.7, 
        difficulty=3,
        config_overrides={"num_obstacles": 6, "comm_noise": 0.2}
    ),
    CurriculumStage(
        threshold=0.85, 
        difficulty=4,
        config_overrides={"num_obstacles": 10, "comm_noise": 0.3}
    ),
]
