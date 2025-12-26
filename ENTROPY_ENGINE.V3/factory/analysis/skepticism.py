"""
Adaptive Skepticism Dial
Dynamically adjusts expert error_rate based on AI performance.
"""
import logging
from typing import Any, Dict

class AdaptiveSkepticismDial:
    """
    Dynamically adjusts error_rate based on AI performance metrics.
    Prevents destroying model with too aggressive fallible expert.
    """
    def __init__(self, config: Any):
        self.error_rate = config.get("start_rate", 0.0)
        self.min_rate = config.get("min_rate", 0.0)
        self.max_rate = config.get("max_rate", 0.20)
        self.adjust_interval = config.get("adjust_interval", 1000)
        
        self.success_threshold = config.get("success_threshold", 0.95)
        self.collision_threshold = config.get("collision_threshold", 0.02)
        self.crisis_threshold = config.get("crisis_threshold", 0.70)
        
        self.increase_step = config.get("increase_step", 0.01)
        self.decrease_step = config.get("decrease_step", 0.02)
        
        self.logger = logging.getLogger("SkepticismDial")

    def update(self, metrics: Dict[str, float]) -> float:
        """
        Update error_rate based on current performance.
        
        Args:
            metrics: dict with 'success_rate', 'collision_rate'
        Returns:
            Updated error_rate
        """
        success_rate = metrics.get("success_rate", 0.0)
        collision_rate = metrics.get("collision_rate", 1.0)
        
        # AI is stable and successful -> increase skepticism
        if success_rate > self.success_threshold and collision_rate < self.collision_threshold:
            self.error_rate = min(self.error_rate + self.increase_step, self.max_rate)
            self.logger.info(f"📈 AI stable, skepticism -> {self.error_rate:.1%}")
            
        # AI is struggling -> emergency brake
        elif success_rate < self.crisis_threshold:
            self.error_rate = max(self.error_rate - self.decrease_step, self.min_rate)
            self.logger.warning(f"🚨 AI struggling, skepticism -> {self.error_rate:.1%}")
            
        return self.error_rate

    def get_error_rate(self) -> float:
        return self.error_rate
