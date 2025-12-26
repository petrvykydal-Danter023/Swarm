from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional

class Station(ABC):
    """
    Base interface for all Factory Stations.
    Each station represents a logical training block in the assembly line.
    """
    def __init__(self, config: Any):
        self.config = config
        self.name = self.__class__.__name__

    def record_milestone(self, model: Any, milestone_name: str, env_class: Any = None):
        """
        Records a validation video for the current model state.
        Args:
            model: Current trained model/state
            milestone_name: Name suffix for the video file
            env_class: Optional EnvWrapper class/config to use
        """
        try:
            from entropy.visuals.recorder import Recorder
            from entropy.training.env_wrapper import EntropyGymWrapper
            import os
            
            output_dir = self.config.get("output_dir", "outputs/recordings")
            recorder = Recorder(output_dir=output_dir)
            
            # Use provided env or default
            if env_class is None:
                 # Create default test config
                 class DefaultConfig:
                     class Env:
                         num_agents = 10
                         arena_width = 800.0
                         arena_height = 600.0
                         max_steps = 400
                     class Model:
                         context_dim = 64
                     env = Env()
                     model = Model()
                 env_wrapper = EntropyGymWrapper(DefaultConfig())
            else:
                 env_wrapper = env_class
                 
            filename = f"{self.name}_{milestone_name}.gif"
            path = recorder.record_episode(env_wrapper, model, model.params, filename)
            print(f"[{self.name}] 🎥 Recorded validation milestone: {path}")
            return path
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to record milestone '{milestone_name}': {e}")
            return None

    @abstractmethod
    def warmup(self, model: Optional[Any] = None) -> bool:
        """
        Dry run to ensure station is ready.
        Returns:
            bool: True if ready, False otherwise.
        """
        pass

    @abstractmethod
    def train(self, model: Optional[Any] = None) -> Tuple[bool, Any]:
        """
        Main training loop for the station.
        Args:
            model: Input model/checkpoint/state from previous station.
                   Can be None for Station 0 or 1.
        Returns:
            (success, trained_model): Tuple of success flag and result.
        """
        pass

    @abstractmethod
    def validate(self, model: Any) -> bool:
        """
        QA Gate check.
        Args:
            model: The model to validate.
        Returns:
            bool: True if model passes QA criteria.
        """
        pass
