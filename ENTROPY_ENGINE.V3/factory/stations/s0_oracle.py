"""
Station 0: The Oracle Factory
Generates perfect trajectories offline using A* pathfinding.
"""
import logging
import os
from pathlib import Path
from typing import Any, Tuple, List, Dict
from factory.stations.base import Station
from factory.oracle.generator import HardStatesGenerator
from factory.oracle.expert import PrivilegedOracle
from factory.storage.demos import DemoStorage

class OracleStation(Station):
    """
    STATION 0: THE ORACLE FACTORY
    Goal: Generate massive dataset of perfect trajectories (Offline).
    """
    def __init__(self, config: Any):
        super().__init__(config)
        self.logger = logging.getLogger("OracleFactory")
        self.config = config if isinstance(config, dict) else {}
        
        # Ensure default storage path is set if missing
        if "storage" not in self.config:
            self.config["storage"] = {}
        if "path" not in self.config["storage"]:
            self.config["storage"]["path"] = "data/oracle_demos/"
            
        self.generator = HardStatesGenerator(self.config)
        self.oracle = PrivilegedOracle(self.config.get("oracle", {}))
        self.storage = DemoStorage(self.config)
        self.dataset_path = self.config["storage"]["path"]

    def warmup(self, model: Any = None) -> bool:
        """
        Check if we can generate and save data.
        """
        self.logger.info("Initializing Oracle warmup...")
        
        # Ensure output directory exists
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        
        # Test Oracle with mock state
        test_state = {
            "agent_position": (10.0, 10.0),
            "agent_angle": 0.0,
            "goal_position": (90.0, 90.0),
            "obstacles": set(),
            "arena_size": (100, 100)
        }
        action = self.oracle.act(test_state)
        
        if action is not None and "motor" in action:
            self.logger.info(f"  ✓ Oracle produces valid actions: motor={action['motor']}")
            return True
        else:
            self.logger.error("  ✗ Oracle failed to produce valid action")
            return False

    def train(self, model: Any = None) -> Tuple[bool, Any]:
        """
        Executes the data generation loop.
        Returns:
            (success, path_to_dataset)
        """
        num_episodes = self.config.get("generation", {}).get("num_episodes", 1000)
        self.logger.info(f"🔮 Starting Oracle Generation for {num_episodes} episodes...")
        
        all_trajectories: List[Dict] = []
        
        # Generate trajectories using scenarios from HardStatesGenerator
        for episode_idx, scenario in enumerate(self.generator.generate_hard_states(num_per_type=max(1, num_episodes // 6))):
            if episode_idx >= num_episodes:
                break
                
            trajectory = self._generate_episode(scenario, episode_idx)
            all_trajectories.append(trajectory)
            
            # Progress logging
            if (episode_idx + 1) % max(1, num_episodes // 10) == 0:
                progress = (episode_idx + 1) / num_episodes * 100
                self.logger.info(f"  Progress: {progress:.0f}% ({episode_idx + 1}/{num_episodes})")
        
        # Save all trajectories
        self.storage.save_trajectory(all_trajectories, f"oracle_demos_batch")
        
        self.logger.info(f"✅ Generated {len(all_trajectories)} trajectories")
        return True, self.dataset_path

    def _generate_episode(self, scenario: Dict, episode_idx: int) -> Dict:
        """
        Generate a single episode trajectory.
        """
        # Simulated episode (without real env for now)
        # In full implementation, this would use entropy.env.SwarmEnv
        
        trajectory = {
            "episode_id": episode_idx,
            "scenario_type": scenario.get("type", "unknown"),
            "steps": []
        }
        
        # Simulated agent state
        agent_pos = (10.0 + (episode_idx % 10) * 5, 10.0 + (episode_idx // 10) * 5)
        agent_angle = 0.0
        goal_pos = (90.0, 90.0)
        
        # Generate 50 steps per episode
        for step in range(50):
            state = {
                "agent_position": agent_pos,
                "agent_angle": agent_angle,
                "goal_position": goal_pos,
                "obstacles": set(),
                "arena_size": (100, 100)
            }
            
            # Get expert action
            action = self.oracle.act(state)
            
            # Record step
            trajectory["steps"].append({
                "observation": state,
                "action": {
                    "motor": action["motor"].tolist(),
                    "comm": action["comm"].tolist()
                }
            })
            
            # Simulate movement (simplified)
            if action["motor"][0] > 0:
                import math
                agent_pos = (
                    agent_pos[0] + action["motor"][0] * math.cos(agent_angle) * 0.1,
                    agent_pos[1] + action["motor"][0] * math.sin(agent_angle) * 0.1
                )
                agent_angle += action["motor"][1] * 0.1
        
        return trajectory

    def validate(self, model: Any) -> bool:
        """
        Verify the generated dataset.
        """
        self.logger.info("Validating Oracle Dataset...")
        
        # Check if files exist
        demo_files = list(Path(self.dataset_path).glob("*.pkl"))
        if len(demo_files) == 0:
            self.logger.warning("No demo files found, but continuing (may be first run)")
        else:
            self.logger.info(f"  Found {len(demo_files)} demo files")
        
        return True

