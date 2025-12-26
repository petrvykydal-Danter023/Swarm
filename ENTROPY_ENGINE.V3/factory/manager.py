import logging
from typing import List, Any, Tuple
from factory.stations.base import Station
from factory.stations.s0_oracle import OracleStation
from factory.stations.s1_kindergarten import KindergartenStation
from factory.stations.s2_gym import GymStation
from factory.stations.s3_language_school import LanguageSchoolStation
from factory.stations.s4_team_building import TeamBuildingStation
from factory.stations.s5_war_room import WarRoomStation
from factory.stations.s55_domain_randomization import DomainRandomizationStation

class FactoryManager:
    """
    Orchestrator for the Swarm AI Factory.
    Manages the lifecycle of models as they pass through stations.
    """
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("FactoryManager")
        self.stations: List[Station] = []
        self._setup_stations()

    def _setup_stations(self):
        """Initialize stations based on config."""
        self.logger.info("Initializing Factory Stations...")
        
        stations_config = self.config.get("stations", {})
        
        # Full pipeline: Oracle -> Kindergarten -> Gym -> Language -> Team -> War -> DR
        self.stations = [
            OracleStation(stations_config.get("oracle", {})),
            KindergartenStation(stations_config.get("kindergarten", {})),
            GymStation(stations_config.get("gym", {})),
            LanguageSchoolStation(stations_config.get("language_school", {})),
            TeamBuildingStation(stations_config.get("team_building", {})),
            WarRoomStation(stations_config.get("war_room", {})),
            DomainRandomizationStation(stations_config.get("domain_randomization", {})),
        ]
        
        self.logger.info(f"Loaded {len(self.stations)} stations: {[s.name for s in self.stations]}")

    def run_pipeline(self, initial_model: Any = None) -> Tuple[bool, Any]:
        """
        Run the full factory pipeline.
        """
        from entropy.dashboard.reporter import DashboardReporter
        dashboard = DashboardReporter()
        
        current_model = initial_model
        
        for i, station in enumerate(self.stations):
            self.logger.info(f"🏭 Entering Station: {station.name}")
            dashboard.update(station=station.name)
            dashboard.log(f"Entering Station: {station.name}")
            
            # 1. Warmup
            dashboard.log(f"{station.name}: Warmup...")
            if not station.warmup(current_model):
                self.logger.error(f"❌ Warmup failed at {station.name}. Aborting.")
                return False, None
                
            # 2. Train with retries
            success = False
            model_candidate = None
            max_retries = self.config.get("auto_retry", 3)
            
            for attempt in range(max_retries):
                dashboard.log(f"{station.name}: Training attempt {attempt+1}/{max_retries}")
                dashboard.update(episode_current=attempt, episode_total=max_retries)
                
                success, model_candidate = station.train(current_model)
                if success:
                    break
                self.logger.warning(f"⚠️ Attempt {attempt+1}/{max_retries} failed at {station.name}.")
            
            if not success:
                self.logger.error(f"🛑 Station {station.name} FATAL FAIL. Discarding batch.")
                dashboard.log(f"🛑 Station {station.name} FAILED FATALLY.")
                return False, None
                
            # 3. QA Gate
            dashboard.log(f"{station.name}: QA Gate Validation...")
            if not station.validate(model_candidate):
                self.logger.error(f"📉 QA Gate failed at {station.name}.")
                dashboard.log(f"📉 QA Gate FAILED at {station.name}.")
                return False, None
                
            # 4. Proceed
            current_model = model_candidate
            self.logger.info(f"✅ Station {station.name} passed. Proceeding.")
            dashboard.log(f"✅ Station {station.name} PASSED.")
            
        self.logger.info("🎉 Factory Pipeline Completed Successfully!")
        dashboard.log("🎉 Pipeline COMPLETED Successfully!")
        dashboard.update(station="Completed", metrics={"success": 1.0})
        return True, current_model
