from typing import Any, Dict
import logging

class QAGate:
    """
    Quality Assurance Gate.
    Verifies if a model/dataset meets the criteria to proceed to the next station.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("QAGate")

    def check(self, metrics: Dict[str, float], station_name: str) -> bool:
        """
        Check metrics against thresholds.
        """
        thresholds = self.config.get("stations", {}).get(station_name, {})
        
        passed = True
        
        if "target_accuracy" in thresholds:
            if metrics.get("accuracy", 0.0) < thresholds["target_accuracy"]:
                self.logger.warning(f"QA Fail: Accuracy {metrics.get('accuracy')} < {thresholds['target_accuracy']}")
                passed = False
                
        if "target_success_rate" in thresholds:
            if metrics.get("success_rate", 0.0) < thresholds["target_success_rate"]:
                self.logger.warning(f"QA Fail: Success Rate {metrics.get('success_rate')} < {thresholds['target_success_rate']}")
                passed = False

        return passed
