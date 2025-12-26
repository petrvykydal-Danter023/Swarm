"""
Dashboard Reporter helper class.
Connects the training loop to the dashboard server.
"""
from entropy.dashboard.server import DashboardServer
import wandb

class DashboardReporter:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DashboardReporter, cls).__new__(cls)
            cls._instance.server = DashboardServer(port=8080)
            cls._instance.server.start()
        return cls._instance

    def update(self, **kwargs):
        self.server.update(**kwargs)
        
        # Check if WandB just started
        if "wandb_url" not in kwargs and wandb.run is not None:
             self.server.update(wandb_url=wandb.run.get_url())

    def log(self, message: str):
        self.server.add_log(message)
        
    def should_render(self) -> bool:
        return self.server.is_render_enabled()
