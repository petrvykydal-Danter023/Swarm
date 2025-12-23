from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
import time

class RichLogger:
    def __init__(self, total_timesteps: int):
        self.console = Console()
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        
        # We will use a Progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.task = self.progress.add_task("[cyan]Training...", total=total_timesteps)
        
    def start(self):
        self.progress.start()
        
    def update(self, timesteps: int, fps: int, mean_reward: float):
        self.progress.update(
            self.task, 
            completed=timesteps, 
            description=f"[cyan]Training... [green]FPS: {fps} [yellow]Reward: {mean_reward:.2f}"
        )
        
    def finish(self):
        self.progress.stop()
        self.console.print(f"[bold green]Training Completed in {time.time() - self.start_time:.2f}s![/bold green]")

    def log_message(self, message: str, level: str="info"):
        if level == "info":
            self.console.print(f"[blue][INFO][/blue] {message}")
        elif level == "success":
            self.console.print(f"[green][SUCCESS][/green] {message}")
        elif level == "warning":
            self.console.print(f"[yellow][WARNING][/yellow] {message}")
        elif level == "error":
            self.console.print(f"[red][ERROR][/red] {message}")
