from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, ProgressColumn
from rich.text import Text
from rich.style import Style
import time
import os
import json
from typing import Dict, Optional


class GradientBarColumn(ProgressColumn):
    """A progress bar with a gradient from blue to cyan to green."""
    
    def __init__(self, bar_width: int = 40):
        super().__init__()
        self.bar_width = bar_width
        # Gradient colors: blue → cyan → green
        self.colors = [
            (66, 135, 245),   # Blue
            (66, 180, 245),   # Light blue
            (66, 220, 220),   # Cyan
            (66, 235, 180),   # Cyan-green
            (66, 245, 135),   # Green
        ]
    
    def _interpolate_color(self, progress: float, position: float) -> tuple:
        """Get color at a specific position in the gradient."""
        # Scale position to gradient
        scaled = position * (len(self.colors) - 1)
        idx = int(scaled)
        frac = scaled - idx
        
        if idx >= len(self.colors) - 1:
            return self.colors[-1]
        
        c1 = self.colors[idx]
        c2 = self.colors[idx + 1]
        
        r = int(c1[0] + (c2[0] - c1[0]) * frac)
        g = int(c1[1] + (c2[1] - c1[1]) * frac)
        b = int(c1[2] + (c2[2] - c1[2]) * frac)
        
        return (r, g, b)
    
    def render(self, task) -> Text:
        completed = task.completed
        total = task.total or 1
        progress = min(1.0, completed / total)
        
        filled_width = int(self.bar_width * progress)
        empty_width = self.bar_width - filled_width
        
        text = Text()
        
        # Filled portion with gradient (thick blocks)
        for i in range(filled_width):
            pos = i / max(1, self.bar_width - 1)
            r, g, b = self._interpolate_color(progress, pos)
            text.append("█", style=Style(color=f"rgb({r},{g},{b})"))
        
        # Empty portion (darker blocks)
        text.append("░" * empty_width, style="dim")
        
        return text

# ASCII Art Banner
BANNER = """[bold cyan]
███████╗███╗   ██╗████████╗██████╗  ██████╗ ██████╗ ██╗   ██╗
██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗╚██╗ ██╔╝
█████╗  ██╔██╗ ██║   ██║   ██████╔╝██║   ██║██████╔╝ ╚████╔╝ 
██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██║   ██║██╔═══╝   ╚██╔╝  
███████╗██║ ╚████║   ██║   ██║  ██║╚██████╔╝██║        ██║   
╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝   
[/bold cyan][dim]               ═══ ENGINE V2 ═══[/dim]
"""

class TrainingState:
    """Persistent training state for auto-resume."""
    
    def __init__(self, save_dir: str):
        self.save_path = os.path.join(save_dir, "training_state.json")
        self.data = {
            "run_name": None,
            "last_timesteps": 0,
            "best_reward": float('-inf'),
            "config": {},
        }
        
    def save(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
            
    def load(self) -> bool:
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.data = json.load(f)
            return True
        return False


class RichLogger:
    """
    Lightweight training progress bar.
    Updates every 2 seconds to minimize CPU usage.
    """
    
    def __init__(self, total_timesteps: int, start_offset: int = 0,
                 run_name: str = "training", wandb_url: str = None,
                 save_dir: str = None, config: Dict = None):
        self.console = Console()
        self.total_timesteps = total_timesteps
        self.start_offset = start_offset
        self.start_time = time.time()
        self.run_name = run_name
        self.wandb_url = wandb_url
        
        # State persistence
        self.save_dir = save_dir or os.getcwd()
        self.state = TrainingState(self.save_dir)
        self.config = config or {}
        self.state.load()
        self.state.data["run_name"] = run_name
        self.state.data["config"] = self.config
        
        # Stats
        self.current_fps = 0
        self.current_reward = 0.0
        self.current_loss = 0.0
        self.current_explained_var = 0.0
        self.current_steps = 0
        self.best_reward = self.state.data.get("best_reward", float('-inf'))
        
        # Update throttling
        self.last_update = 0
        self.update_interval = 2.0  # seconds
        
        # Fancy gradient progress bar
        self.progress = Progress(
            SpinnerColumn(spinner_name="moon", style="bold bright_cyan"),
            TextColumn("[bold cyan]⟨[/][bold white]{task.description}[/][bold cyan]⟩[/]"),
            GradientBarColumn(bar_width=50),
            TextColumn("[bold green]{task.percentage:>5.1f}%[/]"),
            TextColumn("[dim]│[/]"),
            TextColumn("[cyan]FPS:[/] [bold white]{task.fields[fps]}[/]"),
            console=self.console,
            transient=False,
            refresh_per_second=2,
            expand=False,
        )
        self.task = None
        
    def start(self):
        # Print ASCII banner
        self.console.print(BANNER)
        self.console.print(f"[dim]Run: {self.run_name}[/]")
        if self.wandb_url:
            self.console.print(f"[dim]WandB: {self.wandb_url}[/]")
        self.console.print()
        
        self.progress.start()
        self.task = self.progress.add_task(
            description=self.run_name,
            total=self.total_timesteps,
            fps="0"
        )
        
    def update(self, timesteps: int, fps: int, mean_reward: float,
               loss: float = 0.0, explained_var: float = 0.0):
        # Throttle updates
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        self.last_update = now
        
        actual_progress = timesteps - self.start_offset
        
        self.current_fps = fps
        self.current_reward = mean_reward
        self.current_loss = loss
        self.current_explained_var = explained_var
        self.current_steps = timesteps
        
        # Track best reward
        if mean_reward > self.best_reward and mean_reward != 0:
            self.best_reward = mean_reward
            self.state.data["best_reward"] = mean_reward
        
        if self.task is not None:
            self.progress.update(
                self.task,
                completed=min(actual_progress, self.total_timesteps),
                fps=str(fps)
            )
        
        # Save state periodically
        if timesteps % 50000 == 0:
            self.state.data["last_timesteps"] = timesteps
            self.state.save()
        
    def finish(self):
        if self.progress:
            self.progress.stop()
        
        self.state.data["last_timesteps"] = self.current_steps
        self.state.save()
        
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        
        self.console.print()
        self.console.print(f"[bold green]✓ Training Complete[/]")
        self.console.print(f"  Duration: {mins}m {secs}s | Steps: {self.current_steps:,} | Best Reward: {self.best_reward:.2f}")
        if self.wandb_url:
            self.console.print(f"  [blue]{self.wandb_url}[/]")

    def log_message(self, message: str, level: str = "info"):
        colors = {"info": "blue", "success": "green", "warning": "yellow", "error": "red"}
        self.console.print(f"[{colors.get(level, 'white')}]▶[/] {message}")
