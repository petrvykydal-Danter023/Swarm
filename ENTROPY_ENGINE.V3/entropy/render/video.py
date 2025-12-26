"""
Entropy Engine V3 - Video Recording
Export simulation recordings to GIF/MP4.
"""
import imageio
import numpy as np
from datetime import datetime
from pathlib import Path

class VideoRecorder:
    """
    Records frames from EntropyViewer canvas and exports to video file.
    
    Usage:
        recorder = VideoRecorder("output.mp4", fps=30)
        # In update loop:
        recorder.add_frame(viewer.canvas)
        # When done:
        recorder.close()
    """
    def __init__(self, output_path: str = None, fps: int = 30, format: str = "mp4"):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recordings/entropy_recording_{timestamp}.{format}"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.output_path = output_path
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self.is_recording = False
        
    def start(self):
        """Begin recording."""
        self.writer = imageio.get_writer(self.output_path, fps=self.fps)
        self.is_recording = True
        print(f"Recording started: {self.output_path}")
        
    def add_frame(self, canvas):
        """Capture current canvas state as a frame."""
        if not self.is_recording or self.writer is None:
            return
            
        # Render canvas to numpy array
        img = canvas.render()
        
        # imageio expects RGB, vispy returns RGBA
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        self.writer.append_data(img)
        self.frame_count += 1
        
    def stop(self):
        """Stop recording and finalize file."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.is_recording = False
            print(f"Recording saved: {self.output_path} ({self.frame_count} frames)")
            
    def close(self):
        """Alias for stop()."""
        self.stop()


class GifRecorder(VideoRecorder):
    """Convenience class for GIF output."""
    def __init__(self, output_path: str = None, fps: int = 15):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recordings/entropy_recording_{timestamp}.gif"
        super().__init__(output_path=output_path, fps=fps, format="gif")
