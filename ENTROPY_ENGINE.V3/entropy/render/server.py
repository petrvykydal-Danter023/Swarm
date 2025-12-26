"""
Entropy Engine V3 - Rendering Server
Broadcasts simulation state to visualization clients via ZeroMQ.
"""
import zmq
import pickle
import numpy as np
from .schema import RenderFrame

class RenderServer:
    """
    Runs within the simulation process.
    Publishes world state to any connected listeners (vispy, dashboard).
    """
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        try:
            self.socket.bind(f"tcp://*:{port}")
            print(f"✅ Render Server broadcasting on tcp://*:{port}")
        except zmq.ZMQError as e:
            print(f"❌ Failed to bind Render Server to port {port}: {e}")
            raise
        
    def publish_frame(self, frame: RenderFrame):
        """Serialize and broadcast a frame."""
        try:
            data = pickle.dumps(frame)
            self.socket.send(data)
        except Exception as e:
            print(f"Error publishing frame: {e}")

    def close(self):
        self.socket.close()
        self.context.term()
