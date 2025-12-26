"""
Entropy Engine V3 - Rendering Client
Receives simulation state from RenderServer.
"""
import zmq
import pickle
from typing import Optional
from .schema import RenderFrame

class RenderClient:
    """
    Runs within the visualization process.
    Subscribes to world state updates.
    """
    def __init__(self, server_address: str = "tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        print(f"🔌 Connecting to Render Server at {server_address}...")
        self.socket.connect(server_address)
        
        # Subscribe to all topics (empty string)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
    def receive_frame(self, timeout_ms: int = 100) -> Optional[RenderFrame]:
        """
        Polls for a new frame. Returns None if simulation is too slow.
        """
        if self.socket.poll(timeout_ms):
            try:
                data = self.socket.recv()
                return pickle.loads(data)
            except Exception as e:
                print(f"Error receiving/deserializing frame: {e}")
                return None
        return None

    def close(self):
        self.socket.close()
        self.context.term()
