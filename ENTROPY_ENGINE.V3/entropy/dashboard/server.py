"""
Lightweight threaded HTTP server for the dashboard.
Zero-dependency (uses standard library).
"""
import threading
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List
from entropy.dashboard.html_template import HTML_TEMPLATE

# Shared state
STATE = {
    "station": "Initializing...",
    "episode_current": 0,
    "episode_total": 0,
    "fps": 0.0,
    "metrics": {},
    "wandb_url": None,
    "render_enabled": False,
    "logs": []
}

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
            
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(STATE).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/control':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            if 'render_enabled' in data:
                STATE['render_enabled'] = bool(data['render_enabled'])
                
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

class DashboardServer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.server = HTTPServer(('0.0.0.0', port), DashboardHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.logger = logging.getLogger("Dashboard")

    def start(self):
        self.logger.info(f"🚀 Dashboard running at http://localhost:{self.port}")
        self.thread.start()

    def update(self, **kwargs):
        """Update shared state."""
        for k, v in kwargs.items():
            if k in STATE:
                STATE[k] = v
    
    def add_log(self, message: str):
        STATE["logs"].insert(0, message)
        if len(STATE["logs"]) > 50:
            STATE["logs"].pop()

    def is_render_enabled(self) -> bool:
        return STATE["render_enabled"]
