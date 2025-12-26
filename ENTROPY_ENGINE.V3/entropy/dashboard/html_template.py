HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entropy Engine V3 Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a1a; color: #e0e0e0; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 2px solid #333; padding-bottom: 10px; }
        h1 { margin: 0; color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.3); }
        .card { background: #252525; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .stat-item { background: #333; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-label { font-size: 0.9em; color: #aaa; margin-bottom: 5px; }
        .stat-value { font-size: 1.5em; font-weight: bold; color: #fff; }
        .btn { display: inline-block; padding: 12px 24px; background: #007acc; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; border: none; cursor: pointer; transition: all 0.2s; }
        .btn:hover { background: #005f9e; transform: translateY(-2px); }
        .btn-wandb { background: #ffbe0b; color: #000; }
        .btn-wandb:hover { background: #d9a200; }
        .btn-toggle { background: #444; width: 100%; margin-top: 10px; }
        .btn-toggle.active { background: #ff4757; }
        .status-dot { height: 12px; width: 12px; background: #555; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-dot.online { background: #00ff88; box-shadow: 0 0 8px #00ff88; }
        #log-container { height: 200px; overflow-y: auto; background: #111; padding: 10px; font-family: monospace; border-radius: 8px; color: #ccc; }
        .log-line { margin: 2px 0; border-bottom: 1px solid #222; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1><span id="status-dot" class="status-dot"></span>Entropy Factory</h1>
                <div style="font-size: 0.9em; color: #888; margin-top: 5px;">V3.5 Swarm Intelligence</div>
            </div>
            <div>
                <a id="wandb-link" href="#" target="_blank" class="btn btn-wandb" style="display: none;">View on WandB ↗</a>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-top: 0;">Live Status</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-label">Current Station</div>
                    <div id="station-name" class="stat-value">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Episodes</div>
                    <div id="episodes" class="stat-value">0 / 0</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">FPS</div>
                    <div id="fps" class="stat-value">0.0</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Success Rate</div>
                    <div id="success-rate" class="stat-value">--</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2 style="margin-top: 0;">Controls</h2>
            <p style="color: #aaa; margin-bottom: 15px;">Enabling Live Preview may reduce training speed.</p>
            <button id="btn-render" onclick="toggleRender()" class="btn btn-toggle">Enable Live Preview</button>
        </div>

        <div class="card">
            <h3 style="margin-top: 0;">Recent Logs</h3>
            <div id="log-container"></div>
        </div>
    </div>

    <script>
        let isRenderActive = false;
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status-dot').classList.add('online');
                    document.getElementById('station-name').textContent = data.station || "Idle";
                    document.getElementById('episodes').textContent = `${data.episode_current} / ${data.episode_total}`;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('success-rate').textContent = data.metrics.success_rate ? (data.metrics.success_rate * 100).toFixed(1) + '%' : '--';
                    
                    if (data.wandb_url) {
                        const btn = document.getElementById('wandb-link');
                        btn.style.display = 'inline-block';
                        btn.href = data.wandb_url;
                    }

                    // Update Render Button state
                    isRenderActive = data.render_enabled;
                    const btnRender = document.getElementById('btn-render');
                    if (isRenderActive) {
                        btnRender.textContent = "Disable Live Preview (Running)";
                        btnRender.classList.add('active');
                    } else {
                        btnRender.textContent = "Enable Live Preview";
                        btnRender.classList.remove('active');
                    }
                    
                    // Logs
                    const logContainer = document.getElementById('log-container');
                    logContainer.innerHTML = data.logs.map(l => `<div class="log-line">${l}</div>`).join('');
                })
                .catch(() => {
                    document.getElementById('status-dot').classList.remove('online');
                });
        }

        function toggleRender() {
            fetch('/api/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ render_enabled: !isRenderActive })
            }).then(() => updateStatus());
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""
