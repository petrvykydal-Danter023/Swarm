import os
import subprocess
import sys

# Ensure models/saved exists
if not os.path.exists("models/saved"):
    print("Creating models/saved directory...")
    os.makedirs("models/saved")

print("Running training (test_bridge.py)...")
try:
    subprocess.run([sys.executable, "tests/scenarios/test_bridge.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Training failed with code {e.returncode}")
    sys.exit(1)

print("Running video generation (generate_video.py)...")
try:
    subprocess.run([sys.executable, "generate_video.py"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Video generation failed with code {e.returncode}")
    sys.exit(1)

print("Pipeline completed successfully!")
