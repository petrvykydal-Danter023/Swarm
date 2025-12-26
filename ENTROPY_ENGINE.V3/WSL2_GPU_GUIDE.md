# 🚀 E.E.V3 - WARP SPEED GUIDE (WSL2 GPU SETUP)

> [!WARNING]
> **Windows Native JAX GPU Support is DEAD.**
> The only way to unlock the full power of your RTX 3060 (50,000+ FPS) is via **WSL2 (Windows Subsystem for Linux)**.

## 1. Install WSL2 (If not installed)
Open PowerShell as Administrator and run:
```powershell
wsl --install
```
*Restart your computer required.*

## 2. Setup Ubuntu Environment
After restart, open "Ubuntu" from Start Menu. Setup username/password.

## 3. Install NVIDIA Drivers (On Windows)
Ensure you have the latest **Game Ready Driver** or **Studio Driver** installed on your **Windows** host. WSL2 uses the Windows driver directly.

## 4. Install CUDA Toolkit (Inside WSL2)
Run these commands inside the Ubuntu terminal:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python & PIP
sudo apt install python3-pip python3-venv -y

# Install CUDA Support libraries
sudo apt install nvidia-cuda-toolkit -y
```

## 5. Clone Repository
```bash
git clone https://github.com/petrvykydal-Danter023/Swarm.git
cd Swarm/ENTROPY_ENGINE.V3
```

## 6. Create Virtual Env & Install JAX GPU ⚡
```bash
python3 -m venv venv
source venv/bin/activate

# Install JAX with CUDA 12 support
pip install -U "jax[cuda12]"

# Install other dependencies
pip install flax distrax numpy matplotlib imageio
```

## 7. Verify Warp Speed 🏎️
Run the benchmark inside WSL2:
```bash
python3 benchmark_pure.py
```
**Expected Result:**
- JAX Device: `GpuDevice` (NVIDIA GeForce RTX 3060)
- Agent FPS: **> 10,000,000** (Yes, millions)

---
**Why WSL2?**
- **Linux Kernel**: JAX is optimized for Linux.
- **Direct GPU Access**: WSL2 pipelines GPU calls almost natively.
- **No Compilation Issues**: Windows `int64` vs `int32` issues disappear.
