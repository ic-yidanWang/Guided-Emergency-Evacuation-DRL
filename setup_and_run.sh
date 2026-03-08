#!/bin/bash
# ============================================================
# Guided Emergency Evacuation DRL - 一键安装 + 运行全部
# 用法：bash setup_and_run.sh
# 会依次完成：安装环境 → 跑仿真生成GIF → 跑RL训练
# ============================================================

set -e
PYTHON=${PYTHON:-python3}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="train_${TIMESTAMP}.log"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${BLUE}======== $* ========${NC}"; }

cd "$(dirname "$0")"
info "Working directory: $(pwd)"

# ── 1. Python ────────────────────────────────────────────────
section "Step 1: Check Python"
if ! command -v "$PYTHON" &>/dev/null; then
    command -v python &>/dev/null && PYTHON=python || error "Python not found."
fi
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python: $PYTHON $PY_VER"

# ── 2. Virtual environment ───────────────────────────────────
section "Step 2: Virtual Environment"
if [ ! -d ".venv" ]; then
    info "Creating .venv ..."
    "$PYTHON" -m venv .venv
fi
[ -f ".venv/bin/activate" ] && source .venv/bin/activate \
    || { [ -f ".venv/Scripts/activate" ] && source .venv/Scripts/activate; } \
    || error "Cannot activate virtual environment"
info "Virtual environment activated"

# ── 3. Install dependencies ──────────────────────────────────
section "Step 3: Install Dependencies"
pip install --upgrade pip -q

if ! python -c "import torch" &>/dev/null 2>&1; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        info "CUDA $CUDA_VER detected"
        if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        elif [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        fi
        pip install torch --index-url "$TORCH_INDEX" -q
    else
        warn "No GPU found, installing CPU-only PyTorch"
        pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    fi
    pip install "numpy>=1.21.0" "matplotlib>=3.5.0" -q
    info "Dependencies installed"
else
    info "Dependencies already installed, skipping"
fi

# ── 4. Verify ────────────────────────────────────────────────
section "Step 4: Verify Environment"
python -c "
import torch, numpy, matplotlib
print(f'  PyTorch  : {torch.__version__}')
print(f'  NumPy    : {numpy.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
print(f'  GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')
"

# ── 5. Create output dirs ────────────────────────────────────
mkdir -p output/guided output/guided/conformal output/logs

# ── 6. Run simulation → GIF ──────────────────────────────────
section "Step 5: Run Simulation (GIF)"
info "Running run_guided_visualize.py ..."
python run_guided_visualize.py --config config/simulation_config.json
info "GIF saved to output/guided/guided_animation.gif"

# ── 7. Train guide agent ─────────────────────────────────────
section "Step 6: Train Guide Agent (RL)"
info "Running train_guide.py --no-viz ... (log: output/logs/$LOG_FILE)"
python train_guide.py --config config/simulation_config.json --no-viz \
    2>&1 | tee "output/logs/$LOG_FILE"
info "Training complete. Model saved to output/guided/guide_agent.pt"

# ── 8. Done ──────────────────────────────────────────────────
section "All Done"
echo ""
echo "  Simulation GIF : output/guided/guided_animation.gif"
echo "  Trained model  : output/guided/guide_agent.pt"
echo "  Checkpoints    : output/guided/guide_agent_ep*.pt"
echo "  Conformal plots: output/guided/conformal/"
echo "  Training log   : output/logs/$LOG_FILE"
echo ""
info "Finished successfully."
