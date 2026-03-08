#!/bin/bash
# ============================================================
# Guided Emergency Evacuation DRL - Server Runner
# Usage:
#   bash run.sh train        # Train the guide agent (default)
#   bash run.sh visualize    # Run simulation & generate GIF
#   bash run.sh setup        # Only install dependencies
#   bash run.sh train --config config/large_scale.json
# ============================================================

set -e  # exit on error

# ---------- config ----------
MODE=${1:-train}
CONFIG=${2:-config/simulation_config.json}
PYTHON=${PYTHON:-python3}
LOG_DIR="output/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# ----------------------------

# colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 0. move to script directory ──────────────────────────────
cd "$(dirname "$0")"
info "Working directory: $(pwd)"

# ── 1. Python check ──────────────────────────────────────────
if ! command -v "$PYTHON" &>/dev/null; then
    # fallback: try python
    if command -v python &>/dev/null; then
        PYTHON=python
    else
        error "Python not found. Please install Python >= 3.8"
    fi
fi

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using $PYTHON $PY_VER"

# ── 2. GPU detection ─────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    info "GPU detected: $GPU_INFO"
else
    warn "No GPU detected, will run on CPU"
fi

# ── 3. Virtual environment ───────────────────────────────────
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# activate
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    error "Cannot activate virtual environment"
fi
info "Virtual environment activated: $VENV_DIR"

# ── 4. Install dependencies ──────────────────────────────────
install_deps() {
    info "Installing dependencies..."
    pip install --upgrade pip -q

    # detect CUDA version for correct torch wheel
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
        info "Installing PyTorch from: $TORCH_INDEX"
        pip install torch --index-url "$TORCH_INDEX" -q
    else
        warn "nvcc not found, installing CPU-only PyTorch"
        pip install torch --index-url https://download.pytorch.org/whl/cpu -q
    fi

    pip install numpy>=1.21.0 matplotlib>=3.5.0 -q
    info "Dependencies installed."
}

# only reinstall if torch is missing
if ! "$PYTHON" -c "import torch" &>/dev/null 2>&1; then
    install_deps
else
    info "Dependencies already installed, skipping."
fi

# ── 5. Verify config ─────────────────────────────────────────
[ -f "$CONFIG" ] || error "Config file not found: $CONFIG"
info "Using config: $CONFIG"

# ── 6. Create output/log dirs ────────────────────────────────
mkdir -p "$LOG_DIR" output/guided output/guided/conformal

# ── 7. Run ───────────────────────────────────────────────────
case "$MODE" in

  train)
    LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
    info "Starting training... (log: $LOG_FILE)"
    echo "================================================" | tee "$LOG_FILE"
    echo "  Mode   : train"                                  | tee -a "$LOG_FILE"
    echo "  Config : $CONFIG"                                | tee -a "$LOG_FILE"
    echo "  Time   : $(date)"                                | tee -a "$LOG_FILE"
    echo "================================================" | tee -a "$LOG_FILE"
    "$PYTHON" train_guide.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
    info "Training complete. Model saved to output/guided/"
    ;;

  visualize)
    LOG_FILE="$LOG_DIR/vis_${TIMESTAMP}.log"
    info "Starting visualization... (log: $LOG_FILE)"
    "$PYTHON" run_guided_visualize.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"
    info "Visualization complete. Check output/guided/"
    ;;

  setup)
    install_deps
    info "Setup complete. Run: bash run.sh train"
    ;;

  *)
    echo ""
    echo "Usage: bash run.sh [MODE] [--config CONFIG_FILE]"
    echo ""
    echo "  MODE:"
    echo "    train      Train the guide agent (default)"
    echo "    visualize  Run simulation and generate GIF"
    echo "    setup      Install dependencies only"
    echo ""
    echo "  Examples:"
    echo "    bash run.sh train"
    echo "    bash run.sh train config/large_scale.json"
    echo "    bash run.sh visualize config/with_obstacles.json"
    echo ""
    exit 1
    ;;
esac
