#!/bin/bash

# ==============================================================================
# Navi-Bench Environment Setup Script
# ==============================================================================
# This script fixes all prerequisites needed to run run_navi_bench.sh:
#   1. Creates config_navi_bench.json if missing
#   2. Installs navi-bench package (pip install -e .)
#   3. Sets required VisualWebArena environment variables (dummy values)
#   4. Verifies the setup
#
# Usage:
#   source setup_navi_bench.sh        # MUST use 'source' to export env vars
#   # OR
#   . ./setup_navi_bench.sh
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[*] $1${NC}"; }
log_success() { echo -e "${GREEN}[+] $1${NC}"; }
log_warn()    { echo -e "${YELLOW}[!] $1${NC}"; }
log_error()   { echo -e "${RED}[!] $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}=====================================${NC}"
echo -e "${CYAN}  Navi-Bench Environment Setup${NC}"
echo -e "${CYAN}=====================================${NC}"
echo ""

# ==========================================================================
# Fix 1: Create config_navi_bench.json if missing
# ==========================================================================
log_info "Checking config_navi_bench.json..."

if [ ! -f "config_navi_bench.json" ]; then
    log_warn "config_navi_bench.json not found. Creating..."
    cat > config_navi_bench.json << 'CONFIGEOF'
{
    "model": {
        "provider": "openai",
        "model": "qwen3-max",
        "mode": "chat",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 8192
    },
    "browser": {
        "headless": true,
        "slow_mo": 0,
        "viewport_width": 1280,
        "viewport_height": 720,
        "sleep_after_execution": 2.0,
        "save_trace_enabled": true
    },
    "observation": {
        "observation_type": "image_som",
        "action_set_tag": "som",
        "current_viewport_only": true
    },
    "memory": {
        "enable_memory": false,
        "enable_memory_store": false
    },
    "monitor": {
        "enable_monitor": true
    },
    "output": {
        "result_dir": "results/navi_bench",
        "clear_result_dir": false,
        "save_images": false,
        "save_trace": true,
        "verbose": true
    }
}
CONFIGEOF
    log_success "Created config_navi_bench.json"
else
    log_success "config_navi_bench.json already exists"
fi

# ==========================================================================
# Fix 2: Install navi-bench package
# ==========================================================================
log_info "Checking navi-bench installation..."

PYTHON_CMD=""
if command -v python3 &>/dev/null; then PYTHON_CMD="python3"
elif command -v python &>/dev/null; then PYTHON_CMD="python"
else log_error "Python not found!"; return 1 2>/dev/null || exit 1; fi

# Check if navi_bench is importable
NB_CHECK=$($PYTHON_CMD -c "import navi_bench; print('OK')" 2>&1 || true)

if [[ "$NB_CHECK" != *"OK"* ]]; then
    if [ -d "navi-bench" ]; then
        log_warn "navi-bench not installed. Running pip install -e ./navi-bench ..."
        $PYTHON_CMD -m pip install -e ./navi-bench
        log_success "navi-bench installed"
    else
        log_error "navi-bench directory not found! Clone it first:"
        log_error "  git clone https://github.com/yutori-ai/navi-bench.git"
        return 1 2>/dev/null || exit 1
    fi
else
    log_success "navi-bench already installed"
fi

# ==========================================================================
# Fix 3: Set VisualWebArena environment variables
# ==========================================================================
# browser_env/env_config.py asserts these env vars exist at import time.
# For Navi-Bench evaluation (which uses real public websites, not VWA sites),
# we set dummy placeholder values so the assertion passes.
# These values are NOT used during Navi-Bench evaluation.
# ==========================================================================
log_info "Setting required VisualWebArena environment variables..."

export DATASET="${DATASET:-visualwebarena}"

# Only set dummy values if they are not already configured
export REDDIT="${REDDIT:-http://placeholder.reddit.example}"
export SHOPPING="${SHOPPING:-http://placeholder.shopping.example}"
export WIKIPEDIA="${WIKIPEDIA:-http://placeholder.wiki.example}"
export HOMEPAGE="${HOMEPAGE:-http://placeholder.homepage.example}"
export CLASSIFIEDS="${CLASSIFIEDS:-http://placeholder.classifieds.example}"
export CLASSIFIEDS_RESET_TOKEN="${CLASSIFIEDS_RESET_TOKEN:-placeholder_token}"

log_success "Environment variables set:"
echo "    DATASET=$DATASET"
echo "    REDDIT=$REDDIT"
echo "    SHOPPING=$SHOPPING"
echo "    WIKIPEDIA=$WIKIPEDIA"
echo "    HOMEPAGE=$HOMEPAGE"
echo "    CLASSIFIEDS=$CLASSIFIEDS"
echo "    CLASSIFIEDS_RESET_TOKEN=${CLASSIFIEDS_RESET_TOKEN:0:20}..."

# ==========================================================================
# Fix 4: Verify
# ==========================================================================
echo ""
log_info "Verifying setup..."

ERRORS=0

# Check navi_bench import
NB_VERIFY=$($PYTHON_CMD -c "from navi_bench.base import DatasetItem, instantiate; print('OK')" 2>&1 || true)
if [[ "$NB_VERIFY" == *"OK"* ]]; then
    log_success "navi_bench importable"
else
    log_error "navi_bench import failed: $NB_VERIFY"
    ERRORS=$((ERRORS + 1))
fi

# Check browser_env import
BE_VERIFY=$($PYTHON_CMD -c "from browser_env import ScriptBrowserEnv; print('OK')" 2>&1 || true)
if [[ "$BE_VERIFY" == *"OK"* ]]; then
    log_success "browser_env importable"
else
    log_error "browser_env import failed: $BE_VERIFY"
    ERRORS=$((ERRORS + 1))
fi

# Check agent import
AG_VERIFY=$($PYTHON_CMD -c "from agent.multi_agent_coordinator import MultiAgentCoordinator; print('OK')" 2>&1 || true)
if [[ "$AG_VERIFY" == *"OK"* ]]; then
    log_success "Agent framework importable"
else
    log_error "Agent framework import failed: $AG_VERIFY"
    ERRORS=$((ERRORS + 1))
fi

# Check config
if [ -f "config_navi_bench.json" ]; then
    log_success "config_navi_bench.json exists"
else
    log_error "config_navi_bench.json still missing!"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo -e "${CYAN}=====================================${NC}"
if [ $ERRORS -eq 0 ]; then
    log_success "All fixes applied successfully!"
    echo ""
    log_info "You can now run:"
    echo "    ./run_navi_bench.sh --max_tasks 5 --dry_run"
    echo "    ./test_run_navi_bench.sh"
else
    log_error "$ERRORS verification(s) still failing."
    log_error "Check the errors above and fix manually."
fi
echo -e "${CYAN}=====================================${NC}"
echo ""
