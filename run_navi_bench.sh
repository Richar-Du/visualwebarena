#!/bin/bash

# ==============================================================================
# Navi-Bench Evaluation Runner Script (Shell Version)
# ==============================================================================
# Usage: ./run_navi_bench.sh [options]
#
# Options:
#   -c, --config_file FILE    Path to config file (default: config_navi_bench.json)
#   -d, --domains LIST        Comma-separated list of domains
#       --difficulties LIST   Comma-separated list of difficulties
#   -t, --task_ids LIST       Comma-separated list of task IDs
#       --max_tasks INT       Maximum number of tasks
#       --max_steps INT       Max steps per task
#   -o, --result_dir DIR      Override result directory
#       --no_trace            Disable saving traces
#       --monitor BOOL        Enable/Disable monitor (true/false)
#       --dry_run             Dry run mode
# ==============================================================================

set -euo pipefail

# ===========================
# Globals
# ===========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
CONFIG_FILE="config_navi_bench.json"
DOMAINS=""
DIFFICULTIES=""
TASK_IDS=""
MAX_TASKS=""
MAX_STEPS=""
RESULT_DIR=""
NO_TRACE=""
MONITOR=""
DRY_RUN=""

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
GRAY='\033[1;30m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[*] $1${NC}"; }
log_success() { echo -e "${GREEN}[+] $1${NC}"; }
log_warn()    { echo -e "${YELLOW}[!] $1${NC}"; }
log_error()   { echo -e "${RED}[!] $1${NC}"; }
log_detail()  { echo -e "${GRAY}    $1${NC}"; }

TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check_pass() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    log_success "$1"
}

check_fail() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    log_error "$1"
    if [ -n "${2:-}" ]; then
        log_detail "Details: $2"
    fi
}

# ===========================
# Header
# ===========================
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Navi-Bench Evaluation Runner (Shell)${NC}"
echo -e "${CYAN}  Benchmark: yutori-ai/navi-bench${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ===========================
# Argument Parsing (before checks, so --help works early)
# ===========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config_file)  CONFIG_FILE="$2";  shift 2 ;;
        -d|--domains)      DOMAINS="$2";      shift 2 ;;
        --difficulties)    DIFFICULTIES="$2"; shift 2 ;;
        -t|--task_ids)     TASK_IDS="$2";     shift 2 ;;
        --max_tasks)       MAX_TASKS="$2";    shift 2 ;;
        --max_steps)       MAX_STEPS="$2";    shift 2 ;;
        -o|--result_dir)   RESULT_DIR="$2";   shift 2 ;;
        --no_trace)        NO_TRACE="true";   shift   ;;
        --monitor)         MONITOR="$2";      shift 2 ;;
        --dry_run)         DRY_RUN="true";    shift   ;;
        -h|--help)
            head -19 "$0" | tail -14
            exit 0 ;;
        *)
            log_error "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ==============================================================================
# Phase 1: Static Checks — Environment & Dependencies
# ==============================================================================
echo -e "${BOLD}Phase 1: Environment & Dependency Checks${NC}"
echo "----------------------------------------"

# --- 1.1 Python ---
log_info "Checking Python interpreter..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    check_fail "Python interpreter not found" "Install Python 3.10+ and ensure it is on PATH"
else
    PY_VERSION_FULL=$($PYTHON_CMD --version 2>&1)
    PY_VERSION_NUM=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1)

    # Check minimum version (3.10 for match/case syntax)
    PY_OK=$($PYTHON_CMD -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>&1)
    if [ "$PY_OK" == "1" ]; then
        check_pass "Python: $PY_VERSION_FULL"
    else
        check_fail "Python version too old: $PY_VERSION_FULL" "Requires Python >= 3.10 (match/case syntax used in browser_env)"
    fi
fi

# --- 1.2 Required Python packages ---
log_info "Checking required Python packages..."

# List of (import_name, pip_name, description) tuples
REQUIRED_PACKAGES=(
    "torch|torch|PyTorch (deep learning)"
    "openai|openai|OpenAI API client"
    "PIL|Pillow|Image processing"
    "playwright|playwright|Browser automation"
    "datasets|datasets|HuggingFace datasets"
    "loguru|loguru|Structured logging"
    "beartype|beartype|Runtime type checking"
    "pydantic|pydantic|Data validation"
    "numpy|numpy|Numerical computing"
    "gymnasium|gymnasium|Env interface"
)

MISSING_PACKAGES=()
for pkg_entry in "${REQUIRED_PACKAGES[@]}"; do
    IFS='|' read -r import_name pip_name description <<< "$pkg_entry"
    OUTPUT=$($PYTHON_CMD -c "import ${import_name}" 2>&1)
    if [ $? -eq 0 ]; then
        check_pass "Package: ${pip_name} (${description})"
    else
        check_fail "Missing package: ${pip_name} (${description})" "${OUTPUT}"
        MISSING_PACKAGES+=("$pip_name")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    log_warn "Install missing packages with:"
    log_detail "pip install ${MISSING_PACKAGES[*]}"
    echo ""
fi

# --- 1.3 navi-bench local package ---
log_info "Checking navi-bench package..."
NAVI_CHECK=$($PYTHON_CMD -c "
import sys, os
nb_dir = os.path.join(os.getcwd(), 'navi-bench')
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
from navi_bench.base import DatasetItem, instantiate
print('OK')
" 2>&1)

if [ "$NAVI_CHECK" == "OK" ]; then
    check_pass "navi-bench package importable"
else
    check_fail "navi-bench package import failed" "$NAVI_CHECK"
fi

# --- 1.4 browser_env import ---
log_info "Checking browser_env package..."
BROWSER_CHECK=$($PYTHON_CMD -c "
from browser_env import ScriptBrowserEnv, Action, ActionTypes
print('OK')
" 2>&1)

if [ "$BROWSER_CHECK" == "OK" ]; then
    check_pass "browser_env importable"
else
    check_fail "browser_env import failed" "$BROWSER_CHECK"
fi

# --- 1.5 agent framework import ---
log_info "Checking agent framework..."
AGENT_CHECK=$($PYTHON_CMD -c "
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent import PromptAgent
print('OK')
" 2>&1)

if [ "$AGENT_CHECK" == "OK" ]; then
    check_pass "Agent framework importable"
else
    check_fail "Agent framework import failed" "$AGENT_CHECK"
fi

# --- 1.6 run_navi_bench.py syntax ---
log_info "Checking run_navi_bench.py syntax..."
SYNTAX_CHECK=$($PYTHON_CMD -c "
import ast
ast.parse(open('run_navi_bench.py', encoding='utf-8').read())
print('OK')
" 2>&1)

if [ "$SYNTAX_CHECK" == "OK" ]; then
    check_pass "run_navi_bench.py syntax valid"
else
    check_fail "run_navi_bench.py has syntax error" "$SYNTAX_CHECK"
fi

# --- 1.7 Config file ---
log_info "Checking config file..."
if [ -f "$CONFIG_FILE" ]; then
    CONFIG_CHECK=$($PYTHON_CMD -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    c = json.load(f)
required = ['model', 'browser', 'output']
missing = [k for k in required if k not in c]
if missing:
    print(f'MISSING_KEYS:{missing}')
else:
    print('OK')
" 2>&1)
    if [[ "$CONFIG_CHECK" == "OK" ]]; then
        check_pass "Config file valid: $CONFIG_FILE"
    else
        check_fail "Config file has issues" "$CONFIG_CHECK"
    fi
else
    check_fail "Config file not found: $CONFIG_FILE" "Create config_navi_bench.json or specify with --config_file"
fi

# --- 1.8 API Key ---
log_info "Checking API key..."
if [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY_PREVIEW="${OPENAI_API_KEY:0:8}..."
    check_pass "OPENAI_API_KEY set ($KEY_PREVIEW)"
else
    check_fail "OPENAI_API_KEY not set" "export OPENAI_API_KEY='sk-...'"
fi

# --- Check Summary ---
echo ""
echo "----------------------------------------"
echo -e "${BOLD}Pre-flight Check Summary: ${PASSED_CHECKS}/${TOTAL_CHECKS} passed${NC}"
if [ $FAILED_CHECKS -gt 0 ]; then
    log_error "${FAILED_CHECKS} check(s) failed"
    echo ""
    log_warn "Some checks failed. The script may not execute correctly."
    log_warn "Fix the issues above before running, or press Ctrl+C to abort."
    log_warn "Continuing in 5 seconds..."
    sleep 5
else
    log_success "All checks passed!"
fi
echo "----------------------------------------"
echo ""

# ==============================================================================
# Phase 2: Build Command
# ==============================================================================

CMD_ARGS=("run_navi_bench.py" "--config_file" "$CONFIG_FILE")

if [ -n "$DOMAINS" ]; then
    CMD_ARGS+=("--domains" "$DOMAINS")
    log_info "Domains: $DOMAINS"
fi

if [ -n "$DIFFICULTIES" ]; then
    CMD_ARGS+=("--difficulties" "$DIFFICULTIES")
    log_info "Difficulties: $DIFFICULTIES"
fi

if [ -n "$TASK_IDS" ]; then
    CMD_ARGS+=("--task_ids" "$TASK_IDS")
    log_info "Task IDs: $TASK_IDS"
fi

if [ -n "$MAX_TASKS" ]; then
    CMD_ARGS+=("--max_tasks" "$MAX_TASKS")
    log_info "Max tasks: $MAX_TASKS"
fi

if [ -n "$MAX_STEPS" ]; then
    CMD_ARGS+=("--max_steps" "$MAX_STEPS")
    log_info "Max steps: $MAX_STEPS"
fi

if [ -n "$RESULT_DIR" ]; then
    CMD_ARGS+=("--result_dir" "$RESULT_DIR")
    log_info "Result dir: $RESULT_DIR"
fi

if [ -n "$NO_TRACE" ]; then
    CMD_ARGS+=("--no_trace")
    log_info "Traces disabled"
fi

if [ -n "$MONITOR" ]; then
    if [ "$MONITOR" == "true" ]; then
        CMD_ARGS+=("--enable_monitor")
        log_success "Monitor: ENABLED"
    else
        CMD_ARGS+=("--disable_monitor")
        log_warn "Monitor: DISABLED"
    fi
fi

if [ -n "$DRY_RUN" ]; then
    CMD_ARGS+=("--dry_run")
    log_warn "DRY RUN mode"
fi

# ==============================================================================
# Phase 3: Display Config Summary
# ==============================================================================
echo ""
echo -e "${GRAY}----------------------------------------${NC}"
$PYTHON_CMD -c "
import json
try:
    with open('$CONFIG_FILE', 'r') as f:
        c = json.load(f)
    print(f'  Model:')
    print(f'    Provider: {c.get(\"model\", {}).get(\"provider\", \"N/A\")}')
    print(f'    Model:    {c.get(\"model\", {}).get(\"model\", \"N/A\")}')
    print(f'  Browser:')
    print(f'    Headless: {c.get(\"browser\", {}).get(\"headless\", \"N/A\")}')
    print(f'  Monitor:')
    print(f'    Enabled:  {c.get(\"monitor\", {}).get(\"enable_monitor\", \"N/A\")}')
    print(f'  Output:')
    print(f'    Result Dir: {c.get(\"output\", {}).get(\"result_dir\", \"N/A\")}')
except Exception as e:
    print(f'  [Could not parse config: {e}]')
" 2>&1
echo -e "${GRAY}----------------------------------------${NC}"
echo ""

# ==============================================================================
# Phase 4: Execute
# ==============================================================================
log_success "Starting Navi-Bench evaluation..."
echo -e "${GRAY}[*] Command: $PYTHON_CMD ${CMD_ARGS[*]}${NC}"
echo ""
echo -e "${CYAN}========================================${NC}"
echo ""

START_TIME=$(date +%s)

# Execute — use array expansion to preserve quoting
set +e
$PYTHON_CMD "${CMD_ARGS[@]}" 2>&1 | tee /dev/stderr
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
H=$((DURATION / 3600))
M=$(( (DURATION % 3600) / 60 ))
S=$((DURATION % 60))

echo ""
echo -e "${CYAN}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Evaluation completed successfully!"
else
    log_error "Evaluation failed with exit code: $EXIT_CODE"
    log_error "Review the full output above for error details."
fi

log_info "Total time: $(printf "%02d:%02d:%02d" $H $M $S)"

# Show result directory
RESULT_DIR_VAL=$($PYTHON_CMD -c "import json; print(json.load(open('$CONFIG_FILE')).get('output', {}).get('result_dir', 'results/navi_bench'))" 2>/dev/null || echo "")
if [ -n "$RESULT_DIR_VAL" ] && [ -d "$RESULT_DIR_VAL" ]; then
    log_success "Results: $RESULT_DIR_VAL"
    SUMMARY_FILE="$RESULT_DIR_VAL/navi_bench_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        log_info "Summary: $SUMMARY_FILE"
    fi
fi

echo -e "${CYAN}========================================${NC}"
exit $EXIT_CODE
