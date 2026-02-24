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
NC='\033[0m' # No Color

log_info() { echo -e "${CYAN}[*] $1${NC}"; }
log_success() { echo -e "${GREEN}[+] $1${NC}"; }
log_warn() { echo -e "${YELLOW}[!] $1${NC}"; }
log_error() { echo -e "${RED}[!] $1${NC}"; }

# Header
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Navi-Bench Evaluation Runner (Shell)${NC}"
echo -e "${CYAN}  Benchmark: yutori-ai/navi-bench${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check Prerequisites
log_info "Checking prerequisites..."

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    log_error "Python not found"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1)
log_success "Found: $PY_VERSION"

if [ -z "$OPENAI_API_KEY" ]; then
    log_warn "OPENAI_API_KEY not set"
else
    KEY_PREVIEW="${OPENAI_API_KEY:0:8}..."
    log_success "OPENAI_API_KEY set ($KEY_PREVIEW)"
fi
echo ""

# Argument Parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config_file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--domains)
            DOMAINS="$2"
            shift 2
            ;;
        --difficulties)
            DIFFICULTIES="$2"
            shift 2
            ;;
        -t|--task_ids)
            TASK_IDS="$2"
            shift 2
            ;;
        --max_tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -o|--result_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --no_trace)
            NO_TRACE="true"
            shift
            ;;
        --monitor)
            MONITOR="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="true"
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate Config File
if [ ! -f "$CONFIG_FILE" ]; then
    log_warn "Config file not found: $CONFIG_FILE"
    # Check if simpler path works
    if [ -f "./$CONFIG_FILE" ]; then
         CONFIG_FILE="./$CONFIG_FILE"
    else
         log_error "Config file not found!"
         exit 1
    fi
fi
log_success "Config: $CONFIG_FILE"

# Build Python Arguments
CMD_ARGS="--config_file \"$CONFIG_FILE\""

if [ -n "$DOMAINS" ]; then
    CMD_ARGS="$CMD_ARGS --domains $DOMAINS"
    log_info "Domains: $DOMAINS"
fi

if [ -n "$DIFFICULTIES" ]; then
    CMD_ARGS="$CMD_ARGS --difficulties $DIFFICULTIES"
    log_info "Difficulties: $DIFFICULTIES"
fi

if [ -n "$TASK_IDS" ]; then
    CMD_ARGS="$CMD_ARGS --task_ids \"$TASK_IDS\""
    log_info "Task IDs: $TASK_IDS"
fi

if [ -n "$MAX_TASKS" ]; then
    CMD_ARGS="$CMD_ARGS --max_tasks $MAX_TASKS"
    log_info "Max tasks: $MAX_TASKS"
fi

if [ -n "$MAX_STEPS" ]; then
    CMD_ARGS="$CMD_ARGS --max_steps $MAX_STEPS"
    log_info "Max steps: $MAX_STEPS"
fi

if [ -n "$RESULT_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --result_dir \"$RESULT_DIR\""
    log_info "Result dir: $RESULT_DIR"
fi

if [ -n "$NO_TRACE" ]; then
    CMD_ARGS="$CMD_ARGS --no_trace"
    log_info "Traces disabled"
fi

if [ -n "$MONITOR" ]; then
    if [ "$MONITOR" == "true" ]; then
        CMD_ARGS="$CMD_ARGS --enable_monitor"
        log_success "Monitor: ENABLED"
    else
        CMD_ARGS="$CMD_ARGS --disable_monitor"
        log_warn "Monitor: DISABLED"
    fi
fi

if [ -n "$DRY_RUN" ]; then
    CMD_ARGS="$CMD_ARGS --dry_run"
    log_warn "DRY RUN mode"
fi

# ===========================
# Display Config Summary
# ===========================
echo ""
echo -e "\033[1;30m----------------------------------------${NC}"

# Simple python one-liner to parse config for display (avoiding jq dependency)
$PYTHON_CMD -c "
import json
try:
    with open('$CONFIG_FILE', 'r') as f:
        c = json.load(f)
    print(f\"  Model:\\n    Provider: {c.get('model', {}).get('provider')}\\n    Model:    {c.get('model', {}).get('model')}\")
    print(f\"  Browser:\\n    Headless: {c.get('browser', {}).get('headless')}\")
    print(f\"  Output:\\n    Result Dir: {c.get('output', {}).get('result_dir')}\")
except:
    print('  [Could not parse config]')
"
echo -e "\033[1;30m----------------------------------------${NC}"
echo ""

# ===========================
# Execute
# ===========================

log_success "Starting Navi-Bench evaluation..."
echo -e "\033[1;30m[*] Command: $PYTHON_CMD run_navi_bench.py $CMD_ARGS${NC}"
echo ""
echo -e "${CYAN}========================================${NC}"
echo ""

START_TIME=$(date +%s)

# Execute command
eval $PYTHON_CMD run_navi_bench.py $CMD_ARGS
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
H=$((DURATION / 3600))
M=$(( (DURATION % 3600) / 60 ))
S=$((DURATION % 60))

echo ""
echo -e "${CYAN}========================================${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Evaluation completed!"
else
    log_error "Exited with code: $EXIT_CODE"
fi

log_info "Total time: $(printf "%02d:%02d:%02d" $H $M $S)"

# Check results (using extraction logic)
RESULT_DIR_VAL=$($PYTHON_CMD -c "import json; print(json.load(open('$CONFIG_FILE')).get('output', {}).get('result_dir', 'results/navi_bench'))" 2>/dev/null)
if [ -n "$RESULT_DIR_VAL" ] && [ -d "$RESULT_DIR_VAL" ]; then
    log_success "Results: $RESULT_DIR_VAL"
    SUMMARY_FILE="$RESULT_DIR_VAL/navi_bench_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        log_info "Summary: $SUMMARY_FILE"
    fi
fi

echo -e "${CYAN}========================================${NC}"
exit $EXIT_CODE
