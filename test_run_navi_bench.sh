#!/bin/bash

# ==============================================================================
# Test Script for run_navi_bench.sh
# ==============================================================================
# Runs a comprehensive set of checks and then executes 5 sample tasks to verify
# the reliability of the Navi-Bench evaluation pipeline.
#
# Usage: ./test_run_navi_bench.sh
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET_SCRIPT="./run_navi_bench.sh"
MAIN_PY="run_navi_bench.py"
CONFIG_FILE="config_navi_bench.json"
TEST_RESULT_DIR="results/test_reliability_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${TEST_RESULT_DIR}/test_output.log"

# ── Accessible domains only ──────────────────────────────────────────
# Excluded domains (not reachable from test environment):
#   - apartments   : HTTP 403 (blocked by Cloudflare)
#   - opentable    : TIMEOUT  (geo-restricted / rate-limited)
# Accessible domains:
#   - craigslist   : OK
#   - google_flights : OK
#   - resy         : OK
TEST_DOMAINS="craigslist,google_flights,resy"

# Colors
PASS='\033[0;32m'
FAIL='\033[0;31m'
INFO='\033[0;36m'
WARN='\033[0;33m'
GRAY='\033[1;30m'
BOLD='\033[1m'
NC='\033[0m'

print_pass() { echo -e "${PASS}  [PASS] $1${NC}"; }
print_fail() { echo -e "${FAIL}  [FAIL] $1${NC}"; }
print_info() { echo -e "${INFO}  [INFO] $1${NC}"; }
print_warn() { echo -e "${WARN}  [WARN] $1${NC}"; }
print_detail() { echo -e "${GRAY}         $1${NC}"; }

TOTAL=0
PASSED=0
FAILED=0
WARNINGS=0

assert_pass() {
    TOTAL=$((TOTAL + 1))
    PASSED=$((PASSED + 1))
    print_pass "$1"
}

assert_fail() {
    TOTAL=$((TOTAL + 1))
    FAILED=$((FAILED + 1))
    print_fail "$1"
    if [ -n "${2:-}" ]; then
        # Show up to 5 lines of detail
        echo "$2" | head -5 | while IFS= read -r line; do
            print_detail "$line"
        done
    fi
}

assert_warn() {
    WARNINGS=$((WARNINGS + 1))
    print_warn "$1"
    if [ -n "${2:-}" ]; then
        print_detail "$2"
    fi
}

# ===========================
# Header
# ===========================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Navi-Bench Pipeline Reliability Test       ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ===========================
# Pre-setup: VisualWebArena env vars (needed for browser_env import)
# ===========================
export DATASET="${DATASET:-visualwebarena}"
export REDDIT="${REDDIT:-http://placeholder.reddit.example}"
export SHOPPING="${SHOPPING:-http://placeholder.shopping.example}"
export WIKIPEDIA="${WIKIPEDIA:-http://placeholder.wiki.example}"
export HOMEPAGE="${HOMEPAGE:-http://placeholder.homepage.example}"
export CLASSIFIEDS="${CLASSIFIEDS:-http://placeholder.classifieds.example}"
export CLASSIFIEDS_RESET_TOKEN="${CLASSIFIEDS_RESET_TOKEN:-placeholder_token}"

# ==============================================================================
# Section 1: Static File Checks
# ==============================================================================
echo -e "${BOLD}── Section 1: Static File Checks ──${NC}"
echo ""

# 1.1 Scripts exist
if [ -f "$TARGET_SCRIPT" ]; then
    assert_pass "run_navi_bench.sh exists"
else
    assert_fail "run_navi_bench.sh not found"
    echo -e "${FAIL}Cannot proceed without the target script.${NC}"
    exit 1
fi

if [ -f "$MAIN_PY" ]; then
    assert_pass "run_navi_bench.py exists"
else
    assert_fail "run_navi_bench.py not found"
fi

# 1.2 Config file exists and is valid JSON
if [ -f "$CONFIG_FILE" ]; then
    assert_pass "Config file exists: $CONFIG_FILE"
else
    assert_fail "Config file not found: $CONFIG_FILE"
fi

# 1.3 navi-bench directory
if [ -d "navi-bench" ]; then
    assert_pass "navi-bench/ directory exists"
else
    assert_fail "navi-bench/ directory not found"
fi

if [ -f "navi-bench/navi_bench/base.py" ]; then
    assert_pass "navi-bench core module exists (base.py)"
else
    assert_fail "navi-bench core module missing (base.py)" "Run: git clone https://github.com/yutori-ai/navi-bench.git"
fi

# Check if navi-bench is pip-installed (required for proper imports)
_PY=$(command -v python3 || command -v python || echo "")
if [ -n "$_PY" ]; then
    NB_INSTALLED=$($_PY -c "import navi_bench" 2>&1 && echo "OK" || echo "NOT_INSTALLED")
    if [[ "$NB_INSTALLED" == *"OK"* ]]; then
        assert_pass "navi-bench pip-installed"
    else
        assert_warn "navi-bench NOT pip-installed" "Run: pip install -e ./navi-bench"
    fi
fi

echo ""

# ==============================================================================
# Section 2: Python Environment Checks
# ==============================================================================
echo -e "${BOLD}── Section 2: Python Environment ──${NC}"
echo ""

# Find Python
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    assert_fail "No Python interpreter found on PATH"
    echo -e "${FAIL}Cannot proceed without Python.${NC}"
    exit 1
else
    PY_VERSION=$($PYTHON_CMD --version 2>&1)
    assert_pass "Python found: $PY_VERSION"
fi

# 2.1 Version check (>= 3.10)
PY_VERSION_OK=$($PYTHON_CMD -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>&1)
if [ "$PY_VERSION_OK" == "1" ]; then
    assert_pass "Python version >= 3.10 (required for match/case)"
else
    assert_fail "Python version < 3.10" "browser_env/actions.py uses match/case syntax which requires Python 3.10+"
fi

# 2.2 Syntax check on main script
SYNTAX_OUTPUT=$($PYTHON_CMD -c "
import ast, sys
try:
    ast.parse(open('$MAIN_PY', encoding='utf-8').read())
    print('OK')
except SyntaxError as e:
    print(f'SyntaxError at line {e.lineno}: {e.msg}')
    sys.exit(1)
" 2>&1)
SYNTAX_EXIT=$?

if [ $SYNTAX_EXIT -eq 0 ]; then
    assert_pass "$MAIN_PY syntax valid"
else
    assert_fail "$MAIN_PY syntax error" "$SYNTAX_OUTPUT"
fi

# 2.3 Individual package imports
echo ""
print_info "Checking Python package imports..."
echo ""

REQUIRED_PACKAGES=(
    "torch|torch|PyTorch (deep learning framework)"
    "openai|openai|OpenAI API client"
    "PIL|Pillow|Image processing"
    "playwright.sync_api|playwright|Browser automation"
    "datasets|datasets|HuggingFace datasets loader"
    "loguru|loguru|Structured logging (navi-bench)"
    "beartype|beartype|Runtime type checking"
    "pydantic|pydantic|Data validation"
    "numpy|numpy|Numerical computing"
    "gymnasium|gymnasium|Gym environment interface"
)

MISSING_PACKAGES=()
for pkg_entry in "${REQUIRED_PACKAGES[@]}"; do
    IFS='|' read -r import_name pip_name description <<< "$pkg_entry"

    IMPORT_OUTPUT=$($PYTHON_CMD -c "import ${import_name}" 2>&1)
    IMPORT_EXIT=$?

    if [ $IMPORT_EXIT -eq 0 ]; then
        # Also get version if possible
        PKG_VER=$($PYTHON_CMD -c "
try:
    import ${import_name}
    ver = getattr(${import_name}, '__version__', '?')
    print(ver)
except:
    print('?')
" 2>&1)
        assert_pass "${pip_name} ${PKG_VER} — ${description}"
    else
        # Extract last meaningful line from error
        ERR_LINE=$(echo "$IMPORT_OUTPUT" | grep -i "error\|No module" | tail -1)
        assert_fail "Missing: ${pip_name} — ${description}" "${ERR_LINE:-$IMPORT_OUTPUT}"
        MISSING_PACKAGES+=("$pip_name")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    print_warn "Install missing packages:"
    print_detail "pip install ${MISSING_PACKAGES[*]}"
fi

# 2.4 Project module imports
echo ""
print_info "Checking project module imports..."
echo ""

# browser_env
BE_OUTPUT=$($PYTHON_CMD -c "from browser_env import ScriptBrowserEnv, Action, ActionTypes; print('OK')" 2>&1)
if [[ "$BE_OUTPUT" == *"OK"* ]]; then
    assert_pass "browser_env importable (ScriptBrowserEnv, Action, ActionTypes)"
else
    ERR=$(echo "$BE_OUTPUT" | grep -i "error" | tail -1)
    assert_fail "browser_env import failed" "${ERR:-$BE_OUTPUT}"
fi

# agent framework
AG_OUTPUT=$($PYTHON_CMD -c "
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent import PromptAgent
print('OK')
" 2>&1)
if [[ "$AG_OUTPUT" == *"OK"* ]]; then
    assert_pass "Agent framework importable (MultiAgentCoordinator, PromptAgent)"
else
    ERR=$(echo "$AG_OUTPUT" | grep -i "error" | tail -1)
    assert_fail "Agent framework import failed" "${ERR:-$AG_OUTPUT}"
fi

# navi-bench
NB_OUTPUT=$($PYTHON_CMD -c "
import sys, os
nb_dir = os.path.join(os.getcwd(), 'navi-bench')
if nb_dir not in sys.path:
    sys.path.insert(0, nb_dir)
from navi_bench.base import DatasetItem, instantiate
print('OK')
" 2>&1)
if [[ "$NB_OUTPUT" == *"OK"* ]]; then
    assert_pass "navi-bench importable (DatasetItem, instantiate)"
else
    ERR=$(echo "$NB_OUTPUT" | grep -i "error" | tail -1)
    assert_fail "navi-bench import failed" "${ERR:-$NB_OUTPUT}"
fi

echo ""

# ==============================================================================
# Section 3: Config Validation
# ==============================================================================
echo -e "${BOLD}── Section 3: Config File Validation ──${NC}"
echo ""

if [ -f "$CONFIG_FILE" ]; then
    CFG_OUTPUT=$($PYTHON_CMD -c "
import json, sys

with open('$CONFIG_FILE', 'r') as f:
    c = json.load(f)

errors = []
warnings = []

# Required top-level keys
for key in ['model', 'browser', 'output']:
    if key not in c:
        errors.append(f'Missing required section: {key}')

# Model config
m = c.get('model', {})
if not m.get('provider'):
    errors.append('model.provider not set')
if not m.get('model'):
    errors.append('model.model not set')

# Browser config
b = c.get('browser', {})
if 'headless' not in b:
    warnings.append('browser.headless not set (defaults to False)')

# Output config
o = c.get('output', {})
if not o.get('result_dir'):
    warnings.append('output.result_dir not set')

# Monitor config
mon = c.get('monitor', {})
if 'enable_monitor' not in mon:
    warnings.append('monitor.enable_monitor not set')

# Report
for e in errors:
    print(f'ERROR:{e}')
for w in warnings:
    print(f'WARN:{w}')
if not errors:
    print('OK')
" 2>&1)

    # Parse output
    while IFS= read -r line; do
        if [[ "$line" == ERROR:* ]]; then
            assert_fail "Config: ${line#ERROR:}"
        elif [[ "$line" == WARN:* ]]; then
            assert_warn "Config: ${line#WARN:}"
        elif [[ "$line" == "OK" ]]; then
            assert_pass "Config file structure valid"
        fi
    done <<< "$CFG_OUTPUT"
else
    assert_fail "Cannot validate config: file not found"
fi

# API Key
if [ -n "${OPENAI_API_KEY:-}" ]; then
    KEY_PREVIEW="${OPENAI_API_KEY:0:8}..."
    assert_pass "OPENAI_API_KEY is set ($KEY_PREVIEW)"
else
    assert_fail "OPENAI_API_KEY not set" "export OPENAI_API_KEY='sk-...'"
fi

echo ""

# ==============================================================================
# Section 4: Static Check Summary
# ==============================================================================
echo -e "${BOLD}── Static Check Summary ──${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  Total:    ${BOLD}${TOTAL}${NC}"
echo -e "  Passed:   ${PASS}${PASSED}${NC}"
echo -e "  Failed:   ${FAIL}${FAILED}${NC}"
echo -e "  Warnings: ${WARN}${WARNINGS}${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $FAILED -gt 0 ]; then
    print_fail "${FAILED} static check(s) failed."
    echo ""
    print_info "Fix the issues above before running the evaluation."
    print_info "The 5-sample execution test is SKIPPED due to failed checks."
    echo ""
    exit 1
fi

print_pass "All static checks passed. Proceeding to execution test."
echo ""

# ==============================================================================
# Section 5: Dry Run Test
# ==============================================================================
echo -e "${BOLD}── Section 5: Dry Run Test (5 tasks) ──${NC}"
echo ""

print_info "Domains:  $TEST_DOMAINS"
print_info "Running: $TARGET_SCRIPT --domains $TEST_DOMAINS --max_tasks 5 --dry_run --monitor true"
echo ""

# Capture BOTH stdout and stderr
DRY_OUTPUT=$($TARGET_SCRIPT --domains "$TEST_DOMAINS" --max_tasks 5 --dry_run --monitor true 2>&1)
DRY_EXIT=$?

if [ $DRY_EXIT -eq 0 ]; then
    assert_pass "Dry run exited with code 0"
else
    assert_fail "Dry run failed with exit code $DRY_EXIT" "$DRY_OUTPUT"
    echo ""
    echo -e "${BOLD}Full dry-run output:${NC}"
    echo "───────────────────────────────────────────"
    echo "$DRY_OUTPUT"
    echo "───────────────────────────────────────────"
    echo ""
    print_info "Skipping execution test due to dry-run failure."
    exit 1
fi

echo ""

# ==============================================================================
# Section 6: Execution Test — 5 Sample Tasks
# ==============================================================================
echo -e "${BOLD}── Section 6: Execution Test (5 sample tasks) ──${NC}"
echo ""

# Create result directory
mkdir -p "$TEST_RESULT_DIR"

print_info "Domains:    $TEST_DOMAINS"
print_info "Max tasks:  5"
print_info "Monitor:    enabled"
print_info "Output dir: $TEST_RESULT_DIR"
print_info "Log file:   $LOG_FILE"
echo ""

# Run with full output, also tee to log file
print_info "Executing..."
echo "═══════════════════════════════════════════"

set +e
$TARGET_SCRIPT \
    --domains "$TEST_DOMAINS" \
    --max_tasks 5 \
    --result_dir "$TEST_RESULT_DIR" \
    --no_trace \
    --monitor true 2>&1 | tee "$LOG_FILE"

EXEC_EXIT=${PIPESTATUS[0]}
set -e

echo "═══════════════════════════════════════════"
echo ""

if [ $EXEC_EXIT -eq 0 ]; then
    assert_pass "Execution finished with exit code 0"
else
    assert_fail "Execution finished with exit code $EXEC_EXIT" "Check $LOG_FILE for full output"
fi

# ==============================================================================
# Section 7: Artifact Verification
# ==============================================================================
echo ""
echo -e "${BOLD}── Section 7: Output Artifact Verification ──${NC}"
echo ""

# 7.1 Result directory
if [ -d "$TEST_RESULT_DIR" ]; then
    assert_pass "Result directory created: $TEST_RESULT_DIR"
else
    assert_fail "Result directory NOT created"
fi

# 7.2 Summary file
SUMMARY_FILE="$TEST_RESULT_DIR/navi_bench_summary.json"
if [ -f "$SUMMARY_FILE" ]; then
    assert_pass "Summary report generated: $SUMMARY_FILE"

    # Validate JSON
    JSON_CHECK=$($PYTHON_CMD -c "
import json
with open('$SUMMARY_FILE', 'r') as f:
    data = json.load(f)
n = len(data.get('task_results', []))
s = data.get('overall', {}).get('average_score', 'N/A')
print(f'OK tasks={n} score={s}')
" 2>&1)

    if [[ "$JSON_CHECK" == OK* ]]; then
        assert_pass "Summary JSON valid - $JSON_CHECK"
    else
        assert_fail "Summary JSON malformed" "$JSON_CHECK"
    fi
else
    assert_fail "Summary report NOT generated"
fi

# 7.3 Task subdirectories
DIR_COUNT=$(find "$TEST_RESULT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [ "$DIR_COUNT" -gt 0 ]; then
    assert_pass "Task result directories found: $DIR_COUNT"
    # List them
    find "$TEST_RESULT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -10 | while read -r dirpath; do
        print_detail "-> $(basename "$dirpath")"
    done
else
    assert_fail "No task result directories found in $TEST_RESULT_DIR"
fi

# ==============================================================================
# Final Summary
# ==============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║           Test Results Summary               ║${NC}"
echo -e "${BOLD}╠══════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}║${NC}  Total Checks:  ${BOLD}${TOTAL}${NC}"
echo -e "${BOLD}║${NC}  Passed:        ${PASS}${PASSED}${NC}"
echo -e "${BOLD}║${NC}  Failed:        ${FAIL}${FAILED}${NC}"
echo -e "${BOLD}║${NC}  Warnings:      ${WARN}${WARNINGS}${NC}"
echo -e "${BOLD}║${NC}  Execution:     $([ $EXEC_EXIT -eq 0 ] && echo -e "${PASS}SUCCESS${NC}" || echo -e "${FAIL}FAILED (code $EXEC_EXIT)${NC}")"
echo -e "${BOLD}║${NC}  Log:           $LOG_FILE"
echo -e "${BOLD}╚══════════════════════════════════════════════╝${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $EXEC_EXIT -eq 0 ]; then
    echo -e "${PASS}${BOLD}✓ All tests passed! Pipeline is reliable.${NC}"
else
    echo -e "${FAIL}${BOLD}✗ Some tests failed. Review output above for details.${NC}"
fi

echo ""
exit $EXEC_EXIT
