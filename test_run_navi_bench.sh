#!/bin/bash

# ==============================================================================
# Test Script for run_navi_bench.sh
# ==============================================================================
# This script specifically tests the reliability of the run_navi_bench.sh script
# by executing a small batch of 5 tasks and verifying the output.

TARGET_SCRIPT="./run_navi_bench.sh"
TEST_RESULT_DIR="results/test_reliability_$(date +%Y%m%d_%H%M%S)"

# Colors for output
PASS='\033[0;32m'
FAIL='\033[0;31m'
INFO='\033[0;36m'
NC='\033[0m'

function print_pass { echo -e "${PASS}[PASS] $1${NC}"; }
function print_fail { echo -e "${FAIL}[FAIL] $1${NC}"; }
function print_info { echo -e "${INFO}[INFO] $1${NC}"; }

# Check if target exists
if [ ! -f "$TARGET_SCRIPT" ]; then
    print_fail "Target script $TARGET_SCRIPT not found!"
    exit 1
fi

print_info "Starting reliability test for $TARGET_SCRIPT"

# ------------------------------------------------------------------------------
# Test Case 1: Argument Parsing and Dry Run
# ------------------------------------------------------------------------------
print_info "Test 1: Dry run configuration check (5 tasks)..."

$TARGET_SCRIPT --max_tasks 5 --dry_run --monitor true > /dev/null
DRY_RUN_EXIT=$?

if [ $DRY_RUN_EXIT -eq 0 ]; then
    print_pass "Dry run completed successfully (arguments valid)."
else
    print_fail "Dry run failed with exit code $DRY_RUN_EXIT"
    exit 1
fi

# ------------------------------------------------------------------------------
# Test Case 2: Execution of 5 Tasks
# ------------------------------------------------------------------------------
print_info "Test 2: Executing 5 sample tasks (using Craigslist for stability)..."
print_info "Output directory: $TEST_RESULT_DIR"

# We use craigslist as it's often more stable for quick tests if the env setup is partial
# We limit to 5 tasks as requested.
# We disable traces to save space/time during test
$TARGET_SCRIPT \
    --domains "craigslist" \
    --max_tasks 5 \
    --result_dir "$TEST_RESULT_DIR" \
    --no_trace \
    --monitor true

EXEC_EXIT=$?

if [ $EXEC_EXIT -eq 0 ]; then
    print_pass "Execution finished with success code."
else
    print_fail "Execution finished with error code $EXEC_EXIT"
    # Proceed to check artifacts anyway to see partial results
fi

# ------------------------------------------------------------------------------
# Test Case 3: Artifact Verification
# ------------------------------------------------------------------------------
print_info "Test 3: Verifying output artifacts..."

if [ -d "$TEST_RESULT_DIR" ]; then
    SUMMARY_FILE="$TEST_RESULT_DIR/navi_bench_summary.json"
    
    if [ -f "$SUMMARY_FILE" ]; then
        print_pass "Summary report generated: $SUMMARY_FILE"
        
        # Simple check if the JSON is valid and contains "task_results"
        if grep -q "task_results" "$SUMMARY_FILE"; then
            print_pass "Summary file valid (contains task_results)."
        else
            print_fail "Summary file appears malformed or empty."
        fi
        
        # Check if we actually attempted 5 tasks (or fewer if filtered)
        # Count subdirectories
        DIR_COUNT=$(find "$TEST_RESULT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        print_info "Found $DIR_COUNT task result directories."
        
        if [ "$DIR_COUNT" -gt 0 ]; then
            print_pass "Task directories created."
        else
            print_fail "No task directories found!"
        fi
        
    else
        print_fail "Summary report NOT found!"
    fi
else
    print_fail "Result directory was NOT created!"
fi

echo ""
print_info "Test sequence completed."
exit $EXEC_EXIT
