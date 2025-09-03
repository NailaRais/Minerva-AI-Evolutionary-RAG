#!/bin/bash
set -e

echo "🚀 Running Minerva Test Suite"
echo "=============================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to run tests with timing
run_test() {
    local test_name=$1
    local command=$2
    echo -e "${YELLOW}Running $test_name...${NC}"
    start_time=$(date +%s)
    
    if $command; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${GREEN}✓ $test_name passed in ${duration}s${NC}"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo -e "${RED}✗ $test_name failed after ${duration}s${NC}"
        return 1
    fi
}

# Run different test types
run_test "unit tests" "pytest tests/unit/ -v --tb=short"
run_test "integration tests" "pytest tests/integration/ -v --tb=short"
run_test "storage check" "python scripts/utils/check_storage.py --max-size 3.0"

# Run accuracy test if dataset exists
if [ -f "benchmark_data/sample.json" ]; then
    run_test "accuracy test" "python scripts/test/test_accuracy.py --quick"
fi

echo -e "\n${GREEN}All tests completed!${NC}"
