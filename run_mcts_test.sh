#!/bin/bash
# Simple test runner for MCTS and Rule Partition tests

echo "Running MCTS and Rule Partition tests..."
echo "========================================"

# Run the test suite
python -m pytest tests/test_mcts_rule_partition.py -v

# If pytest is not available, use unittest
if [ $? -ne 0 ]; then
    echo "Pytest not available, trying unittest..."
    python -m unittest tests.test_mcts_rule_partition -v
fi

echo ""
echo "Test completed!"

