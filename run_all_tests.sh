#!/bin/bash
# Run all tests for the geo project

echo "Running Geo Test Suite"
echo "====================="

# CA Tests
echo -e "\n[1/3] Running CA tests..."
if [ -d "ca/tests" ]; then
    python -m pytest ca/tests/ -v
else
    echo "No CA tests found"
fi

# Flux Tests  
echo -e "\n[2/3] Running Flux tests..."
if [ -d "flux/tests" ]; then
    python -m pytest flux/tests/ -v
else
    echo "No Flux tests found"
fi

# SPH Tests
echo -e "\n[3/3] Running SPH tests..."
if [ -d "sph/tests" ]; then
    python -m pytest sph/tests/ -v
else
    echo "No SPH tests found"
fi

echo -e "\nTest suite complete!"