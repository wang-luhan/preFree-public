#!/bin/bash

# PreFree SpMV One-Click Auto-tuning Script (Fixed Version)

echo "================================================"
echo "PreFree SpMV Auto-tuning and Code Generation"
echo "================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found!"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking Python dependencies..."
MISSING_PACKAGES=""

if ! python3 -c "import jinja2" &> /dev/null; then
    MISSING_PACKAGES="$MISSING_PACKAGES jinja2"
fi

if ! python3 -c "import yaml" &> /dev/null; then
    MISSING_PACKAGES="$MISSING_PACKAGES pyyaml"
fi

if ! python3 -c "import numpy" &> /dev/null; then
    MISSING_PACKAGES="$MISSING_PACKAGES numpy"
fi

if [ ! -z "$MISSING_PACKAGES" ]; then
    echo "Installing missing Python packages: $MISSING_PACKAGES"
    pip3 install $MISSING_PACKAGES
fi

# Get the project root directory (where this script is located)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if required files exist
echo "Checking required files..."
MISSING_FILES=""

if [ ! -f "$PROJECT_ROOT/scripts/profile_and_generate.py" ]; then
    MISSING_FILES="$MISSING_FILES\n  - scripts/profile_and_generate.py"
fi

if [ ! -f "$PROJECT_ROOT/scripts/config.yaml" ]; then
    MISSING_FILES="$MISSING_FILES\n  - scripts/config.yaml"
fi

if [ ! -f "$PROJECT_ROOT/scripts/templates/preFreeSpMV_template.cu.j2" ]; then
    MISSING_FILES="$MISSING_FILES\n  - scripts/templates/preFreeSpMV_template.cu.j2"
fi

if [ ! -z "$MISSING_FILES" ]; then
    echo ""
    echo "Error: The following required files are missing:"
    echo -e "$MISSING_FILES"
    echo ""
    echo "Please ensure all files are in their correct locations:"
    echo "  $PROJECT_ROOT/scripts/profile_and_generate.py"
    echo "  $PROJECT_ROOT/scripts/config.yaml"
    echo "  $PROJECT_ROOT/scripts/templates/preFreeSpMV_template.cu.j2"
    exit 1
fi

# All files exist, proceed with auto-tuning
echo "All required files found."
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the auto-tuning process
echo "Starting auto-tuning process..."
echo "This may take a while depending on the number of test matrices..."
echo ""

python3 scripts/profile_and_generate.py scripts/config.yaml

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Auto-tuning completed successfully!"
    echo "================================================"
    echo ""
    echo "You can now test the optimized version:"
    echo "  cd $PROJECT_ROOT/build"
    echo "  ./cuda_perftest <path_to_matrix.mtx>"
    echo ""
    echo "The optimized code has been generated at:"
    echo "  $PROJECT_ROOT/src/spmv_kernel/preFreeSpMV_optimized.cu"
    echo ""
    echo "Profiling results saved at:"
    echo "  $PROJECT_ROOT/scripts/profiling_results.json"
else
    echo ""
    echo "Error: Auto-tuning failed!"
    echo "Please check the error messages above."
    exit 1
fi