#!/bin/bash

# Check setup script for PreFree SpMV Auto-tuning System

echo "Checking PreFree SpMV Auto-tuning System setup..."
echo "================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check directory structure
echo ""
echo "Current directory: $SCRIPT_DIR"
echo ""

# Check for required directories
echo "Checking directories..."
if [ -d "$SCRIPT_DIR/scripts" ]; then
    echo "✓ scripts/ directory exists"
else
    echo "✗ scripts/ directory NOT found"
fi

if [ -d "$SCRIPT_DIR/scripts/templates" ]; then
    echo "✓ scripts/templates/ directory exists"
else
    echo "✗ scripts/templates/ directory NOT found"
fi

if [ -d "$SCRIPT_DIR/build" ]; then
    echo "✓ build/ directory exists"
else
    echo "✗ build/ directory NOT found (will be created during build)"
fi

# Check for required files
echo ""
echo "Checking required files..."

FILES_TO_CHECK=(
    "scripts/profile_and_generate.py"
    "scripts/config.yaml"
    "scripts/templates/preFreeSpMV_template.cu.j2"
    "run_autotuning.sh"
    "CMakeLists.txt"
    "src/spmv_kernel/preFreeSpMV.cu"
)

ALL_GOOD=true

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file NOT found"
        ALL_GOOD=false
    fi
done

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
if command -v python3 &> /dev/null; then
    echo "✓ Python 3 is installed"
    
    # Check individual packages
    for package in jinja2 yaml numpy; do
        if python3 -c "import $package" &> /dev/null; then
            echo "✓ Python package '$package' is installed"
        else
            echo "✗ Python package '$package' is NOT installed"
            ALL_GOOD=false
        fi
    done
else
    echo "✗ Python 3 is NOT installed"
    ALL_GOOD=false
fi

# Check CUDA
echo ""
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA compiler (nvcc) found"
    nvcc --version | grep "release" | head -1
else
    echo "✗ CUDA compiler (nvcc) NOT found"
    ALL_GOOD=false
fi

# Summary
echo ""
echo "================================================"
if [ "$ALL_GOOD" = true ]; then
    echo "✓ All checks passed! System is ready for auto-tuning."
    echo ""
    echo "Next steps:"
    echo "1. Edit scripts/config.yaml to set your matrix file paths"
    echo "2. Run ./run_autotuning.sh to start the auto-tuning process"
else
    echo "✗ Some checks failed. Please fix the issues above before running auto-tuning."
    echo ""
    echo "Quick fix suggestions:"
    echo "- Missing directories: mkdir -p scripts/templates"
    echo "- Missing Python packages: pip3 install jinja2 pyyaml numpy"
    echo "- Missing files: Make sure all artifact files are created in the correct locations"
fi
echo "================================================"