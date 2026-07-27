#!/bin/bash
# Release preparation script for SwarmOpt
# This script helps prepare a PyPI release

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Could not find a Python interpreter" >&2
    exit 1
fi

echo "🚀 SwarmOpt Release Preparation"
echo "================================"
echo ""

# Get current version from setup.py
CURRENT_VERSION=$(grep "version=" setup.py | sed "s/.*version='\(.*\)'.*/\1/")
echo "📦 Current version: $CURRENT_VERSION"
echo ""

# Ask for new version
read -p "Enter new version (or press Enter to keep $CURRENT_VERSION): " NEW_VERSION
NEW_VERSION=${NEW_VERSION:-$CURRENT_VERSION}

echo ""
echo "📋 Pre-release Checklist:"
echo "========================="
echo ""

# Check if tests exist and can be run
if [ -f "infra/run_tests.py" ]; then
    echo "✅ Found infra/run_tests.py"
    read -p "Run tests now? (y/n): " RUN_TESTS
    if [ "$RUN_TESTS" = "y" ]; then
        echo "Running tests..."
        "$PYTHON_BIN" infra/run_tests.py
    fi
else
    echo "⚠️  No infra/run_tests.py found"
fi

echo ""
echo "🧹 Cleaning old build artifacts..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
echo "✅ Cleaned"

echo ""
echo "📦 Building distribution packages..."
"$PYTHON_BIN" -m build
echo "✅ Built successfully"

echo ""
echo "📝 Files created in dist/:"
ls -lh dist/

echo ""
echo "🧪 Testing installation locally..."
shopt -s nullglob
WHEEL_FILES=(dist/swarmopt-"${NEW_VERSION}"*.whl)
shopt -u nullglob

if [ ${#WHEEL_FILES[@]} -eq 0 ]; then
    echo "❌ No wheel found for version ${NEW_VERSION}" >&2
    exit 1
fi

if [ ${#WHEEL_FILES[@]} -gt 1 ]; then
    echo "❌ Multiple wheels found for version ${NEW_VERSION}: ${WHEEL_FILES[*]}" >&2
    exit 1
fi

INSTALL_TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$INSTALL_TEST_DIR"' EXIT

"$PYTHON_BIN" -m pip install --force-reinstall "${WHEEL_FILES[0]}" > /dev/null
(
    cd "$INSTALL_TEST_DIR"
    "$PYTHON_BIN" -c "from swarmopt import Swarm; print('✅ Import test passed!')"
)

echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Review the built packages:"
echo "   ls -lh dist/"
echo ""
echo "2. Test on TestPyPI (recommended):"
echo "   twine upload --repository testpypi dist/*"
echo "   pip install --index-url https://test.pypi.org/simple/ swarmopt"
echo ""
echo "3. Publish to PyPI:"
echo "   twine upload dist/*"
echo ""
echo "4. Create git tag:"
echo "   git tag v${NEW_VERSION}"
echo "   git push origin v${NEW_VERSION}"
echo ""
echo "5. Create GitHub release with CHANGELOG notes"
echo ""
echo "✅ Release preparation complete!"
