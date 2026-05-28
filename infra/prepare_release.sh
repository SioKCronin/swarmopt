#!/bin/bash
# Release preparation script for SwarmOpt
# This script helps prepare a PyPI release

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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
        python infra/run_tests.py || echo "⚠️  Tests failed, but continuing..."
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
python -m build
echo "✅ Built successfully"

echo ""
echo "📝 Files created in dist/:"
ls -lh dist/

echo ""
echo "🧪 Testing installation locally..."
python -m pip install --force-reinstall dist/swarmopt-${NEW_VERSION}-py3-none-any.whl > /dev/null 2>&1 || \
python -m pip install --force-reinstall dist/swarmopt-${NEW_VERSION}*.whl > /dev/null 2>&1 || \
echo "⚠️  Could not test installation automatically"

python -c "from swarmopt import Swarm; print('✅ Import test passed!')" || echo "⚠️  Import test failed"

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
