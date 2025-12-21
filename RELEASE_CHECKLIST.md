# 🚀 PyPI Release Checklist for SwarmOpt

## Pre-Release Steps

### 1. ✅ Version Check
- [x] Current version in `setup.py`: **0.2.0**
- [ ] Decide if bumping version or releasing 0.2.0
- [ ] Update version in `setup.py` if needed
- [ ] Update `CHANGELOG.md` with actual release date

### 2. ✅ Code Quality
- [ ] Run tests: `python run_tests.py`
- [ ] Verify all imports work: `python -c "from swarmopt import Swarm"`
- [ ] Check for any uncommitted changes: `git status`
- [ ] Ensure README.md is up to date

### 3. ✅ Documentation
- [x] README.md is comprehensive
- [x] CHANGELOG.md is updated
- [x] PUBLISHING.md has instructions
- [ ] All examples work correctly

### 4. ✅ Build Preparation
- [ ] Clean old builds: `rm -rf build/ dist/ *.egg-info/`
- [ ] Install build tools: `pip install --upgrade build twine`
- [ ] Verify PyPI credentials in `~/.pypirc` (if using)

## Build & Test

### 5. Build Distribution
```bash
cd /Users/carl/code/swarmopt
python -m build
```

This creates:
- `dist/swarmopt-0.2.0.tar.gz` (source distribution)
- `dist/swarmopt-0.2.0-py3-none-any.whl` (wheel)

### 6. Test Installation Locally
```bash
# Install from local wheel
pip install --force-reinstall dist/swarmopt-0.2.0-py3-none-any.whl

# Test import
python -c "from swarmopt import Swarm; print('✅ Success!')"
```

### 7. Test on TestPyPI (Recommended)
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ swarmopt

# Verify it works
python -c "from swarmopt import Swarm; print('✅ TestPyPI install works!')"
```

## Publish to PyPI

### 8. Upload to Production PyPI
```bash
twine upload dist/*
```

**Note**: You'll need PyPI credentials. Options:
- Use `~/.pypirc` file with token
- Enter credentials when prompted
- Use environment variables

### 9. Verify on PyPI
- Visit: https://pypi.org/project/swarmopt/
- Check package page renders correctly
- Verify README displays properly

### 10. Test Installation from PyPI
```bash
# Uninstall local version first
pip uninstall swarmopt -y

# Install from PyPI
pip install swarmopt

# Test
python -c "from swarmopt import Swarm; print('✅ PyPI install works!')"
```

## Post-Release

### 11. Git Tagging
```bash
git tag v0.2.0
git push origin v0.2.0
```

### 12. GitHub Release
- Go to: https://github.com/SioKCronin/swarmopt/releases/new
- Tag: `v0.2.0`
- Title: `SwarmOpt v0.2.0`
- Description: Copy from CHANGELOG.md
- Mark as latest release

### 13. Announcement (Optional)
- Update README if needed
- Share on social media
- Update any documentation sites

## Quick Command Reference

```bash
# Full workflow
cd /Users/carl/code/swarmopt

# Clean
rm -rf build/ dist/ *.egg-info/

# Build
python -m build

# Test locally
pip install --force-reinstall dist/swarmopt-0.2.0-py3-none-any.whl

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Tag release
git tag v0.2.0
git push origin v0.2.0
```

## Troubleshooting

### "Package already exists" error
- Version 0.2.0 already published? Bump version in setup.py
- Check: https://pypi.org/project/swarmopt/

### "Invalid credentials" error
- Regenerate PyPI token: https://pypi.org/manage/account/token/
- Update `~/.pypirc` with new token
- Use `__token__` as username

### Import errors after installation
- Check `packages` in setup.py includes all modules
- Verify `__init__.py` exists in all packages
- Check `install_requires` lists all dependencies

## Current Status

- ✅ Version: 0.2.0
- ✅ setup.py configured
- ✅ README.md ready
- ✅ CHANGELOG.md updated
- ✅ Distribution files exist in dist/
- ⏳ Ready to publish!

