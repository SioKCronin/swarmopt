# 📦 Publishing SwarmOpt to PyPI

This guide walks you through publishing SwarmOpt to PyPI so users can install it with `pip install swarmopt`.

## 🎯 Prerequisites

### 1. Create PyPI Account
- **Production PyPI**: https://pypi.org/account/register/
- **Test PyPI** (for testing): https://test.pypi.org/account/register/

### 2. Install Publishing Tools

```bash
pip install --upgrade build twine
```

### 3. Configure PyPI Credentials

Create `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

Get tokens from:
- **PyPI**: https://pypi.org/manage/account/token/
- **TestPyPI**: https://test.pypi.org/manage/account/token/

## 🚀 Publishing Steps

### Step 1: Update Version Number

Edit `setup.py` and bump the version:

```python
version='0.2.0',  # Current version with new features
```

**Versioning Guide:**
- **Major** (1.0.0): Breaking changes
- **Minor** (0.2.0): New features, backward compatible
- **Patch** (0.2.1): Bug fixes

### Step 2: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/ dist/ *.egg-info/
```

### Step 3: Build Distribution

```bash
# Build source distribution and wheel
python -m build
```

This creates:
- `dist/swarmopt-0.2.0.tar.gz` (source distribution)
- `dist/swarmopt-0.2.0-py3-none-any.whl` (wheel)

### Step 4: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ swarmopt
```

### Step 5: Publish to Production PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

### Step 6: Verify Installation

```bash
# Install from PyPI
pip install swarmopt

# Test it works
python -c "from swarmopt import Swarm; print('✅ SwarmOpt installed!')"
```

## 📋 Pre-Publishing Checklist

Before publishing, ensure:

- [ ] Version number updated in `setup.py`
- [ ] All tests pass (`python run_tests.py`)
- [ ] README.md is up to date
- [ ] CHANGELOG.md documents new features
- [ ] All code committed to git
- [ ] Git tag created for version: `git tag v0.2.0`
- [ ] Changes pushed to GitHub
- [ ] Documentation is complete

## 🔄 Quick Publishing Commands

```bash
# Full publishing workflow
cd /Users/siobhan/code/swarmopt

# 1. Clean old builds
rm -rf build/ dist/ *.egg-info/

# 2. Build new distribution
python -m build

# 3. Test on TestPyPI (optional but recommended)
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ swarmopt

# 4. Upload to production PyPI
twine upload dist/*

# 5. Create git tag
git tag v0.2.0
git push origin v0.2.0
```

## 🎯 What's New in v0.2.0

### Major Features
- ✅ **Respect Boundary**: Safety-critical standoff distance optimization
- ✅ **Multiobjective PSO**: NSGA-II and SPEA2 inspired algorithms
- ✅ **Cooperative PSO**: Multiple collaborating swarms
- ✅ **Proactive PSO**: Knowledge gain guided exploration
- ✅ **Diversity Monitoring**: Prevent premature convergence
- ✅ **Variation Operators**: Escape local optima
- ✅ **Inertia Variations**: 8 different strategies
- ✅ **Velocity Clamping**: 11 clamping variations
- ✅ **Airfoil Optimization**: CST, Hicks-Henne, Bézier, NACA-k

### Documentation
- ✅ Comprehensive README
- ✅ Respect boundary guide
- ✅ Cancer TDA project plan
- ✅ Interactive visualizations
- ✅ 10+ test scripts and examples

## 🐛 Troubleshooting

### "Invalid credentials" error
- Regenerate your PyPI token
- Update `~/.pypirc` with new token
- Use `__token__` as username, not your PyPI username

### "Package already exists" error
- You've already published this version
- Bump version number in `setup.py`
- Rebuild and upload again

### "README rendering error" on PyPI
- Ensure README.md is valid markdown
- Check long_description in setup.py
- Test locally: `python -m readme_renderer README.md`

### Import errors after installation
- Check `packages` in setup.py includes all modules
- Ensure `__init__.py` exists in all packages
- Verify `install_requires` lists all dependencies

## 📊 Post-Publishing

After publishing:

1. **Verify on PyPI**: https://pypi.org/project/swarmopt/
2. **Test installation**: `pip install swarmopt`
3. **Update documentation**: Link to PyPI package
4. **Announce**: Twitter, Reddit, LinkedIn
5. **Create GitHub release**: With changelog

## 🔗 Useful Links

- **PyPI Project**: https://pypi.org/project/swarmopt/
- **TestPyPI Project**: https://test.pypi.org/project/swarmopt/
- **PyPI Packaging Guide**: https://packaging.python.org/
- **Twine Documentation**: https://twine.readthedocs.io/

## 📝 Creating a CHANGELOG

Create `CHANGELOG.md`:

```markdown
# Changelog

## [0.2.0] - 2025-01-XX

### Added
- Respect boundary feature for safety-critical applications
- Multiobjective optimization (NSGA-II, SPEA2)
- Cooperative PSO with multiple swarms
- Proactive PSO with knowledge gain
- Diversity monitoring and intervention
- Variation operators for local optima escape
- 8 inertia weight variations
- 11 velocity clamping strategies
- Airfoil optimization suite

### Changed
- Improved repository structure
- Enhanced documentation
- Added comprehensive test suite

### Fixed
- Various bug fixes and performance improvements

## [0.1.0] - 2018-XX-XX
- Initial release
```

## 🎉 Success!

Once published, users can install with:

```bash
pip install swarmopt
```

And use immediately:

```python
from swarmopt import Swarm
import numpy as np

swarm = Swarm(
    n_particles=20,
    dims=2,
    c1=2.0, c2=2.0, w=0.9,
    epochs=50,
    obj_func=lambda x: np.sum(x**2),
    target_position=[0, 0]  # With automatic respect boundary!
)

swarm.optimize()
print(f"Best position: {swarm.best_pos}")
```

---

**Questions?** Open an issue at https://github.com/siokcronin/swarmopt/issues

