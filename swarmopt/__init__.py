from .swarm import Swarm

__all__ = ['Swarm']

# Optional ETDA integration (if available as submodule)
try:
    import sys
    from pathlib import Path
    # Add etda submodule to path if it exists
    etda_path = Path(__file__).parent.parent / 'etda' / 'etda'
    if etda_path.exists() and str(etda_path) not in sys.path:
        sys.path.insert(0, str(etda_path.parent))
    from etda import ETDAOptimizer
    __all__.append('ETDAOptimizer')
except ImportError:
    # ETDA submodule not available or not installed
    pass

