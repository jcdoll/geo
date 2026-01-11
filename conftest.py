"""Root pytest configuration for the geo project."""
import os
import sys
from pathlib import Path
from typing import Any


def pytest_configure(config):
    """Configure pytest environment."""
    # Note: We do NOT add workspace root to sys.path here because
    # the ca/, flux/, and sph/ directories each have modules with
    # the same names (e.g., gravity_solver.py, materials.py).
    # Each sub-project's conftest.py handles its own path configuration.
    
    # Set SDL to use dummy video driver for headless operation
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['MPLBACKEND'] = 'Agg'


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    """Ignore problematic directories during test collection."""
    # Ignore archive directories - contain deprecated tests
    if 'archive' in collection_path.parts:
        return True
    
    # Ignore deprecated directories
    if 'deprecated' in collection_path.parts:
        return True
    
    # Handle Windows path issues (symlinks that can't be stat'ed)
    try:
        _ = collection_path.is_dir()
    except OSError:
        return True
    
    return False
