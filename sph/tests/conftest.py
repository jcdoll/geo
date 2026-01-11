"""Pytest configuration for SPH tests."""
import os
import sys
from pathlib import Path
from typing import Any


def pytest_configure(config):
    """Configure pytest environment for SPH tests."""
    # Add workspace root to Python path for sph package imports
    workspace_root = Path(__file__).parent.parent.parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    
    # Set SDL to use dummy video driver for headless operation
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['MPLBACKEND'] = 'Agg'


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    """Ignore archive directory which contains deprecated tests."""
    # Ignore the archive directory - contains tests with missing dependencies (torch)
    # and deprecated API usage (kernel.W instead of kernel.W_vectorized)
    if 'archive' in collection_path.parts:
        return True
    
    # Also handle Windows path issues
    try:
        _ = collection_path.is_dir()
    except OSError:
        return True
    
    return False
