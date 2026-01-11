"""Pytest configuration for flux tests."""
import os
import sys
from pathlib import Path
import pytest


def pytest_configure(config):
    """Configure pytest environment for flux tests."""
    # Add flux directory to Python path so imports like 'from simulation import ...' work
    flux_dir = Path(__file__).parent.parent
    if str(flux_dir) not in sys.path:
        sys.path.insert(0, str(flux_dir))
    
    # Set SDL to use dummy video driver for headless operation
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['MPLBACKEND'] = 'Agg'


@pytest.fixture(params=[32, 64])
def grid_size(request):
    """Fixture providing different grid sizes for testing."""
    return request.param
