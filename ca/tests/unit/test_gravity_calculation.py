"""Test suite for gravity calculation"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geo_game import GeoGame as GeologySimulation
from materials import MaterialType
import pytest


class TestGravityCalculation:
    """Test suite for gravity field calculations"""
    
    def setup_method(self):
        """Set up test simulation"""
        self.sim = GeologySimulation(width=16, height=16, cell_size=1.0, setup_planet=False)
        
    def test_gravity_points_toward_center(self):
        """Test that gravitational forces point toward center of mass"""
        # Create a spherical planet
        center_x, center_y = 8, 8
        radius = 5
        
        # Fill with space
        self.sim.material_types[:] = MaterialType.SPACE
        self.sim.temperature[:] = 100.0
        
        # Create planet
        for y in range(16):
            for x in range(16):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    self.sim.material_types[y, x] = MaterialType.GRANITE
                    self.sim.temperature[y, x] = 300.0
        
        # Update properties and calculate gravity
        self.sim._update_material_properties()
        self.sim.calculate_self_gravity()
        
        # Check gravity vectors point toward center
        for y in range(16):
            for x in range(16):
                if self.sim.material_types[y, x] != MaterialType.SPACE:
                    # Vector from cell to center
                    dx = center_x - x
                    dy = center_y - y
                    
                    # Gravity components
                    gx = self.sim.gravity_x[y, x]
                    gy = self.sim.gravity_y[y, x]
                    
                    # Skip if at center (gravity should be near zero)
                    if abs(dx) < 0.5 and abs(dy) < 0.5:
                        continue
                    
                    # Gravity should point toward center
                    # (same direction as vector to center)
                    if abs(dx) > 0.1:
                        assert gx * dx > 0, f"Gravity x-component at ({x},{y}) points wrong way"
                    if abs(dy) > 0.1:
                        assert gy * dy > 0, f"Gravity y-component at ({x},{y}) points wrong way"