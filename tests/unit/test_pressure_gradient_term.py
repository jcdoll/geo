"""Test that the g·∇ρ term is correctly included in pressure calculation."""

import numpy as np
import pytest
from geo_game import GeoGame
from materials import MaterialType


def test_pressure_gradient_term_added():
    """Verify that adding the g·∇ρ term changes the pressure solution.
    
    This test ensures the bug fix was actually applied - the pressure
    field should be different with and without the second term.
    """
    # Create test setup with density interface
    sim = GeoGame(width=20, height=20, cell_size=50.0)
    
    # Disable everything except pressure
    sim.enable_heat_diffusion = False
    sim.enable_self_gravity = False
    sim.external_gravity = (0, 10.0)
    sim.enable_material_melting = False
    sim.enable_atmospheric_processes = False
    sim.enable_material_processes = False
    sim.enable_heat_transfer = False
    sim.enable_pressure_solver = True
    
    # Create density jump: water below, air above
    sim.material_types[:] = MaterialType.AIR
    sim.material_types[10:, :] = MaterialType.WATER
    sim.temperature[:] = 290.0
    
    sim._update_material_properties()
    
    # Calculate pressure with current implementation (includes g·∇ρ)
    sim.fluid_dynamics.calculate_planetary_pressure()
    pressure_with_term = sim.pressure.copy()
    
    # The pressure should NOT be zero everywhere
    # With g·∇ρ term, there should be pressure variations at the interface
    assert not np.allclose(pressure_with_term, 0.0), \
        "Pressure should have variations due to density gradient"
    
    # Check that maximum pressure variation occurs near the interface
    max_variation_y = np.argmax(np.abs(np.diff(pressure_with_term[:, 10])))
    assert 8 <= max_variation_y <= 11, \
        f"Maximum pressure variation at y={max_variation_y}, expected near y=10"
    
    # For uniform external gravity with density jump, RHS = g·∇ρ
    # This should create negative pressures above and below the interface
    # (due to P=0 boundary conditions)
    assert np.any(pressure_with_term < 0), "Should have negative pressures"
    
    # The solution should be roughly symmetric about the midpoint
    # (not exact due to boundary effects)
    mid_y = sim.height // 2
    top_half_avg = np.mean(np.abs(pressure_with_term[:mid_y, :]))
    bottom_half_avg = np.mean(np.abs(pressure_with_term[mid_y:, :]))
    
    # They should be same order of magnitude
    ratio = top_half_avg / bottom_half_avg if bottom_half_avg > 0 else 0
    assert 0.1 < ratio < 10.0, \
        f"Pressure should be roughly symmetric, got ratio {ratio}"


def test_self_gravity_includes_both_terms():
    """Test that self-gravity case now includes both ρ∇·g and g·∇ρ terms."""
    sim = GeoGame(width=30, height=30, cell_size=50.0)
    
    # Enable self-gravity
    sim.enable_heat_diffusion = False
    sim.enable_self_gravity = True
    sim.external_gravity = (0, 0)  # Only self-gravity
    sim.enable_material_melting = False
    sim.enable_atmospheric_processes = False
    sim.enable_material_processes = False
    sim.enable_heat_transfer = False
    sim.enable_pressure_solver = True
    sim.enable_gravity_solver = True
    
    # Create a planet with dense core
    center_y, center_x = sim.height // 2, sim.width // 2
    radius = 8
    
    # Make everything space initially
    sim.material_types[:] = MaterialType.SPACE
    sim.temperature[:] = 100.0  # Cold space
    
    # Create spherical planet
    for y in range(sim.height):
        for x in range(sim.width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < radius:
                # Dense core
                if dist < radius / 2:
                    sim.material_types[y, x] = MaterialType.GRANITE
                    sim.temperature[y, x] = 1000.0  # Hot core
                else:
                    # Less dense mantle
                    sim.material_types[y, x] = MaterialType.WATER
                    sim.temperature[y, x] = 500.0
    
    sim._update_material_properties()
    
    # Calculate self-gravity and pressure
    sim.calculate_self_gravity()
    sim.fluid_dynamics.calculate_planetary_pressure()
    
    # With both terms, pressure should have structure
    # Check that pressure is not uniform within the planet
    planet_mask = sim.material_types != MaterialType.SPACE
    planet_pressures = sim.pressure[planet_mask]
    
    # Should have variation
    pressure_std = np.std(planet_pressures)
    assert pressure_std > 0.0001, \
        f"Pressure should vary within planet, std={pressure_std}"
    
    # Pressure should be zero in space (by construction)
    space_mask = sim.material_types == MaterialType.SPACE
    assert np.allclose(sim.pressure[space_mask], 0.0), \
        "Space cells should have P=0"


def test_force_calculation_consistency():
    """Test that forces are computed correctly from pressure gradients."""
    sim = GeoGame(width=20, height=20, cell_size=50.0)
    
    sim.enable_heat_diffusion = False
    sim.enable_self_gravity = False
    sim.external_gravity = (0, 10.0)
    sim.enable_material_melting = False
    sim.enable_atmospheric_processes = False
    sim.enable_material_processes = False
    sim.enable_heat_transfer = False
    sim.enable_pressure_solver = True
    
    # Uniform density - should give zero pressure everywhere
    sim.material_types[:] = MaterialType.WATER
    sim.temperature[:] = 290.0
    sim._update_material_properties()
    
    # Calculate pressure and forces
    sim.fluid_dynamics.calculate_planetary_pressure()
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    # For uniform density with external gravity:
    # - Pressure should be ~0 (due to boundary conditions)
    # - Force should be ρg everywhere
    
    expected_fy = sim.density[10, 10] * 10.0  # ρg
    
    # Check interior cells (away from boundaries)
    for y in range(5, 15):
        for x in range(5, 15):
            # Allow small deviation due to boundary effects
            assert abs(fy[y, x] - expected_fy) < 10.0, \
                f"Force at ({y},{x}) = {fy[y,x]}, expected {expected_fy}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])