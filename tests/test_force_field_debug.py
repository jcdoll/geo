#!/usr/bin/env python3
"""
Debug force field calculations to understand why forces are too large.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from materials import MaterialType
from geo_game import GeoGame


def debug_force_field():
    """Debug force field calculations"""
    print("FORCE FIELD DEBUG")
    print("=================")
    
    # Create simple test case: single magma cell surrounded by basalt
    sim = GeoGame(20, 20, cell_size=100.0, quality=1, log_level="WARNING")
    sim.material_types.fill(MaterialType.SPACE)
    sim.temperature.fill(273.15)
    
    center = 10
    
    # Single magma cell
    sim.material_types[center, center] = MaterialType.MAGMA
    sim.temperature[center, center] = 1773.15  # Hot magma
    
    # Surround with basalt (3x3 ring)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            y, x = center + dy, center + dx
            sim.material_types[y, x] = MaterialType.BASALT
            sim.temperature[y, x] = 800.0  # Warm solid
    
    sim._properties_dirty = True
    sim._update_material_properties()
    
    print(f"Magma density: {sim.density[center, center]:.1f} kg/m³")
    print(f"Basalt density: {sim.density[center+1, center]:.1f} kg/m³")
    print(f"Magma temperature: {sim.temperature[center, center]:.1f} K")
    print(f"Basalt temperature: {sim.temperature[center+1, center]:.1f} K")
    
    # Calculate different force components
    print("\nFORCE COMPONENT ANALYSIS:")
    print("========================")
    
    # 1. Gravity forces
    if hasattr(sim, 'calculate_self_gravity'):
        sim.calculate_self_gravity()
        gx_self = sim.gravity_x[center, center]
        gy_self = sim.gravity_y[center, center]
        print(f"Self-gravity at magma: ({gx_self:.2e}, {gy_self:.2e}) m/s²")
    
    # 2. External gravity
    if hasattr(sim, 'external_gravity'):
        g_ext_x, g_ext_y = sim.external_gravity
        print(f"External gravity: ({g_ext_x:.2e}, {g_ext_y:.2e}) m/s²")
    
    # 3. Pressure field
    sim.fluid_dynamics.calculate_planetary_pressure()
    pressure_magma = sim.pressure[center, center]
    pressure_basalt = sim.pressure[center+1, center]
    print(f"Pressure at magma: {pressure_magma:.2e} MPa")
    print(f"Pressure at basalt: {pressure_basalt:.2e} MPa")
    
    # 4. Total force field
    fx, fy = sim.fluid_dynamics.compute_force_field()
    force_magma = np.hypot(fx[center, center], fy[center, center])
    force_basalt = np.hypot(fx[center+1, center], fy[center+1, center])
    
    print(f"Total force at magma: {force_magma:.2e} N")
    print(f"Total force at basalt: {force_basalt:.2e} N")
    
    # 5. Force components breakdown
    rho = sim.density[center, center]
    gx_total = sim.gravity_x[center, center] + sim.external_gravity[0]
    gy_total = sim.gravity_y[center, center] + sim.external_gravity[1]
    
    gravity_force = rho * np.hypot(gx_total, gy_total)
    print(f"Gravity force component: {gravity_force:.2e} N")
    
    # 6. Pressure gradient force
    P_pa = sim.pressure * 1e6  # MPa → Pa
    dx = sim.cell_size
    
    # Pressure gradient at magma cell
    if center > 0 and center < sim.width - 1:
        dP_dx = (P_pa[center, center+1] - P_pa[center, center-1]) / (2 * dx)
        dP_dy = (P_pa[center+1, center] - P_pa[center-1, center]) / (2 * dx)
        pressure_gradient_force = np.hypot(dP_dx, dP_dy)
        print(f"Pressure gradient force: {pressure_gradient_force:.2e} N/m³")
    
    # 7. Binding threshold comparison
    temp_avg = 0.5 * (sim.temperature[center, center] + sim.temperature[center+1, center])
    binding_threshold = sim.fluid_dynamics._binding_threshold(
        MaterialType.MAGMA, MaterialType.BASALT, temp_avg
    )
    
    print(f"\nFORCE vs BINDING COMPARISON:")
    print(f"===========================")
    print(f"Force magnitude: {force_magma:.2e} N")
    print(f"Binding threshold: {binding_threshold:.2e} N")
    print(f"Force/Binding ratio: {force_magma/binding_threshold:.1f}")
    
    if force_magma > binding_threshold:
        print("PROBLEM: Force exceeds binding - swapping will occur!")
        
        # Analyze what's driving the large force
        if gravity_force > binding_threshold:
            print(f"  -> Gravity force ({gravity_force:.2e}) exceeds binding")
        if pressure_gradient_force > binding_threshold:
            print(f"  -> Pressure gradient ({pressure_gradient_force:.2e}) exceeds binding")
    else:
        print("OK: Binding contains the force")
    
    return {
        'force_magnitude': force_magma,
        'binding_threshold': binding_threshold,
        'gravity_force': gravity_force,
        'pressure_gradient': pressure_gradient_force if 'pressure_gradient_force' in locals() else 0,
        'ratio': force_magma / binding_threshold
    }


def test_different_scenarios():
    """Test different material configurations"""
    print("\n\nTESTING DIFFERENT SCENARIOS:")
    print("============================")
    
    scenarios = [
        ("Cold magma vs basalt", 1000.0, 800.0),
        ("Hot magma vs basalt", 1773.0, 800.0),
        ("Hot magma vs cold basalt", 1773.0, 500.0),
        ("Moderate temps", 1200.0, 600.0),
    ]
    
    for name, magma_temp, basalt_temp in scenarios:
        print(f"\nScenario: {name}")
        print("-" * 30)
        
        sim = GeoGame(10, 10, cell_size=100.0, quality=1, log_level="WARNING")
        sim.material_types.fill(MaterialType.SPACE)
        sim.temperature.fill(273.15)
        
        center = 5
        sim.material_types[center, center] = MaterialType.MAGMA
        sim.temperature[center, center] = magma_temp
        
        sim.material_types[center+1, center] = MaterialType.BASALT
        sim.temperature[center+1, center] = basalt_temp
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
        fx, fy = sim.fluid_dynamics.compute_force_field()
        force = np.hypot(fx[center, center], fy[center, center])
        
        temp_avg = 0.5 * (magma_temp + basalt_temp)
        binding = sim.fluid_dynamics._binding_threshold(
            MaterialType.MAGMA, MaterialType.BASALT, temp_avg
        )
        
        print(f"  Force: {force:.2e} N")
        print(f"  Binding: {binding:.2e} N")
        print(f"  Ratio: {force/binding:.1f}")
        print(f"  Will swap: {'YES' if force > binding else 'NO'}")


if __name__ == "__main__":
    result = debug_force_field()
    test_different_scenarios()
    
    print(f"\n\nSUMMARY:")
    print(f"========")
    print(f"The force/binding ratio is {result['ratio']:.1f}")
    print(f"This means forces are {result['ratio']:.0f}x larger than binding thresholds!")
    
    if result['ratio'] > 10:
        print("\nRECOMMENDATION: The force calculation is generating unrealistically large forces.")
        print("Possible solutions:")
        print("1. Scale down the force calculation")
        print("2. Increase binding force magnitudes")
        print("3. Add force damping/limiting")
        print("4. Check pressure field calculation for errors") 