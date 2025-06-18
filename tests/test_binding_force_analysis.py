#!/usr/bin/env python3
"""
Detailed analysis of binding force temperature dependence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from materials import MaterialType
from geo_game import GeoGame


def analyze_binding_forces():
    """Analyze how temperature affects binding forces"""
    sim = GeoGame(10, 10, cell_size=100.0, quality=1, log_level="WARNING")
    
    # Test temperatures from cold to very hot
    temperatures = [
        273.15,      # 0°C - freezing
        500.0,       # 227°C - warm
        800.0,       # 527°C - hot solid
        1200.0,      # 927°C - very hot solid
        1500.0,      # 1227°C - magma temperature
        1773.15,     # 1500°C - very hot magma
        2000.0,      # 1727°C - extremely hot
    ]
    
    print("Temperature vs Binding Force Analysis:")
    print("=====================================")
    print(f"{'Temp (K)':<10} {'Temp (°C)':<10} {'Factor':<8} {'Magma-Basalt (N)':<15} {'% Original':<12}")
    print("-" * 70)
    
    base_binding = 1e-4  # Base magma-basalt binding force
    
    for temp in temperatures:
        temp_celsius = temp - 273.15
        
        # Current formula (fixed)
        melting_point = 1473.15  # 1200°C in Kelvin
        if temp > melting_point:
            temp_factor = max(0.5, 1.0 - (temp - melting_point) / 1000.0)
        else:
            temp_factor = 1.0
        binding_force = base_binding * temp_factor
        percent_original = temp_factor * 100
        
        print(f"{temp:<10.1f} {temp_celsius:<10.1f} {temp_factor:<8.3f} {binding_force:<15.2e} {percent_original:<12.1f}%")
    
    print("\nPROBLEM ANALYSIS:")
    print("At magma temperatures (1500°C), binding force is reduced to only 10% of original!")
    print("This makes magma easily overcome solid binding forces.")
    
    # Test different formulas
    print("\n\nTesting Alternative Temperature Weakening Formulas:")
    print("===================================================")
    
    test_temp = 1773.15  # Hot magma temperature
    
    # Current formula
    current = max(0.1, 1.0 - (test_temp - 273.15) / 1500.0)
    
    # Alternative 1: Less aggressive weakening
    alt1 = max(0.3, 1.0 - (test_temp - 273.15) / 3000.0)  # Slower decay, higher minimum
    
    # Alternative 2: Exponential decay
    alt2 = max(0.2, np.exp(-(test_temp - 273.15) / 2000.0))
    
    # Alternative 3: Only weaken above melting point
    melting_point = 1473.15  # ~1200°C
    if test_temp > melting_point:
        alt3 = max(0.5, 1.0 - (test_temp - melting_point) / 1000.0)
    else:
        alt3 = 1.0
    
    print(f"Current formula at {test_temp-273.15:.0f}°C: {current:.3f} ({current*100:.1f}%)")
    print(f"Alternative 1 (gentler): {alt1:.3f} ({alt1*100:.1f}%)")
    print(f"Alternative 2 (exponential): {alt2:.3f} ({alt2*100:.1f}%)")
    print(f"Alternative 3 (above melting only): {alt3:.3f} ({alt3*100:.1f}%)")
    
    print(f"\nRECOMMENDATION: Use Alternative 3 - only weaken binding above melting point")
    print(f"This prevents solid rock from losing cohesion at moderate temperatures.")


def test_force_magnitudes():
    """Test what forces are actually generated in the simulation"""
    sim = GeoGame(20, 20, cell_size=100.0, quality=1, log_level="WARNING")
    
    # Create simple magma-basalt interface
    sim.material_types.fill(MaterialType.SPACE)
    sim.temperature.fill(273.15)
    
    # Single magma cell surrounded by basalt
    center = 10
    sim.material_types[center, center] = MaterialType.MAGMA
    sim.temperature[center, center] = 1773.15  # Hot magma
    
    # Surround with basalt
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            y, x = center + dy, center + dx
            sim.material_types[y, x] = MaterialType.BASALT
            sim.temperature[y, x] = 800.0  # Warm solid
    
    sim._properties_dirty = True
    sim._update_material_properties()
    
    # Calculate forces
    fx, fy = sim.fluid_dynamics.compute_force_field()
    
    # Check force magnitudes at magma-basalt interface
    magma_force = np.hypot(fx[center, center], fy[center, center])
    
    # Check binding thresholds
    temp_avg = 0.5 * (sim.temperature[center, center] + sim.temperature[center, center+1])
    binding_threshold = sim.fluid_dynamics._binding_threshold(
        MaterialType.MAGMA, MaterialType.BASALT, temp_avg
    )
    
    print(f"\nForce vs Binding Analysis:")
    print(f"==========================")
    print(f"Force on magma cell: {magma_force:.2e} N")
    print(f"Magma-Basalt binding threshold: {binding_threshold:.2e} N")
    print(f"Force/Binding ratio: {magma_force/binding_threshold:.1f}")
    
    if magma_force > binding_threshold:
        print("PROBLEM: Force exceeds binding threshold - magma will expand!")
    else:
        print("OK: Binding threshold contains the force")


if __name__ == "__main__":
    analyze_binding_forces()
    test_force_magnitudes() 