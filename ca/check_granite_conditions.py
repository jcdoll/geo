#!/usr/bin/env python3
"""Check conditions for granite in space."""

import numpy as np
from geo_game import GeoGame
from materials import MaterialType

def check_granite_conditions():
    """Check temperature and pressure conditions for granite."""
    # Create small simulation
    sim = GeoGame(width=10, height=10, cell_size=50.0)
    
    # Initialize everything as space
    sim.material_types[:] = MaterialType.SPACE
    sim._update_material_properties()
    
    # Add single granite cell
    sim.material_types[5, 5] = MaterialType.GRANITE
    sim._update_material_properties()
    
    # Check conditions
    T_kelvin = sim.temperature[5, 5]
    T = T_kelvin - 273.15  # Convert to Celsius
    P_pascals = sim.pressure[5, 5]
    P = P_pascals / 1e6  # Convert Pa to MPa
    
    print(f"Granite conditions:")
    print(f"  Temperature (raw): {T_kelvin} K")
    print(f"  Temperature: {T:.1f}°C ({T_kelvin:.1f}K)")
    print(f"  Pressure (raw): {P_pascals} Pa")
    print(f"  Pressure: {P:.3f} MPa")
    
    # Check transition rules
    from materials import MaterialDatabase
    mat_db = MaterialDatabase()
    granite_props = mat_db.get_properties(MaterialType.GRANITE)
    
    print(f"\nGranite transitions:")
    for transition in granite_props.transitions:
        print(f"  {transition.target.name}: T={transition.min_temp}-{transition.max_temp}°C, P={transition.min_pressure}-{transition.max_pressure} MPa")
        if transition.probability:
            print(f"    Probability: {transition.probability}")
        
        # Check if this transition applies
        if (transition.min_temp <= T <= transition.max_temp and 
            transition.min_pressure <= P <= transition.max_pressure):
            print(f"    ⚠️  THIS TRANSITION CAN OCCUR!")
    
    # Test what happens with a step
    print(f"\nTesting material processes...")
    sim.material_processes.apply_metamorphism()
    
    new_mat = sim.material_types[5, 5]
    if new_mat != MaterialType.GRANITE:
        print(f"  ❌ Granite transformed to {new_mat.name}!")
    else:
        print(f"  ✓ Granite remained granite")

if __name__ == "__main__":
    check_granite_conditions()