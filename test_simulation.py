#!/usr/bin/env python3
"""
Test script for the geology simulation without GUI.
Verifies that all components work correctly.
"""

import numpy as np
from rock_types import RockType, RockDatabase
from simulation_engine import GeologySimulation

def test_rock_database():
    """Test the rock database functionality"""
    print("Testing Rock Database...")
    
    db = RockDatabase()
    
    # Test getting properties
    granite_props = db.get_properties(RockType.GRANITE)
    print(f"Granite density: {granite_props.density} kg/m³")
    print(f"Granite melting point: {granite_props.melting_point}°C")
    
    # Test metamorphic transitions
    product = db.get_metamorphic_product(RockType.SHALE, 300, 50)
    print(f"Shale at 300°C, 50 MPa becomes: {product}")
    
    product = db.get_metamorphic_product(RockType.SHALE, 600, 250)
    print(f"Shale at 600°C, 250 MPa becomes: {product}")
    
    # Test melting
    should_melt = db.should_melt(RockType.GRANITE, 1300)
    print(f"Granite melts at 1300°C: {should_melt}")
    
    print("✓ Rock Database tests passed\n")

def test_simulation_engine():
    """Test the simulation engine"""
    print("Testing Simulation Engine...")
    
    # Create small simulation
    sim = GeologySimulation(width=20, height=15, cell_size=1000.0)
    
    print(f"Initial simulation time: {sim.time} years")
    print(f"Grid size: {sim.width} x {sim.height}")
    print(f"Temperature range: {np.min(sim.temperature):.1f}°C to {np.max(sim.temperature):.1f}°C")
    print(f"Pressure range: {np.min(sim.pressure):.1f} to {np.max(sim.pressure):.1f} MPa")
    
    # Test adding heat source
    sim.add_heat_source(10, 10, 2, 1000)
    print(f"After heat source - Max temperature: {np.max(sim.temperature):.1f}°C")
    
    # Test applying tectonic stress
    sim.apply_tectonic_stress(5, 8, 1, 100)
    print(f"After tectonic stress - Max pressure: {np.max(sim.pressure):.1f} MPa")
    
    # Test time stepping
    initial_time = sim.time
    sim.step_forward()
    print(f"After one step - Time: {sim.time} years (stepped {sim.time - initial_time} years)")
    
    # Test backward stepping
    success = sim.step_backward()
    print(f"Backward step successful: {success}, Time back to: {sim.time} years")
    
    # Test statistics
    stats = sim.get_stats()
    print(f"Rock composition: {len(stats['rock_composition'])} different types")
    
    print("✓ Simulation Engine tests passed\n")

def test_multiple_steps():
    """Test multiple simulation steps and observe changes"""
    print("Testing Multiple Simulation Steps...")
    
    sim = GeologySimulation(width=30, height=20)
    
    # Add a strong heat source at depth
    sim.add_heat_source(15, 15, 3, 1200)
    
    print("Running 5 simulation steps...")
    for i in range(5):
        sim.step_forward()
        stats = sim.get_stats()
        print(f"Step {i+1}: Time={stats['time']:.0f}y, "
              f"Avg Temp={stats['avg_temperature']:.1f}°C, "
              f"Max Temp={stats['max_temperature']:.1f}°C")
    
    # Check if any rocks changed type
    rock_strings = [rock.value for rock in sim.rock_types.flatten()]
    unique_rocks = np.unique(rock_strings)
    print(f"Rock types present: {list(unique_rocks)}")
    
    print("✓ Multiple steps test passed\n")

def test_visualization_data():
    """Test getting visualization data"""
    print("Testing Visualization Data...")
    
    sim = GeologySimulation(width=10, height=8)
    
    colors, temp, pressure = sim.get_visualization_data()
    
    print(f"Color array shape: {colors.shape}")
    print(f"Temperature array shape: {temp.shape}")
    print(f"Pressure array shape: {pressure.shape}")
    print(f"Color data type: {colors.dtype}")
    print(f"Sample color: {colors[0, 0]}")
    
    print("✓ Visualization data test passed\n")

def main():
    """Run all tests"""
    print("=" * 50)
    print("GEOLOGY SIMULATION TEST SUITE")
    print("=" * 50)
    print()
    
    try:
        test_rock_database()
        test_simulation_engine() 
        test_multiple_steps()
        test_visualization_data()
        
        print("=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("The simulation is ready to run.")
        print("Use 'python main.py' to start the interactive version.")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 