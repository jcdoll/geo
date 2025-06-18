#!/usr/bin/env python3
"""
Debug swap conditions to understand why magma still expands.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from materials import MaterialType
from geo_game import GeoGame


def debug_swap_conditions():
    """Debug the three swap conditions in detail"""
    print("SWAP CONDITIONS DEBUG")
    print("====================")
    
    # Create simple test case
    sim = GeoGame(10, 10, cell_size=100.0, quality=1, log_level="WARNING")
    sim.material_types.fill(MaterialType.SPACE)
    sim.temperature.fill(273.15)
    
    center = 5
    
    # Single magma cell surrounded by basalt
    sim.material_types[center, center] = MaterialType.MAGMA
    sim.temperature[center, center] = 1773.15  # Hot magma
    
    sim.material_types[center+1, center] = MaterialType.BASALT
    sim.temperature[center+1, center] = 800.0  # Warm solid
    
    sim._properties_dirty = True
    sim._update_material_properties()
    
    # Run one simulation step to generate velocities
    sim.step_forward()
    
    # Now analyze the swap conditions
    fx, fy = sim.fluid_dynamics.compute_force_field()
    vx, vy = sim.velocity_x, sim.velocity_y
    mt = sim.material_types
    temp = sim.temperature
    
    # Test magma -> basalt swap
    y, x = center, center
    ny, nx = center+1, center
    
    print(f"Testing swap: MAGMA ({y},{x}) -> BASALT ({ny},{nx})")
    print("=" * 50)
    
    # Get forces and velocities
    fsrc_x, fsrc_y = fx[y, x], fy[y, x]
    ftgt_x, ftgt_y = fx[ny, nx], fy[ny, nx]
    vsrc_x, vsrc_y = vx[y, x], vy[y, x]
    vtgt_x, vtgt_y = vx[ny, nx], vy[ny, nx]
    
    print(f"Source (magma) force: ({fsrc_x:.2e}, {fsrc_y:.2e}) N")
    print(f"Target (basalt) force: ({ftgt_x:.2e}, {ftgt_y:.2e}) N")
    print(f"Source (magma) velocity: ({vsrc_x:.2e}, {vsrc_y:.2e}) m/s")
    print(f"Target (basalt) velocity: ({vtgt_x:.2e}, {vtgt_y:.2e}) m/s")
    
    # Calculate differences
    dFx, dFy = fsrc_x - ftgt_x, fsrc_y - ftgt_y
    F_net = np.hypot(dFx, dFy)
    dVx, dVy = vsrc_x - vtgt_x, vsrc_y - vtgt_y
    V_diff = np.hypot(dVx, dVy)
    
    print(f"\nForce difference: ({dFx:.2e}, {dFy:.2e}) N")
    print(f"Net force magnitude: {F_net:.2e} N")
    print(f"Velocity difference: ({dVx:.2e}, {dVy:.2e}) m/s")
    print(f"Velocity difference magnitude: {V_diff:.2e} m/s")
    
    # Check the three conditions
    print(f"\nCONDITION ANALYSIS:")
    print(f"==================")
    
    # Condition 1: Force threshold
    temp_avg = 0.5 * (temp[y, x] + temp[ny, nx])
    threshold = sim.fluid_dynamics._binding_threshold(mt[y, x], mt[ny, nx], temp_avg)
    cond_force = F_net > threshold
    
    print(f"1. FORCE CONDITION:")
    print(f"   Force magnitude: {F_net:.2e} N")
    print(f"   Binding threshold: {threshold:.2e} N")
    print(f"   F_net > threshold: {cond_force}")
    print(f"   Force/Binding ratio: {F_net/threshold:.2e}")
    
    # Condition 2: Velocity threshold
    dv_thresh = sim.fluid_dynamics.dv_thresh
    cond_velocity = V_diff >= dv_thresh
    
    print(f"\n2. VELOCITY CONDITION:")
    print(f"   Velocity difference: {V_diff:.2e} m/s")
    print(f"   Velocity threshold: {dv_thresh:.2e} m/s")
    print(f"   V_diff >= threshold: {cond_velocity}")
    print(f"   Velocity/Threshold ratio: {V_diff/dv_thresh:.2e}")
    
    # Condition 3: Source binding
    _ref_solid = MaterialType.GRANITE
    src_bind = sim.fluid_dynamics._binding_threshold(mt[y, x], _ref_solid, temp[y, x])
    
    # Directional force projection
    dx_dir, dy_dir = nx - x, ny - y  # Direction from source to target
    proj_src = fsrc_x * dx_dir + fsrc_y * dy_dir
    
    if src_bind > 0:  # Solid material
        cond_src = abs(proj_src) > src_bind
        src_type = "SOLID"
    else:  # Fluid material
        cond_src = True
        src_type = "FLUID"
    
    print(f"\n3. SOURCE BINDING CONDITION:")
    print(f"   Source material type: {src_type}")
    print(f"   Source binding force: {src_bind:.2e} N")
    print(f"   Projected force: {proj_src:.2e} N")
    print(f"   |proj_src| > src_bind: {cond_src}")
    if src_bind > 0:
        print(f"   Force/Binding ratio: {abs(proj_src)/src_bind:.2e}")
    
    # Overall result
    will_swap = cond_force and cond_velocity and cond_src
    
    print(f"\nOVERALL RESULT:")
    print(f"==============")
    print(f"Condition 1 (Force): {cond_force}")
    print(f"Condition 2 (Velocity): {cond_velocity}")
    print(f"Condition 3 (Source): {cond_src}")
    print(f"Will swap: {will_swap}")
    
    if will_swap:
        print("\nPROBLEM: Swap will occur!")
        if cond_force:
            print("  -> Force condition is allowing swap (forces too large or binding too weak)")
        if cond_velocity:
            print("  -> Velocity condition is allowing swap (velocities too large or threshold too low)")
        if cond_src:
            print("  -> Source condition is allowing swap (fluid has no binding resistance)")
    else:
        print("\nGOOD: Swap will be prevented!")
    
    return {
        'force_condition': cond_force,
        'velocity_condition': cond_velocity,
        'source_condition': cond_src,
        'will_swap': will_swap,
        'force_ratio': F_net/threshold,
        'velocity_ratio': V_diff/dv_thresh
    }


def test_velocity_sources():
    """Test what's generating the velocities"""
    print("\n\nVELOCITY SOURCE ANALYSIS")
    print("========================")
    
    sim = GeoGame(10, 10, cell_size=100.0, quality=1, log_level="WARNING")
    sim.material_types.fill(MaterialType.SPACE)
    sim.temperature.fill(273.15)
    
    center = 5
    sim.material_types[center, center] = MaterialType.MAGMA
    sim.temperature[center, center] = 1773.15
    sim.material_types[center+1, center] = MaterialType.BASALT
    sim.temperature[center+1, center] = 800.0
    
    sim._properties_dirty = True
    sim._update_material_properties()
    
    print("Initial velocities (should be zero):")
    print(f"Magma velocity: ({sim.velocity_x[center, center]:.2e}, {sim.velocity_y[center, center]:.2e}) m/s")
    print(f"Basalt velocity: ({sim.velocity_x[center+1, center]:.2e}, {sim.velocity_y[center+1, center]:.2e}) m/s")
    
    # Test what happens after force application
    dt = 0.01  # 10 ms timestep
    sim.fluid_dynamics.apply_unified_kinematics(dt)
    
    print(f"\nAfter apply_unified_kinematics({dt} s):")
    print(f"Magma velocity: ({sim.velocity_x[center, center]:.2e}, {sim.velocity_y[center, center]:.2e}) m/s")
    print(f"Basalt velocity: ({sim.velocity_x[center+1, center]:.2e}, {sim.velocity_y[center+1, center]:.2e}) m/s")
    
    # Calculate velocity difference
    dVx = sim.velocity_x[center, center] - sim.velocity_x[center+1, center]
    dVy = sim.velocity_y[center, center] - sim.velocity_y[center+1, center]
    V_diff = np.hypot(dVx, dVy)
    
    print(f"Velocity difference: {V_diff:.2e} m/s")
    print(f"Velocity threshold: {sim.fluid_dynamics.dv_thresh:.2e} m/s")
    print(f"Exceeds threshold: {V_diff >= sim.fluid_dynamics.dv_thresh}")


if __name__ == "__main__":
    result = debug_swap_conditions()
    test_velocity_sources()
    
    print(f"\n\nSUMMARY:")
    print(f"========")
    if result['will_swap']:
        print("PROBLEM: Magma will still expand!")
        print("Root causes:")
        if result['force_condition']:
            print(f"  - Forces ({result['force_ratio']:.1e}x binding) are too large")
        if result['velocity_condition']:
            print(f"  - Velocities ({result['velocity_ratio']:.1e}x threshold) are too large")
        if result['source_condition']:
            print("  - Magma (fluid) has no binding resistance")
    else:
        print("SUCCESS: Magma expansion should be contained!") 