"""Test if particle simulation achieves hydrostatic equilibrium.

Key question: Does ∇P = ρg in the bulk regions?
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_hybrid_optimized import FastParticleHybrid
import time


def setup_stratified_test(sim):
    """Set up a stratified water column for testing."""
    # Clear particles
    sim.n_particles = 10000
    sim.pos = np.zeros((sim.n_particles, 2), dtype=np.float32)
    sim.vel = np.zeros((sim.n_particles, 2), dtype=np.float32)
    sim.mass = np.ones(sim.n_particles, dtype=np.float32)
    sim.material = np.zeros(sim.n_particles, dtype=np.int32)
    
    # Create stratified layers
    n_per_layer = sim.n_particles // 3
    domain_x = sim.width * sim.cell_size
    domain_y = sim.height * sim.cell_size
    
    idx = 0
    # Air layer (top 30%)
    for i in range(n_per_layer):
        sim.pos[idx] = [np.random.uniform(0, domain_x),
                        np.random.uniform(0, domain_y * 0.3)]
        sim.material[idx] = 0  # air
        sim.mass[idx] = 0.1
        idx += 1
    
    # Water layer (middle 40%)  
    for i in range(n_per_layer * 2):
        sim.pos[idx] = [np.random.uniform(0, domain_x),
                        np.random.uniform(domain_y * 0.3, domain_y * 0.7)]
        sim.material[idx] = 1  # water
        sim.mass[idx] = 100.0
        idx += 1
    
    # Remaining particles as water at bottom
    while idx < sim.n_particles:
        sim.pos[idx] = [np.random.uniform(0, domain_x),
                        np.random.uniform(domain_y * 0.7, domain_y)]
        sim.material[idx] = 1  # water
        sim.mass[idx] = 100.0
        idx += 1


def analyze_hydrostatic_equilibrium(sim):
    """Analyze if the system is in hydrostatic equilibrium."""
    # Let system settle
    print("Letting system settle...")
    for i in range(100):
        sim.step(dt=0.01)
        if i % 20 == 0:
            max_vel = np.max(np.abs(sim.vel))
            print(f"  Step {i}: max velocity = {max_vel:.3f} m/s")
    
    # Analyze the pressure gradient vs gravity
    print("\nAnalyzing hydrostatic equilibrium...")
    
    # Compute pressure gradient on grid
    dx = sim.cell_size
    grad_p_y = np.zeros_like(sim.grid_pressure)
    grad_p_y[1:-1, :] = (sim.grid_pressure[2:, :] - sim.grid_pressure[:-2, :]) / (2 * dx)
    
    # Expected: ∇P = ρg
    expected_grad = sim.grid_density * sim.gravity_y
    
    # Sample along vertical line at center
    x_center = sim.width // 2
    
    print("\nVertical profile at x = center:")
    print("   y | Height(m) | ρ (kg/m³) | P (Pa) | ∇P (Pa/m) | ρg (Pa/m) | Error")
    print("-" * 75)
    
    # Sample every 5 cells
    for y in range(5, sim.height-5, 5):
        height = y * dx
        density = sim.grid_density[y, x_center]
        pressure = sim.grid_pressure[y, x_center]
        grad_p = grad_p_y[y, x_center]
        rho_g = expected_grad[y, x_center]
        error = abs(grad_p - rho_g)
        
        # Material type based on height
        if height < sim.height * dx * 0.3:
            mat = "AIR"
        else:
            mat = "WATER"
            
        print(f"{y:4d} | {height:9.1f} | {density:9.1f} | {pressure:6.0f} | "
              f"{grad_p:9.1f} | {rho_g:9.1f} | {error:6.1f} ({mat})")
    
    # Check bulk regions
    print("\nBulk region analysis:")
    
    # Air region (y = 5-10)
    air_region = grad_p_y[5:10, :]
    air_expected = expected_grad[5:10, :]
    air_error = np.abs(air_region - air_expected)
    print(f"Air region:")
    print(f"  Mean |∇P|: {np.mean(np.abs(air_region)):.1f} Pa/m")
    print(f"  Mean |ρg|: {np.mean(np.abs(air_expected)):.1f} Pa/m")
    print(f"  Mean error: {np.mean(air_error):.1f} Pa/m")
    print(f"  Max error: {np.max(air_error):.1f} Pa/m")
    
    # Water region (y = 25-35)
    water_region = grad_p_y[25:35, :]
    water_expected = expected_grad[25:35, :]
    water_error = np.abs(water_region - water_expected)
    print(f"\nWater region:")
    print(f"  Mean |∇P|: {np.mean(np.abs(water_region)):.1f} Pa/m")
    print(f"  Mean |ρg|: {np.mean(np.abs(water_expected)):.1f} Pa/m")
    print(f"  Mean error: {np.mean(water_error):.1f} Pa/m")
    print(f"  Max error: {np.max(water_error):.1f} Pa/m")
    
    # Relative error
    water_rel_error = np.mean(water_error) / np.mean(np.abs(water_expected)) * 100
    print(f"  Relative error: {water_rel_error:.1f}%")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Density
    im1 = axes[0, 0].imshow(sim.grid_density, origin='lower', cmap='viridis')
    axes[0, 0].set_title('Density (kg/m³)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Pressure
    im2 = axes[0, 1].imshow(sim.grid_pressure, origin='lower', cmap='plasma')
    axes[0, 1].set_title('Pressure (Pa)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pressure gradient
    im3 = axes[1, 0].imshow(grad_p_y, origin='lower', cmap='RdBu', 
                             vmin=-10000, vmax=10000)
    axes[1, 0].set_title('∇P_y (Pa/m)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Error
    error_map = np.abs(grad_p_y - expected_grad)
    im4 = axes[1, 1].imshow(error_map, origin='lower', cmap='hot')
    axes[1, 1].set_title('|∇P - ρg| (Pa/m)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('particle_hydrostatic_test.png')
    print("\nSaved visualization to particle_hydrostatic_test.png")
    
    return water_rel_error < 20  # Success if < 20% error


def test_pressure_equation():
    """Test what pressure equation is actually being solved."""
    print("\nPRESSURE EQUATION ANALYSIS")
    print("=" * 60)
    
    print("\nCurrent implementation:")
    print("  P = k * (ρ - ρ₀)")
    print("  where k = 1000 Pa/(kg/m³)")
    print("  and ρ₀ = 1000 kg/m³ (water reference)")
    
    print("\nThis gives:")
    print("  ∇P = k * ∇ρ")
    
    print("\nFor hydrostatic equilibrium we need:")
    print("  ∇P = ρg")
    
    print("\nSo we need:")
    print("  k * ∇ρ = ρg")
    
    print("\nThis only works if:")
    print("  1. Density variations are large (∇ρ ≠ 0)")
    print("  2. k is tuned correctly")
    print("  3. Or we use a different pressure model")
    
    print("\nThe fundamental issue:")
    print("  In constant density regions, ∇ρ = 0")
    print("  So ∇P = k * 0 = 0")
    print("  But we need ∇P = ρg ≠ 0!")


if __name__ == "__main__":
    # Create simulation
    sim = FastParticleHybrid(width=64, height=64, n_particles=10000)
    
    # Set up stratified test
    setup_stratified_test(sim)
    
    # Test pressure equation
    test_pressure_equation()
    
    # Analyze equilibrium
    print("\n" + "="*60)
    success = analyze_hydrostatic_equilibrium(sim)
    
    print("\n" + "="*60)
    if success:
        print("SUCCESS: Hydrostatic equilibrium achieved!")
    else:
        print("FAILURE: Hydrostatic equilibrium NOT achieved")
        print("\nThe issue: P = k(ρ - ρ₀) gives ∇P = k∇ρ")
        print("In constant density regions, ∇ρ = 0, so ∇P = 0")
        print("But we need ∇P = ρg!")