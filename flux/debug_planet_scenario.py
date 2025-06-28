"""Debug the planet scenario to understand the mass conservation failure."""

import numpy as np
import matplotlib.pyplot as plt
from simulation import FluxSimulation
from materials import MaterialType
import os

# Disable GUI backend
import matplotlib
matplotlib.use('Agg')

def capture_state(sim, title, filename):
    """Capture simulation state and save as image."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)
    
    # Material distribution
    ax = axes[0, 0]
    # Create composite view of dominant material
    dominant_mat = np.argmax(sim.state.vol_frac, axis=0)
    im = ax.imshow(dominant_mat, cmap='tab20', interpolation='nearest')
    ax.set_title('Dominant Material')
    ax.axis('off')
    
    # Density
    ax = axes[0, 1]
    im = ax.imshow(sim.state.density, cmap='viridis', interpolation='nearest')
    ax.set_title(f'Density (kg/m³)\nmin={np.min(sim.state.density):.1e}, max={np.max(sim.state.density):.1e}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Temperature
    ax = axes[0, 2]
    im = ax.imshow(sim.state.temperature, cmap='hot', interpolation='nearest')
    ax.set_title(f'Temperature (K)\nmin={np.min(sim.state.temperature):.0f}, max={np.max(sim.state.temperature):.0f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # Velocity magnitude
    ax = axes[1, 0]
    vel_mag = np.sqrt(sim.state.velocity_x**2 + sim.state.velocity_y**2)
    if np.all(np.isfinite(vel_mag)):
        im = ax.imshow(vel_mag, cmap='plasma', interpolation='nearest')
        ax.set_title(f'Velocity (m/s)\nmax={np.max(vel_mag):.1e}')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, f'NaN velocities!\ncount={np.sum(~np.isfinite(vel_mag))}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Velocity (m/s) - NaN')
    ax.axis('off')
    
    # Pressure
    ax = axes[1, 1]
    if np.all(np.isfinite(sim.state.pressure)):
        im = ax.imshow(sim.state.pressure, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f'Pressure (Pa)\nmin={np.min(sim.state.pressure):.1e}, max={np.max(sim.state.pressure):.1e}')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, f'NaN pressure!\ncount={np.sum(~np.isfinite(sim.state.pressure))}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pressure (Pa) - NaN')
    ax.axis('off')
    
    # Mass info
    ax = axes[1, 2]
    ax.axis('off')
    total_mass = np.sum(sim.state.density) * sim.state.dx * sim.state.dx
    info_text = f"Total mass: {total_mass:.2e} kg\n"
    info_text += f"Time: {sim.state.time:.3f} s\n"
    info_text += f"Step: {sim.step_count}\n"
    info_text += f"dt: {sim.state.dt:.1e} s\n"
    
    # Count materials
    info_text += "\nMaterial inventory:\n"
    for mat_type in MaterialType:
        vol = np.sum(sim.state.vol_frac[mat_type]) * sim.state.dx**2
        if vol > 0:
            info_text += f"  {mat_type.name}: {vol:.1e} m²\n"
    
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, va='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def main():
    # Create simulation with same parameters as test
    print("Creating planet scenario (64x64 grid)...")
    sim = FluxSimulation(nx=64, ny=64, dx=50.0, scenario="planet")
    
    # Capture initial state
    capture_state(sim, "Planet Scenario - Initial State", "planet_initial.png")
    
    # Get initial mass
    initial_mass = np.sum(sim.state.density) * sim.state.dx * sim.state.dx
    print(f"Initial total mass: {initial_mass:.2e} kg")
    
    # Run for a few steps to see what happens
    sim.paused = False
    steps_completed = 0
    
    try:
        # First step - often where issues start
        print("\nRunning first step...")
        sim.step_forward()
        steps_completed += 1
        capture_state(sim, "Planet Scenario - After 1 Step", "planet_step_001.png")
        
        # Check for NaN
        if np.any(~np.isfinite(sim.state.velocity_x)) or np.any(~np.isfinite(sim.state.velocity_y)):
            print("WARNING: NaN detected in velocities after step 1!")
            
        # Run a few more steps
        for i in range(9):
            sim.step_forward()
            steps_completed += 1
            
        capture_state(sim, "Planet Scenario - After 10 Steps", "planet_step_010.png")
        
        # Run to 100 steps (what the test does)
        for i in range(90):
            sim.step_forward()
            steps_completed += 1
            
        capture_state(sim, "Planet Scenario - After 100 Steps", "planet_step_100.png")
        
    except Exception as e:
        print(f"ERROR: Simulation crashed after {steps_completed} steps: {e}")
        capture_state(sim, f"Planet Scenario - Crashed at Step {steps_completed}", "planet_crashed.png")
    
    # Calculate final mass
    final_mass = np.sum(sim.state.density) * sim.state.dx * sim.state.dx
    mass_change = abs(final_mass - initial_mass) / initial_mass if initial_mass > 0 else 0
    
    print(f"\nFinal total mass: {final_mass:.2e} kg")
    print(f"Mass change: {mass_change*100:.2f}%")
    print(f"Test threshold: 5%")
    print(f"Test would {'FAIL' if mass_change > 0.05 else 'PASS'}")
    
    # Check for NaN in final state
    nan_density = np.sum(~np.isfinite(sim.state.density))
    nan_velocity = np.sum(~np.isfinite(sim.state.velocity_x)) + np.sum(~np.isfinite(sim.state.velocity_y))
    nan_temp = np.sum(~np.isfinite(sim.state.temperature))
    
    print(f"\nNaN counts in final state:")
    print(f"  Density: {nan_density}")
    print(f"  Velocity: {nan_velocity}")
    print(f"  Temperature: {nan_temp}")

if __name__ == "__main__":
    main()