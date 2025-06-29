"""
Adaptive timestepping for SPH simulation.

Computes optimal timestep based on CFL condition, viscous diffusion,
and force constraints.
"""

import numpy as np
from typing import Optional
from ..core.particles import ParticleArrays
from ..physics.materials import MaterialDatabase, MaterialType
from ..physics.pressure_overlap_prevention import get_stable_bulk_modulus_improved


def compute_adaptive_timestep(particles: ParticleArrays, n_active: int,
                            material_db: MaterialDatabase,
                            cfl_number: float = 0.5,
                            force_factor: float = 0.5,
                            viscous_factor: float = 0.25,
                            alpha_visc: float = 1.0,
                            min_dt: float = 1e-3,
                            max_dt: float = 1.0) -> float:
    """Compute adaptive timestep based on CFL and other stability criteria.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        material_db: Material properties database
        cfl_number: CFL safety factor (0.3 typical)
        force_factor: Force criterion safety factor
        viscous_factor: Viscous criterion safety factor
        alpha_visc: Artificial viscosity parameter
        min_dt: Minimum allowed timestep
        max_dt: Maximum allowed timestep
        
    Returns:
        Optimal timestep
    """
    # Arrays for per-particle timestep constraints
    dt_cfl = np.full(n_active, np.inf, dtype=np.float32)
    dt_force = np.full(n_active, np.inf, dtype=np.float32)
    dt_viscous = np.full(n_active, np.inf, dtype=np.float32)
    
    # Get particle velocities and accelerations
    vel_x = particles.velocity_x[:n_active]
    vel_y = particles.velocity_y[:n_active]
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    
    # Accelerations from forces
    masses = particles.mass[:n_active]
    accel_x = particles.force_x[:n_active] / masses
    accel_y = particles.force_y[:n_active] / masses
    accel_mag = np.sqrt(accel_x**2 + accel_y**2)
    
    # Process each material type
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type.value
        if not np.any(mask):
            continue
            
        # Get material properties
        props = material_db.get_properties(mat_type)
        bulk_modulus = get_stable_bulk_modulus_improved(mat_type.value)
        
        # Use effective bulk modulus (same as in pressure calculation)
        # This ensures timestep is consistent with actual pressure forces
        bulk_modulus_effective = bulk_modulus * 0.01
        
        # Sound speed: c = sqrt(K_eff/ρ)
        sound_speed = np.sqrt(bulk_modulus_effective / props.density_ref)
        
        # Get smoothing lengths for this material
        h = particles.smoothing_h[:n_active][mask]
        
        # CFL condition: dt < CFL * h / (c + v)
        dt_cfl[mask] = cfl_number * h / (sound_speed + vel_mag[mask])
        
        # Force condition: dt < sqrt(h / a)
        # Only apply where acceleration is significant
        accel_mask = accel_mag[mask] > 1e-6
        if np.any(accel_mask):
            dt_force[mask][accel_mask] = force_factor * np.sqrt(
                h[accel_mask] / accel_mag[mask][accel_mask]
            )
        
        # Viscous condition (artificial viscosity)
        # dt < viscous_factor * h / (c * alpha)
        dt_viscous[mask] = viscous_factor * h / (sound_speed * alpha_visc)
        
    # Take minimum across all particles and criteria
    dt_optimal = np.min([
        np.min(dt_cfl),
        np.min(dt_force),
        np.min(dt_viscous)
    ])
    
    # Clamp to allowed range
    dt_optimal = np.clip(dt_optimal, min_dt, max_dt)
    
    return dt_optimal


def compute_timestep_diagnostics(particles: ParticleArrays, n_active: int,
                                material_db: MaterialDatabase,
                                current_dt: float) -> dict:
    """Compute detailed timestep diagnostics for debugging.
    
    Returns dict with:
        - limiting_factor: 'cfl', 'force', or 'viscous'
        - safety_margin: ratio of current_dt to limit
        - material_limits: per-material timestep limits
    """
    diagnostics = {
        'current_dt': current_dt,
        'limiting_factor': None,
        'safety_margin': np.inf,
        'material_limits': {}
    }
    
    # Compute optimal timestep
    dt_optimal = compute_adaptive_timestep(particles, n_active, material_db)
    diagnostics['optimal_dt'] = dt_optimal
    diagnostics['speedup_possible'] = dt_optimal / current_dt
    
    # Detailed per-material analysis
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type.value
        if not np.any(mask):
            continue
            
        props = material_db.get_properties(mat_type)
        bulk_modulus = get_stable_bulk_modulus_improved(mat_type.value)
        bulk_modulus_effective = bulk_modulus * 0.01
        sound_speed = np.sqrt(bulk_modulus_effective / props.density_ref)
        
        # Average values for this material
        h_avg = np.mean(particles.smoothing_h[:n_active][mask])
        vel_avg = np.mean(np.sqrt(
            particles.velocity_x[:n_active][mask]**2 + 
            particles.velocity_y[:n_active][mask]**2
        ))
        
        dt_cfl = 0.3 * h_avg / (sound_speed + vel_avg)
        
        diagnostics['material_limits'][mat_type.name] = {
            'sound_speed': sound_speed,
            'avg_velocity': vel_avg,
            'dt_cfl': dt_cfl,
            'particles': np.sum(mask)
        }
    
    return diagnostics