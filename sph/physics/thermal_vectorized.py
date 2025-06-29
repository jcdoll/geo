"""
Vectorized thermal physics for SPH including heat transfer and phase transitions.

Implements:
- SPH heat conduction (Cleary & Monaghan 1999)
- Phase transitions with latent heat
- Radiative cooling
- Heat generation (radioactive decay)
"""

import numpy as np
from typing import Tuple, Optional
from ..core.particles import ParticleArrays
from ..core.kernel_vectorized import CubicSplineKernel
from .materials import MaterialDatabase, MaterialType


def compute_heat_conduction_vectorized(particles: ParticleArrays, kernel: CubicSplineKernel,
                                      material_db: MaterialDatabase, n_active: int) -> np.ndarray:
    """Compute SPH heat conduction with variable thermal properties.
    
    Uses the Cleary & Monaghan (1999) formulation for variable conductivity.
    
    Args:
        particles: Particle arrays with temperature
        kernel: SPH kernel
        material_db: Material properties database
        n_active: Number of active particles
        
    Returns:
        dT/dt array for temperature evolution
    """
    # Get thermal properties for all particles
    k_array, cp_array = material_db.get_thermal_properties(particles.material_id[:n_active])
    
    # Initialize temperature rate
    dT_dt = np.zeros(n_active, dtype=np.float32)
    
    # Process each particle
    for i in range(n_active):
        n_neighbors = particles.neighbor_count[i]
        if n_neighbors == 0:
            continue
        
        # Get neighbor slice
        neighbor_slice = slice(0, n_neighbors)
        neighbor_ids = particles.neighbor_ids[i, neighbor_slice]
        distances = particles.neighbor_distances[i, neighbor_slice]
        
        # Skip if too close (avoid singularities)
        valid_mask = distances > 1e-6
        if not np.any(valid_mask):
            continue
        
        # Filter neighbors
        neighbor_ids = neighbor_ids[valid_mask]
        distances = distances[valid_mask]
        
        # Position differences
        dx = particles.position_x[i] - particles.position_x[neighbor_ids]
        dy = particles.position_y[i] - particles.position_y[neighbor_ids]
        
        # Temperature differences
        dT = particles.temperature[i] - particles.temperature[neighbor_ids]
        
        # Get neighbor thermal properties
        k_j = k_array[neighbor_ids]
        cp_j = cp_array[neighbor_ids]
        
        # Harmonic mean conductivity (for material interfaces)
        k_i = k_array[i]
        k_harmonic = 2.0 * k_i * k_j / (k_i + k_j + 1e-10)
        
        # Kernel gradient
        grad_x, grad_y = kernel.gradW_vectorized(
            dx.reshape(1, -1), dy.reshape(1, -1),
            distances.reshape(1, -1),
            particles.smoothing_h[i]
        )
        
        # Radial component of gradient: (r · ∇W) / |r|
        r_dot_grad = (dx * grad_x[0] + dy * grad_y[0]) / distances
        
        # SPH heat flux
        # dT/dt = Σ (4 * m_j * k_ij * (T_i - T_j) * r·∇W) / (ρ_j * (cp_i + cp_j) * |r|)
        heat_flux = (4.0 * particles.mass[neighbor_ids] * k_harmonic * dT * r_dot_grad) / (
            particles.density[neighbor_ids] * (cp_array[i] + cp_j) * distances
        )
        
        # Sum contributions
        dT_dt[i] += np.sum(heat_flux) / cp_array[i]
    
    return dT_dt


def compute_radiative_cooling(particles: ParticleArrays, material_db: MaterialDatabase,
                             n_active: int, ambient_temp: float = 2.7) -> np.ndarray:
    """Compute radiative cooling using Stefan-Boltzmann law.
    
    Args:
        particles: Particle arrays
        material_db: Material properties
        n_active: Number of active particles
        ambient_temp: Background temperature (K)
        
    Returns:
        dT/dt from radiation
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    
    # Get material properties
    emissivity = np.zeros(n_active, dtype=np.float32)
    cp_array = np.zeros(n_active, dtype=np.float32)
    
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type
        if np.any(mask):
            props = material_db.get_properties(mat_type)
            emissivity[mask] = props.emissivity
            cp_array[mask] = props.specific_heat
    
    # Radiative heat loss: Q = ε σ A (T⁴ - T_amb⁴)
    # For particles, assume surface area proportional to h²
    surface_area = 4.0 * particles.smoothing_h[:n_active]**2
    
    # Heat loss rate
    Q_rad = emissivity * sigma * surface_area * (
        particles.temperature[:n_active]**4 - ambient_temp**4
    )
    
    # Temperature rate: dT/dt = -Q / (m * cp)
    dT_dt_rad = -Q_rad / (particles.mass[:n_active] * cp_array)
    
    return dT_dt_rad


def compute_heat_generation(particles: ParticleArrays, material_db: MaterialDatabase,
                           n_active: int) -> np.ndarray:
    """Compute internal heat generation (e.g., radioactive decay).
    
    Args:
        particles: Particle arrays
        material_db: Material properties
        n_active: Number of active particles
        
    Returns:
        dT/dt from internal heat generation
    """
    # Get heat generation rates
    heat_gen = np.zeros(n_active, dtype=np.float32)
    cp_array = np.zeros(n_active, dtype=np.float32)
    
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type
        if np.any(mask):
            props = material_db.get_properties(mat_type)
            heat_gen[mask] = props.heat_generation
            cp_array[mask] = props.specific_heat
    
    # Temperature rate: dT/dt = Q / cp
    dT_dt_gen = heat_gen / cp_array
    
    return dT_dt_gen


def handle_phase_transitions(particles: ParticleArrays, material_db: MaterialDatabase,
                            n_active: int, dt: float) -> int:
    """Handle material phase transitions with latent heat.
    
    Args:
        particles: Particle arrays
        material_db: Material properties
        n_active: Number of active particles
        dt: Time step
        
    Returns:
        Number of particles that transitioned
    """
    n_transitions = 0
    
    # Check each material type for transitions
    for mat_type in MaterialType:
        mask = particles.material_id[:n_active] == mat_type
        if not np.any(mask):
            continue
        
        # Get indices of particles with this material
        indices = np.where(mask)[0]
        props = material_db.get_properties(mat_type)
        
        # Check each transition rule
        for transition in props.transitions:
            # Find particles meeting transition conditions
            temp_mask = (particles.temperature[indices] >= transition.temp_min) & \
                       (particles.temperature[indices] <= transition.temp_max)
            pressure_mask = (particles.pressure[indices] >= transition.pressure_min) & \
                           (particles.pressure[indices] <= transition.pressure_max)
            
            transition_mask = temp_mask & pressure_mask
            
            if transition.water_required:
                # Check for nearby water (simplified - just check if any water exists)
                water_exists = np.any(particles.material_id[:n_active] == MaterialType.WATER)
                if not water_exists:
                    transition_mask[:] = False
            
            if not np.any(transition_mask):
                continue
            
            # Get transitioning particle indices
            transition_indices = indices[transition_mask]
            
            # Apply transition probability
            if transition.rate < 1.0:
                prob = transition.rate * dt
                random_mask = np.random.random(len(transition_indices)) < prob
                transition_indices = transition_indices[random_mask]
            
            if len(transition_indices) == 0:
                continue
            
            # Apply latent heat
            if transition.latent_heat != 0:
                target_props = material_db.get_properties(transition.target)
                # ΔT = -L / cp (negative because we remove heat for endothermic)
                dT_latent = -transition.latent_heat / target_props.specific_heat
                particles.temperature[transition_indices] += dT_latent
            
            # Change material
            particles.material_id[transition_indices] = transition.target
            
            # Update density to match new material
            new_density_ref = material_db.get_properties(transition.target).density_ref
            old_density_ref = props.density_ref
            
            # Conserve mass: ρ_new = ρ_old * (ρ_ref_new / ρ_ref_old)
            density_ratio = new_density_ref / old_density_ref
            particles.density[transition_indices] *= density_ratio
            
            n_transitions += len(transition_indices)
    
    return n_transitions


def update_temperature_full(particles: ParticleArrays, kernel: CubicSplineKernel,
                           material_db: MaterialDatabase, n_active: int, dt: float,
                           enable_radiation: bool = True,
                           enable_transitions: bool = True) -> dict:
    """Complete temperature update including all thermal processes.
    
    Args:
        particles: Particle arrays
        kernel: SPH kernel
        material_db: Material properties
        n_active: Number of active particles
        dt: Time step
        enable_radiation: Whether to include radiative cooling
        enable_transitions: Whether to allow phase transitions
        
    Returns:
        Dictionary with statistics
    """
    # Heat conduction
    dT_dt_conduction = compute_heat_conduction_vectorized(
        particles, kernel, material_db, n_active
    )
    
    # Internal heat generation
    dT_dt_generation = compute_heat_generation(particles, material_db, n_active)
    
    # Radiative cooling (optional)
    if enable_radiation:
        dT_dt_radiation = compute_radiative_cooling(particles, material_db, n_active)
    else:
        dT_dt_radiation = np.zeros(n_active)
    
    # Total temperature rate
    dT_dt_total = dT_dt_conduction + dT_dt_generation + dT_dt_radiation
    
    # Update temperature
    particles.temperature[:n_active] += dT_dt_total * dt
    
    # Clamp to physical range
    particles.temperature[:n_active] = np.maximum(particles.temperature[:n_active], 0.1)
    particles.temperature[:n_active] = np.minimum(particles.temperature[:n_active], 1e5)
    
    # Handle phase transitions
    if enable_transitions:
        n_transitions = handle_phase_transitions(particles, material_db, n_active, dt)
    else:
        n_transitions = 0
    
    # Return statistics
    return {
        'max_dT_dt': np.max(np.abs(dT_dt_total)),
        'avg_temperature': np.mean(particles.temperature[:n_active]),
        'min_temperature': np.min(particles.temperature[:n_active]),
        'max_temperature': np.max(particles.temperature[:n_active]),
        'n_transitions': n_transitions
    }