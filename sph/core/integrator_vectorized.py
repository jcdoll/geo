"""
Vectorized time integration for SPH particles.

Includes:
- Symplectic integrators (leapfrog, velocity Verlet)
- Boundary conditions
- Adaptive timestepping
"""

import numpy as np
from typing import Tuple, Optional
from .particles import ParticleArrays


def integrate_leapfrog_vectorized(particles: ParticleArrays, n_active: int, dt: float,
                                  domain_bounds: Tuple[float, float, float, float],
                                  damping: float = 0.8):
    """Fully vectorized leapfrog integration.
    
    Leapfrog is symplectic (energy-conserving) and second-order accurate.
    
    Args:
        particles: Particle arrays with forces computed
        n_active: Number of active particles
        dt: Time step
        domain_bounds: (xmin, xmax, ymin, ymax)
        damping: Velocity damping factor for wall collisions
    """
    # Update velocities (vectorized)
    particles.velocity_x[:n_active] += particles.force_x[:n_active] / particles.mass[:n_active] * dt
    particles.velocity_y[:n_active] += particles.force_y[:n_active] / particles.mass[:n_active] * dt
    
    # Update positions (vectorized)
    particles.position_x[:n_active] += particles.velocity_x[:n_active] * dt
    particles.position_y[:n_active] += particles.velocity_y[:n_active] * dt
    
    # Apply boundary conditions (vectorized)
    apply_reflective_boundaries_vectorized(particles, n_active, domain_bounds, damping)


def integrate_verlet_vectorized(particles: ParticleArrays, n_active: int, dt: float,
                               domain_bounds: Tuple[float, float, float, float],
                               damping: float = 0.8,
                               prev_accel_x: Optional[np.ndarray] = None,
                               prev_accel_y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity Verlet integration (vectorized).
    
    More accurate than leapfrog for variable time steps.
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        dt: Time step
        domain_bounds: Domain boundaries
        damping: Wall collision damping
        prev_accel_x: Previous acceleration in x (if None, uses current)
        prev_accel_y: Previous acceleration in y (if None, uses current)
    
    Returns:
        (accel_x, accel_y) current accelerations for next step
    """
    # Current acceleration
    accel_x = particles.force_x[:n_active] / particles.mass[:n_active]
    accel_y = particles.force_y[:n_active] / particles.mass[:n_active]
    
    if prev_accel_x is None:
        prev_accel_x = accel_x.copy()
        prev_accel_y = accel_y.copy()
    
    # Update positions: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
    particles.position_x[:n_active] += (particles.velocity_x[:n_active] * dt + 
                                       0.5 * prev_accel_x * dt * dt)
    particles.position_y[:n_active] += (particles.velocity_y[:n_active] * dt + 
                                       0.5 * prev_accel_y * dt * dt)
    
    # Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    particles.velocity_x[:n_active] += 0.5 * (prev_accel_x + accel_x) * dt
    particles.velocity_y[:n_active] += 0.5 * (prev_accel_y + accel_y) * dt
    
    # Apply boundaries
    apply_reflective_boundaries_vectorized(particles, n_active, domain_bounds, damping)
    
    return accel_x, accel_y


def apply_reflective_boundaries_vectorized(particles: ParticleArrays, n_active: int,
                                          bounds: Tuple[float, float, float, float],
                                          damping: float = 0.8):
    """Apply reflective boundary conditions (vectorized).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        bounds: (xmin, xmax, ymin, ymax)
        damping: Velocity reduction factor on collision
    """
    xmin, xmax, ymin, ymax = bounds
    
    # Left boundary
    mask_left = particles.position_x[:n_active] < xmin
    particles.position_x[:n_active][mask_left] = 2*xmin - particles.position_x[:n_active][mask_left]
    particles.velocity_x[:n_active][mask_left] *= -damping
    
    # Right boundary
    mask_right = particles.position_x[:n_active] > xmax
    particles.position_x[:n_active][mask_right] = 2*xmax - particles.position_x[:n_active][mask_right]
    particles.velocity_x[:n_active][mask_right] *= -damping
    
    # Bottom boundary
    mask_bottom = particles.position_y[:n_active] < ymin
    particles.position_y[:n_active][mask_bottom] = 2*ymin - particles.position_y[:n_active][mask_bottom]
    particles.velocity_y[:n_active][mask_bottom] *= -damping
    
    # Top boundary
    mask_top = particles.position_y[:n_active] > ymax
    particles.position_y[:n_active][mask_top] = 2*ymax - particles.position_y[:n_active][mask_top]
    particles.velocity_y[:n_active][mask_top] *= -damping


def apply_periodic_boundaries_vectorized(particles: ParticleArrays, n_active: int,
                                        bounds: Tuple[float, float, float, float]):
    """Apply periodic boundary conditions (vectorized).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        bounds: (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin
    
    # Wrap positions
    particles.position_x[:n_active] = xmin + np.fmod(particles.position_x[:n_active] - xmin + width, width)
    particles.position_y[:n_active] = ymin + np.fmod(particles.position_y[:n_active] - ymin + height, height)


def compute_adaptive_timestep(particles: ParticleArrays, n_active: int,
                             cfl_factor: float = 0.3,
                             force_factor: float = 0.25,
                             visc_factor: float = 0.125,
                             h_min: Optional[float] = None) -> float:
    """Compute adaptive timestep based on CFL and stability conditions.
    
    dt = min(dt_cfl, dt_force, dt_visc)
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        cfl_factor: CFL safety factor (< 1)
        force_factor: Force-based timestep factor
        visc_factor: Viscous timestep factor
        h_min: Minimum smoothing length (if None, uses min from particles)
    
    Returns:
        Safe timestep
    """
    if h_min is None:
        h_min = np.min(particles.smoothing_h[:n_active])
    
    # Maximum velocity
    v_max = np.sqrt(np.max(particles.velocity_x[:n_active]**2 + 
                           particles.velocity_y[:n_active]**2))
    
    # CFL condition: dt < CFL * h / v_max
    dt_cfl = cfl_factor * h_min / (v_max + 1e-6)
    
    # Force condition: dt < sqrt(h / a_max)
    a_max = np.sqrt(np.max(particles.force_x[:n_active]**2 + 
                          particles.force_y[:n_active]**2) / 
                    np.min(particles.mass[:n_active]))
    dt_force = force_factor * np.sqrt(h_min / (a_max + 1e-6))
    
    # Viscous condition: dt < h² / (kinematic viscosity)
    # Assuming kinematic viscosity ~ 0.01 m²/s for water
    nu_max = 0.01
    dt_visc = visc_factor * h_min**2 / nu_max
    
    # Take minimum
    dt = min(dt_cfl, dt_force, dt_visc)
    
    # Clamp to reasonable range
    dt = np.clip(dt, 1e-6, 0.01)
    
    return dt


def integrate_temperature_vectorized(particles: ParticleArrays, n_active: int,
                                    dT_dt: np.ndarray, dt: float):
    """Update temperature using forward Euler (vectorized).
    
    Args:
        particles: Particle arrays
        n_active: Number of active particles
        dT_dt: Temperature rate of change
        dt: Time step
    """
    particles.temperature[:n_active] += dT_dt[:n_active] * dt
    
    # Clamp to physical range
    particles.temperature[:n_active] = np.maximum(particles.temperature[:n_active], 0.0)