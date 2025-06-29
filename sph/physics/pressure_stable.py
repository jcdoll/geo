"""
Stable pressure calculation for SPH.

The standard Tait equation can produce extreme pressures that destabilize
the simulation. This module provides more stable alternatives.
"""

import numpy as np
from typing import Optional


def compute_pressure_stable(density: np.ndarray, 
                          density_ref: np.ndarray,
                          bulk_modulus: np.ndarray,
                          gamma: float = 7.0,
                          max_compression: float = 2.0,
                          max_expansion: float = 0.5) -> np.ndarray:
    """
    Compute pressure with stability limits.
    
    Uses modified Tait equation with:
    - Clamped density ratios to prevent extreme pressures
    - Optional pressure smoothing
    - Artificial pressure to prevent tensile instability
    
    Args:
        density: Current density array
        density_ref: Reference density array
        bulk_modulus: Bulk modulus array (reduced for stability)
        gamma: Tait equation exponent (default 7)
        max_compression: Maximum density/density_ref ratio
        max_expansion: Minimum density/density_ref ratio
        
    Returns:
        Pressure array
    """
    # Clamp density ratios to reasonable range
    ratio = density / density_ref
    ratio = np.clip(ratio, max_expansion, max_compression)
    
    # Modified Tait equation
    # P = B * ((ρ/ρ₀)^γ - 1)
    # But we use reduced bulk modulus for stability
    B_stable = bulk_modulus * 0.01  # Reduce by 100x for stability
    
    pressure = B_stable * (ratio**gamma - 1.0)
    
    # Add small artificial pressure to prevent tensile instability
    # This helps particles maintain spacing
    artificial_pressure = 0.01 * B_stable * ratio
    pressure += artificial_pressure
    
    # Clamp extreme pressures
    max_pressure = 10.0 * B_stable  # Maximum ~10x bulk modulus
    pressure = np.clip(pressure, -B_stable, max_pressure)
    
    return pressure


def compute_pressure_weakly_compressible(density: np.ndarray,
                                       density_ref: np.ndarray, 
                                       sound_speed: float = 100.0) -> np.ndarray:
    """
    Weakly compressible pressure for low Mach number flows.
    
    Uses: P = c²(ρ - ρ₀)
    
    This is more stable than Tait equation for slow flows.
    
    Args:
        density: Current density
        density_ref: Reference density
        sound_speed: Artificial sound speed (m/s)
        
    Returns:
        Pressure array
    """
    return sound_speed**2 * (density - density_ref)


def compute_pressure_ideal_gas(density: np.ndarray,
                             temperature: np.ndarray,
                             gas_constant: float = 287.0) -> np.ndarray:
    """
    Ideal gas pressure for gas/atmosphere particles.
    
    P = ρRT
    
    Args:
        density: Density array
        temperature: Temperature array
        gas_constant: Specific gas constant (J/kg/K)
        
    Returns:
        Pressure array
    """
    return density * gas_constant * temperature


def get_stable_bulk_modulus(material_type: int) -> float:
    """
    Get reduced bulk modulus for stable SPH simulation.
    
    Real bulk moduli are too stiff for explicit time integration.
    We use artificially reduced values.
    
    Args:
        material_type: Material type enum value
        
    Returns:
        Stable bulk modulus in Pa
    """
    from .materials import MaterialType
    
    # Map real bulk moduli to stable values
    # Real values cause timestep restrictions that are too severe
    stable_moduli = {
        MaterialType.WATER.value: 2.2e7,      # Real: 2.2e9
        MaterialType.WATER_VAPOR.value: 1.0e5, # Gas
        MaterialType.ICE.value: 9.0e7,        # Real: 9.0e9
        MaterialType.ROCK.value: 5.0e8,       # Generic rock
        MaterialType.SAND.value: 2.0e8,       # Loose material
        MaterialType.MAGMA.value: 1.0e7,      # Real: 1.0e9
        MaterialType.URANIUM.value: 1.0e9,    # Real: 1.0e11
        MaterialType.AIR.value: 1.0e5,        # Real: 1.0e5
        MaterialType.SPACE.value: 1.0e3,      # Very soft
    }
    
    return stable_moduli.get(material_type, 1.0e8)