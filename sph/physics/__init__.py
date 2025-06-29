"""Physics modules for SPH: forces, materials, thermal, and gravity."""

from .density_vectorized import (
    compute_density_vectorized,
    compute_density_vectorized_batch,
    compute_density_continuity_vectorized
)
from .forces_vectorized import (
    compute_forces_vectorized,
    compute_viscous_forces_vectorized,
    tait_equation_of_state,
    ideal_gas_eos,
    compute_acceleration
)
from .materials import (
    MaterialType,
    MaterialProperties,
    MaterialDatabase,
    TransitionRule
)
from .thermal_vectorized import (
    compute_heat_conduction_vectorized,
    compute_radiative_cooling,
    compute_heat_generation,
    handle_phase_transitions,
    update_temperature_full
)
from .gravity_vectorized import (
    compute_gravity_uniform,
    compute_gravity_direct_vectorized,
    compute_gravity_direct_batched,
    compute_gravity_barnes_hut,
    compute_gravity_potential_energy,
    compute_center_of_mass
)
from .cohesion import (
    compute_cohesive_forces,
    compute_cohesive_forces_simple
)
from .repulsion import (
    compute_repulsion_forces,
    compute_boundary_force
)
from .pressure_mixed import (
    compute_pressure_mixed,
    smooth_interface_pressure,
    compute_artificial_pressure
)

__all__ = [
    # Density
    'compute_density_vectorized',
    'compute_density_vectorized_batch',
    'compute_density_continuity_vectorized',
    # Forces
    'compute_forces_vectorized',
    'compute_viscous_forces_vectorized',
    'tait_equation_of_state',
    'ideal_gas_eos',
    'compute_acceleration',
    # Materials
    'MaterialType',
    'MaterialProperties',
    'MaterialDatabase',
    'TransitionRule',
    # Thermal
    'compute_heat_conduction_vectorized',
    'compute_radiative_cooling',
    'compute_heat_generation',
    'handle_phase_transitions',
    'update_temperature_full',
    # Gravity
    'compute_gravity_uniform',
    'compute_gravity_direct_vectorized',
    'compute_gravity_direct_batched',
    'compute_gravity_barnes_hut',
    'compute_gravity_potential_energy',
    'compute_center_of_mass',
    # Cohesion
    'compute_cohesive_forces',
    'compute_cohesive_forces_simple',
    # Repulsion
    'compute_repulsion_forces',
    'compute_boundary_force',
    # Pressure
    'compute_pressure_mixed',
    'smooth_interface_pressure',
    'compute_artificial_pressure'
]