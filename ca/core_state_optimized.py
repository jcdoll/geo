"""
Optimized time series recording for CoreState.
"""

import numpy as np
from collections import deque
from typing import Dict, Any
try:
    from .materials import MaterialType
except ImportError:
    from materials import MaterialType  # type: ignore


def create_optimized_time_series_recorder(sim):
    """Create an optimized time series recorder for the simulation.
    
    Key optimizations:
    1. Remove unused planet_radius calculation (saves distance calc for every cell)
    2. Use deque with maxlen instead of list trimming
    3. Cache material counts between frames if materials haven't changed
    4. Use pre-allocated arrays for statistics
    """
    
    # Use deque with automatic size limiting
    max_length = sim.max_time_series_length
    
    # Replace lists with deques that auto-trim
    time_series_data = {
        'time': deque(maxlen=max_length),
        'avg_temperature': deque(maxlen=max_length),
        'max_temperature': deque(maxlen=max_length),
        'min_temperature': deque(maxlen=max_length),
        'avg_pressure': deque(maxlen=max_length),
        'max_pressure': deque(maxlen=max_length),
        'min_pressure': deque(maxlen=max_length),
        'thermal_flux_solar': deque(maxlen=max_length),
        'thermal_flux_radiative': deque(maxlen=max_length),
        'thermal_flux_internal': deque(maxlen=max_length),
        'thermal_flux_net': deque(maxlen=max_length),
        'material_counts': deque(maxlen=max_length),
        'center_of_mass_x': deque(maxlen=max_length),
        'center_of_mass_y': deque(maxlen=max_length),
        'atmospheric_mass': deque(maxlen=max_length),
        'total_energy': deque(maxlen=max_length),
    }
    
    # Cache for material counts
    material_cache = {
        'last_hash': None,
        'counts': None
    }
    
    # Pre-allocate work arrays
    work_arrays = {
        'temps_buffer': np.zeros(sim.material_types.size, dtype=np.float32),
        'press_buffer': np.zeros(sim.material_types.size, dtype=np.float32),
        'dens_buffer': np.zeros(sim.material_types.size, dtype=np.float32),
        'spec_heat_buffer': np.zeros(sim.material_types.size, dtype=np.float32),
    }
    
    def record_time_series_optimized():
        """Optimized time series recording."""
        # Skip if time hasn't advanced
        if len(time_series_data['time']) > 0 and sim.time == time_series_data['time'][-1]:
            return
        
        # Get non-space mask once
        non_space_mask = sim.material_types != MaterialType.SPACE
        has_matter = np.any(non_space_mask)
        
        if has_matter:
            # Flatten arrays and extract non-space data efficiently
            n_non_space = np.count_nonzero(non_space_mask)
            
            # Use pre-allocated buffers (resize if needed)
            if n_non_space > work_arrays['temps_buffer'].size:
                for key in work_arrays:
                    work_arrays[key] = np.zeros(n_non_space, dtype=np.float32)
            
            # Extract data using boolean indexing (faster than fancy indexing)
            temps = work_arrays['temps_buffer'][:n_non_space]
            press = work_arrays['press_buffer'][:n_non_space]
            dens = work_arrays['dens_buffer'][:n_non_space]
            spec_heat = work_arrays['spec_heat_buffer'][:n_non_space]
            
            np.compress(non_space_mask.ravel(), sim.temperature.ravel(), out=temps)
            np.compress(non_space_mask.ravel(), sim.pressure.ravel(), out=press)
            np.compress(non_space_mask.ravel(), sim.density.ravel(), out=dens)
            np.compress(non_space_mask.ravel(), sim.specific_heat.ravel(), out=spec_heat)
            
            # Statistics - all vectorized
            avg_temp = float(np.mean(temps))
            max_temp = float(np.max(temps))
            min_temp = float(np.min(temps))
            
            avg_pressure = float(np.mean(press))
            max_pressure = float(np.max(press))
            min_pressure = float(np.min(press))
            
            # Thermal energy
            cell_volume = sim.cell_size ** 2 * sim.cell_depth
            thermal_energy = np.sum(dens * spec_heat * temps) * cell_volume
            total_energy = float(thermal_energy)
            
        else:
            # No matter - use defaults
            avg_temp = max_temp = min_temp = sim.space_temperature
            avg_pressure = max_pressure = min_pressure = 0.0
            total_energy = 0.0
        
        # Material counts - cache if unchanged
        mat_hash = hash(sim.material_types.tobytes())
        if material_cache['last_hash'] == mat_hash:
            material_counts = material_cache['counts']
        else:
            # Only recalculate if materials changed
            unique_materials, counts = np.unique(sim.material_types, return_counts=True)
            material_counts = {mat.value: int(count) for mat, count in zip(unique_materials, counts)}
            material_cache['last_hash'] = mat_hash
            material_cache['counts'] = material_counts
        
        # Atmospheric mass - vectorized
        is_atmospheric = (sim.material_types == MaterialType.AIR) | (sim.material_types == MaterialType.WATER_VAPOR)
        atmospheric_mass = float(np.sum(sim.density[is_atmospheric]))
        
        # Append data (deque handles size limiting automatically)
        time_series_data['time'].append(sim.time)
        time_series_data['avg_temperature'].append(avg_temp)
        time_series_data['max_temperature'].append(max_temp)
        time_series_data['min_temperature'].append(min_temp)
        time_series_data['avg_pressure'].append(avg_pressure)
        time_series_data['max_pressure'].append(max_pressure)
        time_series_data['min_pressure'].append(min_pressure)
        time_series_data['thermal_flux_solar'].append(sim.thermal_fluxes.get('solar_input', 0.0))
        time_series_data['thermal_flux_radiative'].append(sim.thermal_fluxes.get('radiative_output', 0.0))
        time_series_data['thermal_flux_internal'].append(sim.thermal_fluxes.get('internal_heating', 0.0))
        time_series_data['thermal_flux_net'].append(sim.thermal_fluxes.get('net_flux', 0.0))
        time_series_data['material_counts'].append(material_counts)
        time_series_data['center_of_mass_x'].append(sim.center_of_mass[0])
        time_series_data['center_of_mass_y'].append(sim.center_of_mass[1])
        time_series_data['atmospheric_mass'].append(atmospheric_mass)
        time_series_data['total_energy'].append(total_energy)
    
    return record_time_series_optimized, time_series_data


def patch_core_state_recording(sim):
    """Patch the CoreState instance with optimized recording."""
    recorder, time_series = create_optimized_time_series_recorder(sim)
    
    # Replace the method
    sim._record_time_series_data = recorder
    
    # Replace time series data (convert existing if needed)
    if hasattr(sim, 'time_series_data'):
        # Copy existing data to new deques
        for key in time_series:
            if key in sim.time_series_data and key != 'planet_radius':
                existing = sim.time_series_data[key]
                if isinstance(existing, list):
                    # Convert last N items to deque
                    time_series[key].extend(existing[-sim.max_time_series_length:])
    
    sim.time_series_data = time_series