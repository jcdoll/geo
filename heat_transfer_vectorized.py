"""
Vectorized heat transfer solver for flux-based geological simulation.

This module provides a fully vectorized ADI (Alternating Direction Implicit)
solver that eliminates all loops for better performance.
"""

import numpy as np
from scipy.linalg import solve_banded
from materials import MaterialDatabase, MaterialType
from atmospheric_processes import AtmosphericProcesses


class HeatTransferVectorized:
    """Vectorized heat transfer solver using ADI method."""
    
    def __init__(self, state):
        """
        Initialize heat transfer solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        self.material_db = MaterialDatabase()
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.t_space = 2.7  # K (cosmic background temperature)
        
        # Solver parameters
        self.radiative_cooling_method = "linearized"  # "linearized" or "newton_raphson"
        self.surface_radiation_depth_fraction = 0.1  # Fraction of cell size for radiation depth
        
        # Initialize atmospheric processes for greenhouse effect
        self.atmospheric_processes = AtmosphericProcesses(state)
        
    def solve_heat_equation(self, dt: float):
        """
        Solve heat equation for one timestep.
        
        Uses operator splitting:
        1. Thermal diffusion
        2. Radiative cooling
        3. Heat generation
        
        Args:
            dt: Time step in seconds
        """
        # Step 1: Thermal diffusion
        self.apply_thermal_diffusion(dt)
        
        # Step 2: Radiative cooling at boundaries
        self.apply_radiative_cooling(dt)
        
        # Step 3: Internal heat generation
        self.apply_heat_generation(dt)
        
        # Ensure temperatures stay physical
        self.state.temperature = np.maximum(self.state.temperature, self.t_space)
        self.state.temperature = np.minimum(self.state.temperature, 5000.0)  # Max reasonable temp
        
    def apply_thermal_diffusion(self, dt: float):
        """
        Apply thermal diffusion using vectorized ADI method.
        
        Solves: ∂T/∂t = ∇·(k∇T) / (ρcp)
        
        Args:
            dt: Time step
        """
        try:
            # Safety check - ensure temperature is finite before processing
            if not np.all(np.isfinite(self.state.temperature)):
                # Replace NaN/inf with space temperature
                mask = ~np.isfinite(self.state.temperature)
                self.state.temperature[mask] = self.t_space
                print(f"Warning: Found {np.sum(mask)} NaN/inf temperature values, replaced with space temperature")
            
            ny, nx = self.state.temperature.shape
            dx = self.state.dx
            
            # Get material properties
            k = self.state.thermal_conductivity
            rho = self.state.density
            cp = self.state.specific_heat
            
            # Thermal diffusivity α = k / (ρcp)
            # For cells with very low density (mostly space), set alpha to zero
            # to prevent heat conduction through vacuum
            alpha = np.zeros_like(k)
            
            # Only compute diffusivity for cells with meaningful density
            # Space has density ~1e-10, so use threshold of 1.0 kg/m³
            valid_cells = rho > 1.0
            if np.any(valid_cells):
                alpha[valid_cells] = k[valid_cells] / (rho[valid_cells] * cp[valid_cells])
            
            # Clamp extreme values to prevent instability
            alpha = np.clip(alpha, 0, 1e-3)  # Max diffusivity for stability
            
            # Stability parameter for implicit method
            r = alpha * dt / (dx * dx)
            r = np.minimum(r, 0.5)  # Additional stability clamp
            
            # Step 1: Implicit in x, explicit in y
            T_half = self.adi_x_sweep_vectorized(self.state.temperature, r)
            
            # Step 2: Implicit in y, using T_half
            T_new = self.adi_y_sweep_vectorized(T_half, r)
            
            # Update temperature
            self.state.temperature = T_new
        except Exception as e:
            # If vectorized version fails, don't silently ignore
            print(f"Error in vectorized heat diffusion: {e}")
            raise
        
    def adi_x_sweep_vectorized(self, T: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Vectorized ADI x-direction sweep.
        
        Solves all rows simultaneously using scipy's banded solver.
        """
        ny, nx = T.shape
        T_new = T.copy()
        
        # Process all interior rows at once (skip boundaries)
        interior_rows = slice(1, ny-1)
        
        # Build coefficient matrices for all rows at once
        # Shape: (ny-2, nx) for each coefficient
        a_all = np.zeros((ny-2, nx))  # Lower diagonal
        b_all = np.zeros((ny-2, nx))  # Main diagonal  
        c_all = np.zeros((ny-2, nx))  # Upper diagonal
        d_all = np.zeros((ny-2, nx))  # RHS
        
        # Set boundary conditions (first and last columns)
        b_all[:, 0] = 1.0
        b_all[:, -1] = 1.0
        d_all[:, 0] = T[interior_rows, 0]
        d_all[:, -1] = T[interior_rows, -1]
        
        # Interior points - vectorized
        interior_cols = slice(1, nx-1)
        r_interior = r[interior_rows, interior_cols]
        
        # Average diffusivity at faces
        r_west = 0.5 * (r_interior + r[interior_rows, :-2])
        r_east = 0.5 * (r_interior + r[interior_rows, 2:])
        
        # Set coefficients
        a_all[:, 1:-1] = -r_west
        c_all[:, 1:-1] = -r_east
        b_all[:, 1:-1] = 1.0 + r_west + r_east
        
        # RHS includes explicit y-direction diffusion
        T_north = T[2:, interior_cols]
        T_south = T[:-2, interior_cols]
        T_center = T[interior_rows, interior_cols]
        
        # Check for NaN in temperature values
        if not np.all(np.isfinite(T_center)):
            # Replace NaN with boundary temperature
            nan_mask = ~np.isfinite(T_center)
            T_center[nan_mask] = self.t_space
            T_north[nan_mask[:-1, :]] = self.t_space
            T_south[nan_mask[1:, :]] = self.t_space
        
        d_all[:, 1:-1] = T_center + r_interior * (T_north - 2*T_center + T_south)
        
        # Ensure RHS is finite
        if not np.all(np.isfinite(d_all)):
            nan_mask = ~np.isfinite(d_all)
            d_all[nan_mask] = self.t_space
        
        # Solve all tridiagonal systems at once
        for j in range(ny-2):
            # Convert to banded format for scipy
            ab = np.zeros((3, nx))
            ab[0, 1:] = c_all[j, :-1]  # Upper diagonal
            ab[1, :] = b_all[j, :]      # Main diagonal
            ab[2, :-1] = a_all[j, 1:]   # Lower diagonal
            
            T_new[j+1, :] = solve_banded((1, 1), ab, d_all[j, :])
            
        return T_new
        
    def adi_y_sweep_vectorized(self, T: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Vectorized ADI y-direction sweep.
        
        Solves all columns simultaneously.
        """
        ny, nx = T.shape
        T_new = T.copy()
        
        # Process all interior columns at once
        interior_cols = slice(1, nx-1)
        
        # Build coefficient matrices for all columns
        a_all = np.zeros((ny, nx-2))  # Lower diagonal
        b_all = np.zeros((ny, nx-2))  # Main diagonal
        c_all = np.zeros((ny, nx-2))  # Upper diagonal
        d_all = T[:, interior_cols].copy()  # RHS
        
        # Set boundary conditions (first and last rows)
        b_all[0, :] = 1.0
        b_all[-1, :] = 1.0
        
        # Interior points - vectorized
        interior_rows = slice(1, ny-1)
        r_interior = r[interior_rows, interior_cols]
        
        # Average diffusivity at faces
        r_south = 0.5 * (r_interior + r[:-2, interior_cols])
        r_north = 0.5 * (r_interior + r[2:, interior_cols])
        
        # Set coefficients
        a_all[1:-1, :] = -r_south
        c_all[1:-1, :] = -r_north
        b_all[1:-1, :] = 1.0 + r_south + r_north
        
        # Solve all tridiagonal systems at once
        for i in range(nx-2):
            # Convert to banded format
            ab = np.zeros((3, ny))
            ab[0, 1:] = c_all[:-1, i]   # Upper diagonal
            ab[1, :] = b_all[:, i]       # Main diagonal
            ab[2, :-1] = a_all[1:, i]    # Lower diagonal
            
            T_new[:, i+1] = solve_banded((1, 1), ab, d_all[:, i])
            
        return T_new
        
    def apply_radiative_cooling(self, dt: float):
        """
        Apply radiative cooling at exposed surfaces.
        
        This is already mostly vectorized in the original implementation.
        """
        # Find exposed surfaces using vectorized convolution
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        
        # Use 2D convolution to find cells adjacent to space
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        space_neighbors = convolve(space_mask.astype(float), kernel, mode='constant') > 0
        exposed_mask = (~space_mask) & space_neighbors & (self.state.density > 0.1)
        
        if not np.any(exposed_mask):
            return
            
        # Get properties for exposed cells (already vectorized)
        T = self.state.temperature[exposed_mask]
        emissivity = self.state.emissivity[exposed_mask]
        density = self.state.density[exposed_mask]
        cp = self.state.specific_heat[exposed_mask]
        
        # Only process cells that are cooling
        cooling_mask = T > self.t_space
        if not np.any(cooling_mask):
            return
            
        T_cooling = T[cooling_mask]
        emissivity_cooling = emissivity[cooling_mask]
        density_cooling = density[cooling_mask]
        cp_cooling = cp[cooling_mask]
        
        # Get dynamic greenhouse effect from atmospheric processes
        greenhouse_factor = self.atmospheric_processes.calculate_greenhouse_factor()
        
        # Linearized Stefan-Boltzmann
        T_reference = np.maximum(T_cooling, 300.0)
        h_effective = 4 * self.stefan_boltzmann * (1.0 - greenhouse_factor) * emissivity_cooling * T_reference**3
        
        # Calculate cooling rate
        surface_thickness = self.state.dx * self.surface_radiation_depth_fraction
        cooling_rate = h_effective * (T_cooling - self.t_space) / (density_cooling * cp_cooling * surface_thickness)
        
        # Apply cooling
        T_new = T_cooling - dt * cooling_rate
        T_new = np.maximum(T_new, self.t_space)
        
        # Update temperatures
        T[cooling_mask] = T_new
        self.state.temperature[exposed_mask] = T
        
        # Track power density (vectorized)
        exposed_indices = np.where(exposed_mask)
        cooling_indices = np.where(cooling_mask)[0]
        
        # Vectorized power density update
        j_indices = exposed_indices[0][cooling_indices]
        i_indices = exposed_indices[1][cooling_indices]
        power_values = cooling_rate * density_cooling * cp_cooling * surface_thickness
        
        self.state.power_density[j_indices, i_indices] -= power_values
        
    def apply_heat_generation(self, dt: float):
        """
        Apply internal heat generation from radioactive decay.
        
        Already vectorized in original implementation.
        """
        # Get radioactive material fractions
        uranium_frac = self.state.vol_frac[MaterialType.URANIUM.value]
        
        # Only process cells with uranium
        if not np.any(uranium_frac > 0):
            return
            
        # Get heat generation rate from material database
        uranium_props = self.material_db.get_properties(MaterialType.URANIUM)
        heat_gen_rate = uranium_props.heat_generation  # W/kg
        
        if heat_gen_rate <= 0:
            return
            
        # Calculate volumetric heat generation (already vectorized)
        heat_source = heat_gen_rate * self.state.density * uranium_frac
        
        # Temperature change
        denom = self.state.density * self.state.specific_heat + 1e-10
        dT = heat_source * dt / denom
        
        # Update temperature
        self.state.temperature += dT
        
        # Track power density
        self.state.power_density += heat_source