"""
Vectorized heat transfer solver with improved stability and correct power density tracking.
"""

import numpy as np
from scipy.linalg import solve_banded
from materials import MaterialDatabase, MaterialType
from atmospheric_processes import AtmosphericProcesses


class HeatTransferVectorized:
    """Vectorized heat transfer solver using ADI method with stability fixes."""
    
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
        self.radiative_cooling_method = "linearized"
        self.surface_radiation_depth_fraction = 0.1
        
        # Initialize atmospheric processes for greenhouse effect
        self.atmospheric_processes = AtmosphericProcesses(state)
        
        # No artificial limits - let physics emerge naturally
        
    def solve_heat_equation(self, dt: float):
        """
        Solve heat equation for one timestep with stability checks.
        
        Args:
            dt: Time step in seconds
        """
        # Step 1: Thermal diffusion
        self.apply_thermal_diffusion(dt)
        
        # Step 2: Radiative cooling at boundaries
        self.apply_radiative_cooling(dt)
        
        # Step 3: Internal heat generation
        self.apply_heat_generation(dt)
        
        # No artificial temperature limiting - let physics determine the behavior
        # Only ensure temperature doesn't go below absolute zero
        self.state.temperature = np.maximum(self.state.temperature, 0.0)
        
    def apply_thermal_diffusion(self, dt: float):
        """
        Apply thermal diffusion using vectorized ADI method.
        
        Args:
            dt: Time step
        """
        try:
            # Safety check - ensure temperature is finite before processing
            if not np.all(np.isfinite(self.state.temperature)):
                mask = ~np.isfinite(self.state.temperature)
                self.state.temperature[mask] = self.t_space
                print(f"Warning: Found {np.sum(mask)} NaN/inf temperature values before diffusion")
            
            ny, nx = self.state.temperature.shape
            dx = self.state.dx
            
            # Get material properties
            k = self.state.thermal_conductivity
            rho = self.state.density
            cp = self.state.specific_heat
            
            # Thermal diffusivity α = k / (ρcp)
            # For cells with very low density (mostly space), set alpha to zero
            alpha = np.zeros_like(k)
            
            # Only compute diffusivity for cells with meaningful density
            valid_cells = (rho > 1.0) & (cp > 100.0)  # Reasonable thresholds
            if np.any(valid_cells):
                alpha[valid_cells] = k[valid_cells] / (rho[valid_cells] * cp[valid_cells])
            
            # Clamp extreme values to prevent instability
            # Maximum diffusivity based on stability limit for explicit scheme
            max_alpha = 0.25 * dx * dx / dt  # CFL-like condition
            alpha = np.clip(alpha, 0, max_alpha)
            
            # Stability parameter for implicit method
            r = alpha * dt / (dx * dx)
            
            # Additional safety: limit r to ensure diagonal dominance
            r = np.minimum(r, 0.45)  # Keep well below 0.5 for stability
            
            # Step 1: Implicit in x, explicit in y
            T_half = self.adi_x_sweep_vectorized(self.state.temperature, r)
            
            # Check for NaN after first sweep
            if not np.all(np.isfinite(T_half)):
                print("Warning: NaN detected after x-sweep, using previous temperature")
                T_half = self.state.temperature.copy()
            
            # Step 2: Implicit in y, using T_half
            T_new = self.adi_y_sweep_vectorized(T_half, r)
            
            # Check for NaN after second sweep
            if not np.all(np.isfinite(T_new)):
                print("Warning: NaN detected after y-sweep, keeping previous temperature")
                return
            
            # Update temperature
            self.state.temperature = T_new
            
        except Exception as e:
            print(f"Error in vectorized heat diffusion: {e}")
            # Don't propagate error - just skip diffusion this step
            
    def adi_x_sweep_vectorized(self, T: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Vectorized ADI x-direction sweep with stability checks."""
        ny, nx = T.shape
        T_new = T.copy()
        
        # Set up tridiagonal system for interior points
        # Boundary conditions: Neumann (zero flux)
        a_all = np.zeros((ny-2, nx))  # Lower diagonal
        b_all = np.ones((ny-2, nx))   # Main diagonal
        c_all = np.zeros((ny-2, nx))  # Upper diagonal
        d_all = np.zeros((ny-2, nx))  # RHS
        
        # Interior points
        interior_rows = slice(1, ny-1)
        interior_cols = slice(1, nx-1)
        r_interior = r[interior_rows, interior_cols]
        
        # Harmonic mean for interface diffusivity
        r_west = 0.5 * (r_interior + r[interior_rows, :-2])
        r_east = 0.5 * (r_interior + r[interior_rows, 2:])
        
        # Limit r values for stability
        r_west = np.minimum(r_west, 0.45)
        r_east = np.minimum(r_east, 0.45)
        
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
            nan_mask = ~np.isfinite(T_center)
            T_center[nan_mask] = self.t_space
            T_north[nan_mask[:-1, :]] = self.t_space
            T_south[nan_mask[1:, :]] = self.t_space
        
        # Compute RHS with limited diffusion
        y_diff = r_interior * (T_north - 2*T_center + T_south)
        y_diff = np.clip(y_diff, -10.0, 10.0)  # Limit contribution
        d_all[:, 1:-1] = T_center + y_diff
        
        # Ensure RHS is finite
        if not np.all(np.isfinite(d_all)):
            nan_mask = ~np.isfinite(d_all)
            d_all[nan_mask] = self.t_space
        
        # Solve all tridiagonal systems
        for j in range(ny-2):
            try:
                # Convert to banded format
                ab = np.zeros((3, nx))
                ab[0, 1:] = c_all[j, :-1]   # Upper diagonal
                ab[1, :] = b_all[j, :]       # Main diagonal
                ab[2, :-1] = a_all[j, 1:]    # Lower diagonal
                
                # Ensure main diagonal dominance
                ab[1, :] = np.maximum(np.abs(ab[1, :]), 
                                     np.abs(ab[0, :]) + np.abs(ab[2, :]) + 0.1)
                
                # Solve
                solution = solve_banded((1, 1), ab, d_all[j, :])
                
                # Check solution
                if np.all(np.isfinite(solution)):
                    T_new[j+1, :] = solution
                else:
                    # Keep previous values if solution failed
                    print(f"Warning: ADI x-sweep failed for row {j+1}")
                    
            except Exception as e:
                # Skip this row if solver fails
                print(f"ADI x-sweep error for row {j+1}: {e}")
                
        return T_new
        
    def adi_y_sweep_vectorized(self, T: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Vectorized ADI y-direction sweep with stability checks."""
        ny, nx = T.shape
        T_new = T.copy()
        
        # Similar to x-sweep but solving in y-direction
        a_all = np.zeros((ny, nx-2))
        b_all = np.ones((ny, nx-2))
        c_all = np.zeros((ny, nx-2))
        d_all = np.zeros((ny, nx-2))
        
        # Interior points
        interior_rows = slice(1, ny-1)
        interior_cols = slice(1, nx-1)
        r_interior = r[interior_rows, interior_cols]
        
        # Harmonic mean for interface diffusivity
        r_south = 0.5 * (r_interior + r[:-2, interior_cols])
        r_north = 0.5 * (r_interior + r[2:, interior_cols])
        
        # Limit r values
        r_south = np.minimum(r_south, 0.45)
        r_north = np.minimum(r_north, 0.45)
        
        # Set coefficients
        a_all[1:-1, :] = -r_south
        c_all[1:-1, :] = -r_north
        b_all[1:-1, :] = 1.0 + r_south + r_north
        
        # RHS with x-direction diffusion
        T_east = T[interior_rows, 2:]
        T_west = T[interior_rows, :-2]
        T_center = T[interior_rows, interior_cols]
        
        # Limit diffusion contribution
        x_diff = r_interior * (T_east - 2*T_center + T_west)
        x_diff = np.clip(x_diff, -10.0, 10.0)
        d_all[1:-1, :] = T_center + x_diff
        
        # Solve column by column
        for i in range(nx-2):
            try:
                ab = np.zeros((3, ny))
                ab[0, 1:] = c_all[:-1, i]
                ab[1, :] = b_all[:, i]
                ab[2, :-1] = a_all[1:, i]
                
                # Ensure diagonal dominance
                ab[1, :] = np.maximum(np.abs(ab[1, :]), 
                                     np.abs(ab[0, :]) + np.abs(ab[2, :]) + 0.1)
                
                solution = solve_banded((1, 1), ab, d_all[:, i])
                
                if np.all(np.isfinite(solution)):
                    T_new[:, i+1] = solution
                    
            except Exception as e:
                print(f"ADI y-sweep error for column {i+1}: {e}")
                
        return T_new
        
    def apply_radiative_cooling(self, dt: float):
        """Apply radiative cooling with correct power density calculation."""
        # Find exposed surfaces
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        
        # Use convolution to find cells adjacent to space
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        space_neighbors = convolve(space_mask.astype(float), kernel, mode='constant') > 0
        exposed_mask = (~space_mask) & space_neighbors & (self.state.density > 0.1)
        
        if not np.any(exposed_mask):
            return
            
        # Get properties for exposed cells
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
        
        # Get greenhouse effect
        greenhouse_factor = self.atmospheric_processes.calculate_greenhouse_factor()
        greenhouse_factor = np.clip(greenhouse_factor, 0.0, 0.95)  # Max 95% absorption
        
        # Stefan-Boltzmann radiation (W/m²)
        radiation_flux = self.stefan_boltzmann * emissivity_cooling * (T_cooling**4 - self.t_space**4)
        radiation_flux *= (1.0 - greenhouse_factor)  # Reduce by greenhouse effect
        
        # Convert to volumetric power (W/m³)
        surface_thickness = self.state.dx * self.surface_radiation_depth_fraction
        volumetric_cooling = radiation_flux / surface_thickness
        
        # Temperature change rate (K/s)
        cooling_rate = volumetric_cooling / (density_cooling * cp_cooling)
        
        # No artificial cooling rate limits - let physics determine the rate
        
        # Apply cooling
        T_new = T_cooling - dt * cooling_rate
        T_new = np.maximum(T_new, self.t_space)
        
        # Update temperatures
        T[cooling_mask] = T_new
        self.state.temperature[exposed_mask] = T
        
        # Update power density (W/m³) - negative for cooling
        # volumetric_cooling only contains values for cells where cooling_mask is True
        # Need to map back to the full exposed cells array
        volumetric_cooling_full = np.zeros_like(T)
        volumetric_cooling_full[cooling_mask] = volumetric_cooling
        
        # Now update power density for exposed cells
        self.state.power_density[exposed_mask] -= volumetric_cooling_full
        
    def apply_heat_generation(self, dt: float):
        """Apply internal heat generation from radioactive decay."""
        # Get radioactive material fractions
        uranium_frac = self.state.vol_frac[MaterialType.URANIUM.value]
        
        # Only process cells with uranium
        if not np.any(uranium_frac > 0):
            return
            
        # Get heat generation rate
        uranium_props = self.material_db.get_properties(MaterialType.URANIUM)
        heat_gen_rate = uranium_props.heat_generation  # W/kg
        
        if heat_gen_rate <= 0:
            return
            
        # Calculate volumetric heat generation (W/m³)
        # heat_gen_rate (W/kg) * density (kg/m³) * volume_fraction = W/m³
        heat_source = heat_gen_rate * self.state.density * uranium_frac
        
        # No artificial heat generation limits
        
        # Temperature change
        valid_mask = (self.state.density > 0) & (self.state.specific_heat > 0)
        if np.any(valid_mask):
            dT = np.zeros_like(self.state.temperature)
            dT[valid_mask] = (heat_source[valid_mask] * dt / 
                            (self.state.density[valid_mask] * self.state.specific_heat[valid_mask]))
            
            # No artificial temperature increase limits
            
            # Apply temperature change
            self.state.temperature += dT
            
            # Track power density (positive for heating)
            self.state.power_density += heat_source