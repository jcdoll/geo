"""
Unified heat transfer solver supporting both ADI and multigrid methods.
Consolidated from multiple implementations with space temperature forcing and atmospheric smoothing.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy import ndimage
from typing import Optional
from materials import MaterialDatabase, MaterialType
from atmospheric_processes import AtmosphericProcesses
from multigrid import solve_mac_poisson_vectorized, BoundaryCondition


class HeatTransfer:
    """Heat transfer solver with support for ADI and multigrid methods."""
    
    def __init__(self, state, solver_method: str = "adi"):
        """
        Initialize heat transfer solver.
        
        Args:
            state: FluxState instance
            solver_method: "adi" or "multigrid" (default: "adi")
        """
        self.state = state
        self.material_db = MaterialDatabase()
        self.solver_method = solver_method
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.t_space = 2.7  # K (cosmic background temperature)
        
        # Solver parameters
        self.radiative_cooling_method = "linearized"
        self.surface_radiation_depth_fraction = 0.1
        
        # Initialize atmospheric processes for greenhouse effect
        self.atmospheric_processes = AtmosphericProcesses(state)
        
        # Cache for multigrid warm start
        self.prev_temperature = None
        
    def solve_heat_equation(self, dt: float):
        """
        Solve heat equation for one timestep with stability checks.
        
        Args:
            dt: Time step in seconds
        """
        # Update material properties if using multigrid
        if self.solver_method == "multigrid":
            self.state.update_mixture_properties(self.material_db)
            
            # Skip if no material present
            if np.max(self.state.density) < 1e-3:
                return
        
        # Step 1: Thermal diffusion
        if self.solver_method == "adi":
            self.apply_thermal_diffusion(dt)
        else:  # multigrid
            self._solve_diffusion_multigrid(dt)
        
        # Step 2: Radiative cooling at boundaries
        if self.solver_method == "adi":
            self.apply_radiative_cooling(dt)
        else:  # multigrid uses different radiation method
            self._apply_radiation_boundary_multigrid(dt)
        
        # Step 3: Internal heat generation
        if self.solver_method == "adi":
            self.apply_heat_generation(dt)
        else:  # multigrid
            self._apply_heat_sources_multigrid(dt)
        
        # Step 4: Force space cells to cosmic background temperature
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        self.state.temperature[space_mask] = self.t_space
        
        # Step 5: Apply smoothing to atmospheric cells to prevent instability
        air_mask = self.state.vol_frac[MaterialType.AIR.value] > 0.5
        if np.any(air_mask):
            # Simple 3x3 averaging for air cells
            kernel = np.array([[0.1, 0.1, 0.1],
                             [0.1, 0.2, 0.1],
                             [0.1, 0.1, 0.1]])
            smoothed = ndimage.convolve(self.state.temperature, kernel, mode='constant', cval=self.t_space)
            self.state.temperature[air_mask] = smoothed[air_mask]
        
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
            # Lower threshold to include air (density ~1.2 kg/m³)
            valid_cells = (rho > 0.1) & (cp > 100.0)  # Include air cells
            if np.any(valid_cells):
                alpha[valid_cells] = k[valid_cells] / (rho[valid_cells] * cp[valid_cells])
            
            # Clamp extreme values to prevent instability
            # Air has very high thermal diffusivity due to low density
            # Need much stricter limits for air cells
            air_mask = self.state.vol_frac[MaterialType.AIR.value] > 0.5
            
            # Maximum diffusivity based on stability limit
            max_alpha = 0.25 * dx * dx / dt  # CFL-like condition
            max_alpha_air = 0.05 * dx * dx / dt  # Much stricter for air
            
            # Apply different limits for air vs other materials
            alpha = np.clip(alpha, 0, max_alpha)
            if np.any(air_mask):
                alpha[air_mask] = np.clip(alpha[air_mask], 0, max_alpha_air)
            
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
        # Boundary conditions: Neumann (zero flux) or fixed temperature for space
        a_all = np.zeros((ny-2, nx))  # Lower diagonal
        b_all = np.ones((ny-2, nx))   # Main diagonal
        c_all = np.zeros((ny-2, nx))  # Upper diagonal
        d_all = np.zeros((ny-2, nx))  # RHS
        
        # Interior points
        interior_rows = slice(1, ny-1)
        interior_cols = slice(1, nx-1)
        r_interior = r[interior_rows, interior_cols]
        
        # Get space mask for interior region
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        space_interior = space_mask[interior_rows, interior_cols]
        
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
        
        # For space cells, set identity (no diffusion)
        if np.any(space_interior):
            a_all[:, 1:-1][space_interior] = 0.0
            c_all[:, 1:-1][space_interior] = 0.0
            b_all[:, 1:-1][space_interior] = 1.0
        
        # Check for space boundaries and apply fixed temperature
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        for j in range(ny-2):
            if space_mask[j+1, 0]:  # West boundary
                b_all[j, 0] = 1.0
                a_all[j, 0] = 0.0
                c_all[j, 0] = 0.0
                d_all[j, 0] = self.t_space
            if space_mask[j+1, -1]:  # East boundary
                b_all[j, -1] = 1.0
                a_all[j, -1] = 0.0
                c_all[j, -1] = 0.0
                d_all[j, -1] = self.t_space
        
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
        
        # For space cells, RHS is just the space temperature
        if np.any(space_interior):
            d_all[:, 1:-1][space_interior] = self.t_space
        
        
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
        
        # Get space mask for interior region
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        space_interior = space_mask[interior_rows, interior_cols]
        
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
        
        # For space cells, set identity (no diffusion)
        if np.any(space_interior):
            a_all[1:-1, :][space_interior] = 0.0
            c_all[1:-1, :][space_interior] = 0.0
            b_all[1:-1, :][space_interior] = 1.0
        
        # Check for space boundaries
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        for i in range(nx-2):
            if space_mask[0, i+1]:  # South boundary
                b_all[0, i] = 1.0
                a_all[0, i] = 0.0
                c_all[0, i] = 0.0
                d_all[0, i] = self.t_space
            if space_mask[-1, i+1]:  # North boundary
                b_all[-1, i] = 1.0
                a_all[-1, i] = 0.0
                c_all[-1, i] = 0.0
                d_all[-1, i] = self.t_space
        
        # RHS with x-direction diffusion
        T_east = T[interior_rows, 2:]
        T_west = T[interior_rows, :-2]
        T_center = T[interior_rows, interior_cols]
        
        # Limit diffusion contribution
        x_diff = r_interior * (T_east - 2*T_center + T_west)
        x_diff = np.clip(x_diff, -10.0, 10.0)
        d_all[1:-1, :] = T_center + x_diff
        
        # For space cells, RHS is just the space temperature
        if np.any(space_interior):
            d_all[1:-1, :][space_interior] = self.t_space
        
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
        # Find exposed surfaces - following CA implementation
        space_mask = self.state.vol_frac[MaterialType.SPACE.value] > 0.9
        
        # Identify atmosphere cells
        atmosphere_mask = ((self.state.vol_frac[MaterialType.AIR.value] > 0.5) | 
                          (self.state.vol_frac[MaterialType.WATER_VAPOR.value] > 0.5))
        
        # Use convolution to find cells adjacent to space
        from scipy.ndimage import convolve, binary_dilation
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Outer atmosphere: atmospheric cells adjacent to space
        space_neighbors = binary_dilation(space_mask, structure=kernel)
        outer_atmo_mask = atmosphere_mask & space_neighbors
        
        # Surface solids: non-atmospheric, non-space cells adjacent to outer atmosphere or space
        non_space_mask = ~space_mask
        solid_mask = non_space_mask & ~atmosphere_mask & (self.state.density > 0.1)
        surface_candidates = binary_dilation(outer_atmo_mask | space_mask, structure=kernel)
        surface_solid_mask = surface_candidates & solid_mask
        
        # Combine: both outer atmosphere AND surface solids can radiate
        exposed_mask = outer_atmo_mask | surface_solid_mask
        
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
            
    # ========== Multigrid-specific methods ==========
    
    def _solve_diffusion_multigrid(self, dt: float):
        """
        Solve thermal diffusion implicitly using multigrid.
        
        The backward Euler discretization gives:
        (I - dt*L)T^{n+1} = T^n
        
        Where L is the diffusion operator: L = ∇·(α∇)
        
        We solve this iteratively using multigrid-preconditioned iteration.
        """
        st = self.state
        ny, nx = st.temperature.shape
        dx = st.dx
        
        # Compute thermal diffusivity α = k/(ρ*C_p)
        k = st.thermal_conductivity
        rho = st.density
        cp = st.specific_heat
        
        # Avoid division by zero
        valid_mask = (rho > 1e-3) & (cp > 1e-3)
        alpha = np.zeros_like(k)
        alpha[valid_mask] = k[valid_mask] / (rho[valid_mask] * cp[valid_mask])
        
        # Initial guess (use previous temperature if available)
        if self.prev_temperature is not None and self.prev_temperature.shape == st.temperature.shape:
            T = self.prev_temperature.copy()
        else:
            T = st.temperature.copy()
            
        T_old = st.temperature.copy()
        
        # Fixed-point iteration with multigrid preconditioning
        max_iter = 5  # Usually converges in 2-3 iterations
        omega = 0.8   # Under-relaxation for stability
        
        for iter in range(max_iter):
            # Compute diffusion term: ∇·(α∇T)
            diffusion = self._compute_diffusion(T, alpha, dx)
            
            # Compute residual: R = T_old + dt*diffusion - T
            residual = T_old + dt * diffusion - T
            
            # Check convergence
            res_norm = np.linalg.norm(residual) / (ny * nx)
            if res_norm < 1e-6:
                break
                
            # Solve for correction using multigrid
            # Create face-centered diffusion coefficients
            alpha_x = np.zeros((ny, nx+1))
            alpha_x[:, 1:-1] = 2.0 / (1.0/(alpha[:, :-1] + 1e-10) + 1.0/(alpha[:, 1:] + 1e-10))
            alpha_x[:, 0] = alpha[:, 0]
            alpha_x[:, -1] = alpha[:, -1]
            
            alpha_y = np.zeros((ny+1, nx))
            alpha_y[1:-1, :] = 2.0 / (1.0/(alpha[:-1, :] + 1e-10) + 1.0/(alpha[1:, :] + 1e-10))
            alpha_y[0, :] = alpha[0, :]
            alpha_y[-1, :] = alpha[-1, :]
            
            # Clamp diffusion coefficients to reasonable range
            alpha_x = np.clip(alpha_x, 0, 1e3)
            alpha_y = np.clip(alpha_y, 0, 1e3)
            
            # RHS for multigrid
            rhs = -residual / dt
            
            # Zero RHS in space regions (no heat diffusion in vacuum)
            space_mask = st.density < 1e-3
            rhs[space_mask] = 0.0
            
            # Clamp RHS to prevent numerical overflow in multigrid
            rhs = np.clip(rhs, -1e6, 1e6)
            
            # Solve for correction
            delta_T = solve_mac_poisson_vectorized(
                rhs, alpha_x, alpha_y, dx,
                bc_type=BoundaryCondition.NEUMANN,
                tol=1e-4,
                max_cycles=20,
                verbose=False
            )
            
            # Update temperature with under-relaxation
            delta_T = np.clip(delta_T, -1000, 1000)  # Limit correction magnitude
            T += omega * delta_T
            
            # Enforce boundary conditions
            T[space_mask] = self.t_space
            T = np.clip(T, 0.1, 10000)  # Keep temperature in reasonable range
            
        # Update state temperature
        st.temperature = T
        self.prev_temperature = T.copy()
        
    def _compute_diffusion(self, T: np.ndarray, alpha: np.ndarray, dx: float) -> np.ndarray:
        """
        Compute the diffusion term ∇·(α∇T) using finite differences.
        
        Args:
            T: Temperature field
            alpha: Thermal diffusivity field
            dx: Grid spacing
            
        Returns:
            Diffusion term
        """
        ny, nx = T.shape
        diffusion = np.zeros_like(T)
        
        # X-direction diffusion
        # ∂/∂x(α ∂T/∂x)
        for j in range(ny):
            for i in range(1, nx-1):
                alpha_e = 0.5 * (alpha[j, i] + alpha[j, i+1])
                alpha_w = 0.5 * (alpha[j, i] + alpha[j, i-1])
                
                flux_e = alpha_e * (T[j, i+1] - T[j, i]) / dx
                flux_w = alpha_w * (T[j, i] - T[j, i-1]) / dx
                
                diffusion[j, i] += (flux_e - flux_w) / dx
        
        # Y-direction diffusion
        # ∂/∂y(α ∂T/∂y)
        for j in range(1, ny-1):
            for i in range(nx):
                alpha_n = 0.5 * (alpha[j, i] + alpha[j+1, i])
                alpha_s = 0.5 * (alpha[j, i] + alpha[j-1, i])
                
                flux_n = alpha_n * (T[j+1, i] - T[j, i]) / dx
                flux_s = alpha_s * (T[j, i] - T[j-1, i]) / dx
                
                diffusion[j, i] += (flux_n - flux_s) / dx
                
        return diffusion
        
    def _apply_heat_sources_multigrid(self, dt: float):
        """Apply heat generation from radioactive decay (multigrid version)."""
        st = self.state
        
        # Compute volumetric heat generation
        Q = np.zeros_like(st.temperature)
        
        for mat_idx in range(st.n_materials):
            if mat_idx == 0:  # Skip space
                continue
                
            mat_type = MaterialType(mat_idx)
            props = self.material_db.get_properties(mat_type)
            
            if props.heat_generation > 0:
                Q += st.vol_frac[mat_idx] * props.heat_generation
        
        # Add external heat sources
        Q += st.heat_source
        
        # Convert to temperature change
        valid = (st.density > 1e-3) & (st.specific_heat > 1e-3)
        dT = np.zeros_like(st.temperature)
        dT[valid] = Q[valid] / (st.density[valid] * st.specific_heat[valid])
        
        st.temperature += dt * dT
        
    def _apply_radiation_boundary_multigrid(self, dt: float):
        """Apply Stefan-Boltzmann radiation at space boundaries using implicit Newton-Raphson."""
        st = self.state
        
        # Find material-space interfaces
        space_mask = st.vol_frac[0] > 0.9
        
        # Collect radiating cells
        radiating_cells = []
        for j in range(1, st.ny-1):
            for i in range(1, st.nx-1):
                if not space_mask[j, i] and st.density[j, i] > 1e-3:
                    # Check if any neighbor is space
                    if (space_mask[j-1, i] or space_mask[j+1, i] or 
                        space_mask[j, i-1] or space_mask[j, i+1]):
                        radiating_cells.append((j, i))
        
        if not radiating_cells:
            return
            
        # Apply Newton-Raphson implicit radiation
        for j, i in radiating_cells:
            T_old = st.temperature[j, i]
            T = T_old
            emissivity = st.emissivity[j, i]
            rho_cp = st.density[j, i] * st.specific_heat[j, i]
            
            if rho_cp < 1e-3:
                continue
                
            # Newton-Raphson iteration for implicit radiation
            # Solve: T_new - T_old + dt * σ * ε * (T_new^4 - T_space^4) / (ρ * cp * dx) = 0
            alpha = self.stefan_boltzmann * emissivity * dt / (rho_cp * st.dx)
            
            for iteration in range(3):  # Usually converges quickly
                # Clamp temperature to prevent overflow
                T_clamped = min(T, 5000.0)
                T_clamped = max(T_clamped, self.t_space)
                
                # Calculate T^4 with overflow protection
                if T_clamped > 2000:
                    T4 = np.exp(4 * np.log(T_clamped))
                    T3 = np.exp(3 * np.log(T_clamped))
                else:
                    T4 = T_clamped**4
                    T3 = T_clamped**3
                    
                T_space4 = self.t_space**4
                
                # Newton-Raphson update
                f = T - T_old + alpha * (T4 - T_space4)
                df_dT = 1.0 + alpha * 4.0 * T3
                
                delta_T = -f / df_dT
                T = T + delta_T
                
                # Keep physical
                T = max(self.t_space, min(5000.0, T))
                
                # Check convergence
                if abs(delta_T) < 0.1:  # 0.1K tolerance
                    break
                    
            st.temperature[j, i] = T