"""
Correct multigrid solver for heat diffusion with variable coefficients.
Based on the working flux multigrid solver but adapted for CA heat equation.
"""

import numpy as np
import time
from typing import List, Tuple, Optional
from enum import Enum

try:
    from .materials import MaterialType, MaterialDatabase
    from .solar_heating import SolarHeating
except ImportError:
    from materials import MaterialType, MaterialDatabase
    from solar_heating import SolarHeating


class BoundaryCondition(Enum):
    """Boundary condition types."""
    NEUMANN = "neumann"     # ∂T/∂n = 0 (insulating)
    DIRICHLET = "dirichlet" # T = T_boundary (fixed temp)


class HeatTransferMultigridCorrect:
    """Multigrid solver for heat diffusion with variable thermal diffusivity"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Multigrid parameters (tuned for heat equation)
        self.max_levels = 4         # Number of multigrid levels
        self.pre_smooth = 2         # Pre-smoothing iterations
        self.post_smooth = 2        # Post-smoothing iterations
        self.bottom_smooth = 20     # Smoothing on coarsest grid
        self.max_cycles = 5         # Maximum V-cycles
        self.tolerance = 1e-6       # Convergence tolerance
        self.omega = 2.0/3.0        # Relaxation parameter for Jacobi
        
        # Boundary condition
        self.bc_type = BoundaryCondition.NEUMANN  # Insulating boundaries
        
        # Solar heating module
        self.solar_heating = SolarHeating(simulation)
        
        # Material database
        self.material_db = MaterialDatabase()
        
        # Cache for material properties
        self._last_material_update = -1
        
        # Timing
        self.last_multigrid_time = 0.0
        self.last_total_time = 0.0
    
    def solve_heat_diffusion(self):
        """Apply operator splitting to solve heat equation with sources."""
        total_start = time.perf_counter()
        
        # Reset power density to show instantaneous power, not accumulated
        self.sim.power_density.fill(0.0)
        
        # Get mask of non-space cells
        non_space_mask = self.sim.material_types != MaterialType.SPACE
        
        # Step 1: Pure diffusion using multigrid
        if self.sim.enable_heat_diffusion:
            working_temp, stability = self._solve_pure_diffusion_multigrid(
                self.sim.temperature, non_space_mask
            )
        else:
            working_temp = self.sim.temperature.copy()
            stability = 1.0
        
        # Step 2: Radiative cooling (if enabled)
        if self.sim.enable_radiative_cooling:
            working_temp = self._solve_radiative_cooling(working_temp, non_space_mask)
        
        # Step 3: Non-radiative heat sources (if enabled)
        working_temp = self._solve_non_radiative_sources(working_temp, non_space_mask)
        
        self.last_total_time = time.perf_counter() - total_start
        return working_temp, stability
    
    def _solve_pure_diffusion_multigrid(self, temperature: np.ndarray, 
                                       non_space_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve heat diffusion using multigrid method"""
        
        mg_start = time.perf_counter()
        
        # Get thermal diffusivity field
        thermal_diffusivity = self._compute_thermal_diffusivity(non_space_mask)
        
        # For implicit method, we solve: (I - dt*L)T_new = T_old
        # where L is the diffusion operator with variable coefficients
        
        # Use full timestep for implicit method (stable for any dt)
        dt = self.sim.dt
        dx = self.sim.cell_size
        
        # Set up the problem
        # We're solving: T_new - dt * div(α grad(T_new)) = T_old
        # Rearranged: T_new - dt/(dx²) * L(α, T_new) = T_old
        
        # Initial guess
        T_new = temperature.copy()
        T_old = temperature.copy()
        
        # Apply multigrid V-cycles
        initial_res_norm = None
        for cycle in range(self.max_cycles):
            T_prev = T_new.copy()
            
            # Compute residual: r = T_old - (T_new - dt*L(T_new))
            residual = self._compute_residual(T_new, T_old, thermal_diffusivity, 
                                            dt, dx, non_space_mask)
            
            # Check convergence
            res_norm = np.sqrt(np.mean(residual[non_space_mask]**2))
            if initial_res_norm is None:
                initial_res_norm = res_norm
                # Debug: check what the residual looks like
                if cycle == 0 and False:  # Disable debug for now
                    L_T = self._apply_diffusion_operator(T_new, thermal_diffusivity, dx, non_space_mask)
                    print(f"  Initial T range: {np.min(T_new):.1f} - {np.max(T_new):.1f}")
                    print(f"  L(T) range: {np.min(L_T):.2e} - {np.max(L_T):.2e}")
                    print(f"  dt*L(T) range: {np.min(dt*L_T):.2e} - {np.max(dt*L_T):.2e}")
            
            if res_norm < self.tolerance:
                break
            
            # V-cycle to solve: A * correction = residual
            correction = self._v_cycle(residual, thermal_diffusivity, dt, dx, 
                                     non_space_mask, level=0)
            
            # Update solution
            T_new = T_prev + correction
            
            # Apply boundary conditions
            self._apply_boundary_conditions(T_new)
        
        # Debug output - disable for production
        if False and initial_res_norm > 1e-10:
            max_alpha = np.max(thermal_diffusivity[non_space_mask])
            max_temp_change = np.max(np.abs(T_new - T_old))
            print(f"Multigrid: {cycle+1} cycles, residual {initial_res_norm:.2e} -> {res_norm:.2e}, max α={max_alpha:.2e}, max ΔT={max_temp_change:.2f}")
        
        self.last_multigrid_time = time.perf_counter() - mg_start
        
        # Implicit method is unconditionally stable
        return T_new, 1.0
    
    def _compute_thermal_diffusivity(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Compute thermal diffusivity field with enhancements"""
        
        # Get base thermal diffusivity (α = k / (ρ * cp))
        valid_thermal = (self.sim.density > 0) & (self.sim.specific_heat > 0) & \
                       (self.sim.thermal_conductivity > 0)
        
        thermal_diffusivity = np.zeros_like(self.sim.thermal_conductivity)
        thermal_diffusivity[valid_thermal] = (
            self.sim.thermal_conductivity[valid_thermal] /
            (self.sim.density[valid_thermal] * self.sim.specific_heat[valid_thermal])
        )
        
        # Enhanced atmospheric convection
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        atmospheric_cells = atmosphere_mask & valid_thermal
        if np.any(atmospheric_cells):
            thermal_diffusivity[atmospheric_cells] *= self.sim.atmospheric_diffusivity_enhancement
        
        # Enhanced diffusion at material interfaces
        interface_mask = self._get_interface_mask(non_space_mask)
        enhancement_mask = interface_mask & valid_thermal
        if np.any(enhancement_mask):
            thermal_diffusivity[enhancement_mask] *= self.sim.interface_diffusivity_enhancement
        
        # Clamp extreme values
        thermal_diffusivity = np.clip(thermal_diffusivity, 0.0, self.sim.max_thermal_diffusivity)
        
        # Set space cells to zero
        thermal_diffusivity[~non_space_mask] = 0.0
        
        return thermal_diffusivity
    
    def _compute_residual(self, T: np.ndarray, T_old: np.ndarray, alpha: np.ndarray,
                         dt: float, dx: float, mask: np.ndarray) -> np.ndarray:
        """Compute residual for the implicit heat equation"""
        
        # Apply diffusion operator
        L_T = self._apply_diffusion_operator(T, alpha, dx, mask)
        
        # Residual: r = T_old - (T - dt*L_T)
        residual = T_old - T + dt * L_T
        
        # Zero out space cells
        residual[~mask] = 0.0
        
        return residual
    
    def _apply_diffusion_operator(self, T: np.ndarray, alpha: np.ndarray, 
                                 dx: float, mask: np.ndarray) -> np.ndarray:
        """Apply variable-coefficient diffusion operator: div(α grad(T))"""
        
        ny, nx = T.shape
        dx2 = dx * dx
        
        # Result array
        L_T = np.zeros_like(T)
        
        # Interior points: use centered differences
        # L(T) = 1/dx² * [(α_E + α_C)/2 * (T_E - T_C) - (α_C + α_W)/2 * (T_C - T_W)]
        #      + 1/dy² * [(α_N + α_C)/2 * (T_N - T_C) - (α_C + α_S)/2 * (T_C - T_S)]
        
        # Get shifted arrays
        T_E = np.roll(T, -1, axis=1)  # East
        T_W = np.roll(T, 1, axis=1)   # West
        T_N = np.roll(T, -1, axis=0)  # North
        T_S = np.roll(T, 1, axis=0)   # South
        
        alpha_E = np.roll(alpha, -1, axis=1)
        alpha_W = np.roll(alpha, 1, axis=1)
        alpha_N = np.roll(alpha, -1, axis=0)
        alpha_S = np.roll(alpha, 1, axis=0)
        
        # Harmonic mean for face values (better for discontinuous coefficients)
        eps = 1e-10
        alpha_e_face = 2.0 * alpha * alpha_E / (alpha + alpha_E + eps)
        alpha_w_face = 2.0 * alpha * alpha_W / (alpha + alpha_W + eps)
        alpha_n_face = 2.0 * alpha * alpha_N / (alpha + alpha_N + eps)
        alpha_s_face = 2.0 * alpha * alpha_S / (alpha + alpha_S + eps)
        
        # Apply operator
        L_T = ((alpha_e_face * (T_E - T) - alpha_w_face * (T - T_W)) +
               (alpha_n_face * (T_N - T) - alpha_s_face * (T - T_S))) / dx2
        
        # Handle boundaries for Neumann BC
        if self.bc_type == BoundaryCondition.NEUMANN:
            # Left boundary: no flux from west
            L_T[:, 0] = ((alpha_e_face[:, 0] * (T_E[:, 0] - T[:, 0])) +
                        (alpha_n_face[:, 0] * (T_N[:, 0] - T[:, 0]) - 
                         alpha_s_face[:, 0] * (T[:, 0] - T_S[:, 0]))) / dx2
            
            # Right boundary: no flux from east
            L_T[:, -1] = ((-alpha_w_face[:, -1] * (T[:, -1] - T_W[:, -1])) +
                         (alpha_n_face[:, -1] * (T_N[:, -1] - T[:, -1]) - 
                          alpha_s_face[:, -1] * (T[:, -1] - T_S[:, -1]))) / dx2
            
            # Top boundary: no flux from north
            L_T[0, :] = ((alpha_e_face[0, :] * (T_E[0, :] - T[0, :]) - 
                         alpha_w_face[0, :] * (T[0, :] - T_W[0, :])) +
                        (-alpha_s_face[0, :] * (T[0, :] - T_S[0, :]))) / dx2
            
            # Bottom boundary: no flux from south
            L_T[-1, :] = ((alpha_e_face[-1, :] * (T_E[-1, :] - T[-1, :]) - 
                          alpha_w_face[-1, :] * (T[-1, :] - T_W[-1, :])) +
                         (alpha_n_face[-1, :] * (T_N[-1, :] - T[-1, :]))) / dx2
            
            # Corners need special treatment
            # Top-left
            L_T[0, 0] = (alpha_e_face[0, 0] * (T_E[0, 0] - T[0, 0]) - 
                        alpha_s_face[0, 0] * (T[0, 0] - T_S[0, 0])) / dx2
            
            # Top-right
            L_T[0, -1] = (-alpha_w_face[0, -1] * (T[0, -1] - T_W[0, -1]) - 
                         alpha_s_face[0, -1] * (T[0, -1] - T_S[0, -1])) / dx2
            
            # Bottom-left
            L_T[-1, 0] = (alpha_e_face[-1, 0] * (T_E[-1, 0] - T[-1, 0]) + 
                         alpha_n_face[-1, 0] * (T_N[-1, 0] - T[-1, 0])) / dx2
            
            # Bottom-right
            L_T[-1, -1] = (-alpha_w_face[-1, -1] * (T[-1, -1] - T_W[-1, -1]) + 
                          alpha_n_face[-1, -1] * (T_N[-1, -1] - T[-1, -1])) / dx2
        
        # Zero out space cells
        L_T[~mask] = 0.0
        
        return L_T
    
    def _v_cycle(self, residual: np.ndarray, alpha: np.ndarray, dt: float, dx: float,
                 mask: np.ndarray, level: int) -> np.ndarray:
        """Multigrid V-cycle"""
        
        ny, nx = residual.shape
        
        # Bottom of V-cycle: solve directly
        if level >= self.max_levels - 1 or min(ny, nx) <= 4:
            # Many iterations of smoother on coarsest grid
            correction = np.zeros_like(residual)
            for _ in range(self.bottom_smooth):
                correction = self._smooth_jacobi(correction, residual, alpha, dt, dx, mask)
            return correction
        
        # Pre-smoothing
        correction = np.zeros_like(residual)
        for _ in range(self.pre_smooth):
            correction = self._smooth_jacobi(correction, residual, alpha, dt, dx, mask)
        
        # Compute residual
        r = residual - self._apply_matrix(correction, alpha, dt, dx, mask)
        
        # Restrict to coarse grid
        r_coarse = self._restrict(r)
        alpha_coarse = self._restrict(alpha)
        mask_coarse = self._restrict_mask(mask)
        
        # Coarse grid correction (recursive call)
        correction_coarse = self._v_cycle(r_coarse, alpha_coarse, dt, 2*dx, 
                                        mask_coarse, level + 1)
        
        # Prolongate back to fine grid
        correction += self._prolongate(correction_coarse, ny, nx)
        
        # Post-smoothing
        for _ in range(self.post_smooth):
            correction = self._smooth_jacobi(correction, residual, alpha, dt, dx, mask)
        
        return correction
    
    def _apply_matrix(self, u: np.ndarray, alpha: np.ndarray, dt: float, 
                     dx: float, mask: np.ndarray) -> np.ndarray:
        """Apply the matrix A = I - dt*L for the implicit heat equation"""
        
        L_u = self._apply_diffusion_operator(u, alpha, dx, mask)
        Au = u - dt * L_u
        Au[~mask] = 0.0
        return Au
    
    def _smooth_jacobi(self, u: np.ndarray, f: np.ndarray, alpha: np.ndarray,
                      dt: float, dx: float, mask: np.ndarray) -> np.ndarray:
        """Weighted Jacobi smoother for variable coefficient problem"""
        
        # Compute A*u
        Au = self._apply_matrix(u, alpha, dt, dx, mask)
        
        # Compute diagonal of A (approximation for variable coefficients)
        # Diagonal of I - dt*L ≈ 1 + dt*(4α/dx²) for interior points
        dx2 = dx * dx
        diag = 1.0 + dt * 4.0 * alpha / dx2
        
        # Jacobi update: u_new = u + omega * (f - Au) / diag
        u_new = u + self.omega * (f - Au) / (diag + 1e-10)
        
        # Preserve space cells
        u_new[~mask] = 0.0
        
        # Apply boundary conditions if Dirichlet
        if self.bc_type == BoundaryCondition.DIRICHLET:
            # For correction, BC should be homogeneous (zero)
            u_new[0, :] = 0.0
            u_new[-1, :] = 0.0
            u_new[:, 0] = 0.0
            u_new[:, -1] = 0.0
        
        return u_new
    
    def _restrict(self, fine: np.ndarray) -> np.ndarray:
        """Full-weighting restriction from fine to coarse grid"""
        ny, nx = fine.shape
        ny_coarse = ny // 2
        nx_coarse = nx // 2
        
        coarse = np.zeros((ny_coarse, nx_coarse), dtype=fine.dtype)
        
        # Full weighting stencil:
        # 1/16 * [1 2 1]
        #        [2 4 2]
        #        [1 2 1]
        
        # Vectorized implementation
        for j in range(ny_coarse):
            for i in range(nx_coarse):
                # Map to fine grid indices
                jf = 2 * j
                if_idx = 2 * i
                
                # Apply stencil with boundary handling
                val = 4.0 * fine[jf, if_idx]
                
                if jf > 0:
                    val += 2.0 * fine[jf-1, if_idx]
                    if if_idx > 0:
                        val += fine[jf-1, if_idx-1]
                    if if_idx < nx - 1:
                        val += fine[jf-1, if_idx+1]
                
                if jf < ny - 1:
                    val += 2.0 * fine[jf+1, if_idx]
                    if if_idx > 0:
                        val += fine[jf+1, if_idx-1]
                    if if_idx < nx - 1:
                        val += fine[jf+1, if_idx+1]
                
                if if_idx > 0:
                    val += 2.0 * fine[jf, if_idx-1]
                if if_idx < nx - 1:
                    val += 2.0 * fine[jf, if_idx+1]
                
                coarse[j, i] = val / 16.0
        
        return coarse
    
    def _restrict_mask(self, mask: np.ndarray) -> np.ndarray:
        """Restrict boolean mask to coarse grid"""
        ny, nx = mask.shape
        ny_coarse = ny // 2
        nx_coarse = nx // 2
        
        # A coarse cell is active if any of its fine cells are active
        mask_coarse = np.zeros((ny_coarse, nx_coarse), dtype=bool)
        for j in range(ny_coarse):
            for i in range(nx_coarse):
                mask_coarse[j, i] = np.any(mask[2*j:2*j+2, 2*i:2*i+2])
        
        return mask_coarse
    
    def _prolongate(self, coarse: np.ndarray, ny_fine: int, nx_fine: int) -> np.ndarray:
        """Bilinear prolongation from coarse to fine grid"""
        ny_coarse, nx_coarse = coarse.shape
        fine = np.zeros((ny_fine, nx_fine), dtype=coarse.dtype)
        
        # Handle size carefully for odd dimensions
        ny_copy = min(ny_coarse, (ny_fine + 1) // 2)
        nx_copy = min(nx_coarse, (nx_fine + 1) // 2)
        
        # Direct injection at coincident points
        fine[::2, ::2][:ny_copy, :nx_copy] = coarse[:ny_copy, :nx_copy]
        
        # Linear interpolation along x (horizontal)
        for j in range(0, ny_fine, 2):
            for i in range(1, nx_fine, 2):
                if i-1 >= 0 and i+1 < nx_fine:
                    fine[j, i] = 0.5 * (fine[j, i-1] + fine[j, i+1])
                elif i-1 >= 0:
                    fine[j, i] = fine[j, i-1]
        
        # Linear interpolation along y (vertical)
        for j in range(1, ny_fine, 2):
            for i in range(nx_fine):
                if j-1 >= 0 and j+1 < ny_fine:
                    fine[j, i] = 0.5 * (fine[j-1, i] + fine[j+1, i])
                elif j-1 >= 0:
                    fine[j, i] = fine[j-1, i]
        
        return fine
    
    def _apply_boundary_conditions(self, T: np.ndarray):
        """Apply boundary conditions to temperature field"""
        if self.bc_type == BoundaryCondition.NEUMANN:
            # Neumann BC are naturally satisfied by the operator
            pass
        elif self.bc_type == BoundaryCondition.DIRICHLET:
            # Set boundary temperatures
            T[0, :] = self.sim.surface_temperature
            T[-1, :] = self.sim.surface_temperature
            T[:, 0] = self.sim.surface_temperature
            T[:, -1] = self.sim.surface_temperature
    
    def _get_interface_mask(self, non_space_mask: np.ndarray) -> np.ndarray:
        """Get mask of cells at material interfaces"""
        # Find cells adjacent to different materials
        material_boundaries = np.zeros_like(non_space_mask)
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(self.sim.material_types, dy, axis=0), dx, axis=1)
                different_materials = (self.sim.material_types != shifted) & non_space_mask
                material_boundaries |= different_materials
        
        return material_boundaries
    
    def _solve_radiative_cooling(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Apply radiative cooling and solar heating"""
        working_temp = temperature.copy()
        
        # Identify radiating cells (surface cells)
        space_mask = (self.sim.material_types == MaterialType.SPACE)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        transparent_mask = space_mask | atmosphere_mask
        
        # Find surface cells
        surface_cells = np.zeros(self.sim.material_types.shape, dtype=bool)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = np.roll(np.roll(transparent_mask, dy, axis=0), dx, axis=1)
                surface_cells |= (non_space_mask & neighbor)
        
        # Apply Stefan-Boltzmann radiation
        if np.any(surface_cells):
            emissivity = np.ones_like(working_temp) * 0.9  # Default emissivity
            sigma = 5.67e-8  # Stefan-Boltzmann constant
            
            # Radiative power per unit area
            radiative_power = emissivity * sigma * working_temp**4
            
            # Convert to temperature change
            valid_cells = surface_cells & (self.sim.density > 0) & (self.sim.specific_heat > 0)
            if np.any(valid_cells):
                volume = self.sim.cell_size**2 * self.sim.cell_depth
                mass = self.sim.density[valid_cells] * volume
                
                # Energy loss = power * area * time
                area = self.sim.cell_size**2
                energy_loss = radiative_power[valid_cells] * area * self.sim.dt
                
                # Temperature change
                dT = energy_loss / (mass * self.sim.specific_heat[valid_cells])
                working_temp[valid_cells] -= dT
        
        # Apply solar heating
        if hasattr(self.sim, 'enable_solar_heating') and self.sim.enable_solar_heating:
            solar_term = self.solar_heating.calculate_solar_heating(non_space_mask)
            working_temp += solar_term
        
        # Clamp temperatures
        working_temp = np.maximum(working_temp, 0.1)
        
        return working_temp
    
    def _solve_non_radiative_sources(self, temperature: np.ndarray, non_space_mask: np.ndarray) -> np.ndarray:
        """Apply non-radiative heat sources"""
        working_temp = temperature.copy()
        
        # Apply radioactive decay heating
        if hasattr(self.sim, 'power_density'):
            valid_cells = non_space_mask & (self.sim.density > 0) & (self.sim.specific_heat > 0)
            if np.any(valid_cells):
                decay_heating = self.sim.power_density[valid_cells] * self.sim.dt / (
                    self.sim.density[valid_cells] * self.sim.specific_heat[valid_cells]
                )
                working_temp[valid_cells] += decay_heating
        
        return working_temp