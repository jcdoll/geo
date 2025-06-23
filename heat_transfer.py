"""
Heat transfer solver for flux-based geological simulation.

Solves the heat equation with thermal diffusion, radiative cooling,
and heat generation from radioactive decay.

Written from scratch for flux-based framework with volume fractions.
"""

import numpy as np
from typing import Optional
from state import FluxState
from materials import MaterialType, MaterialDatabase


class HeatTransfer:
    """Heat transfer solver for flux-based simulation."""
    
    def __init__(self, state: FluxState):
        """
        Initialize heat transfer solver.
        
        Args:
            state: FluxState instance
        """
        self.state = state
        
        # Physical constants
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        self.t_space = 2.7  # K (cosmic background temperature)
        
        # Material database for heat generation rates
        self.material_db = MaterialDatabase()
        
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
        Apply thermal diffusion using implicit method for stability.
        
        Solves: ∂T/∂t = ∇·(k∇T) / (ρcp)
        
        Args:
            dt: Time step
        """
        ny, nx = self.state.temperature.shape
        dx = self.state.dx
        
        # Get material properties
        k = self.state.thermal_conductivity
        rho = self.state.density
        cp = self.state.specific_heat
        
        # Thermal diffusivity α = k / (ρcp)
        alpha = k / (rho * cp + 1e-10)
        
        # Stability parameter for implicit method
        r = alpha * dt / (dx * dx)
        
        # Use ADI (Alternating Direction Implicit) method for 2D diffusion
        # This is unconditionally stable
        
        # Step 1: Implicit in x, explicit in y
        T_half = self.adi_x_sweep(self.state.temperature, r, dx)
        
        # Step 2: Implicit in y, using T_half
        T_new = self.adi_y_sweep(T_half, r, dx)
        
        # Update temperature
        self.state.temperature = T_new
        
    def adi_x_sweep(self, T: np.ndarray, r: np.ndarray, dx: float) -> np.ndarray:
        """
        ADI x-direction sweep (implicit in x, explicit in y).
        
        Args:
            T: Current temperature
            r: Diffusion parameter array
            dx: Grid spacing
            
        Returns:
            Updated temperature after x-sweep
        """
        ny, nx = T.shape
        T_new = T.copy()
        
        # For each row, solve tridiagonal system
        for j in range(ny):
            # Skip boundary rows
            if j == 0 or j == ny - 1:
                continue
                
            # Build tridiagonal matrix coefficients
            a = np.zeros(nx)  # Lower diagonal
            b = np.zeros(nx)  # Main diagonal
            c = np.zeros(nx)  # Upper diagonal
            d = np.zeros(nx)  # RHS
            
            for i in range(nx):
                if i == 0 or i == nx - 1:
                    # Boundary conditions
                    b[i] = 1.0
                    d[i] = T[j, i]
                else:
                    # Interior points
                    r_avg = 0.5 * (r[j, i] + r[j, i-1])
                    a[i] = -r_avg
                    
                    r_avg = 0.5 * (r[j, i] + r[j, i+1])
                    c[i] = -r_avg
                    
                    b[i] = 1.0 - a[i] - c[i]  # Since a and c are negative
                    
                    # Y-direction diffusion (explicit)
                    d[i] = T[j, i] + r[j, i] * (T[j+1, i] - 2*T[j, i] + T[j-1, i])
                    
            # Solve tridiagonal system
            T_new[j, :] = self.solve_tridiagonal(a, b, c, d)
            
        return T_new
        
    def adi_y_sweep(self, T: np.ndarray, r: np.ndarray, dx: float) -> np.ndarray:
        """
        ADI y-direction sweep (implicit in y, using result from x-sweep).
        
        Args:
            T: Temperature after x-sweep
            r: Diffusion parameter array
            dx: Grid spacing
            
        Returns:
            Final temperature after y-sweep
        """
        ny, nx = T.shape
        T_new = T.copy()
        
        # For each column, solve tridiagonal system
        for i in range(nx):
            # Skip boundary columns
            if i == 0 or i == nx - 1:
                continue
                
            # Build tridiagonal matrix coefficients
            a = np.zeros(ny)  # Lower diagonal
            b = np.zeros(ny)  # Main diagonal
            c = np.zeros(ny)  # Upper diagonal
            d = np.zeros(ny)  # RHS
            
            for j in range(ny):
                if j == 0 or j == ny - 1:
                    # Boundary conditions
                    b[j] = 1.0
                    d[j] = T[j, i]
                else:
                    # Interior points
                    r_avg = 0.5 * (r[j, i] + r[j-1, i])
                    a[j] = -r_avg
                    
                    r_avg = 0.5 * (r[j, i] + r[j+1, i])
                    c[j] = -r_avg
                    
                    b[j] = 1.0 - a[j] - c[j]  # Since a and c are negative
                    
                    # Use temperature from x-sweep
                    d[j] = T[j, i]
                    
            # Solve tridiagonal system
            T_new[:, i] = self.solve_tridiagonal(a, b, c, d)
            
        return T_new
        
    def solve_tridiagonal(self, a: np.ndarray, b: np.ndarray, 
                         c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system using Thomas algorithm.
        
        Args:
            a: Lower diagonal
            b: Main diagonal
            c: Upper diagonal
            d: Right-hand side
            
        Returns:
            Solution vector
        """
        n = len(d)
        
        # Forward elimination
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            if i < n-1:
                c_prime[i] = c[i] / (b[i] - a[i] * c_prime[i-1])
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / (b[i] - a[i] * c_prime[i-1])
            
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
            
        return x
        
    def apply_radiative_cooling(self, dt: float):
        """
        Apply radiative cooling at exposed surfaces.
        
        Uses Stefan-Boltzmann law with atmospheric effects.
        
        Args:
            dt: Time step
        """
        # Find exposed surfaces (top boundary and low-density cells)
        exposed_mask = np.zeros_like(self.state.temperature, dtype=bool)
        
        # Top row is always exposed
        exposed_mask[0, :] = self.state.density[0, :] > 0.1
        
        # Interior cells with very low density (essentially vacuum)
        exposed_mask |= self.state.density < 0.1
        
        if not np.any(exposed_mask):
            return
            
        # Get properties for exposed cells
        T = self.state.temperature[exposed_mask]
        emissivity = self.state.emissivity[exposed_mask]
        density = self.state.density[exposed_mask]
        cp = self.state.specific_heat[exposed_mask]
        
        # Check for atmospheric greenhouse effect
        # Water vapor creates greenhouse effect
        vapor_frac = self.state.vol_frac[MaterialType.WATER_VAPOR.value][exposed_mask]
        greenhouse_factor = 1.0 - 0.5 * np.tanh(vapor_frac * 10)  # 0 to 1
        
        # Radiative heat loss: Q = σ * ε * A * (T⁴ - T_space⁴)
        # Temperature change: dT = -Q * dt / (ρ * cp * V)
        
        # Use Newton's method for implicit solution to avoid instability
        for _ in range(3):  # Usually converges quickly
            heat_loss = self.stefan_boltzmann * emissivity * greenhouse_factor * \
                       (T**4 - self.t_space**4)
            dT_dt = -heat_loss / (density * cp + 1e-10)
            
            # Implicit update with Newton-Raphson
            # f(T_new) = T_new - T_old - dt * dT_dt(T_new) = 0
            T_old = T.copy()
            f = T - T_old - dt * dT_dt
            
            # Derivative: df/dT = 1 + dt * 4σεT³/(ρcp)
            df_dT = 1.0 + dt * 4 * self.stefan_boltzmann * emissivity * \
                    greenhouse_factor * T**3 / (density * cp + 1e-10)
                    
            # Newton update
            T = T - f / (df_dT + 1e-10)
            
            # Keep physical
            T = np.maximum(T, self.t_space)
            
        # Update temperatures
        self.state.temperature[exposed_mask] = T
        
    def apply_heat_generation(self, dt: float):
        """
        Apply internal heat generation from radioactive decay.
        
        Args:
            dt: Time step
        """
        # Get radioactive material fractions
        uranium_frac = self.state.vol_frac[MaterialType.URANIUM.value]
        
        # Only process cells with uranium
        if not np.any(uranium_frac > 0):
            return
            
        # Get heat generation rate from material database
        uranium_props = self.material_db.get_properties(MaterialType.URANIUM)
        heat_gen_rate = uranium_props.get('heat_generation', 0.0)  # W/kg
        
        if heat_gen_rate <= 0:
            return
            
        # Calculate volumetric heat generation
        # Q = heat_gen_rate * density * volume_fraction
        heat_source = heat_gen_rate * self.state.density * uranium_frac
        
        # Temperature change: dT = Q * dt / (ρ * cp)
        denom = self.state.density * self.state.specific_heat + 1e-10
        dT = heat_source * dt / denom
        
        # Update temperature
        self.state.temperature += dT
        
        # Track total heat generation for diagnostics
        if hasattr(self.state, 'total_heat_generated'):
            self.state.total_heat_generated += np.sum(heat_source) * dt
        else:
            self.state.total_heat_generated = np.sum(heat_source) * dt