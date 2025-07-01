"""
Gravity-aware fluid dynamics module using actual gravity vectors.

This module implements proper radial swapping based on the actual gravity field,
not assuming gravity always points "down".
"""

import numpy as np
try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:  # standalone script execution
    from materials import MaterialType, MaterialDatabase  # type: ignore


class FluidDynamics:
    """Gravity-aware fluid dynamics using density-based swapping"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
        
        # Create dummy velocity fields for UI compatibility
        h, w = self.sim.material_types.shape
        self.velocity_x = np.zeros((h, w), dtype=np.float64)
        self.velocity_y = np.zeros((h, w), dtype=np.float64)
        
        # Material swap probabilities
        self.swap_prob_lookup = self._create_swap_probability_lookup()
        
        # Pre-allocate work arrays
        self.random_field = np.zeros((h, w), dtype=np.float32)
        
        # Neighbor offsets (4-connected for now, could extend to 8)
        self.neighbor_offsets = np.array([
            (0, -1),   # left
            (0, 1),    # right
            (-1, 0),   # up
            (1, 0),    # down
        ])
    
    def _create_swap_probability_lookup(self):
        """Create lookup dict for material swap probabilities"""
        probs = {
            # Fluids
            MaterialType.SPACE: 1.0,  # Space can always be displaced
            MaterialType.AIR: 0.8,
            MaterialType.WATER: 0.7,
            MaterialType.ICE: 0.3,
            MaterialType.WATER_VAPOR: 0.8,
            # Special
            MaterialType.MAGMA: 0.4,
            MaterialType.URANIUM: 0.1,
            # Rocks
            MaterialType.SAND: 0.5,
            MaterialType.SHALE: 0.2,
            MaterialType.LIMESTONE: 0.15,
            MaterialType.GRANITE: 0.1,
            MaterialType.BASALT: 0.1,
            MaterialType.SLATE: 0.15,
            MaterialType.SCHIST: 0.1,
            MaterialType.GNEISS: 0.1,
            MaterialType.MARBLE: 0.1,
        }
        return probs
    
    def apply_unified_kinematics(self, dt: float) -> None:
        """Main entry point - apply gravity-aware density-based swapping"""
        # Ensure we have current gravity field
        if not hasattr(self.sim, 'gravity_x') or not hasattr(self.sim, 'gravity_y'):
            # Fallback to simple vertical swapping if no gravity field
            self._apply_simple_vertical_swapping()
            return
        
        # Apply gravity-aware swapping
        self._apply_gravity_aware_swapping()
        
        # Clear velocity fields (we don't use them)
        self.velocity_x.fill(0)
        self.velocity_y.fill(0)
    
    def _apply_gravity_aware_swapping(self):
        """Apply swapping based on actual gravity vectors"""
        h, w = self.sim.material_types.shape
        
        # Get gravity field - includes both self-gravity and external gravity
        gx = self.sim.gravity_x
        gy = self.sim.gravity_y
        
        # Calculate gravity magnitude at each cell
        g_mag = np.sqrt(gx**2 + gy**2)
        
        # Avoid division by zero in regions with no gravity
        g_mag_safe = np.maximum(g_mag, 1e-10)
        
        # Calculate effective density accounting for thermal expansion
        effective_density = self.sim.calculate_effective_density(self.sim.temperature)
        
        # For each cell, find which neighbor is most aligned with gravity
        # This will be our "downhill" direction for that cell
        
        # Process in random order to avoid directional bias
        indices = np.arange(h * w)
        np.random.shuffle(indices)
        
        # Track which cells have been swapped this step to avoid double-swapping
        swapped = np.zeros((h, w), dtype=bool)
        
        for idx in indices:
            y, x = divmod(idx, w)
            
            # Skip if already swapped
            if swapped[y, x]:
                continue
            
            # Skip space cells (they don't move, but can be swapped into)
            if self.sim.material_types[y, x] == MaterialType.SPACE:
                continue
            
            # Get local gravity vector
            local_gx = gx[y, x]
            local_gy = gy[y, x]
            local_gmag = g_mag_safe[y, x]
            
            # Skip if no significant gravity
            if local_gmag < 0.01:  # threshold for "significant" gravity
                continue
            
            # Normalize gravity vector
            gx_norm = local_gx / local_gmag
            gy_norm = local_gy / local_gmag
            
            # Check each neighbor
            best_neighbor = None
            best_alignment = -1.0
            best_density_ratio = 0.0
            
            for dy, dx in self.neighbor_offsets:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if 0 <= ny < h and 0 <= nx < w and not swapped[ny, nx]:
                    # Calculate how aligned this direction is with gravity
                    # Positive = same direction as gravity (downhill)
                    alignment = dx * gx_norm + dy * gy_norm
                    
                    # Only consider downhill directions (alignment > 0)
                    if alignment > 0.1:  # Small threshold to avoid numerical issues
                        # Check density difference using effective density
                        current_density = effective_density[y, x]
                        neighbor_density = effective_density[ny, nx]
                        
                        # Should we sink into this neighbor?
                        if current_density > neighbor_density:  # Any density difference drives flow
                            density_ratio = current_density / neighbor_density
                            
                            # Prefer most aligned direction, break ties by density difference
                            score = alignment * density_ratio
                            if score > best_alignment * best_density_ratio:
                                best_neighbor = (ny, nx)
                                best_alignment = alignment
                                best_density_ratio = density_ratio
            
            # If we found a good neighbor to swap with
            if best_neighbor is not None:
                ny, nx = best_neighbor
                
                # Get swap probabilities
                mat1 = self.sim.material_types[y, x]
                mat2 = self.sim.material_types[ny, nx]
                
                prob1 = self.swap_prob_lookup.get(mat1, 0.1)
                prob2 = self.swap_prob_lookup.get(mat2, 0.1)
                swap_prob = max(prob1, prob2)
                
                # Increase probability based on alignment (more aligned = more likely)
                swap_prob *= (0.5 + 0.5 * best_alignment)
                
                # Random decision
                if np.random.random() < swap_prob:
                    # Perform swap
                    self.sim.material_types[y, x], self.sim.material_types[ny, nx] = \
                        self.sim.material_types[ny, nx], self.sim.material_types[y, x]
                    self.sim.temperature[y, x], self.sim.temperature[ny, nx] = \
                        self.sim.temperature[ny, nx], self.sim.temperature[y, x]
                    self.sim.age[y, x], self.sim.age[ny, nx] = \
                        self.sim.age[ny, nx], self.sim.age[y, x]
                    
                    # Mark both cells as swapped
                    swapped[y, x] = True
                    swapped[ny, nx] = True
        
        # Apply lateral fluid spreading (perpendicular to gravity)
        self._apply_lateral_fluid_spreading(gx, gy, g_mag_safe)
        
        # Update material properties after all swaps
        self.sim._update_material_properties()
    
    def _apply_lateral_fluid_spreading(self, gx, gy, g_mag):
        """Apply fluid spreading perpendicular to gravity"""
        h, w = self.sim.material_types.shape
        
        # Fluid types that can spread laterally
        fluid_types = {MaterialType.WATER, MaterialType.MAGMA, MaterialType.AIR, MaterialType.WATER_VAPOR}
        
        # Process a random subset of cells for performance
        sample_size = min(1000, h * w // 4)  # Process up to 25% of cells
        indices = np.random.choice(h * w, size=sample_size, replace=False)
        
        for idx in indices:
            y, x = divmod(idx, w)
            
            # Skip non-fluids
            if self.sim.material_types[y, x] not in fluid_types:
                continue
            
            # Get local gravity vector
            local_gx = gx[y, x]
            local_gy = gy[y, x]
            local_gmag = g_mag[y, x]
            
            # Skip if no significant gravity
            if local_gmag < 0.01:
                continue
            
            # Normalize gravity vector
            gx_norm = local_gx / local_gmag
            gy_norm = local_gy / local_gmag
            
            # Find neighbors that are perpendicular to gravity
            for dy, dx in self.neighbor_offsets:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if 0 <= ny < h and 0 <= nx < w:
                    # Calculate alignment with gravity
                    alignment = abs(dx * gx_norm + dy * gy_norm)
                    
                    # If mostly perpendicular (alignment near 0)
                    if alignment < 0.3:
                        neighbor_mat = self.sim.material_types[ny, nx]
                        
                        # Can spread into space or other fluids of similar density
                        if neighbor_mat == MaterialType.SPACE or neighbor_mat in fluid_types:
                            current_density = self.sim.density[y, x]
                            neighbor_density = self.sim.density[ny, nx]
                            
                            # Allow spreading if densities are similar
                            if neighbor_density > 0:  # Avoid division by zero
                                density_ratio = abs(current_density - neighbor_density) / neighbor_density
                                if density_ratio < 0.2:  # Within 20%
                                    if np.random.random() < 0.2:  # 20% chance
                                        # Swap
                                        self.sim.material_types[y, x], self.sim.material_types[ny, nx] = \
                                            self.sim.material_types[ny, nx], self.sim.material_types[y, x]
                                        self.sim.temperature[y, x], self.sim.temperature[ny, nx] = \
                                            self.sim.temperature[ny, nx], self.sim.temperature[y, x]
                                        self.sim.age[y, x], self.sim.age[ny, nx] = \
                                            self.sim.age[ny, nx], self.sim.age[y, x]
                                        break  # Only one swap per cell per step
    
    def _apply_simple_vertical_swapping(self):
        """Fallback to simple vertical swapping if no gravity field available"""
        h, w = self.sim.material_types.shape
        
        # Get swap probabilities for all cells
        swap_probs = np.zeros((h, w), dtype=np.float32)
        for j in range(h):
            for i in range(w):
                mat = self.sim.material_types[j, i]
                swap_probs[j, i] = self.swap_prob_lookup.get(mat, 0.0)
        
        # Process row by row from bottom to top
        for row in range(h - 1):
            # Get current and below row data
            current_density = self.sim.density[row, :]
            below_density = self.sim.density[row + 1, :]
            
            current_prob = swap_probs[row, :]
            below_prob = swap_probs[row + 1, :]
            
            # Find where current is denser than below (should sink)
            should_sink = current_density > below_density * 1.1  # 10% threshold
            
            # Get max probability for each pair
            max_prob = np.maximum(current_prob, below_prob)
            
            # Random decision for each cell
            rand_vals = np.random.rand(w)
            will_swap = should_sink & (rand_vals < max_prob)
            
            # Perform swaps where decided
            swap_indices = np.where(will_swap)[0]
            if len(swap_indices) > 0:
                # Swap materials
                self.sim.material_types[row, swap_indices], self.sim.material_types[row + 1, swap_indices] = \
                    self.sim.material_types[row + 1, swap_indices].copy(), self.sim.material_types[row, swap_indices].copy()
                
                # Swap temperatures
                self.sim.temperature[row, swap_indices], self.sim.temperature[row + 1, swap_indices] = \
                    self.sim.temperature[row + 1, swap_indices].copy(), self.sim.temperature[row, swap_indices].copy()
                
                # Swap ages
                self.sim.age[row, swap_indices], self.sim.age[row + 1, swap_indices] = \
                    self.sim.age[row + 1, swap_indices].copy(), self.sim.age[row, swap_indices].copy()
        
        # Update material properties after all swaps
        self.sim._update_material_properties()
    
    # Stub methods for compatibility
    def calculate_planetary_pressure(self) -> None:
        """No pressure calculation in simplified version"""
        self.sim.pressure[:] = 101325.0  # 1 atmosphere
        space_mask = self.sim.material_types == MaterialType.SPACE
        self.sim.pressure[space_mask] = 0.0
    
    def apply_velocity_projection(self, dt: float) -> None:
        """No velocity projection in simplified version"""
        pass