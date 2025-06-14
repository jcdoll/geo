"""
Fluid dynamics module for geological simulation.
Handles pressure calculations, gravitational collapse, density stratification, and fluid migration.
"""

import numpy as np
from typing import Tuple
from scipy import ndimage
try:
    from .materials import MaterialType, MaterialDatabase
    from .pressure_solver import solve_pressure
except ImportError:  # standalone script execution
    from materials import MaterialType, MaterialDatabase  # type: ignore
    from pressure_solver import solve_pressure  # type: ignore


class FluidDynamics:
    """Fluid dynamics calculations for geological simulation"""
    
    def __init__(self, simulation):
        """Initialize with reference to main simulation"""
        self.sim = simulation
    
    def calculate_planetary_pressure(self):
        """Multigrid solve for pressure using self-gravity field."""

        # Ensure gravity field is up-to-date
        if hasattr(self.sim, 'calculate_self_gravity'):
            self.sim.calculate_self_gravity()

        gx = self.sim.gravity_x
        gy = self.sim.gravity_y

        # Divergence of gravity
        div_g = np.zeros_like(gx)
        dx = self.sim.cell_size
        div_g[1:-1, 1:-1] = (
            (gx[1:-1, 2:] - gx[1:-1, :-2]) + (gy[2:, 1:-1] - gy[:-2, 1:-1])
        ) / (2 * dx)

        # Build RHS: -ρ ∇·g  (MPa units -> divide 1e6)
        rhs = (self.sim.density * div_g) / 1e6   # Pa → MPa

        # Solve Poisson
        pressure = solve_pressure(rhs, dx)

        # Apply Dirichlet boundary (space/atmosphere = 0 MPa)
        atmosphere_mask = (
            (self.sim.material_types == MaterialType.SPACE) |
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR)
        )
        pressure[atmosphere_mask] = 0.0

        # Store & add persistent offsets
        self.sim.pressure[:] = np.maximum(0.0, pressure + self.sim.pressure_offset)
    
    def apply_gravitational_collapse(self):
        """Deterministic gravitational collapse (no RNG, no sampling)."""

        neighbor_count = 8  # always full neighbourhood for accuracy

        # Solid voxels that may fall into cavities (non-solid)
        solid_mask = self.sim._get_solid_mask()
        if not np.any(solid_mask):
            return

        # Precompute distance-to-COM once for whole grid
        center_x, center_y = self.sim.center_of_mass
        yy, xx = np.ogrid[:self.sim.height, :self.sim.width]
        dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2

        neighbors = self.sim._get_neighbors(neighbor_count, shuffle=False)
        
        # Find cells that can potentially fall
        center_x, center_y = self.sim.center_of_mass
        
        # Calculate direction toward center for each cell
        y_coords = np.arange(self.sim.height).reshape(-1, 1)
        x_coords = np.arange(self.sim.width).reshape(1, -1)
        
        dx_to_center = center_x - x_coords
        dy_to_center = center_y - y_coords
        
        # Collect all potential swaps
        src_y_list, src_x_list = [], []
        tgt_y_list, tgt_x_list = [], []
        
        for dy, dx in neighbors:
            # Build full H×W neighbor coordinate grids using broadcasting
            neighbor_y = (np.arange(self.sim.height)[:, None] + dy).repeat(self.sim.width, axis=1)
            neighbor_x = (np.arange(self.sim.width)[None, :] + dx).repeat(self.sim.height, axis=0)
            
            # Check bounds
            valid_neighbors = (
                (neighbor_y >= 0) & (neighbor_y < self.sim.height) &
                (neighbor_x >= 0) & (neighbor_x < self.sim.width)
            )
            
            if not np.any(valid_neighbors):
                continue
            
            # Check if neighbor is closer to center
            neighbor_dx = center_x - neighbor_x
            neighbor_dy = center_y - neighbor_y
            neighbor_dist_sq = neighbor_dx**2 + neighbor_dy**2
            current_dist_sq = dx_to_center**2 + dy_to_center**2
            
            closer_to_center = neighbor_dist_sq < current_dist_sq
            
            # Get neighbor materials
            neighbor_materials = np.full_like(self.sim.material_types, MaterialType.SPACE)
            neighbor_materials[valid_neighbors] = self.sim.material_types[neighbor_y[valid_neighbors], neighbor_x[valid_neighbors]]
            
            # Check if neighbor is a cavity (fluid or space)
            cavity_materials = {
                MaterialType.SPACE, MaterialType.AIR, MaterialType.WATER_VAPOR,
                MaterialType.WATER, MaterialType.MAGMA
            }
            is_cavity = np.isin(neighbor_materials, list(cavity_materials))
            
            # Cells that can fall: solid, neighbor is cavity and closer to center
            can_fall = solid_mask & valid_neighbors & is_cavity & closer_to_center
            
            if not np.any(can_fall):
                continue
            
            # Deterministic – all cells that can fall WILL fall
            fall_coords = np.where(can_fall)
            if len(fall_coords[0]) == 0:
                continue

            final_y = fall_coords[0]
            final_x = fall_coords[1]
            
            # Calculate target positions
            target_y = final_y + dy
            target_x = final_x + dx
            
            # Add to swap lists
            src_y_list.extend(final_y)
            src_x_list.extend(final_x)
            tgt_y_list.extend(target_y)
            tgt_x_list.extend(target_x)
        
        if not src_y_list:
            return
        
        # Convert to arrays and deduplicate
        src_y = np.array(src_y_list)
        src_x = np.array(src_x_list)
        tgt_y = np.array(tgt_y_list)
        tgt_x = np.array(tgt_x_list)
        
        src_y, src_x, tgt_y, tgt_x = self.sim._dedupe_swap_pairs(src_y, src_x, tgt_y, tgt_x)
        
        if len(src_y) == 0:
            return
        
        # Perform swaps
        self._perform_material_swaps(src_y, src_x, tgt_y, tgt_x)
    
    def apply_fluid_dynamics(self):
        """Deterministic fluid migration and buoyancy (no RNG, full grid)."""

        # Get fluid materials
        fluid_mask = (
            (self.sim.material_types == MaterialType.AIR) |
            (self.sim.material_types == MaterialType.WATER_VAPOR) |
            (self.sim.material_types == MaterialType.WATER) |
            (self.sim.material_types == MaterialType.MAGMA) |
            (self.sim.material_types == MaterialType.SPACE)
        )
        
        if not np.any(fluid_mask):
            return
        
        # Process **all** fluid cells (deterministic behaviour)
        fluid_coords = np.where(fluid_mask)
        if len(fluid_coords[0]) == 0:
            return
        
        # Get neighbors within radius 2
        neighbors = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                if dx**2 + dy**2 <= 4:  # Within radius 2
                    neighbors.append((dy, dx))
        
        # Collect potential swaps
        src_y_list, src_x_list = [], []
        tgt_y_list, tgt_x_list = [], []
        
        center_x, center_y = self.sim.center_of_mass
        
        for i, (y, x) in enumerate(zip(fluid_coords[0], fluid_coords[1])):
            current_material = self.sim.material_types[y, x]
            current_density = self.sim.density[y, x]
            
            # Skip space (it doesn't migrate)
            if current_material == MaterialType.SPACE:
                continue
            
            # Check neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if ny < 0 or ny >= self.sim.height or nx < 0 or nx >= self.sim.width:
                    continue
                
                neighbor_material = self.sim.material_types[ny, nx]
                
                # Skip space neighbors
                if neighbor_material == MaterialType.SPACE:
                    continue
                
                neighbor_density = self.sim.density[ny, nx]
                
                # Check if neighbor is denser and farther from center
                current_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                neighbor_dist = np.sqrt((nx - center_x)**2 + (ny - center_y)**2)
                
                if (neighbor_density > current_density and neighbor_dist > current_dist):
                    src_y_list.append(y)
                    src_x_list.append(x)
                    tgt_y_list.append(ny)
                    tgt_x_list.append(nx)
                    break  # Only one swap per cell
        
        if not src_y_list:
            return
        
        # Convert to arrays and deduplicate
        src_y = np.array(src_y_list)
        src_x = np.array(src_x_list)
        tgt_y = np.array(tgt_y_list)
        tgt_x = np.array(tgt_x_list)
        
        src_y, src_x, tgt_y, tgt_x = self.sim._dedupe_swap_pairs(src_y, src_x, tgt_y, tgt_x)
        
        if len(src_y) == 0:
            return
        
        # Perform swaps
        self._perform_material_swaps(src_y, src_x, tgt_y, tgt_x)
    
    def apply_density_stratification(self):
        """Deterministic density stratification (no RNG, full grid)."""

        mobile_mask = self.sim._get_mobile_mask()
        if not np.any(mobile_mask):
            return

        sampled_y, sampled_x = np.where(mobile_mask)
        
        # Get 5x5 neighbors for isotropic sampling
        neighbors = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                if dx**2 + dy**2 <= 6:  # Roughly circular
                    neighbors.append((dy, dx))
        
        # Keep fixed neighbour order for deterministic behaviour
        
        # Collect potential swaps
        src_y_list, src_x_list = [], []
        tgt_y_list, tgt_x_list = [], []
        
        center_x, center_y = self.sim.center_of_mass
        
        for i, (y, x) in enumerate(zip(sampled_y, sampled_x)):
            current_density = self.calculate_effective_density_single(y, x)
            current_dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Check neighbors
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                
                # Check bounds
                if ny < 0 or ny >= self.sim.height or nx < 0 or nx >= self.sim.width:
                    continue
                
                # Skip space
                if self.sim.material_types[ny, nx] == MaterialType.SPACE:
                    continue
                
                neighbor_density = self.calculate_effective_density_single(ny, nx)
                neighbor_dist = np.sqrt((nx - center_x)**2 + (ny - center_y)**2)
                
                # Check for density inversion
                density_ratio = max(current_density, neighbor_density) / max(min(current_density, neighbor_density), 1e-10)
                
                if density_ratio < 1.05:  # Not enough density difference
                    continue
                
                # Determine if swap should occur
                should_swap = False
                
                if (current_density > neighbor_density and current_dist > neighbor_dist):
                    # Denser material farther out - should sink inward
                    should_swap = True
                elif (current_density < neighbor_density and current_dist < neighbor_dist):
                    # Lighter material closer in - should rise outward
                    should_swap = True
                
                if should_swap:
                    src_y_list.append(y)
                    src_x_list.append(x)
                    tgt_y_list.append(ny)
                    tgt_x_list.append(nx)
                    break  # Only one swap per cell
        
        if not src_y_list:
            return
        
        # Convert to arrays and deduplicate
        src_y = np.array(src_y_list)
        src_x = np.array(src_x_list)
        tgt_y = np.array(tgt_y_list)
        tgt_x = np.array(tgt_x_list)
        
        src_y, src_x, tgt_y, tgt_x = self.sim._dedupe_swap_pairs(src_y, src_x, tgt_y, tgt_x)
        
        if len(src_y) == 0:
            return
        
        # Perform swaps
        self._perform_material_swaps(src_y, src_x, tgt_y, tgt_x)
    
    def settle_unsupported_chunks(self) -> bool:
        """Settle unsupported chunks of material using vectorized approach"""
        # Get solid materials
        solid_mask = self.sim._get_solid_mask()
        
        if not np.any(solid_mask):
            return False
        
        # Calculate center of mass direction
        center_x, center_y = self.sim.center_of_mass
        
        # Create coordinate grids
        y_coords = np.arange(self.sim.height).reshape(-1, 1)
        x_coords = np.arange(self.sim.width).reshape(1, -1)
        
        # Calculate direction vectors toward center
        dx_to_center = center_x - x_coords
        dy_to_center = center_y - y_coords
        distances = np.sqrt(dx_to_center**2 + dy_to_center**2)
        
        # Normalize direction vectors
        safe_distances = np.maximum(distances, 1e-10)
        unit_dx = dx_to_center / safe_distances
        unit_dy = dy_to_center / safe_distances
        
        # Find the primary inward direction for each cell (quantized to 8 directions)
        angles = np.arctan2(unit_dy, unit_dx)
        direction_indices = np.round(angles / (np.pi / 4)) % 8
        
        # Map to discrete directions
        directions = [
            (0, 1),   # East
            (1, 1),   # Southeast  
            (1, 0),   # South
            (1, -1),  # Southwest
            (0, -1),  # West
            (-1, -1), # Northwest
            (-1, 0),  # North
            (-1, 1)   # Northeast
        ]
        
        # Find unsupported cells
        unsupported_mask = np.zeros_like(solid_mask)
        
        for i, (dy, dx) in enumerate(directions):
            # Cells with this primary direction
            direction_mask = (direction_indices == i) & solid_mask
            
            if not np.any(direction_mask):
                continue
            
            # Check if support exists in the inward direction
            support_y = np.clip(y_coords + dy, 0, self.sim.height - 1)
            support_x = np.clip(x_coords + dx, 0, self.sim.width - 1)
            
            # Support exists if the inward neighbor is also solid
            has_support = solid_mask[support_y, support_x]
            
            # Unsupported cells in this direction
            unsupported_mask |= direction_mask & ~has_support
        
        if not np.any(unsupported_mask):
            return False
        
        # Label connected components
        labeled_chunks, num_chunks = ndimage.label(unsupported_mask, structure=self.sim._circular_kernel_3x3)
        
        if num_chunks == 0:
            return False
        
        moved_any = False
        
        # Process each chunk
        for chunk_id in range(1, num_chunks + 1):
            chunk_mask = (labeled_chunks == chunk_id)
            chunk_coords = np.where(chunk_mask)
            
            if len(chunk_coords[0]) == 0:
                continue
            
            # Calculate average direction for this chunk
            chunk_y = chunk_coords[0]
            chunk_x = chunk_coords[1]
            
            avg_dx = np.mean(unit_dx[chunk_y, chunk_x])
            avg_dy = np.mean(unit_dy[chunk_y, chunk_x])
            
            # Quantize to nearest cardinal/diagonal direction
            angle = np.arctan2(avg_dy, avg_dx)
            direction_idx = int(np.round(angle / (np.pi / 4))) % 8
            dy, dx = directions[direction_idx]
            
            # Calculate how far this chunk can fall
            max_fall_distance = 0
            
            for fall_dist in range(1, int(self.sim.terminal_settle_velocity) + 1):
                # Check if chunk can move this distance
                target_y = chunk_y + dy * fall_dist
                target_x = chunk_x + dx * fall_dist
                
                # Check bounds
                if (np.any(target_y < 0) or np.any(target_y >= self.sim.height) or
                    np.any(target_x < 0) or np.any(target_x >= self.sim.width)):
                    break
                
                # Check if target positions are fluid (can be displaced)
                target_materials = self.sim.material_types[target_y, target_x]
                fluid_materials = {
                    MaterialType.AIR, MaterialType.WATER_VAPOR,
                    MaterialType.WATER, MaterialType.MAGMA, MaterialType.SPACE
                }
                
                if not all(mat in fluid_materials for mat in target_materials):
                    break
                
                # Check density - chunk should be denser than target
                chunk_densities = self.sim.density[chunk_y, chunk_x]
                target_densities = self.sim.density[target_y, target_x]
                
                if not np.all(chunk_densities > target_densities):
                    break
                
                max_fall_distance = fall_dist
            
            if max_fall_distance == 0:
                continue
            
            # Move the chunk
            target_y = chunk_y + dy * max_fall_distance
            target_x = chunk_x + dx * max_fall_distance
            
            # Perform the swap
            self._perform_material_swaps(chunk_y, chunk_x, target_y, target_x)
            moved_any = True
        
        return moved_any
    
    def calculate_effective_density(self, temperature: np.ndarray) -> np.ndarray:
        """Calculate effective density including thermal expansion"""
        # Base density from material properties
        base_density = self.sim.density.copy()
        
        # Apply thermal expansion
        thermal_expansion = np.zeros_like(temperature)
        
        for y in range(self.sim.height):
            for x in range(self.sim.width):
                if self.sim.material_types[y, x] == MaterialType.SPACE:
                    continue
                
                material_props = self.sim.material_db.get_properties(self.sim.material_types[y, x])
                expansion_coeff = getattr(material_props, 'thermal_expansion', 1e-5)
                
                # Effective density with thermal expansion
                temp_diff = temperature[y, x] - self.sim.reference_temperature
                expansion_factor = 1.0 + expansion_coeff * temp_diff
                thermal_expansion[y, x] = expansion_factor
        
        # Avoid division by zero
        thermal_expansion = np.maximum(thermal_expansion, 0.1)
        
        return base_density / thermal_expansion
    
    def calculate_effective_density_single(self, y: int, x: int) -> float:
        """Calculate effective density for a single cell"""
        if self.sim.material_types[y, x] == MaterialType.SPACE:
            return 0.0
        
        material_props = self.sim.material_db.get_properties(self.sim.material_types[y, x])
        expansion_coeff = getattr(material_props, 'thermal_expansion', 1e-5)
        
        temp_diff = self.sim.temperature[y, x] - self.sim.reference_temperature
        expansion_factor = max(1.0 + expansion_coeff * temp_diff, 0.1)
        
        return self.sim.density[y, x] / expansion_factor
    
    def _perform_material_swaps(self, src_y: np.ndarray, src_x: np.ndarray, 
                               tgt_y: np.ndarray, tgt_x: np.ndarray):
        """Perform material and temperature swaps between source and target cells"""
        if len(src_y) == 0:
            return
        
        # Store source values
        src_materials = self.sim.material_types[src_y, src_x].copy()
        src_temperatures = self.sim.temperature[src_y, src_x].copy()
        src_ages = self.sim.age[src_y, src_x].copy()
        
        # Store target values
        tgt_materials = self.sim.material_types[tgt_y, tgt_x].copy()
        tgt_temperatures = self.sim.temperature[tgt_y, tgt_x].copy()
        tgt_ages = self.sim.age[tgt_y, tgt_x].copy()
        
        # Perform swaps
        self.sim.material_types[src_y, src_x] = tgt_materials
        self.sim.temperature[src_y, src_x] = tgt_temperatures
        self.sim.age[src_y, src_x] = tgt_ages
        
        self.sim.material_types[tgt_y, tgt_x] = src_materials
        self.sim.temperature[tgt_y, tgt_x] = src_temperatures
        self.sim.age[tgt_y, tgt_x] = src_ages
        
        # Mark properties as dirty for recalculation
        self.sim._properties_dirty = True 
