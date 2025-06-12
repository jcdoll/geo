"""
Simulation utilities for geological simulation.
Contains helper functions, kernels, and shared calculations.
"""

import numpy as np
from scipy import ndimage
try:
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from materials import MaterialType, MaterialDatabase


class SimulationUtils:
    """Utility functions for geological simulation"""
    
    @staticmethod
    def create_circular_kernel(size: int) -> np.ndarray:
        """Create a circular kernel for morphological operations"""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= center ** 2
        kernel = np.zeros((size, size), dtype=bool)
        kernel[mask] = True
        return kernel
    
    @staticmethod
    def get_distances_from_center(height: int, width: int, center_x: float = None, center_y: float = None) -> np.ndarray:
        """Calculate distances from center for all grid points"""
        if center_x is None:
            center_x = width / 2
        if center_y is None:
            center_y = height / 2
        
        y_coords = np.arange(height).reshape(-1, 1)
        x_coords = np.arange(width).reshape(1, -1)
        
        return np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    @staticmethod
    def get_neighbors(num_neighbors: int = 8, shuffle: bool = True) -> list:
        """Get neighbor offsets for morphological operations"""
        if num_neighbors == 4:
            neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        elif num_neighbors == 8:
            neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        else:
            raise ValueError(f"Unsupported number of neighbors: {num_neighbors}")
        
        if shuffle:
            neighbors = neighbors.copy()
            np.random.shuffle(neighbors)
        
        return neighbors
    
    @staticmethod
    def dedupe_swap_pairs(src_y: np.ndarray, src_x: np.ndarray,
                         tgt_y: np.ndarray, tgt_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate and conflicting swap pairs"""
        if len(src_y) == 0:
            return src_y, src_x, tgt_y, tgt_x
        
        # Create unique identifiers for each swap
        src_ids = src_y * 100000 + src_x
        tgt_ids = tgt_y * 100000 + tgt_x
        
        # Find unique swaps (no duplicate sources or targets)
        unique_mask = np.ones(len(src_y), dtype=bool)
        
        # Remove swaps where source appears multiple times
        _, unique_src_indices = np.unique(src_ids, return_index=True)
        src_mask = np.zeros(len(src_y), dtype=bool)
        src_mask[unique_src_indices] = True
        
        # Remove swaps where target appears multiple times
        _, unique_tgt_indices = np.unique(tgt_ids, return_index=True)
        tgt_mask = np.zeros(len(src_y), dtype=bool)
        tgt_mask[unique_tgt_indices] = True
        
        # Keep only swaps that are unique in both source and target
        unique_mask = src_mask & tgt_mask
        
        # Also remove swaps where source and target are the same
        same_cell_mask = (src_y == tgt_y) & (src_x == tgt_x)
        unique_mask = unique_mask & ~same_cell_mask
        
        return src_y[unique_mask], src_x[unique_mask], tgt_y[unique_mask], tgt_x[unique_mask]
    
    @staticmethod
    def calculate_center_of_mass(material_types: np.ndarray, density: np.ndarray, 
                               height: int, width: int) -> tuple[float, float]:
        """Calculate center of mass for the planet"""
        # Only consider cells that contain matter (not space)
        matter_mask = (material_types != MaterialType.SPACE)
        
        if not np.any(matter_mask):
            return width / 2, height / 2
        
        # Create coordinate grids
        y_coords = np.arange(height).reshape(-1, 1)
        x_coords = np.arange(width).reshape(1, -1)
        
        # Calculate masses (density is already effective density including thermal expansion)
        masses = density * matter_mask.astype(float)
        
        # Calculate center of mass
        total_mass = np.sum(masses)
        if total_mass > 0:
            center_x = np.sum(masses * x_coords) / total_mass
            center_y = np.sum(masses * y_coords) / total_mass
        else:
            center_x = width / 2
            center_y = height / 2
        
        return center_x, center_y
    
    @staticmethod
    def get_planet_radius(material_types: np.ndarray, center_x: float, center_y: float) -> float:
        """Calculate planet radius as distance to farthest non-space cell"""
        non_space_mask = (material_types != MaterialType.SPACE)
        
        if not np.any(non_space_mask):
            return 1.0
        
        height, width = material_types.shape
        distances = SimulationUtils.get_distances_from_center(height, width, center_x, center_y)
        
        return np.max(distances[non_space_mask])
    
    @staticmethod
    def get_solar_direction(solar_angle: float) -> tuple[float, float]:
        """Calculate solar direction vector from angle"""
        # Convert angle to radians
        angle_rad = np.radians(solar_angle)
        
        # Solar direction vector (pointing toward sun)
        solar_dir_x = np.cos(angle_rad)
        solar_dir_y = np.sin(angle_rad)
        
        return solar_dir_x, solar_dir_y
    
    @staticmethod
    def get_mobile_mask(material_types: np.ndarray, temperature: np.ndarray, 
                       hot_solid_threshold: float = 1200.0) -> np.ndarray:
        """Get mask of materials that can move due to density stratification"""
        # Gases are always mobile
        gas_mask = (
            (material_types == MaterialType.AIR) |
            (material_types == MaterialType.WATER_VAPOR)
        )
        
        # Liquids are always mobile
        liquid_mask = (
            (material_types == MaterialType.WATER) |
            (material_types == MaterialType.MAGMA)
        )
        
        # Hot solids become mobile (ductile)
        hot_solid_mask = (
            (material_types != MaterialType.SPACE) &
            ~gas_mask & ~liquid_mask &
            (temperature > hot_solid_threshold + 273.15)  # Convert to Kelvin
        )
        
        # Light solids (ice, pumice) are mobile
        light_solid_mask = (
            (material_types == MaterialType.ICE) |
            (material_types == MaterialType.PUMICE)
        )
        
        return gas_mask | liquid_mask | hot_solid_mask | light_solid_mask
    
    @staticmethod
    def get_solid_mask(material_types: np.ndarray) -> np.ndarray:
        """Get mask of solid materials"""
        fluid_materials = {
            MaterialType.SPACE, MaterialType.AIR, MaterialType.WATER_VAPOR,
            MaterialType.WATER, MaterialType.MAGMA
        }
        
        return ~np.isin(material_types, list(fluid_materials))
    
    @staticmethod
    def create_laplacian_kernel_radius2() -> np.ndarray:
        """Create 13-point isotropic Laplacian kernel"""
        kernel = np.array([
            [0,  0,  1,  0,  0],
            [0,  2,  2,  2,  0],
            [1,  2, -16, 2,  1],
            [0,  2,  2,  2,  0],
            [0,  0,  1,  0,  0]
        ], dtype=np.float64) / 12.0
        
        return kernel 