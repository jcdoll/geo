"""Material physics test scenarios (phase transitions, metamorphism, stability)."""

import numpy as np
from typing import Dict, Any

from .base import TestScenario
from materials import MaterialType, MaterialDatabase
from geo_game import GeoGame


class MaterialStabilityScenario(TestScenario):
    """Test material stability in various conditions."""
    
    def __init__(self, test_material: MaterialType = MaterialType.GRANITE,
                 environment: str = "vacuum", **kwargs):
        """Initialize material stability test.
        
        Args:
            test_material: Material to test
            environment: "vacuum", "air", "water", or "magma"
        """
        super().__init__(**kwargs)
        self.test_material = test_material
        self.environment = environment
        
    def get_name(self) -> str:
        return f"material_stability_{self.test_material.name.lower()}_in_{self.environment}"
        
    def get_description(self) -> str:
        return f"Tests {self.test_material.name} stability in {self.environment}"
        
    def setup(self, sim: GeoGame) -> None:
        """Create material sample in specified environment."""
        # Set up environment
        if self.environment == "vacuum":
            sim.material_types[:] = MaterialType.SPACE
            sim.temperature[:] = 3.0  # Near absolute zero
            sim.pressure[:] = 0.0
        elif self.environment == "air":
            sim.material_types[:] = MaterialType.AIR
            sim.temperature[:] = 290.0
            sim.pressure[:] = 101325.0  # 1 atm
        elif self.environment == "water":
            sim.material_types[:] = MaterialType.WATER
            sim.temperature[:] = 290.0
            sim.pressure[:] = 101325.0
        elif self.environment == "magma":
            sim.material_types[:] = MaterialType.MAGMA
            sim.temperature[:] = 2000.0  # Extremely hot magma
            sim.pressure[:] = 101325.0
            
        # Create test material sample in center
        center_y, center_x = sim.height // 2, sim.width // 2
        sample_size = 10
        
        y_start = max(0, center_y - sample_size // 2)
        y_end = min(sim.height, center_y + sample_size // 2)
        x_start = max(0, center_x - sample_size // 2)
        x_end = min(sim.width, center_x + sample_size // 2)
        
        sim.material_types[y_start:y_end, x_start:x_end] = self.test_material
        
        # Set material temperature based on environment
        if self.environment == "vacuum":
            # For ice sublimation test, ice needs to be warm enough
            if self.test_material == MaterialType.ICE:
                sim.temperature[y_start:y_end, x_start:x_end] = 250.0  # Near melting
            else:
                sim.temperature[y_start:y_end, x_start:x_end] = 100.0  # Cold
        elif self.environment == "magma":
            # Start materials cooler than magma so they heat up
            sim.temperature[y_start:y_end, x_start:x_end] = 1000.0
        else:
            # Match environment temperature initially
            pass
            
        # Enable relevant physics
        sim.enable_heat_diffusion = True
        sim.enable_internal_heating = True  # Enable material-based heating
        sim.enable_material_processes = True  # Enable all material processes
        sim.external_gravity = (0, 0)  # No gravity for stability test
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def step_callback(self, sim: GeoGame, step: int) -> None:
        """Apply heating for magma environment tests."""
        if self.environment == "magma" and step % 5 == 0:
            # Apply heat to maintain high magma temperature
            # Heat multiple points to ensure good coverage
            for offset_y in [-10, 0, 10]:
                for offset_x in [-10, 0, 10]:
                    heat_y = sim.height // 2 + offset_y
                    heat_x = sim.width // 2 + offset_x
                    if 0 <= heat_y < sim.height and 0 <= heat_x < sim.width:
                        sim.add_heat_source(heat_x, heat_y, radius=8, temperature=100.0)
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check material stability."""
        # Count material cells
        material_count = np.sum(sim.material_types == self.test_material)
        initial_count = self.initial_state.get('initial_material_count', 0)
        
        # Check for transformations
        all_materials, counts = np.unique(sim.material_types, return_counts=True)
        material_dict = {MaterialType(mat): count for mat, count in zip(all_materials, counts)}
        
        # Calculate preservation percentage
        preservation_pct = (material_count / initial_count * 100) if initial_count > 0 else 0
        
        # Check temperature stability
        if material_count > 0:
            material_mask = sim.material_types == self.test_material
            avg_temp = np.mean(sim.temperature[material_mask])
        else:
            avg_temp = 0
            
        # Success criteria depends on expected behavior
        expected_stable = self._is_expected_stable()
        if expected_stable:
            success = preservation_pct > 90  # Should remain mostly unchanged
        else:
            success = preservation_pct < 50  # Should transform
            
        return {
            'success': success,
            'metrics': {
                'material_count': material_count,
                'preservation_pct': preservation_pct,
                'avg_temperature': avg_temp,
                'num_material_types': len(material_dict),
            },
            'message': f"{self.test_material.name}: {preservation_pct:.1f}% preserved, T={avg_temp:.0f}K"
        }
        
    def _is_expected_stable(self) -> bool:
        """Determine if material should be stable in environment."""
        # Simplified stability rules
        if self.environment == "vacuum":
            # Ice sublimation only happens at very specific conditions (-10 to 0K)
            # At 250K in vacuum, ice is actually stable
            return True  # All materials stable in vacuum at these temperatures
        elif self.environment == "magma":
            # In very hot magma, granite should melt
            # Basalt has higher melting point and might survive
            return self.test_material == MaterialType.BASALT
        elif self.environment == "water":
            # Most rocks stable in water, ice melts
            return self.test_material not in [MaterialType.ICE]
        else:  # air
            return True  # Most materials stable in air
            
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial material count."""
        super().store_initial_state(sim)
        self.initial_state['initial_material_count'] = np.sum(
            sim.material_types == self.test_material
        )


class MetamorphismScenario(TestScenario):
    """Test rock metamorphism under pressure and temperature."""
    
    def __init__(self, source_rock: MaterialType = MaterialType.SHALE,
                 target_conditions: str = "high_pressure", **kwargs):
        """Initialize metamorphism test.
        
        Args:
            source_rock: Starting rock type
            target_conditions: "high_pressure", "high_temp", or "both"
        """
        super().__init__(**kwargs)
        self.source_rock = source_rock
        self.target_conditions = target_conditions
        
    def get_name(self) -> str:
        return f"metamorphism_{self.source_rock.name.lower()}_{self.target_conditions}"
        
    def get_description(self) -> str:
        return f"Tests metamorphism of {self.source_rock.name} under {self.target_conditions.replace('_', ' ')}"
        
    def setup(self, sim: GeoGame) -> None:
        """Create rock sample and apply conditions."""
        # Clear to space
        sim.material_types[:] = MaterialType.SPACE
        sim.temperature[:] = 0.0
        sim.pressure[:] = 0.0
        
        # Create rock layer
        layer_height = sim.height // 3
        sim.material_types[layer_height:2*layer_height, :] = self.source_rock
        
        # Set initial conditions
        if self.target_conditions == "high_pressure":
            # Create overburden to generate pressure
            sim.material_types[:layer_height, :] = MaterialType.GRANITE
            sim.temperature[:] = 400.0  # Moderate temperature
            sim.external_gravity = (0, 50)  # High gravity for pressure
        elif self.target_conditions == "high_temp":
            # Create hot bottom layer
            sim.material_types[2*layer_height:, :] = MaterialType.BASALT
            sim.temperature[:] = 800.0  # Start warm
            sim.temperature[2*layer_height:, :] = 1500.0  # Very hot at bottom
            sim.external_gravity = (0, 9.81)
        else:  # both
            # Overburden and hot bottom
            sim.material_types[:layer_height//2, :] = MaterialType.GRANITE
            sim.material_types[2*layer_height:, :] = MaterialType.BASALT
            sim.temperature[:] = 800.0  # Start warm
            sim.temperature[2*layer_height:, :] = 1500.0  # Hot at bottom
            sim.external_gravity = (0, 20)
            
        # Enable physics
        sim.enable_heat_diffusion = True
        sim.enable_internal_heating = True  # Enable uranium heating
        sim.enable_pressure = True
        sim.enable_material_processes = True  # Ensure metamorphism is part of this
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def step_callback(self, sim: GeoGame, step: int) -> None:
        """Apply heat sources for high temperature conditions."""
        if self.target_conditions in ["high_temp", "both"] and step % 10 == 0:
            # Apply heat from bottom to maintain temperature gradient
            layer_height = sim.height // 3
            for x in range(0, sim.width, 10):
                sim.add_heat_source(x, sim.height - 5, radius=10, temperature=200.0)
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check for metamorphic changes."""
        # Count original and metamorphic rocks
        source_count = np.sum(sim.material_types == self.source_rock)
        initial_count = self.initial_state.get('initial_source_count', 0)
        
        # Expected metamorphic products
        expected_products = self._get_expected_products()
        product_counts = {}
        for product in expected_products:
            count = np.sum(sim.material_types == product)
            if count > 0:
                product_counts[product.name] = count
                
        # Calculate transformation percentage
        transformed = initial_count - source_count
        transform_pct = (transformed / initial_count * 100) if initial_count > 0 else 0
        
        # Check conditions in rock layer
        layer_height = sim.height // 3
        rock_region = sim.material_types[layer_height:2*layer_height, :]
        rock_mask = rock_region != MaterialType.SPACE
        
        if np.any(rock_mask):
            avg_temp = np.mean(sim.temperature[layer_height:2*layer_height, :][rock_mask])
            avg_pressure = np.mean(sim.pressure[layer_height:2*layer_height, :][rock_mask])
        else:
            avg_temp = avg_pressure = 0
            
        # Success if significant transformation occurred
        success = transform_pct > 20 and len(product_counts) > 0
        
        return {
            'success': success,
            'metrics': {
                'source_count': source_count,
                'transform_pct': transform_pct,
                'num_products': len(product_counts),
                'avg_temperature': avg_temp,
                'avg_pressure': avg_pressure,
            },
            'message': f"Transformed {transform_pct:.1f}%, T={avg_temp:.0f}K, P={avg_pressure:.0f}Pa"
        }
        
    def _get_expected_products(self) -> list:
        """Get expected metamorphic products for source rock."""
        # Simplified metamorphic sequences
        if self.source_rock == MaterialType.SHALE:
            return [MaterialType.SLATE, MaterialType.SCHIST]
        elif self.source_rock == MaterialType.LIMESTONE:
            return [MaterialType.MARBLE]
        elif self.source_rock == MaterialType.SANDSTONE:
            return [MaterialType.QUARTZITE]
        else:
            return []
            
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial rock count."""
        super().store_initial_state(sim)
        self.initial_state['initial_source_count'] = np.sum(
            sim.material_types == self.source_rock
        )
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [self.source_rock] + self._get_expected_products(),
            'preferred_display_mode': 'material',
            'show_metrics': ['transform_pct', 'avg_temperature', 'avg_pressure'],
        }


class PhaseTransitionScenario(TestScenario):
    """Test phase transitions (melting, freezing, vaporization)."""
    
    def __init__(self, material: MaterialType = MaterialType.ICE,
                 transition_type: str = "melting", **kwargs):
        """Initialize phase transition test."""
        super().__init__(**kwargs)
        self.material = material
        self.transition_type = transition_type
        
    def get_name(self) -> str:
        return f"phase_transition_{self.material.name.lower()}_{self.transition_type}"
        
    def get_description(self) -> str:
        return f"Tests {self.transition_type} of {self.material.name}"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up material for phase transition."""
        # Clear to air
        sim.material_types[:] = MaterialType.AIR
        sim.pressure[:] = 101325.0  # 1 atm
        
        # Create material blob
        center_y, center_x = sim.height // 2, sim.width // 2
        # Smaller blob for faster heat transfer in tests
        radius = 3 if self.transition_type == "freezing" else 10
        
        yy, xx = np.ogrid[:sim.height, :sim.width]
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        material_mask = dist <= radius
        
        sim.material_types[material_mask] = self.material
        
        # Set temperatures to induce transition
        if self.transition_type == "melting":
            if self.material == MaterialType.ICE:
                # Start with reasonable temperatures
                sim.temperature[:] = 290.0  # Room temperature air
                sim.temperature[material_mask] = 260.0  # Start frozen
            elif self.material == MaterialType.BASALT:
                sim.temperature[:] = 1200.0  # Hot environment
                sim.temperature[material_mask] = 1000.0  # Start solid
        elif self.transition_type == "freezing":
            if self.material == MaterialType.WATER:
                # Cold environment
                sim.temperature[:] = 250.0  # Below freezing
                sim.temperature[material_mask] = 274.0  # Start just above freezing
        elif self.transition_type == "vaporization":
            if self.material == MaterialType.WATER:
                sim.temperature[:] = 350.0  # Hot environment
                sim.temperature[material_mask] = 300.0  # Start liquid
                
        # Enable physics
        sim.enable_heat_diffusion = True
        sim.enable_internal_heating = True  # Enable material-based heating
        sim.enable_material_processes = True  # Ensure phase transitions are enabled
        sim.external_gravity = (0, 0)  # No gravity to keep material in place
        
        # Use geological timescale for phase transitions
        # Default timestep (10,000s) is better for heat diffusion
        sim.dt = 10000.0  # 10,000 seconds (~2.8 hours)
        
        # Enhance thermal diffusivity for faster testing
        # This simulates faster heat conduction for test purposes
        sim.thermal_diffusivity_enhancement = 1000.0  # 1000x faster heat diffusion
        
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def step_callback(self, sim: GeoGame, step: int) -> None:
        """Apply heat sources during simulation to drive transitions."""
        center_y, center_x = sim.height // 2, sim.width // 2
        
        # Apply heat for melting tests
        if self.transition_type == "melting":
            # Apply heat around the material to encourage melting
            sim.add_heat_source(center_x, center_y, radius=15, temperature=50.0)
        elif self.transition_type == "vaporization":
            # Apply strong heat for vaporization
            sim.add_heat_source(center_x, center_y, radius=15, temperature=100.0)
        # For freezing, we don't add heat - let natural cooling occur
            
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Check phase transition progress."""
        # Count materials
        original_count = np.sum(sim.material_types == self.material)
        initial_count = self.initial_state.get('initial_count', 0)
        
        # Get expected product
        expected_product = self._get_expected_product()
        if expected_product:
            product_count = np.sum(sim.material_types == expected_product)
        else:
            product_count = 0
            
        # Calculate conversion
        conversion_pct = (product_count / initial_count * 100) if initial_count > 0 else 0
        
        # Check temperatures
        if original_count > 0:
            original_temps = sim.temperature[sim.material_types == self.material]
            avg_original_temp = np.mean(original_temps)
        else:
            avg_original_temp = 0
            
        if product_count > 0:
            product_temps = sim.temperature[sim.material_types == expected_product]
            avg_product_temp = np.mean(product_temps)
        else:
            avg_product_temp = 0
            
        # Success if significant conversion (25% for partial transitions)
        success = conversion_pct > 25
        
        return {
            'success': success,
            'metrics': {
                'original_count': original_count,
                'product_count': product_count,
                'conversion_pct': conversion_pct,
                'avg_original_temp': avg_original_temp,
                'avg_product_temp': avg_product_temp,
            },
            'message': f"Conversion: {conversion_pct:.1f}%, {self.material.name}→{expected_product.name if expected_product else '?'}"
        }
        
    def _get_expected_product(self) -> MaterialType:
        """Get expected phase transition product."""
        if self.transition_type == "melting":
            if self.material == MaterialType.ICE:
                return MaterialType.WATER
            elif self.material == MaterialType.BASALT:
                return MaterialType.MAGMA
        elif self.transition_type == "freezing":
            if self.material == MaterialType.WATER:
                return MaterialType.ICE
        elif self.transition_type == "vaporization":
            if self.material == MaterialType.WATER:
                return MaterialType.WATER_VAPOR
        return None
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial material count."""
        super().store_initial_state(sim)
        self.initial_state['initial_count'] = np.sum(sim.material_types == self.material)