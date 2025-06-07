"""
Rock classification system for geological simulation.
Defines rock types, properties, and metamorphic transitions.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

class RockType(Enum):
    """Primary rock classifications"""
    # Igneous rocks
    GRANITE = "granite"
    BASALT = "basalt"
    OBSIDIAN = "obsidian"
    PUMICE = "pumice"
    ANDESITE = "andesite"
    
    # Sedimentary rocks
    SANDSTONE = "sandstone"
    LIMESTONE = "limestone"
    SHALE = "shale"
    CONGLOMERATE = "conglomerate"
    
    # Metamorphic rocks
    GNEISS = "gneiss"
    SCHIST = "schist"
    SLATE = "slate"
    MARBLE = "marble"
    QUARTZITE = "quartzite"
    
    # Special states
    MAGMA = "magma"
    WATER = "water"
    AIR = "air"

@dataclass
class RockProperties:
    """Physical and chemical properties of rock types"""
    density: float  # kg/m³
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    melting_point: float  # °C
    strength: float  # MPa (compressive strength)
    porosity: float  # fraction (0-1)
    metamorphic_threshold_temp: float  # °C
    metamorphic_threshold_pressure: float  # MPa
    color_rgb: Tuple[int, int, int]  # RGB color for visualization

class RockDatabase:
    """Database of rock properties and metamorphic transitions"""
    
    def __init__(self):
        self.properties = self._init_rock_properties()
        self.metamorphic_transitions = self._init_metamorphic_transitions()
        self.weathering_products = self._init_weathering_products()
    
    def _init_rock_properties(self) -> Dict[RockType, RockProperties]:
        """Initialize physical properties for all rock types"""
        return {
            # Igneous rocks
            RockType.GRANITE: RockProperties(
                density=2650, thermal_conductivity=2.9, specific_heat=790,
                melting_point=1215, strength=200, porosity=0.01,
                metamorphic_threshold_temp=650, metamorphic_threshold_pressure=200,
                color_rgb=(255, 182, 193)  # Light pink - felsic igneous
            ),
            RockType.BASALT: RockProperties(
                density=3000, thermal_conductivity=1.7, specific_heat=840,
                melting_point=1200, strength=300, porosity=0.1,
                metamorphic_threshold_temp=500, metamorphic_threshold_pressure=150,
                color_rgb=(47, 79, 79)  # Dark slate gray - mafic igneous
            ),
            RockType.OBSIDIAN: RockProperties(
                density=2400, thermal_conductivity=1.2, specific_heat=840,
                melting_point=1000, strength=50, porosity=0.01,
                metamorphic_threshold_temp=600, metamorphic_threshold_pressure=100,
                color_rgb=(0, 0, 0)  # Black - volcanic glass
            ),
            RockType.PUMICE: RockProperties(
                density=600, thermal_conductivity=0.5, specific_heat=840,
                melting_point=1000, strength=20, porosity=0.8,
                metamorphic_threshold_temp=600, metamorphic_threshold_pressure=100,
                color_rgb=(245, 245, 245)  # White gray - vesicular volcanic
            ),
            RockType.ANDESITE: RockProperties(
                density=2800, thermal_conductivity=1.8, specific_heat=840,
                melting_point=1150, strength=250, porosity=0.05,
                metamorphic_threshold_temp=550, metamorphic_threshold_pressure=150,
                color_rgb=(105, 105, 105)  # Dim gray - intermediate volcanic
            ),
            
            # Sedimentary rocks
            RockType.SANDSTONE: RockProperties(
                density=2200, thermal_conductivity=2.5, specific_heat=830,
                melting_point=1650, strength=100, porosity=0.2,
                metamorphic_threshold_temp=300, metamorphic_threshold_pressure=50,
                color_rgb=(238, 203, 173)  # Tan - sandstone
            ),
            RockType.LIMESTONE: RockProperties(
                density=2600, thermal_conductivity=2.2, specific_heat=880,
                melting_point=825, strength=150, porosity=0.15,
                metamorphic_threshold_temp=400, metamorphic_threshold_pressure=100,
                color_rgb=(255, 255, 224)  # Light yellow - limestone
            ),
            RockType.SHALE: RockProperties(
                density=2400, thermal_conductivity=1.5, specific_heat=800,
                melting_point=1200, strength=80, porosity=0.3,
                metamorphic_threshold_temp=250, metamorphic_threshold_pressure=30,
                color_rgb=(139, 69, 19)  # Brown - mudstone/shale
            ),
            RockType.CONGLOMERATE: RockProperties(
                density=2300, thermal_conductivity=2.0, specific_heat=820,
                melting_point=1500, strength=120, porosity=0.25,
                metamorphic_threshold_temp=350, metamorphic_threshold_pressure=75,
                color_rgb=(205, 133, 63)  # Peru brown - conglomerate
            ),
            
            # Metamorphic rocks
            RockType.GNEISS: RockProperties(
                density=2700, thermal_conductivity=3.0, specific_heat=790,
                melting_point=1250, strength=250, porosity=0.02,
                metamorphic_threshold_temp=800, metamorphic_threshold_pressure=400,
                color_rgb=(128, 128, 128)  # Gray - high-grade metamorphic
            ),
            RockType.SCHIST: RockProperties(
                density=2800, thermal_conductivity=2.8, specific_heat=780,
                melting_point=1300, strength=200, porosity=0.05,
                metamorphic_threshold_temp=700, metamorphic_threshold_pressure=300,
                color_rgb=(85, 107, 47)  # Dark olive green - medium-grade metamorphic
            ),
            RockType.SLATE: RockProperties(
                density=2700, thermal_conductivity=2.0, specific_heat=800,
                melting_point=1400, strength=180, porosity=0.03,
                metamorphic_threshold_temp=500, metamorphic_threshold_pressure=200,
                color_rgb=(112, 128, 144)  # Slate gray - low-grade metamorphic
            ),
            RockType.MARBLE: RockProperties(
                density=2650, thermal_conductivity=2.5, specific_heat=880,
                melting_point=1000, strength=120, porosity=0.02,
                metamorphic_threshold_temp=600, metamorphic_threshold_pressure=250,
                color_rgb=(255, 250, 250)  # Snow white - metamorphosed limestone
            ),
            RockType.QUARTZITE: RockProperties(
                density=2650, thermal_conductivity=6.0, specific_heat=800,
                melting_point=1700, strength=300, porosity=0.01,
                metamorphic_threshold_temp=800, metamorphic_threshold_pressure=500,
                color_rgb=(255, 228, 196)  # Bisque - metamorphosed sandstone
            ),
            
            # Special states
            RockType.MAGMA: RockProperties(
                density=2800, thermal_conductivity=4.0, specific_heat=1200,
                melting_point=2000, strength=0, porosity=0,
                metamorphic_threshold_temp=2000, metamorphic_threshold_pressure=0,
                color_rgb=(255, 69, 0)  # Red orange - molten rock
            ),
            RockType.WATER: RockProperties(
                density=1000, thermal_conductivity=0.6, specific_heat=4186,
                melting_point=0, strength=0, porosity=1,
                metamorphic_threshold_temp=1000, metamorphic_threshold_pressure=0,
                color_rgb=(30, 144, 255)  # Dodger blue - water
            ),
            RockType.AIR: RockProperties(
                density=1.2, thermal_conductivity=0.024, specific_heat=1005,
                melting_point=-200, strength=0, porosity=1,
                metamorphic_threshold_temp=1000, metamorphic_threshold_pressure=0,
                color_rgb=(135, 206, 235)  # Sky blue - atmosphere
            ),
        }
    
    def _init_metamorphic_transitions(self) -> Dict[RockType, List[Tuple[RockType, float, float]]]:
        """Define metamorphic transition rules: parent -> [(product, min_temp, min_pressure), ...]"""
        return {
            RockType.SHALE: [
                (RockType.SLATE, 250, 30),
                (RockType.SCHIST, 500, 200),
                (RockType.GNEISS, 700, 400)
            ],
            RockType.SANDSTONE: [
                (RockType.QUARTZITE, 300, 50)
            ],
            RockType.LIMESTONE: [
                (RockType.MARBLE, 400, 100)
            ],
            RockType.GRANITE: [
                (RockType.GNEISS, 650, 200)
            ],
            RockType.BASALT: [
                (RockType.SCHIST, 500, 150),
                (RockType.GNEISS, 650, 300)
            ],
            RockType.SLATE: [
                (RockType.SCHIST, 500, 200),
                (RockType.GNEISS, 700, 400)
            ],
            RockType.SCHIST: [
                (RockType.GNEISS, 700, 400)
            ],
            RockType.QUARTZITE: [
                # Quartzite is already highly metamorphosed, minimal further transitions
            ],
            RockType.CONGLOMERATE: [
                (RockType.GNEISS, 600, 300)
            ]
        }
    
    def _init_weathering_products(self) -> Dict[RockType, List[RockType]]:
        """Define weathering products for surface processes"""
        return {
            RockType.GRANITE: [RockType.SANDSTONE],
            RockType.BASALT: [RockType.SHALE],
            RockType.GNEISS: [RockType.SANDSTONE],
            RockType.SCHIST: [RockType.SHALE],
            RockType.SLATE: [RockType.SHALE],
            RockType.MARBLE: [RockType.LIMESTONE],
            RockType.QUARTZITE: [RockType.SANDSTONE],
            RockType.PUMICE: [RockType.SANDSTONE],
            RockType.ANDESITE: [RockType.SHALE],
            RockType.CONGLOMERATE: [RockType.SANDSTONE]
        }
    
    def get_metamorphic_product(self, rock_type: RockType, temperature: float, pressure: float) -> Optional[RockType]:
        """Determine metamorphic product based on P-T conditions"""
        if rock_type not in self.metamorphic_transitions:
            return None
        
        transitions = self.metamorphic_transitions[rock_type]
        
        # Find the highest grade metamorphic product that conditions allow
        best_product = None
        for product, min_temp, min_pressure in transitions:
            if temperature >= min_temp and pressure >= min_pressure:
                best_product = product
        
        return best_product
    
    def should_melt(self, rock_type: RockType, temperature: float) -> bool:
        """Check if rock should melt at given temperature"""
        # Magma is already molten, so it doesn't "melt"
        if rock_type == RockType.MAGMA:
            return False
        return temperature >= self.properties[rock_type].melting_point
    
    def get_cooling_product(self, temperature: float, pressure: float, composition: str = "mafic") -> RockType:
        """Determine igneous rock type from cooling magma"""
        if composition == "felsic":
            if pressure > 100:  # Deep intrusive
                return RockType.GRANITE
            else:  # Shallow/extrusive
                return RockType.OBSIDIAN if temperature < 900 else RockType.PUMICE
        else:  # mafic composition
            if pressure > 50:
                return RockType.BASALT
            else:
                return RockType.ANDESITE
    
    def get_properties(self, rock_type: RockType) -> RockProperties:
        """Get properties for a rock type"""
        return self.properties[rock_type] 