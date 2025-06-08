"""
Material classification system for geological simulation.
Defines material types, properties, and transitions.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

class MaterialType(Enum):
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
    ICE = "ice"
    WATER_VAPOR = "water_vapor"
    AIR = "air"
    SPACE = "space"

@dataclass
class TransitionRule:
    """Defines a material transition under specific P-T conditions"""
    target: MaterialType  # What this material transitions to
    min_temp: float  # Minimum temperature (°C)
    max_temp: float  # Maximum temperature (°C) 
    min_pressure: float  # Minimum pressure (MPa)
    max_pressure: float  # Maximum pressure (MPa)
    description: str = ""  # Human-readable description
    
    def is_applicable(self, temperature: float, pressure: float) -> bool:
        """Check if this transition applies under given P-T conditions"""
        return (self.min_temp <= temperature <= self.max_temp and 
                self.min_pressure <= pressure <= self.max_pressure)

@dataclass
class MaterialProperties:
    """Physical and chemical properties of material types"""
    density: float  # kg/m³
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    strength: float  # MPa (compressive strength)
    porosity: float  # fraction (0-1)
    emissivity: float  # thermal emissivity (0-1) for radiative cooling
    albedo: float  # solar albedo (0-1) for solar reflection
    thermal_expansion: float  # volumetric thermal expansion coefficient (1/K)
    color_rgb: Tuple[int, int, int]  # RGB color for visualization
    transitions: List[TransitionRule] = field(default_factory=list)  # All possible transitions for this material
    is_solid: bool = True  # Whether rocks cannot fall through this material (default: solid)
    
    def get_applicable_transition(self, temperature: float, pressure: float) -> Optional[TransitionRule]:
        """Find the first applicable transition for given P-T conditions"""
        for transition in self.transitions:
            if transition.is_applicable(temperature, pressure):
                return transition
        return None
    
    def get_all_applicable_transitions(self, temperature: float, pressure: float) -> List[TransitionRule]:
        """Find all applicable transitions for given P-T conditions (for complex phase diagrams)"""
        return [t for t in self.transitions if t.is_applicable(temperature, pressure)]

class MaterialDatabase:
    """Database of material properties and transitions"""
    
    def __init__(self):
        self.properties = self._init_material_properties()
        self.weathering_products = self._init_weathering_products()
    
    def _init_material_properties(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize physical properties for all rock types"""
        return {
            # Igneous rocks
            MaterialType.GRANITE: MaterialProperties(
                density=2650, thermal_conductivity=2.9, specific_heat=790,
                strength=200, porosity=0.01,
                emissivity=0.9, albedo=0.25,  # High emissivity solid, moderate albedo (light colored)
                thermal_expansion=2.4e-5,  # Volumetric expansion coefficient for granite
                color_rgb=(255, 182, 193),  # Light pink - felsic igneous
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1215, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.GNEISS, 650, 1215, 200, float('inf'), "Metamorphism to gneiss")
                ]
            ),
            MaterialType.BASALT: MaterialProperties(
                density=3000, thermal_conductivity=1.7, specific_heat=840,
                strength=300, porosity=0.1,
                emissivity=0.9, albedo=0.15,  # High emissivity solid, low albedo (dark colored)
                thermal_expansion=2.8e-5,  # Volumetric expansion coefficient for basalt
                color_rgb=(47, 79, 79),  # Dark slate gray - mafic igneous
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1200, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SCHIST, 500, 1200, 150, float('inf'), "Metamorphism to schist")
                ]
            ),
            MaterialType.OBSIDIAN: MaterialProperties(
                density=2400, thermal_conductivity=1.2, specific_heat=840,
                strength=50, porosity=0.01,
                emissivity=0.9, albedo=0.05,  # High emissivity solid, very low albedo (black)
                thermal_expansion=2.2e-5,  # Volumetric expansion coefficient for obsidian (glass)
                color_rgb=(0, 0, 0),  # Black - volcanic glass
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1000, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            MaterialType.PUMICE: MaterialProperties(
                density=600, thermal_conductivity=0.5, specific_heat=840,
                strength=20, porosity=0.8,
                emissivity=0.9, albedo=0.35,  # High emissivity solid, high albedo (white/light)
                thermal_expansion=3.0e-5,  # Volumetric expansion coefficient for pumice (porous volcanic)
                color_rgb=(245, 245, 245),  # White gray - vesicular volcanic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1000, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            MaterialType.ANDESITE: MaterialProperties(
                density=2800, thermal_conductivity=1.8, specific_heat=840,
                strength=250, porosity=0.05,
                emissivity=0.9, albedo=0.20,  # High emissivity solid, moderate-low albedo (gray)
                thermal_expansion=2.6e-5,  # Volumetric expansion coefficient for andesite
                color_rgb=(105, 105, 105),  # Dim gray - intermediate volcanic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1150, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            
            # Sedimentary rocks
            MaterialType.SANDSTONE: MaterialProperties(
                density=2200, thermal_conductivity=2.5, specific_heat=830,
                strength=100, porosity=0.2,
                emissivity=0.9, albedo=0.30,  # High emissivity solid, moderate albedo (tan)
                thermal_expansion=3.6e-5,  # Volumetric expansion coefficient for sandstone (quartz-rich)
                color_rgb=(238, 203, 173),  # Tan - sandstone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1650, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.QUARTZITE, 300, 1650, 50, float('inf'), "Metamorphism to quartzite")
                ]
            ),
            MaterialType.LIMESTONE: MaterialProperties(
                density=2600, thermal_conductivity=2.2, specific_heat=880,
                strength=150, porosity=0.15,
                emissivity=0.9, albedo=0.35,  # High emissivity solid, high albedo (light colored)
                thermal_expansion=3.0e-5,  # Volumetric expansion coefficient for limestone (calcite)
                color_rgb=(255, 255, 224),  # Light yellow - limestone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 825, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.MARBLE, 400, 825, 100, float('inf'), "Metamorphism to marble")
                ]
            ),
            MaterialType.SHALE: MaterialProperties(
                density=2400, thermal_conductivity=1.5, specific_heat=800,
                strength=80, porosity=0.3,
                emissivity=0.9, albedo=0.18,  # High emissivity solid, low albedo (brown/dark)
                thermal_expansion=4.2e-5,  # Volumetric expansion coefficient for shale (clay-rich)
                color_rgb=(139, 69, 19),  # Brown - mudstone/shale
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1200, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SLATE, 250, 1200, 30, float('inf'), "Low-grade metamorphism to slate"),
                    TransitionRule(MaterialType.SCHIST, 500, 1200, 200, float('inf'), "Medium-grade metamorphism to schist"),
                    TransitionRule(MaterialType.GNEISS, 700, 1200, 400, float('inf'), "High-grade metamorphism to gneiss")
                ]
            ),
            MaterialType.CONGLOMERATE: MaterialProperties(
                density=2300, thermal_conductivity=2.0, specific_heat=820,
                strength=120, porosity=0.25,
                emissivity=0.9, albedo=0.22,  # High emissivity solid, moderate albedo (brown)
                thermal_expansion=3.3e-5,  # Volumetric expansion coefficient for conglomerate (mixed composition)
                color_rgb=(205, 133, 63),  # Peru brown - conglomerate
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1500, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.GNEISS, 600, 1500, 300, float('inf'), "High-grade metamorphism to gneiss")
                ]
            ),
            
            # Metamorphic rocks
            MaterialType.GNEISS: MaterialProperties(
                density=2700, thermal_conductivity=3.0, specific_heat=790,
                strength=250, porosity=0.02,
                emissivity=0.9, albedo=0.25,  # High emissivity solid, moderate albedo (gray)
                thermal_expansion=2.7e-5,  # Volumetric expansion coefficient for gneiss
                color_rgb=(128, 128, 128),  # Gray - high-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1250, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            MaterialType.SCHIST: MaterialProperties(
                density=2800, thermal_conductivity=2.8, specific_heat=780,
                strength=200, porosity=0.05,
                emissivity=0.9, albedo=0.16,  # High emissivity solid, low albedo (dark green)
                thermal_expansion=3.3e-5,  # Volumetric expansion coefficient for schist (mica-rich)
                color_rgb=(85, 107, 47),  # Dark olive green - medium-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1300, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.GNEISS, 700, 1300, 400, float('inf'), "High-grade metamorphism to gneiss")
                ]
            ),
            MaterialType.SLATE: MaterialProperties(
                density=2700, thermal_conductivity=2.0, specific_heat=800,
                strength=180, porosity=0.03,
                emissivity=0.9, albedo=0.23,  # High emissivity solid, moderate albedo (slate gray)
                thermal_expansion=3.9e-5,  # Volumetric expansion coefficient for slate (clay-derived)
                color_rgb=(112, 128, 144),  # Slate gray - low-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1400, float('inf'), 0, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SCHIST, 500, 1400, 200, float('inf'), "Metamorphism to schist"),
                    TransitionRule(MaterialType.GNEISS, 700, 1400, 400, float('inf'), "High-grade metamorphism to gneiss")
                ]
            ),
            MaterialType.MARBLE: MaterialProperties(
                density=2650, thermal_conductivity=2.5, specific_heat=880,
                strength=120, porosity=0.02,
                emissivity=0.9, albedo=0.40,  # High emissivity solid, high albedo (white)
                thermal_expansion=3.6e-5,  # Volumetric expansion coefficient for marble (calcite)
                color_rgb=(255, 250, 250),  # Snow white - metamorphosed limestone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 825, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            MaterialType.QUARTZITE: MaterialProperties(
                density=2650, thermal_conductivity=6.0, specific_heat=800,
                strength=300, porosity=0.01,
                emissivity=0.9, albedo=0.32,  # High emissivity solid, moderate-high albedo (light bisque)
                thermal_expansion=3.6e-5,  # Volumetric expansion coefficient for quartzite (quartz)
                color_rgb=(255, 228, 196),  # Bisque - metamorphosed sandstone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1700, float('inf'), 0, float('inf'), "Melting to magma")
                ]
            ),
            
            # Special states
            MaterialType.MAGMA: MaterialProperties(
                density=2800, thermal_conductivity=4.0, specific_heat=1200,
                strength=0, porosity=0,
                emissivity=0.95, albedo=0.10,  # Very high emissivity (hot molten), low albedo (dark/red)
                thermal_expansion=9.0e-5,  # High volumetric expansion coefficient for molten rock
                color_rgb=(255, 69, 0),  # Red orange - molten rock
                transitions=[
                    TransitionRule(MaterialType.OBSIDIAN, float('-inf'), 600, 0, 10, "Rapid cooling to obsidian (shallow/surface)"),
                    TransitionRule(MaterialType.BASALT, 600, 900, 0, 50, "Moderate cooling to basalt"),
                    TransitionRule(MaterialType.GRANITE, 700, 1000, 50, float('inf'), "Slow deep cooling to granite")
                ],
                is_solid=False  # Molten rock - rocks can sink through it
            ),
            MaterialType.WATER: MaterialProperties(
                density=1000, thermal_conductivity=0.6, specific_heat=4186,
                strength=0, porosity=1,
                emissivity=0.96, albedo=0.08,  # Very high emissivity (good blackbody), very low albedo (dark)
                thermal_expansion=2.1e-4,  # Volumetric expansion coefficient for water (temperature dependent)
                color_rgb=(30, 144, 255),  # Dodger blue - water
                transitions=[
                    TransitionRule(MaterialType.ICE, float('-inf'), 0, 0, float('inf'), "Freezing to ice"),
                    TransitionRule(MaterialType.WATER_VAPOR, 100, float('inf'), 0, float('inf'), "Vaporization to water vapor")
                ],
                is_solid=False  # Liquid - rocks can sink through it
            ),
            MaterialType.ICE: MaterialProperties(
                density=920, thermal_conductivity=2.2, specific_heat=2108,
                strength=10, porosity=0.1,
                emissivity=0.95, albedo=0.80,  # High emissivity solid, very high albedo (highly reflective)
                thermal_expansion=1.6e-4,  # Volumetric expansion coefficient for ice
                color_rgb=(173, 216, 230),  # Light blue - ice
                transitions=[
                    TransitionRule(MaterialType.WATER, 0, float('inf'), 0, float('inf'), "Melting to water"),
                    TransitionRule(MaterialType.WATER_VAPOR, -10, 0, 0, 0.1, "Sublimation to water vapor")
                ]
            ),
            MaterialType.WATER_VAPOR: MaterialProperties(
                density=0.6, thermal_conductivity=0.025, specific_heat=2010,
                strength=0, porosity=1,
                emissivity=0.10, albedo=0.70,  # Low emissivity (gas), high albedo (cloud-like)
                thermal_expansion=3.7e-3,  # High volumetric expansion coefficient for gas (ideal gas law)
                color_rgb=(192, 224, 255),  # Light blue-white - humid air/steam
                transitions=[
                    TransitionRule(MaterialType.WATER, float('-inf'), 100, 0, float('inf'), "Condensation to water")
                ],
                is_solid=False  # Gas - rocks can fall through it
            ),
            MaterialType.AIR: MaterialProperties(
                density=1.2, thermal_conductivity=0.024, specific_heat=1005,
                strength=0, porosity=1,
                emissivity=0.05, albedo=0.20,  # Very low emissivity (transparent gas), low albedo
                thermal_expansion=3.7e-3,  # High volumetric expansion coefficient for gas (ideal gas law)
                color_rgb=(245, 245, 255),  # Very light blue/white - dry air
                transitions=[],  # Dry air doesn't transition (no water content to condense)
                is_solid=False  # Gas - rocks can fall through it
            ),
            MaterialType.SPACE: MaterialProperties(
                density=0.0, thermal_conductivity=0.0, specific_heat=0.0,
                strength=0, porosity=1,
                emissivity=0.0, albedo=0.0,  # No emissivity or albedo (perfect vacuum)
                thermal_expansion=0.0,  # No thermal expansion in vacuum
                color_rgb=(0, 0, 0),  # Black - vacuum of space
                transitions=[],  # Space doesn't transition to anything
                is_solid=False  # Vacuum - rocks fall through it infinitely fast
            )
        }
    
    def _init_weathering_products(self) -> Dict[MaterialType, List[MaterialType]]:
        """Define weathering products for surface processes"""
        return {
            MaterialType.GRANITE: [MaterialType.SANDSTONE],
            MaterialType.BASALT: [MaterialType.SHALE],
            MaterialType.GNEISS: [MaterialType.SANDSTONE],
            MaterialType.SCHIST: [MaterialType.SHALE],
            MaterialType.SLATE: [MaterialType.SHALE],
            MaterialType.MARBLE: [MaterialType.LIMESTONE],
            MaterialType.QUARTZITE: [MaterialType.SANDSTONE],
            MaterialType.PUMICE: [MaterialType.SANDSTONE],
            MaterialType.ANDESITE: [MaterialType.SHALE],
            MaterialType.CONGLOMERATE: [MaterialType.SANDSTONE]
        }
    
    def get_metamorphic_product(self, material_type: MaterialType, temperature: float, pressure: float) -> Optional[MaterialType]:
        """Determine metamorphic product based on P-T conditions using new transition system"""
        props = self.get_properties(material_type)
        transition = props.get_applicable_transition(temperature, pressure)
        return transition.target if transition else None
    
    def should_melt(self, material_type: MaterialType, temperature: float) -> bool:
        """Check if material should melt at given temperature using new transition system"""
        # Magma is already molten, so it doesn't "melt"
        if material_type == MaterialType.MAGMA:
            return False
            
        props = self.get_properties(material_type)
        # Check if any transition leads to MAGMA
        for transition in props.transitions:
            if (transition.target == MaterialType.MAGMA and 
                transition.min_temp <= temperature <= transition.max_temp):
                return True
        return False
    
    def get_cooling_product(self, temperature: float, pressure: float, composition: str = "mafic") -> MaterialType:
        """Determine igneous rock type from cooling magma"""
        if composition == "felsic":
            if pressure > 100:  # Deep intrusive
                return MaterialType.GRANITE
            else:  # Shallow/extrusive
                return MaterialType.OBSIDIAN if temperature < 900 else MaterialType.PUMICE
        else:  # mafic composition
            if pressure > 50:
                return MaterialType.BASALT
            else:
                return MaterialType.ANDESITE
    
    def get_properties(self, material_type: MaterialType) -> MaterialProperties:
        """Get properties for a material type"""
        return self.properties[material_type]
    
    def get_weathering_products(self, material_type: MaterialType) -> List[MaterialType]:
        """Get weathering products for a material type"""
        return self.weathering_products.get(material_type, []) 