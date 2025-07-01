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
    # Fluids
    SPACE = "space"
    AIR = "air"
    WATER = "water"
    ICE = "ice"
    WATER_VAPOR = "water_vapor"
    
    # Special
    MAGMA = "magma"
    URANIUM = "uranium"
    
    # Rocks
    SAND = "sand"
    SANDSTONE = "sandstone"
    SHALE = "shale"
    LIMESTONE = "limestone"
    GRANITE = "granite"
    BASALT = "basalt"
    SLATE = "slate"
    SCHIST = "schist"
    GNEISS = "gneiss"
    MARBLE = "marble"

    # ------------------------------------------------------------------
    # Enable ordering so that high-level numpy helpers like np.unique can
    # operate directly on arrays of MaterialType.  Ordering is defined by the
    # lexicographic order of their string value which is deterministic and
    # adequate for set-like operations (no physical meaning implied).
    # ------------------------------------------------------------------
    def __lt__(self, other):  # type: ignore[override]
        if isinstance(other, MaterialType):
            return self.value < other.value
        return NotImplemented

@dataclass
class TransitionRule:
    """Defines a material transition under specific P-T conditions"""
    target: MaterialType  # What this material transitions to
    min_temp: float  # Minimum temperature (°C)
    max_temp: float  # Maximum temperature (°C) 
    description: str = ""  # Human-readable description
    probability: float = 1.0  # Probability of the transition occurring
    
    def is_applicable(self, temperature: float) -> bool:
        """Check if this transition applies under given temperature"""
        return self.min_temp <= temperature <= self.max_temp

@dataclass
class MaterialProperties:
    """Physical and chemical properties of material types"""
    density: float  # kg/m³
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    color_rgb: Tuple[int, int, int]  # RGB color for visualization
    emissivity: float = 0.0  # thermal emissivity (0-1) for radiative cooling
    thermal_expansion: float = 0.0  # volumetric expansion coefficient (1/K)
    transitions: List[TransitionRule] = field(default_factory=list)  # All possible transitions for this material
    heat_generation: float = 0.0  # W/m³ - volumetric heat generation rate (for radioactive materials)
    swap_probability: float = 0.1  # Probability of swapping with another material (0-1)
    
    def get_applicable_transition(self, temperature: float) -> Optional[TransitionRule]:
        """Find the first applicable transition for given temperature"""
        for transition in self.transitions:
            if transition.is_applicable(temperature):
                # Check probability for non-guaranteed transitions
                if transition.probability < 1.0:
                    if np.random.random() > transition.probability:
                        continue  # Skip this transition based on probability
                return transition
        return None
    
    def get_all_applicable_transitions(self, temperature: float) -> List[TransitionRule]:
        """Find all applicable transitions for given temperature"""
        applicable_transitions = []
        for transition in self.transitions:
            if transition.is_applicable(temperature):
                # Check probability for non-guaranteed transitions
                if transition.probability < 1.0:
                    if np.random.random() > transition.probability:
                        continue  # Skip this transition based on probability
                applicable_transitions.append(transition)
        return applicable_transitions

class MaterialDatabase:
    """Database of material properties and transitions"""
    
    def __init__(self):
        self.properties = self._init_material_properties()
        self.weathering_products = self._init_weathering_products()
        self.absorption_coeff = self._init_optical_absorption()
    
    def _init_material_properties(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize physical properties for all rock types"""
        return {
            # Igneous rocks
            MaterialType.GRANITE: MaterialProperties(
                density=2650, thermal_conductivity=2.9, specific_heat=790,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=2.4e-5,  # Volumetric expansion coefficient for granite
                color_rgb=(255, 182, 193),  # Light pink - felsic igneous
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1215, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.GNEISS, 650, 1215, "High-temp metamorphism to gneiss"),
                ],
                swap_probability=0.1  # Solid rock - low swap probability
            ),
            MaterialType.BASALT: MaterialProperties(
                density=3000, thermal_conductivity=1.7, specific_heat=840,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=2.8e-5,  # Volumetric expansion coefficient for basalt
                color_rgb=(47, 79, 79),  # Dark slate gray - mafic igneous
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1200, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SCHIST, 500, 1200, "Mid-temp metamorphism to schist"),
                ],
                swap_probability=0.1  # Solid rock - low swap probability
            ),
            
            # Sedimentary rocks
            MaterialType.LIMESTONE: MaterialProperties(
                density=2600, thermal_conductivity=2.2, specific_heat=880,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.0e-5,  # Volumetric expansion coefficient for limestone (calcite)
                color_rgb=(255, 255, 224),  # Light yellow - limestone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 825, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.MARBLE, 400, 825, "Contact metamorphism to marble"),
                ],
                swap_probability=0.15  # Sedimentary rock - moderate swapping
            ),
            MaterialType.SHALE: MaterialProperties(
                density=2400, thermal_conductivity=1.5, specific_heat=800,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=4.2e-5,  # Volumetric expansion coefficient for shale (clay-rich)
                color_rgb=(139, 69, 19),  # Brown - mudstone/shale
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1200, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SLATE, 250, 500, "Low-grade metamorphism to slate"),
                    TransitionRule(MaterialType.SCHIST, 500, 700, "Medium-grade metamorphism to schist"),
                    TransitionRule(MaterialType.GNEISS, 700, 1200, "High-grade metamorphism to gneiss")
                ],
                swap_probability=0.2  # Softer sedimentary rock
            ),
            MaterialType.SAND: MaterialProperties(
                density=1500, thermal_conductivity=0.8, specific_heat=800,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.6e-5,  # Similar to quartz
                color_rgb=(255, 218, 185),  # Peach puff - loose sand
                transitions=[
                    # Sand doesn't transform without pressure in CA
                ],
                swap_probability=0.5  # Loose sand - easy swapping
            ),
            MaterialType.SANDSTONE: MaterialProperties(
                density=2200, thermal_conductivity=2.0, specific_heat=850,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.5e-5,  # Similar to quartz
                color_rgb=(210, 180, 140),  # Tan - sandstone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1200, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SCHIST, 600, 1200, "Metamorphism to schist"),
                ],
                swap_probability=0.15  # Solid sedimentary rock
            ),
            
            # Metamorphic rocks
            MaterialType.GNEISS: MaterialProperties(
                density=2700, thermal_conductivity=3.0, specific_heat=790,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=2.7e-5,  # Volumetric expansion coefficient for gneiss
                color_rgb=(128, 128, 128),  # Gray - high-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1250, float('inf'), "Melting to magma"),
                ],
                swap_probability=0.1  # Hard metamorphic rock
            ),
            MaterialType.SCHIST: MaterialProperties(
                density=2800, thermal_conductivity=2.8, specific_heat=780,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.3e-5,  # Volumetric expansion coefficient for schist (mica-rich)
                color_rgb=(85, 107, 47),  # Dark olive green - medium-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1300, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.GNEISS, 700, 1300, "Further metamorphism to gneiss"),
                ],
                swap_probability=0.1  # Hard metamorphic rock
            ),
            MaterialType.SLATE: MaterialProperties(
                density=2700, thermal_conductivity=2.0, specific_heat=800,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.9e-5,  # Volumetric expansion coefficient for slate (clay-derived)
                color_rgb=(112, 128, 144),  # Slate gray - low-grade metamorphic
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 1400, float('inf'), "Melting to magma"),
                    TransitionRule(MaterialType.SCHIST, 500, 700, "Further metamorphism to schist"),
                    TransitionRule(MaterialType.GNEISS, 700, 1400, "High-grade metamorphism to gneiss"),
                ],
                swap_probability=0.15  # Medium-hard metamorphic rock
            ),
            MaterialType.MARBLE: MaterialProperties(
                density=2650, thermal_conductivity=2.5, specific_heat=880,
                emissivity=0.9,  # High emissivity solid
                thermal_expansion=3.6e-5,  # Volumetric expansion coefficient for marble (calcite)
                color_rgb=(255, 250, 250),  # Snow white - metamorphosed limestone
                transitions=[
                    TransitionRule(MaterialType.MAGMA, 825, float('inf'), "Melting to magma")
                ],
                swap_probability=0.1  # Hard metamorphic rock
            ),
            
            # Special states
            MaterialType.MAGMA: MaterialProperties(
                density=2800, thermal_conductivity=4.0, specific_heat=1200,
                emissivity=0.95,  # Very high emissivity (hot molten)
                thermal_expansion=9.0e-5,  # High volumetric expansion coefficient for molten rock
                color_rgb=(255, 69, 0),  # Red orange - molten rock
                transitions=[
                    TransitionRule(MaterialType.BASALT, 600, 900, "Fast cooling to basalt"),
                    TransitionRule(MaterialType.GRANITE, 700, 1000, "Slow cooling to granite", probability=0.3),
                ],
                swap_probability=0.4  # Viscous liquid - moderate swapping
            ),
            MaterialType.WATER: MaterialProperties(
                density=1000, thermal_conductivity=2.4, specific_heat=4186,  # Increased from 0.6 to 2.4 (4x)
                emissivity=0.96,  # Very high emissivity (good blackbody)
                thermal_expansion=2.1e-4,  # Volumetric expansion coefficient for water (temperature dependent)
                color_rgb=(30, 144, 255),  # Dodger blue - water
                transitions=[
                    TransitionRule(MaterialType.ICE, float('-inf'), 0, "Freezing to ice"),
                    TransitionRule(MaterialType.WATER_VAPOR, 100, float('inf'), "Boiling to water vapor"),
                    TransitionRule(MaterialType.WATER_VAPOR, 80, 100, "Evaporation", probability=0.05),
                ],
                swap_probability=0.7  # Liquid - easy swapping
            ),
            MaterialType.ICE: MaterialProperties(
                density=920, thermal_conductivity=10.0, specific_heat=2108,  # Increased from 2.2 to 10.0 (4.5x)
                emissivity=0.95,  # High emissivity solid
                thermal_expansion=1.6e-4,  # Volumetric expansion coefficient for ice
                color_rgb=(173, 216, 230),  # Light blue - ice
                transitions=[
                    TransitionRule(MaterialType.WATER, 0, float('inf'), "Melting to water"),
                    TransitionRule(MaterialType.WATER_VAPOR, -10, 0, "Sublimation", probability=0.01),
                ],
                swap_probability=0.3  # Solid ice - harder to swap
            ),
            MaterialType.WATER_VAPOR: MaterialProperties(
                density=0.6, thermal_conductivity=0.025, specific_heat=2010,
                emissivity=0.7,  # Atmospheric water vapor (humid air/cloud-like emissivity)
                thermal_expansion=3.7e-3,  # High volumetric expansion coefficient for gas (ideal gas law)
                color_rgb=(192, 224, 255),  # Light blue-white - humid air/steam
                transitions=[
                    TransitionRule(MaterialType.WATER, float('-inf'), 100, "Condensation to water"),
                    TransitionRule(MaterialType.ICE, float('-inf'), -5, "Deposition to ice", probability=0.05),
                ],
                swap_probability=0.8  # Gas - very easy swapping
            ),
            MaterialType.AIR: MaterialProperties(
                density=1.2, thermal_conductivity=0.1, specific_heat=1005,  # Increased from 0.024 to 0.1 (4x)
                emissivity=0.3,  # Increased emissivity for better simulation cooling
                thermal_expansion=3.7e-3,  # High volumetric expansion coefficient for gas (ideal gas law)
                color_rgb=(245, 245, 255),  # Very light blue/white - dry air
                transitions=[],  # Dry air doesn't transition (no water content to condense)
                swap_probability=0.8  # Gas - very easy swapping
            ),
            MaterialType.SPACE: MaterialProperties(
                density=1e-10, thermal_conductivity=1e-10, specific_heat=1e-10,  # 0 density for vacuum
                emissivity=0.0,  # No emissivity (perfect vacuum)
                thermal_expansion=0.0,  # No thermal expansion in vacuum
                color_rgb=(0, 0, 0),  # Black - vacuum of space
                transitions=[],  # Space doesn't transition to anything
                swap_probability=1.0  # Vacuum - no resistance
            ),
            MaterialType.URANIUM: MaterialProperties(
                density=19000, thermal_conductivity=27.0, specific_heat=116,  # Dense metallic properties
                emissivity=0.8,  # Metallic surface properties
                thermal_expansion=4.2e-5,  # Volumetric expansion coefficient for uranium metal
                color_rgb=(0, 255, 0),  # Bright green for visibility
                transitions=[],  # No phase transitions - remains uranium forever
                heat_generation=5e-4,  # 0.5 mW/m³ - enhanced for simulation visibility
                swap_probability=0.1  # Dense solid - low swap probability
            )
        }
    
    def _init_weathering_products(self) -> Dict[MaterialType, List[MaterialType]]:
        """Define weathering products for surface processes"""
        return {
            MaterialType.GRANITE: [MaterialType.SAND],
            MaterialType.BASALT: [MaterialType.SHALE],
            MaterialType.GNEISS: [MaterialType.SAND],
            MaterialType.SCHIST: [MaterialType.SHALE],
            MaterialType.SLATE: [MaterialType.SHALE],
            MaterialType.MARBLE: [MaterialType.LIMESTONE]
        }
    
    def _init_optical_absorption(self) -> Dict[MaterialType, float]:
        """Per-cell fractional absorption of incoming solar flux (0-1). Values <1 transmit light."""
        return {
            MaterialType.AIR: 0.05,
            MaterialType.WATER_VAPOR: 0.05,
            MaterialType.WATER: 0.1,
            MaterialType.ICE: 0.05
        }
    
    def get_solar_absorption(self, material_type: MaterialType) -> float:
        """Return fractional absorption for material (defaults to 1 for opaque solids)."""
        return self.absorption_coeff.get(material_type, 1.0)
    
    
    def get_properties(self, material_type: MaterialType) -> MaterialProperties:
        """Get properties for a material type"""
        return self.properties[material_type]
    
    def get_weathering_products(self, material_type: MaterialType) -> List[MaterialType]:
        """Get weathering products for a material type"""
        return self.weathering_products.get(material_type, [])

    def effective_density(self, material_type: MaterialType, temperature: float, *, reference_T: float = 273.15) -> float:
        """Return density adjusted for thermal expansion β (volumetric).

        ρ_eff = ρ₀ / (1 + β (T - T₀))
        """
        props = self.get_properties(material_type)
        beta = getattr(props, "thermal_expansion", 0.0)
        return props.density / max(1.0 + beta * (temperature - reference_T), 0.1) 