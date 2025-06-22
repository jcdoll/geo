"""
Material properties and phase transitions for flux-based geological simulation.

Based on the simplified material list from PHYSICS_FLUX.md.
"""

import numpy as np
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


class MaterialType(IntEnum):
    """Material types for the flux-based simulation."""
    SPACE = 0
    AIR = auto()
    WATER = auto()
    WATER_VAPOR = auto()
    ICE = auto()
    ROCK = auto()
    SAND = auto()
    URANIUM = auto()
    MAGMA = auto()


@dataclass
class TransitionRule:
    """Defines a material transition under specific conditions."""
    target: MaterialType
    temp_min: float  # Kelvin
    temp_max: float  # Kelvin
    pressure_min: float  # Pascals
    pressure_max: float  # Pascals
    rate: float = 0.1  # Transition rate (fraction per second)
    latent_heat: float = 0.0  # J/kg (negative for exothermic)
    water_required: bool = False  # For weathering
    description: str = ""
    
    def check_conditions(self, T: float, P: float, water_present: bool = False) -> bool:
        """Check if transition conditions are met."""
        if self.water_required and not water_present:
            return False
        return (self.temp_min <= T <= self.temp_max and 
                self.pressure_min <= P <= self.pressure_max)


@dataclass
class MaterialProperties:
    """Physical properties of a material type."""
    # Basic properties
    density: float  # kg/m³
    viscosity: float  # 0-1 (0=no resistance, 1=solid)
    
    # Thermal properties
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    emissivity: float  # 0-1 for radiative cooling
    heat_generation: float = 0.0  # W/m³ (for radioactive materials)
    
    # Optical properties
    albedo: float  # 0-1 (0=perfect absorber, 1=perfect reflector)
    absorption_coeff: float = 1.0  # For solar radiation
    
    # Phase transitions
    transitions: List[TransitionRule] = field(default_factory=list)
    
    # Visualization
    color_rgb: Tuple[int, int, int] = (128, 128, 128)


class MaterialDatabase:
    """Database of material properties for flux-based simulation."""
    
    def __init__(self):
        self.properties = self._init_properties()
        self._material_list = list(MaterialType)
        
    def _init_properties(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize material properties based on PHYSICS_FLUX.md tables."""
        props = {}
        
        # SPACE - vacuum
        props[MaterialType.SPACE] = MaterialProperties(
            density=0.0,
            viscosity=0.0,
            thermal_conductivity=0.0,
            specific_heat=0.0,
            emissivity=0.0,
            albedo=0.0,
            absorption_coeff=0.0,
            color_rgb=(0, 0, 0),  # Black
        )
        
        # AIR - atmospheric gas
        props[MaterialType.AIR] = MaterialProperties(
            density=1.2,
            viscosity=0.005,
            thermal_conductivity=0.025,
            specific_heat=1005.0,
            emissivity=0.8,
            albedo=0.0,
            absorption_coeff=0.001,  # Nearly transparent
            color_rgb=(135, 206, 250),  # Light sky blue
        )
        
        # WATER - liquid H₂O
        props[MaterialType.WATER] = MaterialProperties(
            density=1000.0,
            viscosity=0.01,
            thermal_conductivity=0.6,
            specific_heat=4186.0,
            emissivity=0.96,
            albedo=0.06,
            absorption_coeff=0.02,  # Moderate absorption
            color_rgb=(0, 119, 190),  # Deep water blue
            transitions=[
                TransitionRule(
                    target=MaterialType.ICE,
                    temp_min=0.0,
                    temp_max=273.15,
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.1,
                    latent_heat=3.34e5,  # Heat of fusion
                    description="Freezing"
                ),
                TransitionRule(
                    target=MaterialType.WATER_VAPOR,
                    temp_min=373.15,
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.05,
                    latent_heat=-2.26e6,  # Heat of vaporization (negative = cooling)
                    description="Evaporation"
                ),
            ]
        )
        
        # WATER_VAPOR - gaseous H₂O
        props[MaterialType.WATER_VAPOR] = MaterialProperties(
            density=0.6,
            viscosity=0.005,
            thermal_conductivity=0.02,
            specific_heat=2080.0,
            emissivity=0.8,
            albedo=0.0,
            absorption_coeff=0.005,  # Slight absorption
            color_rgb=(240, 248, 255),  # Alice blue (light mist)
            transitions=[
                TransitionRule(
                    target=MaterialType.WATER,
                    temp_min=0.0,
                    temp_max=373.15,
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.05,
                    latent_heat=2.26e6,  # Heat released on condensation
                    description="Condensation"
                ),
            ]
        )
        
        # ICE - solid H₂O
        props[MaterialType.ICE] = MaterialProperties(
            density=917.0,
            viscosity=0.15,
            thermal_conductivity=2.2,
            specific_heat=2100.0,
            emissivity=0.97,
            albedo=0.8,
            absorption_coeff=0.01,  # Semi-transparent
            color_rgb=(176, 224, 230),  # Powder blue
            transitions=[
                TransitionRule(
                    target=MaterialType.WATER,
                    temp_min=273.15,
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.1,
                    latent_heat=-3.34e5,  # Heat absorbed on melting
                    description="Melting"
                ),
            ]
        )
        
        # ROCK - generic solid rock
        props[MaterialType.ROCK] = MaterialProperties(
            density=2700.0,
            viscosity=0.35,
            thermal_conductivity=3.0,
            specific_heat=1000.0,
            emissivity=0.95,
            albedo=0.3,
            absorption_coeff=1.0,  # Opaque
            color_rgb=(139, 90, 43),  # Saddle brown
            transitions=[
                TransitionRule(
                    target=MaterialType.SAND,
                    temp_min=0.0,
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=1e5,  # Surface pressure
                    rate=1e-7,  # Very slow weathering
                    water_required=True,
                    description="Weathering"
                ),
                TransitionRule(
                    target=MaterialType.MAGMA,
                    temp_min=1473.0,  # ~1200°C
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.01,
                    latent_heat=-4e5,  # Heat absorbed on melting
                    description="Melting"
                ),
            ]
        )
        
        # SAND - loose sediment
        props[MaterialType.SAND] = MaterialProperties(
            density=1600.0,
            viscosity=0.1,
            thermal_conductivity=0.3,
            specific_heat=830.0,
            emissivity=0.95,
            albedo=0.4,
            absorption_coeff=1.0,  # Opaque
            color_rgb=(238, 203, 173),  # Desert sand
            transitions=[
                TransitionRule(
                    target=MaterialType.ROCK,
                    temp_min=673.0,  # ~400°C
                    temp_max=float('inf'),
                    pressure_min=1e7,  # 10 MPa
                    pressure_max=float('inf'),
                    rate=1e-8,  # Very slow lithification
                    description="Lithification"
                ),
            ]
        )
        
        # URANIUM - radioactive material
        props[MaterialType.URANIUM] = MaterialProperties(
            density=19000.0,
            viscosity=0.4,
            thermal_conductivity=27.0,
            specific_heat=116.0,
            emissivity=0.9,
            albedo=0.15,
            absorption_coeff=1.0,  # Opaque
            heat_generation=5e-4,  # W/m³ - radioactive decay
            color_rgb=(0, 255, 0),  # Bright green for visibility
            # No transitions
        )
        
        # MAGMA - molten rock
        props[MaterialType.MAGMA] = MaterialProperties(
            density=2700.0,
            viscosity=0.05,
            thermal_conductivity=1.5,
            specific_heat=1200.0,
            emissivity=0.95,
            albedo=0.9,
            absorption_coeff=1.0,  # Opaque
            color_rgb=(255, 69, 0),  # Orange red
            transitions=[
                TransitionRule(
                    target=MaterialType.ROCK,
                    temp_min=0.0,
                    temp_max=1273.0,  # ~1000°C
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.01,
                    latent_heat=4e5,  # Heat released on crystallization
                    description="Crystallization"
                ),
            ]
        )
        
        return props
    
    def get_properties(self, material: MaterialType) -> MaterialProperties:
        """Get properties for a material type."""
        return self.properties[material]
    
    def get_properties_by_index(self, index: int) -> MaterialProperties:
        """Get properties by material index."""
        return self.properties[self._material_list[index]]
    
    def get_weathering_rate(self, temperature: float, water_factor: float = 1.0) -> float:
        """
        Calculate chemical weathering rate based on temperature.
        
        From PHYSICS_FLUX.md:
        Chemical rate = exp((T - 288)/14.4) × water_factor
        
        Args:
            temperature: Temperature in Kelvin
            water_factor: 3.0 if water present, 1.0 otherwise
            
        Returns:
            Weathering rate multiplier
        """
        return np.exp((temperature - 288.0) / 14.4) * water_factor