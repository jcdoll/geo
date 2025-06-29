"""
Material properties and phase transitions for SPH geological simulation.

Ported from flux implementation with modifications for SPH physics.
"""

import numpy as np
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


class MaterialType(IntEnum):
    """Material types for SPH simulation."""
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
    """Physical properties of a material type for SPH."""
    # Basic properties
    name: str
    density_ref: float  # Reference density kg/m³
    bulk_modulus: float  # Pa (for Tait EOS)
    
    # SPH-specific viscosity (Pa·s)
    dynamic_viscosity: float  # Physical viscosity
    
    # Thermal properties
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    emissivity: float  # 0-1 for radiative cooling
    
    # Optical properties
    solar_absorption: float  # 0-1 (fraction absorbed)
    
    # Special properties
    heat_generation: float = 0.0  # W/kg (radioactive)
    default_temperature: float = 288.0  # K
    
    # Mechanical properties (for solids)
    cohesion_strength: float = 0.0  # Pa
    tensile_strength: float = 0.0  # Pa
    shear_modulus: float = 0.0  # Pa
    
    # Phase transitions
    transitions: List[TransitionRule] = field(default_factory=list)
    
    # Visualization
    color_rgb: Tuple[int, int, int] = (128, 128, 128)


class MaterialDatabase:
    """Database of material properties for SPH simulation."""
    
    def __init__(self):
        self.properties = self._init_properties()
        self._material_list = list(MaterialType)
        
    def _init_properties(self) -> Dict[MaterialType, MaterialProperties]:
        """Initialize material properties."""
        props = {}
        
        # SPACE - vacuum with minimal properties for stability
        props[MaterialType.SPACE] = MaterialProperties(
            name="space",
            density_ref=0.1,  # Prevents extreme density ratios
            bulk_modulus=1e3,  # Very compressible
            dynamic_viscosity=1e-6,  # Minimal resistance
            thermal_conductivity=1e-6,
            specific_heat=100.0,
            emissivity=1e-3,
            solar_absorption=0.0,
            default_temperature=2.7,
            color_rgb=(0, 0, 0),
        )
        
        # AIR - atmospheric gas
        props[MaterialType.AIR] = MaterialProperties(
            name="air",
            density_ref=1.2,
            bulk_modulus=1.42e5,  # Ideal gas at 1 atm
            dynamic_viscosity=1.8e-5,
            thermal_conductivity=0.025,
            specific_heat=1005.0,
            emissivity=0.8,
            solar_absorption=0.01,
            default_temperature=288.0,
            color_rgb=(135, 206, 250),
        )
        
        # WATER - liquid H₂O
        props[MaterialType.WATER] = MaterialProperties(
            name="water",
            density_ref=1000.0,
            bulk_modulus=2.2e9,
            dynamic_viscosity=1e-3,  # 1 mPa·s at 20°C
            thermal_conductivity=0.6,
            specific_heat=4186.0,
            emissivity=0.96,
            solar_absorption=0.019,
            default_temperature=288.0,
            color_rgb=(0, 119, 190),
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
                    latent_heat=-2.26e6,  # Heat of vaporization
                    description="Evaporation"
                ),
            ]
        )
        
        # WATER_VAPOR - steam
        props[MaterialType.WATER_VAPOR] = MaterialProperties(
            name="steam",
            density_ref=0.6,
            bulk_modulus=1e5,
            dynamic_viscosity=1.2e-5,
            thermal_conductivity=0.02,
            specific_heat=2080.0,
            emissivity=0.8,
            solar_absorption=0.02,
            default_temperature=373.0,
            color_rgb=(240, 248, 255),
            transitions=[
                TransitionRule(
                    target=MaterialType.WATER,
                    temp_min=0.0,
                    temp_max=373.15,
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.05,
                    latent_heat=2.26e6,
                    description="Condensation"
                ),
            ]
        )
        
        # ICE - solid H₂O
        props[MaterialType.ICE] = MaterialProperties(
            name="ice",
            density_ref=917.0,
            bulk_modulus=9.0e9,
            dynamic_viscosity=1e13,  # Very viscous solid
            thermal_conductivity=2.2,
            specific_heat=2100.0,
            emissivity=0.97,
            solar_absorption=0.002,
            default_temperature=263.0,
            cohesion_strength=1e4,  # 10 kPa (reduced for stability)
            tensile_strength=1e6,
            shear_modulus=3.5e9,
            color_rgb=(176, 224, 230),
            transitions=[
                TransitionRule(
                    target=MaterialType.WATER,
                    temp_min=273.15,
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.1,
                    latent_heat=-3.34e5,
                    description="Melting"
                ),
            ]
        )
        
        # ROCK - generic solid rock
        props[MaterialType.ROCK] = MaterialProperties(
            name="rock",
            density_ref=2700.0,
            bulk_modulus=50e9,
            dynamic_viscosity=1e20,  # Effectively rigid
            thermal_conductivity=3.0,
            specific_heat=1000.0,
            emissivity=0.95,
            solar_absorption=0.7,
            default_temperature=288.0,
            cohesion_strength=5e4,  # 50 kPa (reduced for stability)
            tensile_strength=5e6,   # 5 MPa
            shear_modulus=30e9,
            color_rgb=(139, 90, 43),
            transitions=[
                TransitionRule(
                    target=MaterialType.SAND,
                    temp_min=0.0,
                    temp_max=float('inf'),
                    pressure_min=0.0,
                    pressure_max=1e5,
                    rate=1e-7,
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
                    latent_heat=-4e5,
                    description="Melting"
                ),
            ]
        )
        
        # SAND - granular material
        props[MaterialType.SAND] = MaterialProperties(
            name="sand",
            density_ref=1600.0,
            bulk_modulus=35e9,
            dynamic_viscosity=1e8,  # Granular flow
            thermal_conductivity=0.3,
            specific_heat=830.0,
            emissivity=0.95,
            solar_absorption=0.6,
            default_temperature=288.0,
            cohesion_strength=0.0,  # No cohesion when dry
            color_rgb=(238, 203, 173),
            transitions=[
                TransitionRule(
                    target=MaterialType.ROCK,
                    temp_min=673.0,  # ~400°C
                    temp_max=float('inf'),
                    pressure_min=1e7,  # 10 MPa
                    pressure_max=float('inf'),
                    rate=1e-8,
                    description="Lithification"
                ),
            ]
        )
        
        # URANIUM - radioactive material
        props[MaterialType.URANIUM] = MaterialProperties(
            name="uranium",
            density_ref=19000.0,
            bulk_modulus=100e9,
            dynamic_viscosity=1e21,  # Solid metal
            thermal_conductivity=27.0,
            specific_heat=116.0,
            emissivity=0.9,
            solar_absorption=0.85,
            heat_generation=0.1,  # W/kg radioactive decay
            default_temperature=288.0,
            cohesion_strength=1e5,  # 100 kPa (reduced for stability)
            tensile_strength=150e6,
            shear_modulus=80e9,
            color_rgb=(0, 255, 0),
        )
        
        # MAGMA - molten rock
        props[MaterialType.MAGMA] = MaterialProperties(
            name="magma",
            density_ref=2700.0,
            bulk_modulus=35e9,
            dynamic_viscosity=100.0,  # 100 Pa·s (basaltic)
            thermal_conductivity=1.5,
            specific_heat=1200.0,
            emissivity=0.95,
            solar_absorption=0.8,
            default_temperature=1500.0,
            color_rgb=(255, 69, 0),
            transitions=[
                TransitionRule(
                    target=MaterialType.ROCK,
                    temp_min=0.0,
                    temp_max=1273.0,  # ~1000°C
                    pressure_min=0.0,
                    pressure_max=float('inf'),
                    rate=0.01,
                    latent_heat=4e5,
                    description="Crystallization"
                ),
            ]
        )
        
        return props
    
    def get_properties(self, material: MaterialType) -> MaterialProperties:
        """Get properties for a material type."""
        return self.properties[material]
    
    def get_bulk_modulus_array(self, material_ids: np.ndarray) -> np.ndarray:
        """Get bulk modulus for array of material IDs."""
        result = np.zeros_like(material_ids, dtype=np.float32)
        for mat_type in MaterialType:
            mask = material_ids == mat_type
            if np.any(mask):
                result[mask] = self.properties[mat_type].bulk_modulus
        return result
    
    def get_density_ref_array(self, material_ids: np.ndarray) -> np.ndarray:
        """Get reference density for array of material IDs."""
        result = np.zeros_like(material_ids, dtype=np.float32)
        for mat_type in MaterialType:
            mask = material_ids == mat_type
            if np.any(mask):
                result[mask] = self.properties[mat_type].density_ref
        return result
    
    def get_viscosity_array(self, material_ids: np.ndarray) -> np.ndarray:
        """Get dynamic viscosity for array of material IDs."""
        result = np.zeros_like(material_ids, dtype=np.float32)
        for mat_type in MaterialType:
            mask = material_ids == mat_type
            if np.any(mask):
                result[mask] = self.properties[mat_type].dynamic_viscosity
        return result
    
    def get_thermal_properties(self, material_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get thermal conductivity and specific heat arrays."""
        k = np.zeros_like(material_ids, dtype=np.float32)
        cp = np.zeros_like(material_ids, dtype=np.float32)
        
        for mat_type in MaterialType:
            mask = material_ids == mat_type
            if np.any(mask):
                props = self.properties[mat_type]
                k[mask] = props.thermal_conductivity
                cp[mask] = props.specific_heat
        
        return k, cp
    
    def get_cohesion_strength_array(self, material_ids: np.ndarray) -> np.ndarray:
        """Get cohesion strength for solid materials."""
        result = np.zeros_like(material_ids, dtype=np.float32)
        for mat_type in MaterialType:
            mask = material_ids == mat_type
            if np.any(mask):
                result[mask] = self.properties[mat_type].cohesion_strength
        return result
    
    def check_transitions(self, material_id: int, temperature: float, 
                         pressure: float, water_present: bool = False) -> Optional[TransitionRule]:
        """Check if material should transition."""
        material = MaterialType(material_id)
        props = self.properties[material]
        
        for transition in props.transitions:
            if transition.check_conditions(temperature, pressure, water_present):
                return transition
        
        return None