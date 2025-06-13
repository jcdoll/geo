"""
Modular geological simulation engine.
This is a refactored version that delegates to specialized modules.
"""

# Import the original simulation engine and extend it with modular components
from simulation_engine_original import GeologySimulation as OriginalGeologySimulation
from heat_transfer import HeatTransfer
from fluid_dynamics import FluidDynamics
from atmospheric_processes import AtmosphericProcesses
from material_processes import MaterialProcesses

class GeologySimulation(OriginalGeologySimulation):
    """Modular simulation engine that extends the original with specialized modules"""
    
    def __init__(self, *args, **kwargs):
        # Initialize the original simulation
        super().__init__(*args, **kwargs)
        
        # Add missing attributes needed by modular components
        if not hasattr(self, 'quality'):
            self.quality = kwargs.get('quality', 1)
        
        # Initialize modular components
        self.heat_transfer_module = HeatTransfer(self)
        self.fluid_dynamics_module = FluidDynamics(self)
        self.atmospheric_processes_module = AtmosphericProcesses(self)
        self.material_processes_module = MaterialProcesses(self)
        
        # Flag to indicate this is the modular version
        self._is_modular = True
        
        # Unified kinematics toggle
        self.use_unified_kinematics = True  # Default to new unified kinematics
    
    def step_forward_modular(self, dt=None):
        """Alternative step_forward using modular components"""
        if dt is not None:
            self.dt = dt

        # Update material properties if needed
        self._update_material_properties()

        # Calculate center of mass
        self._calculate_center_of_mass()

        # Heat diffusion using modular heat transfer
        new_temperature, stability_factor = self.heat_transfer_module.solve_heat_diffusion()
        self.temperature = new_temperature

        # Fluid dynamics processes
        self.fluid_dynamics_module.calculate_planetary_pressure()
        
        if self.use_unified_kinematics:
            # Use new unified kinematics approach
            self.fluid_dynamics_module.apply_unified_kinematics(self.dt)
        else:
            # Use old discrete approach
            self.fluid_dynamics_module.apply_gravitational_collapse()
            self.fluid_dynamics_module.apply_density_stratification()

        # Atmospheric processes
        self.atmospheric_processes_module.apply_atmospheric_convection()

        # Material processes
        self.material_processes_module.apply_metamorphism()
        self.material_processes_module.apply_phase_transitions()

        # Final update of material properties to ensure cache is clean
        self._update_material_properties()

        # Update time
        self.time += self.dt

    def toggle_kinematics_mode(self):
        """Toggle between unified kinematics and discrete methods"""
        self.use_unified_kinematics = not self.use_unified_kinematics
        mode = "Unified Kinematics" if self.use_unified_kinematics else "Discrete Methods"
        print(f"Switched to: {mode}")
        return mode
    
    def get_kinematics_mode(self):
        """Get current kinematics mode"""
        return "Unified Kinematics" if self.use_unified_kinematics else "Discrete Methods"
    
    def get_modular_info(self):
        """Get information about the modular components"""
        return {
            'is_modular': True,
            'kinematics_mode': self.get_kinematics_mode(),
            'modules': {
                'heat_transfer': type(self.heat_transfer_module).__name__,
                'fluid_dynamics': type(self.fluid_dynamics_module).__name__,
                'atmospheric_processes': type(self.atmospheric_processes_module).__name__,
                'material_processes': type(self.material_processes_module).__name__
            }
        } 
