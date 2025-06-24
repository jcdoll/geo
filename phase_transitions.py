"""
General phase transition system for flux-based geological simulation.

Uses the transition rules defined in materials.py to handle all
material transformations based on temperature and pressure conditions.
"""

import numpy as np
from typing import Optional
from state import FluxState
from materials import MaterialType, MaterialDatabase


class PhaseTransitionSystem:
    """Handles all material phase transitions based on T-P conditions."""
    
    def __init__(self, state: FluxState, material_db: MaterialDatabase):
        """
        Initialize phase transition system.
        
        Args:
            state: FluxState instance
            material_db: MaterialDatabase instance
        """
        self.state = state
        self.material_db = material_db
        
        # Cache transition rules for each material
        self._build_transition_cache()
        
    def _build_transition_cache(self):
        """Build cache of transition rules for efficient access."""
        self.transitions = {}
        
        # For each material type, get its transition rules
        for mat_type in MaterialType:
            mat_props = self.material_db.get_properties(mat_type)
            if mat_props.transitions:
                self.transitions[mat_type] = mat_props.transitions
                
    def apply_transitions(self, dt: float):
        """
        Apply all material phase transitions based on current T-P conditions.
        
        Args:
            dt: Time step in seconds
        """
        T = self.state.temperature
        P = self.state.pressure
        
        # Track heat sources from phase transitions
        heat_source = np.zeros_like(T)
        
        # Process each material type that has transitions
        for source_type, transition_rules in self.transitions.items():
            source_idx = source_type.value
            
            # Skip if this material isn't present
            if not np.any(self.state.vol_frac[source_idx] > 0):
                continue
                
            # Check each transition rule for this material
            for rule in transition_rules:
                target_idx = rule.target.value
                
                # Find cells where transition conditions are met
                temp_condition = (T >= rule.temp_min) & (T <= rule.temp_max)
                pressure_condition = (P >= rule.pressure_min) & (P <= rule.pressure_max)
                material_present = self.state.vol_frac[source_idx] > 0
                
                transition_mask = temp_condition & pressure_condition & material_present
                
                if not np.any(transition_mask):
                    continue
                    
                # Calculate transition amount (limited by available material)
                max_rate = np.minimum(rule.rate * dt, 1.0)
                transition_amount = self.state.vol_frac[source_idx] * max_rate
                
                # Apply transition where conditions are met
                self.state.vol_frac[source_idx][transition_mask] -= transition_amount[transition_mask]
                self.state.vol_frac[target_idx][transition_mask] += transition_amount[transition_mask]
                
                # Apply latent heat
                if rule.latent_heat != 0:
                    # Heat per unit mass times mass transitioned
                    # Note: latent_heat is J/kg, need to multiply by density
                    source_density = self.material_db.get_properties(source_type).density
                    heat_per_cell = rule.latent_heat * source_density * transition_amount
                    heat_source[transition_mask] += heat_per_cell[transition_mask]
        
        # Apply accumulated heat sources
        if np.any(heat_source != 0):
            # Avoid division by zero
            valid_mask = (self.state.density > 0) & (self.state.specific_heat > 0)
            if np.any(valid_mask):
                dT = np.zeros_like(T)
                # Heat source is J/m³, need to divide by volumetric heat capacity
                dT[valid_mask] = (heat_source[valid_mask] / 
                                 (self.state.density[valid_mask] * 
                                  self.state.specific_heat[valid_mask]))
                self.state.temperature += dT
        
        # Normalize volume fractions to ensure they sum to 1
        self.state.normalize_volume_fractions()
        
        # Update mixture properties after transitions
        self.state.update_mixture_properties(self.material_db)