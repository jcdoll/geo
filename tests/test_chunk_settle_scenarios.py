"""
Chunk settling test scenarios using the test framework.

These scenarios test that unsupported solid chunks fall and respect physics constraints.
"""

import numpy as np
from typing import Dict, Any, Optional

from tests.test_framework import TestScenario
from materials import MaterialType
from geo_game import GeoGame


class ChunkSettleScenario(TestScenario):
    """Base scenario for testing chunk settling behavior."""
    
    def __init__(self, chunk_material: MaterialType = MaterialType.BASALT,
                 chunk_size: int = 1, 
                 terminal_velocity: Optional[float] = 3.0,
                 **kwargs):
        """Initialize chunk settle scenario.
        
        Args:
            chunk_material: Material type for the falling chunk
            chunk_size: Size of the chunk (1 = single cell)
            terminal_velocity: Terminal velocity limit for settling
        """
        super().__init__(chunk_material=chunk_material,
                        chunk_size=chunk_size,
                        terminal_velocity=terminal_velocity,
                        **kwargs)
        self.chunk_material = chunk_material
        self.chunk_size = chunk_size
        self.terminal_velocity = terminal_velocity
        
    def get_name(self) -> str:
        return f"chunk_settle_{self.chunk_material.value}_v{self.terminal_velocity}"
        
    def get_description(self) -> str:
        vel_desc = "infinite" if self.terminal_velocity == float('inf') else f"{self.terminal_velocity}"
        return (f"Tests {self.chunk_material.value} chunk settling with "
                f"terminal velocity = {vel_desc} cells/step")
        
    def setup(self, sim: GeoGame) -> None:
        """Set up an unsupported chunk above empty space."""
        # Clear to space with uniform temperature
        sim.material_types[:, :] = MaterialType.SPACE
        sim.temperature[:, :] = 300.0
        
        # Place chunk at top center
        col = sim.width // 2
        row = 0  # Top row
        
        # Create chunk
        if self.chunk_size == 1:
            sim.material_types[row, col] = self.chunk_material
        else:
            # Create larger chunk
            half_size = self.chunk_size // 2
            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    y, x = row + dy + half_size, col + dx
                    if 0 <= y < sim.height and 0 <= x < sim.width:
                        sim.material_types[y, x] = self.chunk_material
        
        # Fix center of mass so direction is purely downward
        sim.center_of_mass = (sim.height // 2, sim.width // 2)
        
        # Set terminal velocity
        if hasattr(sim, 'terminal_settle_velocity'):
            sim.terminal_settle_velocity = self.terminal_velocity
            
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate chunk settling behavior."""
        # Find chunk positions
        chunk_mask = (sim.material_types == self.chunk_material)
        chunk_positions = np.column_stack(np.where(chunk_mask))
        
        if chunk_positions.shape[0] == 0:
            return {
                'success': False,
                'metrics': {},
                'message': f'No {self.chunk_material.value} chunk found!'
            }
            
        # Calculate current position
        avg_y = np.mean(chunk_positions[:, 0])
        avg_x = np.mean(chunk_positions[:, 1])
        
        # Get initial position from stored state
        initial_positions = self.initial_state.get('chunk_positions', [])
        if not initial_positions:
            return {
                'success': False,
                'metrics': {'current_y': avg_y},
                'message': 'No initial position stored!'
            }
            
        initial_y = np.mean([pos[0] for pos in initial_positions])
        
        # Calculate movement
        fall_distance = avg_y - initial_y
        
        # Check constraints
        success = True
        message_parts = []
        
        if self.terminal_velocity != float('inf'):
            # Should not exceed terminal velocity
            if fall_distance > self.terminal_velocity + 0.1:  # Small tolerance
                success = False
                message_parts.append(f"exceeded terminal velocity ({fall_distance:.1f} > {self.terminal_velocity})")
        else:
            # Should fall to bottom
            expected_y = sim.height - 1 - (self.chunk_size // 2)
            if abs(avg_y - expected_y) > 1.0:
                success = False
                message_parts.append(f"didn't reach bottom (y={avg_y:.1f}, expected {expected_y})")
                
        if fall_distance < 0.1:
            success = False
            message_parts.append("chunk didn't fall")
            
        # Build message
        if success:
            message = f"Chunk fell {fall_distance:.1f} cells correctly"
        else:
            message = f"Chunk issues: {', '.join(message_parts)}"
            
        return {
            'success': success,
            'metrics': {
                'initial_y': initial_y,
                'current_y': avg_y,
                'fall_distance': fall_distance,
                'chunk_count': chunk_positions.shape[0],
                'terminal_velocity': self.terminal_velocity
            },
            'message': message
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial chunk positions."""
        super().store_initial_state(sim)
        
        chunk_mask = (sim.material_types == self.chunk_material)
        positions = list(zip(*np.where(chunk_mask)))
        self.initial_state['chunk_positions'] = positions
        
    def get_visualization_hints(self) -> Dict[str, Any]:
        """Provide visualization hints."""
        return {
            'highlight_materials': [self.chunk_material],
            'show_metrics': ['fall_distance', 'current_y', 'terminal_velocity']
        }


class MultiChunkSettleScenario(ChunkSettleScenario):
    """Test multiple chunks falling with different materials."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.materials = [MaterialType.BASALT, MaterialType.GRANITE, MaterialType.ICE]
        
    def get_name(self) -> str:
        return "multi_chunk_settle"
        
    def get_description(self) -> str:
        return "Tests multiple material chunks falling simultaneously"
        
    def setup(self, sim: GeoGame) -> None:
        """Set up multiple chunks of different materials."""
        # Clear to space
        sim.material_types[:, :] = MaterialType.SPACE
        sim.temperature[:, :] = 300.0
        
        # Place different material chunks
        spacing = sim.width // (len(self.materials) + 1)
        
        for i, material in enumerate(self.materials):
            col = spacing * (i + 1)
            row = i * 2  # Stagger heights
            sim.material_types[row, col] = material
            
        # Set center of mass
        sim.center_of_mass = (sim.height // 2, sim.width // 2)
        
        # Set terminal velocity
        if hasattr(sim, 'terminal_settle_velocity'):
            sim.terminal_settle_velocity = self.terminal_velocity
            
        # Update properties
        sim._properties_dirty = True
        sim._update_material_properties()
        
    def evaluate(self, sim: GeoGame) -> Dict[str, Any]:
        """Evaluate multiple chunks settling."""
        all_success = True
        messages = []
        metrics = {}
        
        for material in self.materials:
            mask = (sim.material_types == material)
            if np.any(mask):
                positions = np.column_stack(np.where(mask))
                avg_y = np.mean(positions[:, 0])
                
                # Get initial position
                initial_positions = self.initial_state.get(f'{material.value}_positions', [])
                if initial_positions:
                    initial_y = np.mean([pos[0] for pos in initial_positions])
                    fall_distance = avg_y - initial_y
                    
                    metrics[f'{material.value}_fall'] = fall_distance
                    
                    if fall_distance < 0.1:
                        all_success = False
                        messages.append(f"{material.value} didn't fall")
                else:
                    all_success = False
                    messages.append(f"No initial position for {material.value}")
                    
        message = "All chunks fell correctly" if all_success else f"Issues: {', '.join(messages)}"
        
        return {
            'success': all_success,
            'metrics': metrics,
            'message': message
        }
        
    def store_initial_state(self, sim: GeoGame) -> None:
        """Store initial positions for all materials."""
        super().store_initial_state(sim)
        
        for material in self.materials:
            mask = (sim.material_types == material)
            if np.any(mask):
                positions = list(zip(*np.where(mask)))
                self.initial_state[f'{material.value}_positions'] = positions 