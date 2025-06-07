"""
Interactive real-time visualizer for the geology simulation.
Uses pygame for fast 2D graphics and user interaction.
"""

import pygame
import numpy as np
import sys
from typing import Tuple, Optional
from simulation_engine import GeologySimulation
from rock_types import RockType

class GeologyVisualizer:
    """Interactive visualizer for geological simulation"""
    
    def __init__(self, sim_width: int = 100, sim_height: int = 60, window_width: int = 1200, window_height: int = 800):
        """
        Initialize the visualizer
        
        Args:
            sim_width: Simulation grid width
            sim_height: Simulation grid height
            window_width: Display window width
            window_height: Display window height
        """
        pygame.init()
        
        self.sim_width = sim_width
        self.sim_height = sim_height
        self.window_width = window_width
        self.window_height = window_height
        
        # Calculate display scaling
        self.main_panel_width = int(window_width * 0.75)
        self.sidebar_width = window_width - self.main_panel_width
        
        self.cell_width = self.main_panel_width // sim_width
        self.cell_height = window_height // sim_height
        
        # Initialize pygame
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Geology Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Colors
        self.colors = {
            'background': (30, 30, 30),
            'sidebar': (50, 50, 50), 
            'text': (255, 255, 255),
            'button': (80, 120, 200),
            'button_hover': (100, 140, 220),
            'red': (255, 100, 100),
            'green': (100, 255, 100),
            'blue': (100, 100, 255)
        }
        
        # Simulation state
        self.simulation = GeologySimulation(sim_width, sim_height)
        self.running = True
        self.paused = False
        self.display_mode = 'rocks'  # 'rocks', 'temperature', 'pressure'
        self.auto_step = False
        self.step_interval = 100  # milliseconds
        self.last_step_time = 0
        
        # Interaction state
        self.mouse_tool = 'heat'  # 'heat', 'pressure', 'inspect'
        self.tool_radius = 3
        self.tool_intensity = 100
        
        # UI elements
        self.buttons = self._create_buttons()
        
    def _create_buttons(self) -> list:
        """Create UI buttons"""
        button_width = self.sidebar_width - 20
        button_height = 30
        x = self.main_panel_width + 10
        y_start = 10
        spacing = 5
        
        buttons = []
        y = y_start
        
        # Control buttons
        buttons.append({
            'rect': pygame.Rect(x, y, button_width, button_height),
            'text': 'Play/Pause',
            'action': 'toggle_pause',
            'color': self.colors['button']
        })
        y += button_height + spacing
        
        buttons.append({
            'rect': pygame.Rect(x, y, button_width//2 - 2, button_height),
            'text': 'Step +',
            'action': 'step_forward',
            'color': self.colors['green']
        })
        buttons.append({
            'rect': pygame.Rect(x + button_width//2 + 2, y, button_width//2 - 2, button_height),
            'text': 'Step -',
            'action': 'step_backward',
            'color': self.colors['red']
        })
        y += button_height + spacing
        
        buttons.append({
            'rect': pygame.Rect(x, y, button_width, button_height),
            'text': 'Auto Step',
            'action': 'toggle_auto',
            'color': self.colors['button']
        })
        y += button_height + spacing * 3
        
        # Display mode buttons
        display_modes = [('Rocks', 'rocks'), ('Temperature', 'temperature'), ('Pressure', 'pressure')]
        for text, mode in display_modes:
            buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': text,
                'action': f'display_{mode}',
                'color': self.colors['button']
            })
            y += button_height + spacing
        
        y += spacing * 2
        
        # Tool buttons
        tools = [('Heat Source', 'heat'), ('Pressure', 'pressure'), ('Inspect', 'inspect')]
        for text, tool in tools:
            buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'text': text,
                'action': f'tool_{tool}',
                'color': self.colors['button']
            })
            y += button_height + spacing
        
        return buttons
    
    def _handle_button_click(self, pos: Tuple[int, int]):
        """Handle button clicks"""
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                action = button['action']
                
                if action == 'toggle_pause':
                    self.paused = not self.paused
                elif action == 'step_forward':
                    self.simulation.step_forward()
                elif action == 'step_backward':
                    self.simulation.step_backward()
                elif action == 'toggle_auto':
                    self.auto_step = not self.auto_step
                elif action.startswith('display_'):
                    self.display_mode = action.split('_')[1]
                elif action.startswith('tool_'):
                    self.mouse_tool = action.split('_')[1]
    
    def _handle_mouse_drag(self, pos: Tuple[int, int], buttons: Tuple[bool, bool, bool]):
        """Handle mouse dragging on simulation area"""
        x, y = pos
        
        # Check if mouse is in simulation area
        if x >= self.main_panel_width or y >= self.window_height:
            return
        
        # Convert screen coordinates to simulation coordinates
        sim_x = x // self.cell_width
        sim_y = y // self.cell_height
        
        if sim_x >= self.sim_width or sim_y >= self.sim_height:
            return
        
        # Apply tool
        if buttons[0]:  # Left mouse button
            if self.mouse_tool == 'heat':
                self.simulation.add_heat_source(sim_x, sim_y, self.tool_radius, 
                                               self.simulation.temperature[sim_y, sim_x] + self.tool_intensity)
            elif self.mouse_tool == 'pressure':
                self.simulation.apply_tectonic_stress(sim_x, sim_y, self.tool_radius, self.tool_intensity)
    
    def _get_display_colors(self) -> np.ndarray:
        """Get colors for current display mode"""
        if self.display_mode == 'rocks':
            colors, _, _ = self.simulation.get_visualization_data()
            return colors
        elif self.display_mode == 'temperature':
            temp = self.simulation.temperature
            # Normalize temperature to color range
            temp_norm = np.clip((temp - np.min(temp)) / (np.max(temp) - np.min(temp) + 1e-10), 0, 1)
            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            colors[:, :, 0] = (temp_norm * 255).astype(np.uint8)  # Red channel
            colors[:, :, 2] = ((1 - temp_norm) * 255).astype(np.uint8)  # Blue channel
            return colors
        elif self.display_mode == 'pressure':
            pressure = self.simulation.pressure
            # Normalize pressure to color range
            pressure_norm = np.clip((pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure) + 1e-10), 0, 1)
            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            colors[:, :, 1] = (pressure_norm * 255).astype(np.uint8)  # Green channel
            colors[:, :, 2] = ((1 - pressure_norm) * 128).astype(np.uint8)  # Blue channel
            return colors
    
    def _draw_simulation(self):
        """Draw the simulation grid"""
        colors = self._get_display_colors()
        
        # Create surface for faster blitting
        sim_surface = pygame.Surface((self.main_panel_width, self.window_height))
        
        for y in range(self.sim_height):
            for x in range(self.sim_width):
                color = tuple(colors[y, x])
                rect = pygame.Rect(
                    x * self.cell_width, 
                    y * self.cell_height,
                    self.cell_width, 
                    self.cell_height
                )
                pygame.draw.rect(sim_surface, color, rect)
        
        self.screen.blit(sim_surface, (0, 0))
    
    def _draw_sidebar(self):
        """Draw the control sidebar"""
        sidebar_rect = pygame.Rect(self.main_panel_width, 0, self.sidebar_width, self.window_height)
        pygame.draw.rect(self.screen, self.colors['sidebar'], sidebar_rect)
        
        # Draw buttons
        for button in self.buttons:
            # Highlight active buttons
            color = button['color']
            if button['action'] == f'display_{self.display_mode}':
                color = self.colors['button_hover']
            elif button['action'] == f'tool_{self.mouse_tool}':
                color = self.colors['button_hover']
            elif button['action'] == 'toggle_pause' and self.paused:
                color = self.colors['red']
            elif button['action'] == 'toggle_auto' and self.auto_step:
                color = self.colors['green']
                
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['text'], button['rect'], 2)
            
            # Button text
            text_surface = self.small_font.render(button['text'], True, self.colors['text'])
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)
        
        # Draw statistics
        stats = self.simulation.get_stats()
        y_offset = 350
        x_offset = self.main_panel_width + 10
        
        stat_texts = [
            f"Time: {stats['time']:.0f} years",
            f"Time Step: {stats['dt']:.0f} years",
            f"Avg Temp: {stats['avg_temperature']:.1f}°C",
            f"Max Temp: {stats['max_temperature']:.1f}°C",
            f"Avg Pressure: {stats['avg_pressure']:.1f} MPa",
            f"Max Pressure: {stats['max_pressure']:.1f} MPa",
            f"History: {stats['history_length']}",
            "",
            f"Tool: {self.mouse_tool.title()}",
            f"Radius: {self.tool_radius}",
            f"Intensity: {self.tool_intensity}",
            "",
            "Rock Composition:"
        ]
        
        for i, text in enumerate(stat_texts):
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, y_offset + i * 20))
        
        # Rock composition
        y_offset += len(stat_texts) * 20
        for rock_type, percentage in stats['rock_composition'].items():
            text = f"  {rock_type}: {percentage:.1f}%"
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 18
        
        # Instructions
        y_offset += 20
        instructions = [
            "Controls:",
            "  Left click + drag: Apply tool",
            "  Mouse wheel: Adjust radius",
            "  Shift + wheel: Adjust intensity",
            "  Space: Play/Pause",
            "  R: Step forward",
            "  T: Step backward"
        ]
        
        for instruction in instructions:
            surface = self.small_font.render(instruction, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 18
    
    def _handle_keyboard(self, event):
        """Handle keyboard input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.simulation.step_forward()
            elif event.key == pygame.K_t:
                self.simulation.step_backward()
            elif event.key == pygame.K_1:
                self.display_mode = 'rocks'
            elif event.key == pygame.K_2:
                self.display_mode = 'temperature'
            elif event.key == pygame.K_3:
                self.display_mode = 'pressure'
    
    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel for tool adjustment"""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            # Adjust intensity
            self.tool_intensity = max(1, self.tool_intensity + event.y * 10)
        else:
            # Adjust radius
            self.tool_radius = max(1, min(10, self.tool_radius + event.y))
    
    def run(self):
        """Main visualization loop"""
        while self.running:
            current_time = pygame.time.get_ticks()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_button_click(event.pos)
                elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    self._handle_keyboard(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_mouse_wheel(event)
            
            # Handle mouse dragging
            mouse_buttons = pygame.mouse.get_pressed()
            if any(mouse_buttons):
                mouse_pos = pygame.mouse.get_pos()
                self._handle_mouse_drag(mouse_pos, mouse_buttons)
            
            # Auto-stepping
            if self.auto_step and not self.paused and current_time - self.last_step_time > self.step_interval:
                self.simulation.step_forward()
                self.last_step_time = current_time
            
            # Draw everything
            self.screen.fill(self.colors['background'])
            self._draw_simulation()
            self._draw_sidebar()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

def main():
    """Run the geology simulator"""
    print("Starting Geology Simulator...")
    print("Controls:")
    print("  Left click + drag: Apply selected tool")
    print("  Mouse wheel: Adjust tool radius")
    print("  Shift + mouse wheel: Adjust tool intensity")
    print("  Space: Play/Pause simulation")
    print("  R: Step forward")
    print("  T: Step backward")
    print("  1/2/3: Switch display modes")
    
    visualizer = GeologyVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 