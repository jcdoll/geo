"""
Interactive real-time visualizer for the geology simulation.
Uses pygame for fast 2D graphics and user interaction.
"""

import pygame
import numpy as np
import sys
from typing import Tuple, Optional

# Handle both relative and absolute imports
try:
    from .geo_game import GeoGame as GeologySimulation
    from .materials import MaterialType, MaterialDatabase
except ImportError:
    from geo_game import GeoGame as GeologySimulation
    from materials import MaterialType, MaterialDatabase
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import io
import math

class GeologyVisualizer:
    """Interactive visualizer for geological simulation"""
    
    def _format_power_smart(self, power_watts: float) -> str:
        """Smart formatting of power values with appropriate units and .1f precision"""
        abs_power = abs(power_watts)
        
        if abs_power == 0:
            return "0 W"
        elif abs_power < 1:
            return f"{power_watts * 1e3:.1f} mW"
        elif abs_power < 1e3:
            return f"{power_watts:.1f} W"
        elif abs_power < 1e6:
            return f"{power_watts / 1e3:.1f} kW"
        elif abs_power < 1e9:
            return f"{power_watts / 1e6:.1f} MW"
        elif abs_power < 1e12:
            return f"{power_watts / 1e9:.1f} GW"
        else:
            return f"{power_watts / 1e12:.1f} TW"
    
    def __init__(self, sim_width: int = 120, sim_height: int = 120, window_width: int = 1200, window_height: int = 800):
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
        self.status_bar_height = 30
        self.sim_area_height = window_height - self.status_bar_height
        
        # Calculate cell size to maintain square aspect ratio
        cell_size_x = self.main_panel_width // sim_width
        cell_size_y = self.sim_area_height // sim_height
        self.cell_size = min(cell_size_x, cell_size_y)  # Use smaller dimension for square cells
        self.cell_width = self.cell_size
        self.cell_height = self.cell_size
        
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
            'blue': (100, 100, 255),
            'yellow': (255, 255, 100)
        }
        
        # Simulation state (start in INFO mode; press 'L' to toggle DEBUG)
        self.simulation = GeologySimulation(sim_width, sim_height, log_level="INFO")
        self.running = True
        self.paused = True  # Start paused
        self.display_mode = 'materials'  # 'materials', 'temperature', 'pressure'
        self.speed_multiplier = 1.0  # 0.5x, 1x, 2x, 4x, 8x, 16x
        
        # Performance monitoring
        self.step_times = []  # Track recent step durations
        self.max_step_history = 30  # Keep last 30 step times
        
        # Interaction state
        self.mouse_tool = 'heat'  # 'heat', 'pressure', 'delete', 'add'
        self.tool_radius = 3
        self.tool_intensity = 100
        self.selected_tile = None  # (x, y) coordinates of selected tile
        # Reorder materials to start with common ones (water, air, magma, water_vapor) then others
        common_materials = [MaterialType.WATER, MaterialType.AIR, MaterialType.MAGMA, MaterialType.WATER_VAPOR]
        other_materials = [m for m in MaterialType if m not in (MaterialType.SPACE,) and m not in common_materials]
        self.add_materials = common_materials + other_materials
        self._add_material_index = 0
        
        # UI state
        self.sidebar_tab = 'controls'  # 'controls', 'stats', 'composition', 'graphs', 'info'
        
        # Key repeat state for arrow keys
        self.key_repeat_state = {
            'left_held': False,
            'right_held': False,
            'last_repeat_time': 0,
            'repeat_delay': 500,  # ms before first repeat
            'repeat_rate': 100,   # ms between repeats
            'initial_press_handled': False
        }
        
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
        
        # Speed control buttons
        speed_buttons = [('0.5x', 0.5), ('1x', 1.0), ('2x', 2.0), ('4x', 4.0), ('8x', 8.0), ('16x', 16.0)]
        speed_width = button_width // 6 - 1
        for i, (text, speed) in enumerate(speed_buttons):
            buttons.append({
                'rect': pygame.Rect(x + i * (speed_width + 2), y, speed_width, button_height),
                'text': text,
                'action': f'speed_{speed}',
                'color': self.colors['button']
            })
        y += button_height + spacing * 3
        
        # Display mode buttons in a 2-column grid to save vertical space
        display_modes = [
            ('Materials', 'materials'),
            ('Temperature', 'temperature'),
            ('Power', 'power'),
            ('Pressure', 'pressure'),
            ('Velocity', 'velocity'),
            ('Gravity', 'gravity'),
            ('Potential', 'potential'),
        ]
        display_button_width = button_width // 2 - 2
        display_button_height = button_height - 2  # Slightly shorter
        
        for i, (text, mode) in enumerate(display_modes):
            col = i % 2  # 0 or 1
            row = i // 2  # 0, 1, 2
            button_x = x + col * (display_button_width + 4)
            button_y = y + row * (display_button_height + spacing)
            
            buttons.append({
                'rect': pygame.Rect(button_x, button_y, display_button_width, display_button_height),
                'text': text,
                'action': f'display_{mode}',
                'color': self.colors['button']
            })
        
        # Update y to account for the full grid height
        num_rows = math.ceil(len(display_modes) / 2)
        y += num_rows * (display_button_height + spacing)
        
        y += spacing * 2
        
        # Tool buttons in a 2x2 grid to save vertical space
        tools = [('Heat Source', 'heat'), ('Pressure', 'pressure'), ('Delete', 'delete'), ('Add', 'add')]
        tool_button_width = button_width // 2 - 2
        tool_button_height = button_height - 2  # Slightly shorter
        
        for i, (text, tool) in enumerate(tools):
            col = i % 2  # 0 or 1
            row = i // 2  # 0 or 1
            button_x = x + col * (tool_button_width + 4)
            button_y = y + row * (tool_button_height + spacing)
            
            buttons.append({
                'rect': pygame.Rect(button_x, button_y, tool_button_width, tool_button_height),
                'text': text,
                'action': f'tool_{tool}',
                'color': self.colors['button']
            })
        
        # Update y to account for the grid (2 rows of buttons)
        y += 2 * (tool_button_height + spacing)
        
        y += spacing * 2
        
        # Tab buttons in 2x3 grid
        tabs = [('Controls', 'controls'), ('Stats', 'stats'), ('Composition', 'composition'), ('Graphs', 'graphs'), ('Info', 'info')]
        tab_width = button_width // 3 - 2
        tab_height = button_height - 5
        for i, (text, tab) in enumerate(tabs):
            if i < 3:  # First row: Controls, Stats, Composition
                row = 0
                col = i
            else:  # Second row: Graphs, Info
                row = 1
                col = i - 3
            buttons.append({
                'rect': pygame.Rect(x + col * (tab_width + 2), y + row * (tab_height + 2), tab_width, tab_height),
                'text': text,
                'action': f'tab_{tab}',
                'color': self.colors['button']
            })
        
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
                elif action.startswith('speed_'):
                    self.speed_multiplier = float(action.split('_')[1])
                elif action.startswith('display_'):
                    self.display_mode = action.split('_')[1]
                elif action.startswith('tool_'):
                    self.mouse_tool = action.split('_')[1]
                    if self.mouse_tool == 'add':
                        # Ensure button text shows current material
                        self._update_add_button_text()
                elif action.startswith('tab_'):
                    self.sidebar_tab = action.split('_')[1]
    
    def _get_simulation_offsets(self) -> Tuple[int, int]:
        """Calculate centering offsets for the simulation area
        
        Returns:
            Tuple of (offset_x, offset_y) for centering the simulation in the main panel
        """
        total_sim_width = self.sim_width * self.cell_width
        total_sim_height = self.sim_height * self.cell_height
        offset_x = (self.main_panel_width - total_sim_width) // 2
        offset_y = (self.sim_area_height - total_sim_height) // 2
        return offset_x, offset_y
    
    def _handle_right_click(self, pos: Tuple[int, int]):
        """Handle right-click for selecting tiles and showing info"""
        x, y = pos
        
        # Check if click is in simulation area
        if x >= self.main_panel_width or y < self.status_bar_height:
            return
        
        # Get centering offsets
        offset_x, offset_y = self._get_simulation_offsets()
        
        # Convert screen coordinates to simulation coordinates (accounting for centering)
        sim_x = (x - offset_x) // self.cell_width
        sim_y = (y - self.status_bar_height - offset_y) // self.cell_height
        
        # Validate coordinates and select tile
        if 0 <= sim_x < self.sim_width and 0 <= sim_y < self.sim_height:
            self.selected_tile = (sim_x, sim_y)
    
    def _handle_mouse_drag(self, pos: Tuple[int, int], buttons: Tuple[bool, bool, bool]):
        """Handle mouse dragging on simulation area"""
        x, y = pos
        
        # Check if mouse is in simulation area
        if x >= self.main_panel_width or y >= self.window_height or y < self.status_bar_height:
            return
        
        # Get centering offsets
        offset_x, offset_y = self._get_simulation_offsets()
        
        # Convert screen coordinates to simulation coordinates (accounting for centering)
        sim_x = (x - offset_x) // self.cell_width
        sim_y = (y - self.status_bar_height - offset_y) // self.cell_height
        
        if sim_x < 0 or sim_x >= self.sim_width or sim_y < 0 or sim_y >= self.sim_height:
            return
        
        # Apply tool
        if buttons[0]:  # Left mouse button
            if self.mouse_tool == 'heat':
                # Add heat in Kelvin - convert tool intensity from Celsius
                target_temp = self.simulation.temperature[sim_y, sim_x] + self.tool_intensity
                self.simulation.add_heat_source(sim_x, sim_y, self.tool_radius, target_temp)
            elif self.mouse_tool == 'pressure':
                # Apply pressure increase in MPa
                self.simulation.apply_tectonic_stress(sim_x, sim_y, self.tool_radius, self.tool_intensity)
            elif self.mouse_tool == 'delete':
                self.simulation.delete_material_blob(sim_x, sim_y, self.tool_radius)
            elif self.mouse_tool == 'add':
                mat = self.add_materials[self._add_material_index]
                self.simulation.add_material_blob(sim_x, sim_y, self.tool_radius, mat)
    
    def _get_display_colors(self) -> np.ndarray:
        """Get colors for current display mode"""
        if self.display_mode == 'materials':
            colors, _, _, _ = self.simulation.get_visualization_data()
            return colors
        elif self.display_mode == 'temperature':
            # Convert from Kelvin to Celsius for visualization
            temp_celsius = self.simulation.temperature - 273.15
            # FIXED: Use reasonable geological temperature limits instead of min/max
            # This makes normal temperature variations visible even with extreme outliers
            temp_min = -100.0  # Cold surface/ice temperatures
            temp_max = 1500.0  # Hot mantle/magma temperatures
            
            # Normalize temperature to geological range (clamped)
            temp_norm = np.clip((temp_celsius - temp_min) / (temp_max - temp_min), 0, 1)
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
        elif self.display_mode == 'power':
            # Convert from power density (W/m³) to power per cell (W)
            cell_volume = self.simulation.cell_size ** 3  # m³ per cell
            power_per_cell = self.simulation.power_density * cell_volume  # W per cell
            
            # Separate positive (heating) and negative (cooling) values for different scaling
            heating = np.maximum(0, power_per_cell)
            cooling = np.maximum(0, -power_per_cell)
            
            # Scale heating and cooling separately to make both visible
            max_heating = np.max(heating) if np.any(heating > 0) else 1e-10
            max_cooling = np.max(cooling) if np.any(cooling > 0) else 1e-10
            
            # Normalize each separately (0 to 1)
            heating_norm = np.clip(heating / max_heating, 0, 1)
            cooling_norm = np.clip(cooling / max_cooling, 0, 1)
            
            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            
            # Red for heat generation (scaled separately)
            heat_mask = heating > 0
            colors[heat_mask, 0] = (heating_norm[heat_mask] * 255).astype(np.uint8)
            
            # Blue for heat loss (scaled separately)  
            cool_mask = cooling > 0
            colors[cool_mask, 2] = (cooling_norm[cool_mask] * 255).astype(np.uint8)
            
            return colors
        elif self.display_mode == 'velocity':
            # Get velocity magnitude from fluid dynamics module
            if hasattr(self.simulation, 'fluid_dynamics_module'):
                velocity_x = self.simulation.fluid_dynamics_module.velocity_x
                velocity_y = self.simulation.fluid_dynamics_module.velocity_y
                velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                
                # Convert to mm/year for better geological visualization
                seconds_per_year = 365.25 * 24 * 3600
                velocity_mm_per_year = velocity_magnitude * seconds_per_year * 1000  # m/s to mm/year
                
                # Use logarithmic scaling for velocity to handle wide range of values
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                log_velocity = np.log10(velocity_mm_per_year + epsilon)
                
                # Set reasonable velocity range for geological processes (mm/year)
                # Geological velocities: ~1e-6 mm/year (very slow) to ~100 mm/year (fast flow)
                log_min = -6.0   # 1e-6 mm/year (extremely slow geological processes)
                log_max = 2.0    # 100 mm/year (fast geological flow like landslides)
                
                # Normalize to [0, 1] range
                velocity_norm = np.clip((log_velocity - log_min) / (log_max - log_min), 0, 1)
                
                # Create color map: blue (slow) to red (fast) through green
                colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
                
                # Use a blue-green-red color scheme
                # Blue component: high for low velocities, decreases with velocity
                colors[:, :, 2] = ((1 - velocity_norm) * 255).astype(np.uint8)
                
                # Green component: peaks in the middle range
                green_factor = 4 * velocity_norm * (1 - velocity_norm)  # Parabolic peak at 0.5
                colors[:, :, 1] = (green_factor * 255).astype(np.uint8)
                
                # Red component: high for high velocities
                colors[:, :, 0] = (velocity_norm * 255).astype(np.uint8)
                
                return colors
            else:
                # Fallback if no fluid dynamics module
                colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
                return colors
        elif self.display_mode == 'gravity':
            # Display magnitude of gravity acceleration
            if hasattr(self.simulation, 'gravity_x') and np.any(self.simulation.gravity_x):
                gx = self.simulation.gravity_x
                gy = self.simulation.gravity_y
            else:
                # Compute once if missing
                if hasattr(self.simulation, 'calculate_self_gravity'):
                    self.simulation.calculate_self_gravity()
                    gx = self.simulation.gravity_x
                    gy = self.simulation.gravity_y
                else:
                    gx = gy = np.zeros_like(self.simulation.density)

            g_mag = np.sqrt(gx**2 + gy**2)
            g_norm = np.clip(g_mag / max(1e-10, np.max(g_mag)), 0, 1)

            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            # Map 0 -> blue, 0.5 green, 1 red (same as velocity)
            colors[:, :, 2] = ((1 - g_norm) * 255).astype(np.uint8)
            green_factor = 4 * g_norm * (1 - g_norm)
            colors[:, :, 1] = (green_factor * 255).astype(np.uint8)
            colors[:, :, 0] = (g_norm * 255).astype(np.uint8)
            return colors
        elif self.display_mode == 'potential':
            # Visualise gravitational potential Φ across the grid
            phi = self.simulation.gravitational_potential
            min_val = np.min(phi)
            max_val = np.max(phi)
            span = max_val - min_val if max_val != min_val else 1.0
            norm = (phi - min_val) / span

            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            blue = ((1 - norm) * 255).astype(np.uint8)
            red = (norm * 255).astype(np.uint8)
            green = (4 * norm * (1 - norm) * 255).astype(np.uint8)
            colors[..., 0] = red
            colors[..., 1] = green
            colors[..., 2] = blue
            return colors
    
    def _draw_simulation(self):
        """Draw the simulation grid"""
        colors = self._get_display_colors()
        
        # Get centering offsets for the simulation area
        offset_x, offset_y = self._get_simulation_offsets()
        
        # Create surface for faster blitting
        sim_surface = pygame.Surface((self.main_panel_width, self.sim_area_height))
        sim_surface.fill(self.colors['background'])  # Fill with background color
        
        for y in range(self.sim_height):
            for x in range(self.sim_width):
                color = tuple(colors[y, x])
                rect = pygame.Rect(
                    offset_x + x * self.cell_width, 
                    offset_y + y * self.cell_height,
                    self.cell_width, 
                    self.cell_height
                )
                pygame.draw.rect(sim_surface, color, rect)
        
        self.screen.blit(sim_surface, (0, self.status_bar_height))
        
        # Draw selected tile highlight
        if self.selected_tile:
            x, y = self.selected_tile
            if 0 <= x < self.sim_width and 0 <= y < self.sim_height:
                highlight_rect = pygame.Rect(
                    offset_x + x * self.cell_width,
                    self.status_bar_height + offset_y + y * self.cell_height,
                    self.cell_width,
                    self.cell_height
                )
                # Draw white highlight border (original color)
                pygame.draw.rect(self.screen, (255, 255, 255), highlight_rect, 2)
        
        # Draw selected tile info in top-right of simulation area
        self._draw_selected_tile_info()
        
        # Draw color bar for temperature and pressure modes
        if self.display_mode in ['temperature', 'pressure', 'power', 'velocity', 'gravity', 'potential']:
            self._draw_color_bar()
    
    def _draw_selected_tile_info(self):
        """Draw selected tile information in top-right of simulation area"""
        if not self.selected_tile:
            return
            
        x, y = self.selected_tile
        if not (0 <= x < self.sim_width and 0 <= y < self.sim_height):
            return
            
        # Get tile data
        material = self.simulation.material_types[y, x]
        temp = self.simulation.temperature[y, x] - 273.15  # Convert to Celsius
        pressure = self.simulation.pressure[y, x]
        age = self.simulation.age[y, x]
        
        # Calculate power for this tile (convert from power density W/m³ to power W)
        power_density = self.simulation.power_density[y, x]  # W/m³
        cell_volume = self.simulation.cell_size ** 3  # m³ per cell
        power_watts = power_density * cell_volume  # W per cell
        power_formatted = self._format_power_smart(power_watts)
        
        # Get velocity information if available
        velocity_info = "N/A"
        if hasattr(self.simulation, 'fluid_dynamics_module'):
            vx = self.simulation.fluid_dynamics_module.velocity_x[y, x]
            vy = self.simulation.fluid_dynamics_module.velocity_y[y, x]
            velocity_magnitude = (vx**2 + vy**2)**0.5
            # Convert to mm/year for better readability
            seconds_per_year = 365.25 * 24 * 3600
            velocity_mm_per_year = velocity_magnitude * seconds_per_year * 1000
            if velocity_mm_per_year > 1e-6:  # Only show if significant
                velocity_info = f"{velocity_mm_per_year:.2e} mm/year"
            else:
                velocity_info = "~0 mm/year"
        
        # Position in top-left of simulation area
        info_x = 10  # 10px from left edge
        info_y = self.status_bar_height + 10  # 10px below status bar
        
        # Create info lines
        info_lines = [
            "Selected Tile:",
            f"Position: ({x}, {y})",
            f"Material: {material.name}",
            f"Temperature: {temp:.1f}°C", 
            f"Pressure: {pressure:.2f} MPa",
            f"Power: {power_formatted}",
            f"Velocity: {velocity_info}",
            f"Age: {age:.0f} years"
        ]
        
        # Draw semi-transparent background
        max_width = max(self.small_font.size(line)[0] for line in info_lines)
        bg_width = max_width + 20
        bg_height = len(info_lines) * 16 + 10
        bg_rect = pygame.Rect(info_x - 10, info_y - 5, bg_width, bg_height)
        
        # Create surface with alpha for transparency
        bg_surface = pygame.Surface((bg_width, bg_height))
        bg_surface.set_alpha(200)  # Semi-transparent
        bg_surface.fill((40, 40, 40))  # Dark background
        self.screen.blit(bg_surface, (info_x - 10, info_y - 5))
        
        # Draw border
        pygame.draw.rect(self.screen, self.colors['text'], bg_rect, 1)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            color = self.colors['green'] if i == 0 else self.colors['text']  # Header in green
            text_surface = self.small_font.render(line, True, color)
            self.screen.blit(text_surface, (info_x, info_y + i * 16))
    
    def _draw_color_bar(self):
        """Draw color bar legend for temperature or pressure mode"""
        bar_width = 20
        bar_height = 200
        bar_x = self.main_panel_width - bar_width - 80  # More space from sidebar
        bar_y = self.status_bar_height + 50  # Below status bar
        
        # Get min/max values and create color gradient
        if self.display_mode == 'temperature':
            # FIXED: Use same geological temperature limits as the color mapping
            min_val = -100.0  # Cold surface/ice temperatures
            max_val = 1500.0  # Hot mantle/magma temperatures
            unit = "°C"
            
            # Draw gradient bar (blue to red for temperature)
            for i in range(bar_height):
                normalized = i / bar_height
                color = (
                    int(normalized * 255),  # Red
                    0,                      # Green
                    int((1 - normalized) * 255)  # Blue
                )
                pygame.draw.line(self.screen, color, 
                               (bar_x, bar_y + bar_height - i - 1), 
                               (bar_x + bar_width, bar_y + bar_height - i - 1))
                
        elif self.display_mode == 'pressure':
            pressure = self.simulation.pressure
            min_val = np.min(pressure)
            max_val = np.max(pressure)
            unit = "MPa"
            
            # Draw gradient bar (black to green for pressure)
            for i in range(bar_height):
                normalized = i / bar_height
                color = (
                    0,                      # Red
                    int(normalized * 255),  # Green
                    int((1 - normalized) * 128)  # Blue
                )
                pygame.draw.line(self.screen, color, 
                               (bar_x, bar_y + bar_height - i - 1), 
                               (bar_x + bar_width, bar_y + bar_height - i - 1))
        
        elif self.display_mode == 'power':
            # Convert from power density (W/m³) to power per cell (W)
            cell_volume = self.simulation.cell_size ** 3  # m³ per cell
            power_per_cell = self.simulation.power_density * cell_volume  # W per cell
            
            heating = np.maximum(0, power_per_cell)
            cooling = np.maximum(0, -power_per_cell)
            
            max_heating = np.max(heating) if np.any(heating > 0) else 1e-10
            max_cooling = np.max(cooling) if np.any(cooling > 0) else 1e-10
            
            # Use larger range for main scale reference
            max_val = max_cooling  # Usually much larger (radiative cooling)
            min_val = -max_cooling
            unit = ""  # Unit will be handled by smart formatting
            
            # Draw gradient bar (blue to black to red for power: loss to neutral to generation)
            for i in range(bar_height):
                normalized = i / bar_height  # 0 to 1
                if normalized < 0.5:
                    # Bottom half: blue (heat loss)
                    intensity = (0.5 - normalized) * 2  # 1 to 0
                    color = (0, 0, int(intensity * 255))
                else:
                    # Top half: red (heat generation)
                    intensity = (normalized - 0.5) * 2  # 0 to 1
                    color = (int(intensity * 255), 0, 0)
                
                pygame.draw.line(self.screen, color, 
                               (bar_x, bar_y + bar_height - i - 1), 
                               (bar_x + bar_width, bar_y + bar_height - i - 1))
        
        elif self.display_mode == 'velocity':
            # Velocity range (logarithmic scale)
            log_min = -6.0   # 1e-6 mm/year
            log_max = 2.0    # 100 mm/year
            min_val = 10**log_min
            max_val = 10**log_max
            unit = "mm/year"
            
            # Draw gradient bar (blue to green to red for velocity)
            for i in range(bar_height):
                normalized = i / bar_height
                
                # Blue component: high for low velocities, decreases with velocity
                blue = int((1 - normalized) * 255)
                
                # Green component: peaks in the middle range
                green_factor = 4 * normalized * (1 - normalized)  # Parabolic peak at 0.5
                green = int(green_factor * 255)
                
                # Red component: high for high velocities
                red = int(normalized * 255)
                
                color = (red, green, blue)
                pygame.draw.line(self.screen, color, 
                               (bar_x, bar_y + bar_height - i - 1), 
                               (bar_x + bar_width, bar_y + bar_height - i - 1))
        
        elif self.display_mode == 'gravity':
            # Gravity magnitude (linear scale)
            gx = self.simulation.gravity_x
            gy = self.simulation.gravity_y
            g_mag = np.sqrt(gx**2 + gy**2)
            min_val = 0.0
            max_val = max(1e-5, np.max(g_mag))
            unit = "m/s²"

            for i in range(bar_height):
                normalized = i / bar_height
                # Blue to red via green
                blue = int((1 - normalized) * 255)
                green_factor = 4 * normalized * (1 - normalized)
                green = int(green_factor * 255)
                red = int(normalized * 255)
                color = (red, green, blue)
                pygame.draw.line(self.screen, color,
                                 (bar_x, bar_y + bar_height - i - 1),
                                 (bar_x + bar_width, bar_y + bar_height - i - 1))
        
        elif self.display_mode == 'potential':
            # Visualise gravitational potential Φ across the grid
            phi = self.simulation.gravitational_potential
            min_val = np.min(phi)
            max_val = np.max(phi)
            span = max_val - min_val if max_val != min_val else 1.0
            norm = (phi - min_val) / span

            colors = np.zeros((self.sim_height, self.sim_width, 3), dtype=np.uint8)
            blue = ((1 - norm) * 255).astype(np.uint8)
            red = (norm * 255).astype(np.uint8)
            green = (4 * norm * (1 - norm) * 255).astype(np.uint8)
            colors[..., 0] = red
            colors[..., 1] = green
            colors[..., 2] = blue
            return colors
        
        # Draw border around color bar
        pygame.draw.rect(self.screen, self.colors['text'], 
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Draw scale labels
        label_x = bar_x + bar_width + 5
        
        if self.display_mode == 'power':
            # Show separate scales for heating and cooling using power per cell
            cell_volume = self.simulation.cell_size ** 3  # m³ per cell
            power_per_cell = self.simulation.power_density * cell_volume  # W per cell
            
            heating = np.maximum(0, power_per_cell)
            cooling = np.maximum(0, -power_per_cell)
            max_heating = np.max(heating) if np.any(heating > 0) else 1e-10
            max_cooling = np.max(cooling) if np.any(cooling > 0) else 1e-10
            
            # Top: heating scale with smart formatting
            heat_text = f"+{self._format_power_smart(max_heating)}"
            heat_surface = self.small_font.render(heat_text, True, (255, 100, 100))  # Light red
            self.screen.blit(heat_surface, (label_x, bar_y - 5))
            
            # Middle: zero
            mid_text = "0 W"
            mid_surface = self.small_font.render(mid_text, True, self.colors['text'])
            self.screen.blit(mid_surface, (label_x, bar_y + bar_height // 2 - 10))
            
            # Bottom: cooling scale with smart formatting
            cool_text = f"-{self._format_power_smart(max_cooling)}"
            cool_surface = self.small_font.render(cool_text, True, (100, 100, 255))  # Light blue
            self.screen.blit(cool_surface, (label_x, bar_y + bar_height - 15))
        elif self.display_mode == 'velocity':
            # Special formatting for velocity (logarithmic scale)
            max_text = f"100 {unit}"
            max_surface = self.small_font.render(max_text, True, self.colors['text'])
            self.screen.blit(max_surface, (label_x, bar_y - 5))
            
            mid_text = f"0.01 {unit}"
            mid_surface = self.small_font.render(mid_text, True, self.colors['text'])
            self.screen.blit(mid_surface, (label_x, bar_y + bar_height // 2 - 10))
            
            min_text = f"1e-6 {unit}"
            min_surface = self.small_font.render(min_text, True, self.colors['text'])
            self.screen.blit(min_surface, (label_x, bar_y + bar_height - 15))
        else:
            # Regular scaling (temperature/pressure/gravity)
            def _fmt(val: float) -> str:
                if self.display_mode == 'gravity':
                    # For very small magnitudes use scientific notation
                    return f"{val:.2e} {unit}" if val < 0.1 else f"{val:.1f} {unit}"
                else:
                    return f"{val:.0f}{unit}"

            max_surface = self.small_font.render(_fmt(max_val), True, self.colors['text'])
            self.screen.blit(max_surface, (label_x, bar_y - 5))

            mid_val = (min_val + max_val) / 2
            mid_surface = self.small_font.render(_fmt(mid_val), True, self.colors['text'])
            self.screen.blit(mid_surface, (label_x, bar_y + bar_height // 2 - 10))

            min_surface = self.small_font.render(_fmt(min_val), True, self.colors['text'])
            self.screen.blit(min_surface, (label_x, bar_y + bar_height - 15))
        
        # Draw title above the bar
        if self.display_mode == 'power':
            title = "Power"
        elif self.display_mode == 'temperature':
            title = "Temperature"
        elif self.display_mode == 'velocity':
            title = "Velocity"
        elif self.display_mode == 'gravity':
            title = "Gravity"
        elif self.display_mode == 'potential':
            title = "Potential"
        else:
            title = "Pressure"
        title_surface = self.small_font.render(title, True, self.colors['text'])
        title_x = bar_x + (bar_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, bar_y - 25))
    
    def _draw_graphs(self):
        """Draw time-series graphs"""
        time_series = self.simulation.time_series
        
        if not time_series['time'] or len(time_series['time']) < 2:
            # Not enough data yet
            text = "No data to display yet. Run simulation to collect data."
            text_surface = self.font.render(text, True, self.colors['text'])
            text_x = self.main_panel_width // 2 - text_surface.get_width() // 2
            text_y = self.window_height // 2
            self.screen.blit(text_surface, (text_x, text_y))
            return
        
        # Create matplotlib figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('black')
        
        times = time_series['time']
        
        # Plot 1: Temperature vs Time
        ax1.plot(times, time_series['avg_temperature'], 'r-', label='Average', linewidth=2)
        ax1.plot(times, time_series['max_temperature'], 'orange', label='Maximum', linewidth=1)
        ax1.set_title('Temperature vs Time', color='white', fontsize=12)
        ax1.set_xlabel('Time (years)', color='white')
        ax1.set_ylabel('Temperature (°C)', color='white')
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        # Plot 2: Total Energy vs Time
        ax2.plot(times, time_series['total_energy'], 'b-', linewidth=2)
        ax2.set_title('Total Energy vs Time', color='white', fontsize=12)
        ax2.set_xlabel('Time (years)', color='white')
        ax2.set_ylabel('Energy (J)', color='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        
        # Plot 3: Net Power vs Time
        ax3.plot(times, time_series['net_power'], 'g-', linewidth=2)
        ax3.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax3.set_title('Net Power vs Time', color='white', fontsize=12)
        ax3.set_xlabel('Time (years)', color='white')
        ax3.set_ylabel('Power (W)', color='white')
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor('black')
        ax3.tick_params(colors='white')
        
        # Plot 4: Climate Factors vs Time
        ax4.plot(times, time_series['greenhouse_factor'], 'purple', label='Greenhouse Factor', linewidth=2)
        ax4.plot(times, time_series['planet_albedo'], 'cyan', label='Planet Albedo', linewidth=2)
        ax4.set_title('Climate Factors vs Time', color='white', fontsize=12)
        ax4.set_xlabel('Time (years)', color='white')
        ax4.set_ylabel('Factor (0-1)', color='white')
        ax4.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('black')
        ax4.tick_params(colors='white')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert matplotlib figure to pygame surface (ensure no display interference)
        plt.ioff()  # Turn off interactive mode
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        
        # Handle multiple matplotlib API versions robustly
        try:
            # Method 1: Try newest API first (matplotlib 3.8+)
            renderer = canvas.get_renderer()
            raw_data = renderer.tobytes()
        except AttributeError:
            try:
                # Method 2: Try older API (matplotlib 3.0-3.7)
                renderer = canvas.get_renderer()
                raw_data = renderer.tostring_rgb()
            except AttributeError:
                try:
                    # Method 3: Try even older API (matplotlib 2.x)
                    renderer = canvas.get_renderer()
                    raw_data = renderer.buffer_rgba()
                except AttributeError:
                    try:
                        # Method 4: Canvas buffer (matplotlib 1.x/2.x)
                        raw_data = canvas.tostring_rgb()
                    except AttributeError:
                        # Method 5: Last resort - save and reload
                        import io
                        buf = io.BytesIO()
                        canvas.print_rgb(buf)
                        buf.seek(0)
                        raw_data = buf.read()
        
        # Ensure raw_data is bytes, not memoryview
        if isinstance(raw_data, memoryview):
            raw_data = raw_data.tobytes()
        
        size = canvas.get_width_height()
        
        # Handle different color formats from matplotlib
        expected_rgb_size = size[0] * size[1] * 3  # RGB = 3 bytes per pixel
        expected_rgba_size = size[0] * size[1] * 4  # RGBA = 4 bytes per pixel
        
        if len(raw_data) == expected_rgba_size:
            # Convert RGBA to RGB by removing alpha channel
            import numpy as np
            rgba_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[1], size[0], 4)
            rgb_array = rgba_array[:, :, :3]  # Take only RGB channels
            raw_data = rgb_array.tobytes()
            format_str = 'RGB'
        elif len(raw_data) == expected_rgb_size:
            format_str = 'RGB'
        else:
            # Try to auto-detect format based on size
            channels = len(raw_data) // (size[0] * size[1])
            if channels == 3:
                format_str = 'RGB'
            elif channels == 4:
                format_str = 'RGBA'
            else:
                raise ValueError(f"Unexpected data format: {len(raw_data)} bytes for {size[0]}x{size[1]} image")
        
        # Create pygame surface from matplotlib data
        graph_surface = pygame.image.fromstring(raw_data, size, format_str)
        
        # Scale to fit main panel
        graph_width = self.main_panel_width - 20
        graph_height = self.sim_area_height - 20
        scaled_surface = pygame.transform.scale(graph_surface, (graph_width, graph_height))
        
        # Blit to screen
        self.screen.blit(scaled_surface, (10, self.status_bar_height + 10))
        
        # Close matplotlib figure to free memory and ensure display state is preserved
        plt.close(fig)
        plt.ion()  # Re-enable interactive mode for potential future use
    
    def _draw_status_bar(self):
        """Draw status information at the top"""
        stats = self.simulation.get_stats()
        
        # Draw status bar background
        status_rect = pygame.Rect(0, 0, self.window_width, self.status_bar_height)
        pygame.draw.rect(self.screen, (40, 40, 40), status_rect)
        pygame.draw.line(self.screen, self.colors['text'], (0, self.status_bar_height-1), (self.window_width, self.status_bar_height-1))
        
        # Status text with performance info
        play_status = "PAUSED" if self.paused else f"PLAYING {self.speed_multiplier}x"
        
        # Add performance info
        perf_info = ""
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            perf_info = f" | Step: {avg_step_time:.1f}ms"
        
        status_text = f"{play_status} | Time: {stats['time']:.0f}y | Temp: {stats['avg_temperature']:.0f}°C (max {stats['max_temperature']:.0f}°C) | Pressure: {stats['avg_pressure']:.1f} MPa | Tool: {self.mouse_tool.title()} (R:{self.tool_radius}, I:{self.tool_intensity}){perf_info}"
        
        text_surface = self.small_font.render(status_text, True, self.colors['text'])
        self.screen.blit(text_surface, (5, 5))
    
    def _draw_info_tab(self, x_offset: int, y_offset: int):
        """Draw rock type transition diagram"""
        
        # Title
        title = self.small_font.render("Rock Transitions:", True, self.colors['text'])
        self.screen.blit(title, (x_offset, y_offset))
        y_offset += 25
        
        # Draw simplified transition pathways with geological process focus
        transitions = [
            ("Metamorphism (Heat + Pressure):", self.colors['green']),
            ("  Shale -> Slate -> Schist -> Gneiss", self.colors['text']),
            ("  Sandstone -> Quartzite", self.colors['text']), 
            ("  Limestone -> Marble", self.colors['text']),
            ("  Granite -> Gneiss", self.colors['text']),
            ("", self.colors['text']),
            ("Melting (High Heat):", self.colors['red']),
            ("  Any Rock -> Magma", self.colors['text']),
            ("", self.colors['text']),
            ("Crystallization (Cooling):", self.colors['blue']),
            ("  Slow cooling -> Granite", self.colors['text']),
            ("  Fast cooling -> Basalt", self.colors['text']),
            ("  Very fast -> Obsidian", self.colors['text']),
            ("", self.colors['text']),
            ("Tips:", self.colors['yellow']),
            ("- Use Heat tool to melt rocks", self.colors['text']),
            ("- Use Pressure tool for metamorphism", self.colors['text']),
            ("- Try different intensities!", self.colors['text'])
        ]
        
        for transition_text, color in transitions:
            if transition_text:  # Skip empty lines
                text = self.small_font.render(transition_text, True, color)
                self.screen.blit(text, (x_offset, y_offset))
            y_offset += 14
    
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
            elif button['action'] == f'tab_{self.sidebar_tab}':
                color = self.colors['button_hover']
            elif button['action'] == 'toggle_pause' and self.paused:
                color = self.colors['red']
            elif button['action'] == f'speed_{self.speed_multiplier}':
                color = self.colors['green']
                
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, self.colors['text'], button['rect'], 2)
            
            # Button text
            text_surface = self.small_font.render(button['text'], True, self.colors['text'])
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)
        
        # Draw tabbed content
        self._draw_tabbed_content()
    
    def _draw_tabbed_content(self):
        """Draw content based on selected tab"""
        x_offset = self.main_panel_width + 10
        
        # Calculate dynamic y_offset based on where tab buttons end
        # Find the bottom of the tab buttons by looking at their positions
        tab_buttons = [btn for btn in self.buttons if btn['action'].startswith('tab_')]
        if tab_buttons:
            # Tab buttons are in a 2-row grid, so find the bottom of the second row
            max_bottom = max(btn['rect'].bottom for btn in tab_buttons)
            y_offset = max_bottom + 15  # Add some spacing below the buttons
        else:
            y_offset = 440  # Fallback to old hardcoded value
        
        if self.sidebar_tab == 'controls':
            self._draw_controls_tab(x_offset, y_offset)
        
        elif self.sidebar_tab == 'stats':
            self._draw_stats_tab(x_offset, y_offset)
        
        elif self.sidebar_tab == 'composition':
            self._draw_composition_tab(x_offset, y_offset)
        
        elif self.sidebar_tab == 'graphs':
            self._draw_graphs_tab(x_offset, y_offset)
        
        elif self.sidebar_tab == 'info':
            self._draw_info_tab(x_offset, y_offset)
    
    def _draw_composition_tab(self, x_offset: int, y_offset: int):
        """Draw rock composition information"""
        stats = self.simulation.get_stats()
        
        # Header
        header = self.small_font.render("Material Composition:", True, self.colors['green'])
        self.screen.blit(header, (x_offset, y_offset))
        y_offset += 20
        
        # Material composition with colors (already sorted by percentage in descending order)
        # Filter out space since it's not a geological material
        filtered_composition = {k: v for k, v in stats['material_composition'].items() if k != 'space'}
        for material_type, percentage in filtered_composition.items():
            try:
                material_enum = MaterialType(material_type)
                material_color = self.simulation.material_db.get_properties(material_enum).color_rgb
                
                # Draw color square
                color_rect = pygame.Rect(x_offset, y_offset + 2, 12, 12)
                pygame.draw.rect(self.screen, material_color, color_rect)
                pygame.draw.rect(self.screen, self.colors['text'], color_rect, 1)
                
                text = f"{material_type}: {percentage:.1f}%"
                surface = self.small_font.render(text, True, self.colors['text'])
                self.screen.blit(surface, (x_offset + 18, y_offset))
            except:
                text = f"{material_type}: {percentage:.1f}%"
                surface = self.small_font.render(text, True, self.colors['text'])
                self.screen.blit(surface, (x_offset, y_offset))
            
            y_offset += 16
        
        # Additional stats
        y_offset += 10
        stats_header = self.small_font.render("Statistics:", True, self.colors['green'])
        self.screen.blit(stats_header, (x_offset, y_offset))
        y_offset += 16
        
        stat_texts = [
            f"Max Pressure: {stats['max_pressure']:.1f} MPa",
            f"History Length: {stats['history_length']}",
        ]
        
        for text in stat_texts:
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 14
    
    def _draw_stats_tab(self, x_offset: int, y_offset: int):
        """Draw simulation statistics"""
        stats = self.simulation.get_stats()
        
        # Header
        header = self.small_font.render("Simulation Stats:", True, self.colors['green'])
        self.screen.blit(header, (x_offset, y_offset))
        y_offset += 20
        
        # Time and simulation stats
        stat_texts = [
            f"Time Step: {stats['dt']:.0f} years",
            f"Effective dt: {stats['effective_dt']:.1f} years",
            f"Stability Factor: {stats['stability_factor']:.3f}",
            f"Max Thermal Diff: {stats['max_thermal_diffusivity']:.2e} m²/s",
            "",
            f"Simulation Time: {stats['time']:.0f} years",
            f"History Length: {stats['history_length']} steps",
            "",
            f"Temperature Range:",
            f"  Average: {stats['avg_temperature']:.0f}°C",
            f"  Maximum: {stats['max_temperature']:.0f}°C",
            "",
            f"Pressure Range:",
            f"  Average: {stats['avg_pressure']:.1f} MPa",
            f"  Maximum: {stats['max_pressure']:.1f} MPa",
        ]
        
        for text in stat_texts:
            if text:  # Skip empty lines
                color = self.colors['yellow'] if 'Factor' in text and stats['stability_factor'] < 1.0 else self.colors['text']
                surface = self.small_font.render(text, True, color)
                self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 14
    
    def _draw_controls_tab(self, x_offset: int, y_offset: int):
        """Draw controls information"""
        # Tool info
        tool_header = self.small_font.render("Current Tool:", True, self.colors['green'])
        self.screen.blit(tool_header, (x_offset, y_offset))
        y_offset += 20
        
        play_status = "PAUSED" if self.paused else f"PLAYING {self.speed_multiplier}x"
        # Get current quality level
        current_quality = getattr(self.simulation, '_quality_level', 1)
        quality_names = {1: "Full", 2: "Balanced", 3: "Fast"}
        quality_name = quality_names.get(current_quality, f"Level {current_quality}")
        
        # Get kinematics mode if available
        kinematics_mode = "N/A"
        if hasattr(self.simulation, 'get_kinematics_mode'):
            kinematics_mode = self.simulation.get_kinematics_mode()
        
        tool_texts = [
            f"Status: {play_status}",
            f"Tool: {self.mouse_tool.title()}",
            f"Radius: {self.tool_radius}",
            f"Intensity: {self.tool_intensity}",
            f"Quality: {quality_name} ({current_quality})",
            f"Kinematics: {kinematics_mode}",
        ]
        
        for text in tool_texts:
            surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 14
        
        # Instructions
        y_offset += 15
        instructions = [
            "Controls:",
            "  Left click + drag: Apply tool",
            "  Right click: Select tile (show info)",
            "  Mouse wheel: Adjust radius", 
            "  Shift + wheel: Adjust intensity",
            "  Space: Play/Pause",
            "  +/-: Change playback speed",
            "  R: Step forward",
            "  T: Step backward",
            "  A: Add mass tool (cycle material)",
            "  D: Delete mass tool",
            "  L: Toggle logging (INFO/DEBUG)",
            "  M: Toggle kinematics mode",
            "  1/2/3/4/5: Switch display modes",
            "  Tab: Cycle sidebar tabs",
            "  Q: Change quality setting"
        ]
        
        for i, instruction in enumerate(instructions):
            color = self.colors['green'] if i == 0 else self.colors['text']
            surface = self.small_font.render(instruction, True, color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 14
    
    def _draw_graphs_tab(self, x_offset: int, y_offset: int):
        """Draw compact graphs in the sidebar"""
        time_series = self.simulation.time_series
        
        if not time_series['time'] or len(time_series['time']) < 2:
            # Not enough data yet
            header = self.small_font.render("Time Series Graphs:", True, self.colors['green'])
            self.screen.blit(header, (x_offset, y_offset))
            y_offset += 20
            
            text = "No data yet. Run simulation"
            text_surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (x_offset, y_offset))
            y_offset += 14
            
            text2 = "to collect data."
            text_surface2 = self.small_font.render(text2, True, self.colors['text'])
            self.screen.blit(text_surface2, (x_offset, y_offset))
            return
        
        # Create compact matplotlib figure
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as agg
        
        # Create 2x1 subplot layout focusing on key metrics for better readability
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
        fig.patch.set_facecolor('black')
        
        times = time_series['time']
        
        # Plot 1: Average Temperature vs Time (highly readable)
        ax1.plot(times, time_series['avg_temperature'], 'r-', linewidth=3)
        ax1.set_title('Average Temperature', color='white', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', color='white', fontsize=16)
        ax1.tick_params(colors='white', labelsize=14)
        ax1.grid(False)  # Remove grid for cleaner look
        ax1.set_facecolor('black')
        

        
        # Plot 2: Net Power vs Time (highly readable with prominent zero line)
        ax2.plot(times, time_series['net_power'], 'lime', linewidth=3)
        # Make zero crossing very prominent
        ax2.axhline(y=0, color='yellow', linestyle='-', alpha=1.0, linewidth=3)
        ax2.set_title('Net Power Balance', color='white', fontsize=20, fontweight='bold')
        ax2.set_xlabel('Time (Years)', color='white', fontsize=16)
        ax2.set_ylabel('Net Power (W)', color='white', fontsize=16)
        ax2.tick_params(colors='white', labelsize=14)
        ax2.grid(False)  # Remove grid for cleaner look
        ax2.set_facecolor('black')
        

        
        # Tight layout with better spacing
        plt.tight_layout(pad=0.8)
        
        # Convert to pygame surface (ensure no display interference)
        plt.ioff()  # Turn off interactive mode
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        
        # Handle multiple matplotlib API versions robustly
        try:
            # Method 1: Try newest API first (matplotlib 3.8+)
            renderer = canvas.get_renderer()
            raw_data = renderer.tobytes()
        except AttributeError:
            try:
                # Method 2: Try older API (matplotlib 3.0-3.7)
                renderer = canvas.get_renderer()
                raw_data = renderer.tostring_rgb()
            except AttributeError:
                try:
                    # Method 3: Try even older API (matplotlib 2.x)
                    renderer = canvas.get_renderer()
                    raw_data = renderer.buffer_rgba()
                except AttributeError:
                    try:
                        # Method 4: Canvas buffer (matplotlib 1.x/2.x)
                        raw_data = canvas.tostring_rgb()
                    except AttributeError:
                        # Method 5: Last resort - save and reload
                        import io
                        buf = io.BytesIO()
                        canvas.print_rgb(buf)
                        buf.seek(0)
                        raw_data = buf.read()
        
        # Ensure raw_data is bytes, not memoryview
        if isinstance(raw_data, memoryview):
            raw_data = raw_data.tobytes()
        
        size = canvas.get_width_height()
        
        # Handle different color formats from matplotlib
        expected_rgb_size = size[0] * size[1] * 3  # RGB = 3 bytes per pixel
        expected_rgba_size = size[0] * size[1] * 4  # RGBA = 4 bytes per pixel
        
        if len(raw_data) == expected_rgba_size:
            # Convert RGBA to RGB by removing alpha channel
            import numpy as np
            rgba_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(size[1], size[0], 4)
            rgb_array = rgba_array[:, :, :3]  # Take only RGB channels
            raw_data = rgb_array.tobytes()
            format_str = 'RGB'
        elif len(raw_data) == expected_rgb_size:
            format_str = 'RGB'
        else:
            # Try to auto-detect format based on size
            channels = len(raw_data) // (size[0] * size[1])
            if channels == 3:
                format_str = 'RGB'
            elif channels == 4:
                format_str = 'RGBA'
            else:
                raise ValueError(f"Unexpected data format: {len(raw_data)} bytes for {size[0]}x{size[1]} image")
        
        graph_surface = pygame.image.fromstring(raw_data, size, format_str)
        
        # Scale to fit sidebar width
        sidebar_width = self.sidebar_width - 20
        aspect_ratio = size[1] / size[0]  # height / width
        scaled_height = int(sidebar_width * aspect_ratio)
        scaled_surface = pygame.transform.scale(graph_surface, (sidebar_width, scaled_height))
        
        # Blit to sidebar
        self.screen.blit(scaled_surface, (x_offset, y_offset))
        
        # Close matplotlib figure and ensure display state is preserved
        plt.close(fig)
        plt.ion()  # Re-enable interactive mode for potential future use
    
    def _handle_keyboard(self, event):
        """Handle keyboard input"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # SPACE: Play/Pause simulation
                self.paused = not self.paused
            elif event.key == pygame.K_LEFT:
                # Left arrow: Step backward (handle initial press)
                if not self.key_repeat_state['left_held']:
                    self.simulation.step_backward()
                    self.key_repeat_state['left_held'] = True
                    self.key_repeat_state['last_repeat_time'] = pygame.time.get_ticks()
                    self.key_repeat_state['initial_press_handled'] = True
            elif event.key == pygame.K_RIGHT:
                # Right arrow: Step forward (handle initial press)
                if not self.key_repeat_state['right_held']:
                    self.simulation.step_forward()
                    self.key_repeat_state['right_held'] = True
                    self.key_repeat_state['last_repeat_time'] = pygame.time.get_ticks()
                    self.key_repeat_state['initial_press_handled'] = True
            elif event.key == pygame.K_UP:
                # Up arrow: Increase simulation speed
                speeds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                current_idx = speeds.index(self.speed_multiplier) if self.speed_multiplier in speeds else 1
                if current_idx < len(speeds) - 1:
                    self.speed_multiplier = speeds[current_idx + 1]
            elif event.key == pygame.K_DOWN:
                # Down arrow: Decrease simulation speed
                speeds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                current_idx = speeds.index(self.speed_multiplier) if self.speed_multiplier in speeds else 1
                if current_idx > 0:
                    self.speed_multiplier = speeds[current_idx - 1]
            elif event.key == pygame.K_1:
                # 1: Material view
                self.display_mode = 'materials'
            elif event.key == pygame.K_2:
                # 2: Temperature view
                self.display_mode = 'temperature'
            elif event.key == pygame.K_3:
                # 3: Power view
                self.display_mode = 'power'
            elif event.key == pygame.K_4:
                # 4: Pressure view
                self.display_mode = 'pressure'
            elif event.key == pygame.K_5:
                # 5: Velocity view
                self.display_mode = 'velocity'
            elif event.key == pygame.K_6:
                # 6: Gravity view
                self.display_mode = 'gravity'
            elif event.key == pygame.K_7:
                # 7: Potential view
                self.display_mode = 'potential'
            elif event.key == pygame.K_r:
                # R: Reset simulation (now centralized in GeologySimulation)
                self.simulation.reset()
            elif event.key == pygame.K_q:
                # Q: Change quality setting (cycle between 1=Full, 2=Balanced, 3=Fast)
                current_quality = getattr(self.simulation, '_quality_level', 1)
                new_quality = (current_quality % 3) + 1
                self.simulation._setup_performance_config(new_quality)
                self.simulation._quality_level = new_quality
            elif event.key == pygame.K_TAB:
                # Tab: Cycle through sidebar tabs
                tabs = ['controls', 'stats', 'composition', 'graphs', 'info']
                current_idx = tabs.index(self.sidebar_tab)
                self.sidebar_tab = tabs[(current_idx + 1) % len(tabs)]
            elif event.key == pygame.K_d:
                # D: Delete mass tool
                self.mouse_tool = 'delete'
            elif event.key == pygame.K_a:
                # A: cycle add-mass tool and material
                if self.mouse_tool != 'add':
                    self.mouse_tool = 'add'
                else:
                    # Cycle to next material
                    self._add_material_index = (self._add_material_index + 1) % len(self.add_materials)
                self._update_add_button_text()
            elif event.key == pygame.K_l:
                # L: Toggle verbose logging / performance output
                self.simulation.toggle_logging()
            elif event.key == pygame.K_m:
                # M: Toggle kinematics mode (unified vs discrete)
                if hasattr(self.simulation, 'toggle_kinematics_mode'):
                    mode = self.simulation.toggle_kinematics_mode()
                    print(f"Kinematics mode: {mode}")
                else:
                    print("Kinematics toggle not available (requires modular simulation)")
        
        elif event.type == pygame.KEYUP:
            # Handle key releases for repeat functionality
            if event.key == pygame.K_LEFT:
                self.key_repeat_state['left_held'] = False
            elif event.key == pygame.K_RIGHT:
                self.key_repeat_state['right_held'] = False
    
    def _handle_key_repeat(self):
        """Handle key repeat for arrow keys"""
        current_time = pygame.time.get_ticks()
        
        # Check if left arrow should repeat
        if self.key_repeat_state['left_held']:
            time_since_last = current_time - self.key_repeat_state['last_repeat_time']
            
            # Use delay for first repeat, then rate for subsequent repeats
            threshold = self.key_repeat_state['repeat_delay'] if self.key_repeat_state['initial_press_handled'] else self.key_repeat_state['repeat_rate']
            
            if time_since_last >= threshold:
                self.simulation.step_backward()
                self.key_repeat_state['last_repeat_time'] = current_time
                self.key_repeat_state['initial_press_handled'] = False  # Switch to repeat rate
        
        # Check if right arrow should repeat
        if self.key_repeat_state['right_held']:
            time_since_last = current_time - self.key_repeat_state['last_repeat_time']
            
            # Use delay for first repeat, then rate for subsequent repeats
            threshold = self.key_repeat_state['repeat_delay'] if self.key_repeat_state['initial_press_handled'] else self.key_repeat_state['repeat_rate']
            
            if time_since_last >= threshold:
                self.simulation.step_forward()
                self.key_repeat_state['last_repeat_time'] = current_time
                self.key_repeat_state['initial_press_handled'] = False  # Switch to repeat rate

    def _handle_mouse_wheel(self, event):
        """Handle mouse wheel for tool adjustment"""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            # Adjust intensity
            self.tool_intensity = max(1, self.tool_intensity + event.y * 10)
        else:
            # Adjust radius
            self.tool_radius = max(1, min(20, self.tool_radius + event.y))
    
    def _update_add_button_text(self):
        """Update the label on the Add Mass button to show selected material."""
        material_name = self.add_materials[self._add_material_index].value
        for btn in self.buttons:
            if btn['action'] == 'tool_add':
                btn['text'] = f'Add: {material_name}'
                break
    
    def run(self):
        """Main visualization loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # Right click
                        self._handle_right_click(event.pos)
                    else:  # Left click and others
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
            
            # Handle key repeat for arrow keys
            self._handle_key_repeat()
            
            # Auto-stepping when playing
            if not self.paused:
                if self.speed_multiplier <= 1.0:
                    # For slow speeds, step every few frames
                    frame_skip = int(1.0 / self.speed_multiplier)  # 0.5x = every 2 frames
                    if pygame.time.get_ticks() % (frame_skip * 17) < 17:  # ~60fps timing
                        step_start = pygame.time.get_ticks()
                        self.simulation.step_forward()
                        step_duration = pygame.time.get_ticks() - step_start
                        self.step_times.append(step_duration)
                        if len(self.step_times) > self.max_step_history:
                            self.step_times.pop(0)
                else:
                    # For normal and high speeds, run multiple steps per frame
                    steps_per_frame = int(self.speed_multiplier)
                    
                    # Measure step performance
                    step_start = pygame.time.get_ticks()
                    for _ in range(steps_per_frame):
                        self.simulation.step_forward()
                    step_duration = pygame.time.get_ticks() - step_start
                    
                    # Track performance (duration per single step for comparison)
                    avg_step_duration = step_duration / steps_per_frame if steps_per_frame > 0 else step_duration
                    self.step_times.append(avg_step_duration)
                    if len(self.step_times) > self.max_step_history:
                        self.step_times.pop(0)
            
            # Draw everything
            self.screen.fill(self.colors['background'])
            self._draw_simulation()
            self._draw_status_bar()
            self._draw_sidebar()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

def main():
    """Run the geology simulator"""
    print("Starting Geology Simulator...")
    print("Controls:")
    print("  Mouse: Click and drag to add heat sources")
    print("  Right Click: Select tile (show info)")
    print("  Shift + Right Click: Add tectonic stress")
    print("  SPACE: Play/Pause simulation")
    print("  Left/Right arrows: Step backward/forward")
    print("  Up/Down arrows: Adjust simulation speed")
    print("  1-7: Switch visualization modes (Material, Temperature, Power, Pressure, Velocity, Gravity, Potential)")
    print("  R: Reset simulation")
    print("  G: Toggle graphs display")
    print("  Q: Change quality setting (1=Full, 2=Balanced, 3=Fast)")
    print("  Tab: Cycle through sidebar tabs")
    print("  Mouse wheel: Adjust tool radius")
    print("  Shift + mouse wheel: Adjust tool intensity")
    print("  A: Add mass tool (cycle material)")
    print("  D: Delete mass tool")
    print("  L: Toggle logging (INFO/DEBUG)")
    
    visualizer = GeologyVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main() 