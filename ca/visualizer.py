"""
Visualization for flux-based geological simulation.

Adapted to display volume fractions and continuous fields instead of
discrete materials per cell.
"""

import pygame
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from geo_game import GeoGame
from materials import MaterialType, MaterialDatabase


class DisplayMode(Enum):
    """Available visualization modes."""
    MATERIAL_DOMINANT = "material_dominant"
    TEMPERATURE = "temperature"
    GRAVITY = "gravity"
    POWER = "power"


class GeologyVisualizer:
    """Interactive visualizer for CA-based simulation."""
    
    def __init__(
        self,
        simulation: GeoGame,
        window_width: int = 800,
        window_height: int = 900,
        scale_factor: float = 1.5,
    ):
        """
        Initialize visualizer.
        
        Args:
            simulation: GeoGame instance
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        self.simulation = simulation
        self.state = simulation  # In CA, the simulation IS the state
        self.material_db = simulation.material_db
        self.scale_factor = scale_factor
        
        # Apply scale factor to all dimensions
        scaled_window_width = int(window_width * scale_factor)
        scaled_window_height = int(window_height * scale_factor)
        
        # Pygame setup
        # Initialize pygame without audio to avoid ALSA warnings
        pygame.display.init()
        pygame.font.init()
        
        # Add space for toolbar on the right (also scaled)
        self.toolbar_width = int(200 * scale_factor)
        self.screen = pygame.display.set_mode((scaled_window_width + self.toolbar_width, scaled_window_height))
        pygame.display.set_caption("CA Geological Simulation")
        self.clock = pygame.time.Clock()
        
        # Scale all font sizes
        self.font = pygame.font.Font(None, int(20 * scale_factor))
        self.small_font = pygame.font.Font(None, int(14 * scale_factor))
        self.toolbar_font = pygame.font.Font(None, int(12 * scale_factor))
        
        # Display settings
        self.display_mode = DisplayMode.MATERIAL_DOMINANT
        self.show_info = True
        self.show_help = False
        self.show_performance = False
        self.show_debug = False
        
        # Store scaled window dimensions
        self.window_width = scaled_window_width
        self.window_height = scaled_window_height
        
        # Grid to screen mapping - ensure square pixels
        self.grid_surface = pygame.Surface((self.state.width, self.state.height))
        
        # Calculate scale to maintain square aspect ratio
        # Use the smaller scale to ensure the grid fits in the window
        scale = min(scaled_window_width / self.state.width, scaled_window_height / self.state.height)
        self.scale_x = scale
        self.scale_y = scale
        
        # Calculate actual display dimensions and center the grid
        self.display_width = int(self.state.width * scale)
        self.display_height = int(self.state.height * scale)
        self.display_offset_x = (scaled_window_width - self.display_width) // 2
        self.display_offset_y = (scaled_window_height - self.display_height) // 2
        
        # Interaction state
        self.running = True
        self.paused = False  # CA doesn't have paused, so track it here
        self.mouse_down = False
        self.selected_material = MaterialType.WATER
        self.tool_radius = int(5 * scale_factor)
        self.tool_intensity = 0.1
        self.selected_cell = None  # (x, y) coordinates of selected cell for inspection
        
        # Tool types
        self.tools = [
            {"name": "Material", "desc": "Add/remove materials", "icon": "M"},
            {"name": "Heat", "desc": "Add/remove heat", "icon": "H"},
        ]
        self.current_tool = 0  # Index into tools list
        
        # Sidebar dimensions (scaled)
        self.sidebar_x = scaled_window_width
        self.button_height = int(30 * scale_factor)
        self.button_margin = int(3 * scale_factor)
        
        # Color maps
        self.init_colormaps()
        
        # Colormap registry for easy lookup
        self.colormap_registry = {
            DisplayMode.TEMPERATURE: self.temp_colors,
            DisplayMode.GRAVITY: self.gravity_colors,
        }
        
        # Custom event handler for test runner
        self.custom_event_handler = None
        
    def init_colormaps(self):
        """Initialize color maps for different display modes."""
        # Temperature colormap (black -> dark red -> red -> orange -> yellow -> white)
        # Black at space temperature (2.7K) for visual consistency
        self.temp_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 10:
                # Black for very cold (space)
                self.temp_colors[i] = [0, 0, 0]
            elif i < 64:
                # Black to dark red
                t = (i - 10) / 54.0
                self.temp_colors[i] = [int(139 * t), 0, 0]
            elif i < 128:
                # Dark red to bright red
                t = (i - 64) / 64.0
                self.temp_colors[i] = [139 + int(116 * t), 0, 0]
            elif i < 192:
                # Red to orange
                t = (i - 128) / 64.0
                self.temp_colors[i] = [255, int(165 * t), 0]
            else:
                # Orange to yellow to white
                t = (i - 192) / 64.0
                self.temp_colors[i] = [255, 165 + int(90 * t), int(255 * t)]
                
        # Velocity colormap (black -> red gradient)
        self.velocity_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 5:
                # Black for zero/near-zero velocity
                self.velocity_colors[i] = [0, 0, 0]
            else:
                # Black to red gradient
                t = (i - 5) / 250.0
                self.velocity_colors[i] = [int(255 * t), 0, 0]
                
        # Gravity colormap (black at zero -> red for positive acceleration)
        self.gravity_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 5:
                # Black for zero/near-zero gravity
                self.gravity_colors[i] = [0, 0, 0]
            else:
                # Black to red gradient
                t = (i - 5) / 250.0
                self.gravity_colors[i] = [int(255 * t), 0, 0]
            
    def run(self):
        """Main visualization loop."""
        while self.running:
            self.handle_events()
            
            # Step simulation if not paused
            if not self.paused:
                self.simulation.step_forward()
                
            self.render()
            self.clock.tick(60)  # 60 FPS cap
            
        pygame.quit()
        
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            # Check custom event handler first
            if self.custom_event_handler and self.custom_event_handler(event):
                continue
                
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicking on toolbar
                    if event.pos[0] >= self.sidebar_x:
                        self.handle_toolbar_click(event.pos)
                    else:
                        self.mouse_down = True
                        self.apply_tool(event.pos, event.button)
                elif event.button == 3:  # Right click
                    # Select cell for inspection
                    if event.pos[0] < self.sidebar_x:
                        self.handle_cell_selection(event.pos)
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    self.apply_tool(event.pos, 1)
                    
            elif event.type == pygame.MOUSEWHEEL:
                # Adjust tool radius
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    self.tool_intensity = np.clip(
                        self.tool_intensity + event.y * 0.01, 0.01, 1.0
                    )
                else:
                    self.tool_radius = np.clip(
                        self.tool_radius + event.y, 1, 20
                    )
                    
    def handle_keydown(self, event):
        """Handle keyboard input."""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            
        elif event.key == pygame.K_r:
            # Reset simulation to initial state
            self.simulation.reset()
            self.paused = True
            self.selected_cell = None
            print("Simulation reset")
            
        elif event.key == pygame.K_m:
            # Cycle display mode
            modes = list(DisplayMode)
            idx = modes.index(self.display_mode)
            self.display_mode = modes[(idx + 1) % len(modes)]
            
        elif event.key == pygame.K_TAB:
            # Tab also cycles display mode (more intuitive)
            modes = list(DisplayMode)
            idx = modes.index(self.display_mode)
            self.display_mode = modes[(idx + 1) % len(modes)]
            
        elif event.key == pygame.K_h:
            self.show_help = not self.show_help
            
        elif event.key == pygame.K_i:
            self.show_info = not self.show_info
            
        elif event.key == pygame.K_l:
            self.show_performance = not self.show_performance
            if self.show_performance:
                print("Performance logging enabled")
            else:
                print("Performance logging disabled")
            
        elif event.key == pygame.K_s:
            # Save screenshot
            pygame.image.save(self.screen, "ca_simulation_screenshot.png")
            
        elif event.key == pygame.K_c:
            # Clear board - set everything to space
            self.state.material_types[:] = MaterialType.SPACE
            self.state.temperature[:] = 2.7  # Space temperature
            self.state._update_material_properties()
            print("Board cleared")
            
        elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
            # Select material
            mat_idx = event.key - pygame.K_1
            if mat_idx < len(MaterialType):
                self.selected_material = list(MaterialType)[mat_idx]
                
        elif event.key == pygame.K_t:
            # Cycle through tools
            self.current_tool = (self.current_tool + 1) % len(self.tools)
                
        elif event.key == pygame.K_LEFT:
            # Step backward in time
            self.simulation.step_backward()
            
        elif event.key == pygame.K_RIGHT:
            # Step forward one frame
            if self.paused:
                # Temporarily unpause to allow step
                self.paused = False
                self.simulation.step_forward()
                self.paused = True
                
        elif event.key == pygame.K_LEFTBRACKET:
            # Rotate sun counter-clockwise
            self.simulation.solar_angle -= 15  # 15 degrees
            print(f"Sun angle: {self.simulation.solar_angle % 360:.1f}°")
            
        elif event.key == pygame.K_RIGHTBRACKET:
            # Rotate sun clockwise
            self.simulation.solar_angle += 15  # 15 degrees
            print(f"Sun angle: {self.simulation.solar_angle % 360:.1f}°")
            
        elif event.key == pygame.K_ESCAPE:
            self.running = False
            
    def handle_cell_selection(self, pos):
        """Handle right-click cell selection for inspection."""
        x, y = pos
        # Convert screen coordinates to grid coordinates, accounting for display offset
        mouse_x = x - self.display_offset_x
        mouse_y = y - self.display_offset_y
        
        # Check if mouse is within the display area
        if mouse_x < 0 or mouse_x >= self.display_width or mouse_y < 0 or mouse_y >= self.display_height:
            self.selected_cell = None
            return
            
        grid_x = int(mouse_x / self.scale_x)
        grid_y = int(mouse_y / self.scale_y)
        
        # Validate coordinates
        if 0 <= grid_x < self.state.width and 0 <= grid_y < self.state.height:
            # Toggle selection if clicking same cell
            if self.selected_cell == (grid_x, grid_y):
                self.selected_cell = None
            else:
                self.selected_cell = (grid_x, grid_y)
        else:
            self.selected_cell = None
            
    def handle_toolbar_click(self, mouse_pos: Tuple[int, int]):
        """Handle clicks on the toolbar."""
        x, y = mouse_pos
        
        # Check tool buttons
        button_y = int(50 * self.scale_factor)
        for i, tool in enumerate(self.tools):
            if (x >= self.sidebar_x + self.button_margin and 
                x <= self.sidebar_x + self.toolbar_width - self.button_margin and
                y >= button_y and y <= button_y + self.button_height):
                self.current_tool = i
                return
            button_y += self.button_height + self.button_margin
            
        # Check material buttons - 2 column layout
        mat_button_y = button_y + int(40 * self.scale_factor)  # Same as in render_toolbar
        mat_button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        mat_button_height = int(25 * self.scale_factor)
        
        mat_index = 0
        for mat_type in MaterialType:
            # Skip SPACE - not a selectable material
            if mat_type == MaterialType.SPACE:
                continue
                
            # Calculate position - 2 columns
            col = mat_index % 2
            row = mat_index // 2
            
            mat_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (mat_button_width + self.button_margin),
                mat_button_y + row * (mat_button_height + int(3 * self.scale_factor)),
                mat_button_width,
                mat_button_height
            )
            
            if mat_rect.collidepoint(x, y):
                self.selected_material = mat_type
                return
            
            mat_index += 1
        
        # Calculate where material buttons end
        # Count materials excluding SPACE
        num_materials = len([m for m in MaterialType if m != MaterialType.SPACE])
        num_mat_rows = (num_materials + 1) // 2
        mat_section_end = button_y + int(40 * self.scale_factor) + num_mat_rows * (mat_button_height + int(3 * self.scale_factor))
            
        # Check display mode buttons - 2 columns
        display_button_y = mat_section_end + int(30 * self.scale_factor)  # After materials section
        display_modes = list(DisplayMode)
        button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        button_height = int(22 * self.scale_factor)
        
        for i, mode in enumerate(display_modes):
            col = i % 2
            row = i // 2
            
            mode_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (button_width + self.button_margin),
                display_button_y + row * (button_height + int(3 * self.scale_factor)),
                button_width,
                button_height
            )
            
            if mode_rect.collidepoint(x, y):
                self.display_mode = mode
                return
        
        # Calculate where display modes end
        num_rows = (len(display_modes) + 1) // 2
        display_section_end = display_button_y + num_rows * (button_height + int(3 * self.scale_factor))
            
        # Check physics module checkboxes - after display modes
        physics_section_y = display_section_end + int(30 * self.scale_factor)  # Match render_toolbar
        checkbox_y = physics_section_y + int(35 * self.scale_factor)
        checkbox_size = int(20 * self.scale_factor)  # Matches render_toolbar
        checkbox_margin = int(10 * self.scale_factor)
        
        physics_modules = [
            ("enable_self_gravity", self.simulation.enable_self_gravity),
            ("enable_heat_diffusion", self.simulation.enable_heat_diffusion),
            ("enable_internal_heating", self.simulation.enable_internal_heating),
            ("enable_solar_heating", self.simulation.enable_solar_heating),
            ("enable_radiative_cooling", self.simulation.enable_radiative_cooling),
            ("enable_material_processes", self.simulation.enable_material_processes),
            ("enable_weathering", self.simulation.enable_weathering),
            ("enable_atmospheric_processes", self.simulation.enable_atmospheric_processes),
        ]
        
        for attr_name, current_value in physics_modules:
            checkbox_rect = pygame.Rect(
                self.sidebar_x + checkbox_margin,
                checkbox_y,
                checkbox_size,
                checkbox_size
            )
            
            # Check if click is on this checkbox or the text label
            label_rect = pygame.Rect(
                checkbox_rect.right + int(8 * self.scale_factor),
                checkbox_rect.top,
                self.toolbar_width - checkbox_margin - checkbox_size - int(8 * self.scale_factor),
                checkbox_size
            )
            
            if checkbox_rect.collidepoint(x, y) or label_rect.collidepoint(x, y):
                # Toggle the physics module
                setattr(self.simulation, attr_name, not current_value)
                return
                
            checkbox_y += checkbox_size + int(8 * self.scale_factor)
    
    def apply_tool(self, mouse_pos: Tuple[int, int], button: int):
        """Apply interactive tool at mouse position."""
        # Convert mouse to grid coordinates, accounting for display offset
        mouse_x = mouse_pos[0] - self.display_offset_x
        mouse_y = mouse_pos[1] - self.display_offset_y
        
        # Check if mouse is within the display area
        if mouse_x < 0 or mouse_x >= self.display_width or mouse_y < 0 or mouse_y >= self.display_height:
            return
            
        gx = int(mouse_x / self.scale_x)
        gy = int(mouse_y / self.scale_y)
        
        if 0 <= gx < self.state.width and 0 <= gy < self.state.height:
            # Create circular mask
            y_grid, x_grid = np.ogrid[:self.state.height, :self.state.width]
            dist = np.sqrt((x_grid - gx)**2 + (y_grid - gy)**2)
            mask = dist < self.tool_radius
            
            tool = self.tools[self.current_tool]
            
            if tool["name"] == "Material":
                if button == 1:  # Left click - add material
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        # Remove material (make space)
                        self.state.material_types[mask] = MaterialType.SPACE
                        self.state.temperature[mask] = 2.7  # Space temperature
                    else:
                        # Add selected material
                        self.state.material_types[mask] = self.selected_material
                        
                        # Set appropriate temperature for material
                        if self.selected_material == MaterialType.MAGMA:
                            self.state.temperature[mask] = 1500 + 273.15
                        elif self.selected_material == MaterialType.WATER:
                            self.state.temperature[mask] = 20 + 273.15
                        elif self.selected_material == MaterialType.ICE:
                            self.state.temperature[mask] = -10 + 273.15
                        elif self.selected_material == MaterialType.AIR:
                            self.state.temperature[mask] = 20 + 273.15
                        else:
                            # For rocks, inherit temperature from neighbors
                            for j in range(self.state.height):
                                for i in range(self.state.width):
                                    if mask[j, i]:
                                        # Find non-space neighbors
                                        neighbors = []
                                        for dj, di in [(-1,0), (1,0), (0,-1), (0,1)]:
                                            nj, ni = j + dj, i + di
                                            if (0 <= nj < self.state.height and 0 <= ni < self.state.width and
                                                self.state.material_types[nj, ni] != MaterialType.SPACE):
                                                neighbors.append((nj, ni))
                                        
                                        if neighbors:
                                            # Average neighbor temperatures
                                            avg_temp = np.mean([self.state.temperature[nj, ni] for nj, ni in neighbors])
                                            self.state.temperature[j, i] = avg_temp
                                        else:
                                            self.state.temperature[j, i] = 20 + 273.15
                        
                    # Update material properties
                    self.state._update_material_properties()
                    
            elif tool["name"] == "Heat":
                # Apply heat source/sink
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    # Cool down
                    self.state.temperature[mask] *= 0.95
                else:
                    # Heat up
                    self.state.temperature[mask] += 50.0
                    
                    
            elif tool["name"] == "Velocity":
                # CA doesn't use velocity, skip this tool
                pass
                
    def render(self):
        """Render the current simulation state."""
        self.screen.fill((0, 0, 0))
        
        # Render based on display mode
        if self.display_mode == DisplayMode.MATERIAL_DOMINANT:
            self.render_material_dominant()
        elif self.display_mode == DisplayMode.TEMPERATURE:
            self.render_temperature()
        elif self.display_mode == DisplayMode.GRAVITY:
            self.render_gravity()
        elif self.display_mode == DisplayMode.POWER:
            self.render_power()
            
        # Draw info overlay
        if self.show_info:
            self.render_info()
            
        # Draw help overlay
        if self.show_help:
            self.render_help()
            
        # Draw performance overlay
        if self.show_performance:
            self.render_performance()
            
        # Draw toolbar
        self.render_toolbar()
        
        # Draw selected cell highlight and info
        if self.selected_cell:
            self.render_selected_cell()
            
        # Draw sun direction indicator
        if self.simulation.enable_solar_heating:
            self.render_sun_indicator()
            
        pygame.display.flip()
    
    def render_sun_indicator(self):
        """Render sun-planet system indicator in top right corner."""
        # Position in top right of display area, with margin
        margin = int(40 * self.scale_factor)
        radius = int(25 * self.scale_factor)
        center_x = self.display_offset_x + self.display_width - margin - radius
        center_y = self.display_offset_y + margin + radius
        
        # Get sun angle from simulation (in degrees)
        sun_angle_deg = self.simulation.solar_angle
        sun_angle_rad = np.radians(sun_angle_deg)
        
        # Draw thin circle showing planet's orbital path
        pygame.draw.circle(self.screen, (150, 150, 150), (center_x, center_y), radius, 1)
        
        # Draw sun in center (yellow)
        pygame.draw.circle(self.screen, (255, 255, 0), (center_x, center_y), int(8 * self.scale_factor))
        pygame.draw.circle(self.screen, (255, 200, 0), (center_x, center_y), int(8 * self.scale_factor), 2)
        
        # Calculate planet position on orbit
        # Match the _get_solar_direction convention:
        # solar_dir = (cos(angle), -sin(angle))
        # angle = 0°: sun is on the right (1, 0)
        # angle = 90°: sun is at the bottom (0, -1)
        # angle = 180°: sun is on the left (-1, 0)
        # angle = 270°: sun is at the top (0, 1)
        # So planet position is opposite: (-cos(angle), sin(angle))
        planet_x = center_x - radius * np.cos(sun_angle_rad)
        planet_y = center_y + radius * np.sin(sun_angle_rad)
        
        # Draw planet (dark circle)
        pygame.draw.circle(self.screen, (50, 50, 50), (int(planet_x), int(planet_y)), int(5 * self.scale_factor))
        pygame.draw.circle(self.screen, (100, 100, 100), (int(planet_x), int(planet_y)), int(5 * self.scale_factor), 1)
        
        # Draw a small arrow showing sunlight direction on the planet
        # Arrow points from sun towards planet
        arrow_len = int(12 * self.scale_factor)
        arrow_dir_x = planet_x - center_x
        arrow_dir_y = planet_y - center_y
        arrow_norm = np.sqrt(arrow_dir_x**2 + arrow_dir_y**2)
        if arrow_norm > 0:
            arrow_dir_x /= arrow_norm
            arrow_dir_y /= arrow_norm
            
            # Arrow start and end
            arrow_start_x = planet_x - arrow_dir_x * (int(5 * self.scale_factor) + arrow_len)
            arrow_start_y = planet_y - arrow_dir_y * (int(5 * self.scale_factor) + arrow_len)
            arrow_end_x = planet_x - arrow_dir_x * int(5 * self.scale_factor)
            arrow_end_y = planet_y - arrow_dir_y * int(5 * self.scale_factor)
            
            # Draw arrow
            pygame.draw.line(self.screen, (255, 200, 0), 
                           (int(arrow_start_x), int(arrow_start_y)),
                           (int(arrow_end_x), int(arrow_end_y)), int(2 * self.scale_factor))
        
        # Draw labels
        sun_text = self.small_font.render("SUN", True, (255, 255, 255))
        text_rect = sun_text.get_rect(center=(center_x, center_y + radius + int(20 * self.scale_factor)))
        self.screen.blit(sun_text, text_rect)
        
        # Optional: Show angle in degrees for debugging
        if self.show_debug:
            angle_deg = sun_angle_deg % 360
            angle_text = self.small_font.render(f"{angle_deg:.0f}°", True, (200, 200, 200))
            angle_rect = angle_text.get_rect(center=(center_x, center_y + radius + int(35 * self.scale_factor)))
            self.screen.blit(angle_text, angle_rect)
        
    def _render_field(self, rgb: np.ndarray, min_val: float, max_val: float, unit: str, title: str):
        """Common field rendering logic.
        
        Args:
            rgb: RGB array with shape (height, width, 3)
            min_val: Minimum value for scale bar
            max_val: Maximum value for scale bar
            unit: Unit string for scale bar
            title: Title for scale bar
        """
        # Render to surface
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        
        # Scale to display size
        scaled = pygame.transform.scale(
            self.grid_surface,
            (self.display_width, self.display_height)
        )
        
        # Blit to screen
        self.screen.blit(scaled, (self.display_offset_x, self.display_offset_y))
        
        # Add scale bar
        self.render_scale_bar(min_val, max_val, unit, title)
    
    def render_material_dominant(self):
        """Render material per cell."""
        # Create RGB image
        rgb = np.zeros((self.state.height, self.state.width, 3), dtype=np.uint8)
        
        # For each cell, get its material color
        for j in range(self.state.height):
            for i in range(self.state.width):
                mat_type = self.state.material_types[j, i]
                props = self.material_db.get_properties(mat_type)
                rgb[j, i] = props.color_rgb
        
        # Apply solar illumination if enabled
        if hasattr(self.simulation, 'solar_angle') and self.display_mode == DisplayMode.MATERIAL_DOMINANT:
            rgb = self._apply_solar_illumination(rgb)
            
        # Convert to surface and scale
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface, 
            (self.display_width, self.display_height)
        )
        self.screen.blit(scaled, (self.display_offset_x, self.display_offset_y))
        
    def _apply_solar_illumination(self, rgb: np.ndarray) -> np.ndarray:
        """Apply solar illumination effect to darken the night side."""
        # Get sun direction
        sun_dx, sun_dy = self.simulation._get_solar_direction()
        
        # For each cell, calculate dot product with sun direction
        illuminated = np.zeros((self.state.height, self.state.width), dtype=np.float32)
        
        # Get center of planet
        cx, cy = self.simulation.center_of_mass
        
        for j in range(self.state.height):
            for i in range(self.state.width):
                # Skip space cells
                if self.state.material_types[j, i] == MaterialType.SPACE:
                    illuminated[j, i] = 1.0
                    continue
                    
                # Vector from planet center to cell
                dx = i - cx
                dy = j - cy
                norm = np.sqrt(dx*dx + dy*dy)
                
                if norm > 0:
                    # Normalize
                    dx /= norm
                    dy /= norm
                    
                    # Dot product with sun direction
                    dot = dx * sun_dx + dy * sun_dy
                    
                    # Map to illumination factor (0.3 to 1.0)
                    # dot = 1: fully lit, dot = -1: fully dark
                    illuminated[j, i] = 0.3 + 0.7 * max(0, dot)
                else:
                    illuminated[j, i] = 1.0
        
        # Apply illumination to RGB
        for c in range(3):
            rgb[:, :, c] = (rgb[:, :, c] * illuminated).astype(np.uint8)
            
        return rgb
        
    def render_material_composite(self):
        """Render composite material colors - same as dominant for CA."""
        # CA doesn't have volume fractions, so just render materials
        self.render_material_dominant()
        
    def render_temperature(self):
        """Render temperature field."""
        T = self.state.temperature
        T_min, T_max = np.min(T), np.max(T)
        
        # Normalize and apply colormap
        if T_max > T_min:
            T_norm = np.clip((T - T_min) / (T_max - T_min) * 255, 0, 255).astype(np.uint8)
        else:
            T_norm = np.ones_like(T, dtype=np.uint8) * 128
        
        rgb = self.temp_colors[T_norm]
        
        # Common rendering
        self._render_field(rgb, T_min, T_max, "K", "Temperature")
        
        
    def render_gravity(self):
        """Render gravitational field magnitude."""
        # CA has gravity fields directly on the simulation object
        if hasattr(self.simulation, 'gravity_x') and hasattr(self.simulation, 'gravity_y'):
            gx = self.simulation.gravity_x
            gy = self.simulation.gravity_y
        else:
            # Calculate gravity if not available
            self.simulation.calculate_self_gravity()
            gx = self.simulation.gravity_x
            gy = self.simulation.gravity_y
            
        # Calculate magnitude
        g_mag = np.sqrt(gx**2 + gy**2)
        
        # Auto-scale based on actual values
        g_max = np.max(g_mag)
        g_min = 0  # Gravity magnitude is always positive
        if g_max > 0:
            g_norm = np.clip(g_mag / g_max, 0, 1)
        else:
            g_norm = np.zeros_like(g_mag)
        
        # Convert normalized values to colormap indices
        g_indices = np.clip(g_norm * 255, 0, 255).astype(np.uint8)
        rgb = self.gravity_colors[g_indices]
        
        # Common rendering
        self._render_field(rgb, g_min, g_max, "m/s²", "Gravity")
        
    def render_power(self):
        """Render power generation/dissipation with diverging color scale."""
        # Get power density field
        power = self.state.power_density.copy()
        
        # If power_density is not being updated (all zeros), calculate manually
        if np.max(np.abs(power)) < 1e-10:
            # Add radioactive heat generation as a minimum
            uranium_props = self.material_db.get_properties(MaterialType.URANIUM)
            if hasattr(uranium_props, 'heat_generation') and uranium_props.heat_generation > 0:
                # For CA, check material_types directly
                uranium_mask = self.state.material_types == MaterialType.URANIUM
                if np.any(uranium_mask):
                    # Heat generation is in W/kg, multiply by density
                    power[uranium_mask] = (self.state.density[uranium_mask] * 
                                         uranium_props.heat_generation)
        
        # Find positive and negative extremes separately
        max_power = np.max(power)
        min_power = np.min(power)
        
        # Create RGB array
        rgb = np.zeros((self.state.height, self.state.width, 3), dtype=np.uint8)
        
        if max(abs(max_power), abs(min_power)) > 1e-10:
            # Use logarithmic scaling for better visualization
            # Handle positive and negative values independently
            
            # Define linear thresholds for positive and negative values
            linear_threshold_pos = max(abs(max_power) * 0.001, 1e-10) if max_power > 0 else 1e-10
            linear_threshold_neg = max(abs(min_power) * 0.001, 1e-10) if min_power < 0 else 1e-10
            
            # Create independent log scaling for positive and negative values
            def independent_log_scale(values, threshold_pos, threshold_neg):
                """Log scaling with independent positive/negative thresholds."""
                scaled = np.zeros_like(values)
                
                # Positive values
                pos_mask = values > threshold_pos
                if np.any(pos_mask):
                    scaled[pos_mask] = 1 + np.log10(values[pos_mask] / threshold_pos)
                    
                # Negative values
                neg_mask = values < -threshold_neg
                if np.any(neg_mask):
                    scaled[neg_mask] = -1 - np.log10(-values[neg_mask] / threshold_neg)
                    
                # Linear region near zero
                linear_pos = (values > 0) & (values <= threshold_pos)
                if np.any(linear_pos):
                    scaled[linear_pos] = values[linear_pos] / threshold_pos
                    
                linear_neg = (values < 0) & (values >= -threshold_neg)
                if np.any(linear_neg):
                    scaled[linear_neg] = values[linear_neg] / threshold_neg
                    
                return scaled
            
            # Apply independent log scaling
            power_scaled = independent_log_scale(power, linear_threshold_pos, linear_threshold_neg)
            
            # Find positive and negative scale limits separately
            scale_max_pos = np.max(power_scaled[power_scaled > 0]) if np.any(power_scaled > 0) else 0
            scale_max_neg = abs(np.min(power_scaled[power_scaled < 0])) if np.any(power_scaled < 0) else 0
            
            # Normalize positive and negative values independently
            power_norm = np.zeros_like(power_scaled)
            if scale_max_pos > 0:
                pos_mask = power_scaled > 0
                power_norm[pos_mask] = power_scaled[pos_mask] / scale_max_pos
            if scale_max_neg > 0:
                neg_mask = power_scaled < 0
                power_norm[neg_mask] = power_scaled[neg_mask] / scale_max_neg
                
            # Apply diverging colormap: blue (cooling) -> black (zero) -> red (heating)
            # Vectorized implementation for better performance
            
            # Initialize to black (for zero/near-zero values)
            rgb.fill(0)
            
            # Define threshold for "near zero" to show as black
            zero_threshold = 0.05  # 5% of normalized range
            
            # For positive values (heating): black to red
            heating_mask = power_norm > zero_threshold
            if np.any(heating_mask):
                # Normalize positive values from threshold to 1
                pos_norm = (power_norm[heating_mask] - zero_threshold) / (1.0 - zero_threshold)
                # Red channel increases from 0 to 255
                rgb[heating_mask, 0] = (255 * pos_norm).astype(np.uint8)
                # Green and blue stay at 0
                rgb[heating_mask, 1] = 0
                rgb[heating_mask, 2] = 0
            
            # For negative values (cooling): black to blue
            cooling_mask = power_norm < -zero_threshold
            if np.any(cooling_mask):
                # Normalize negative values from -threshold to -1
                neg_norm = (np.abs(power_norm[cooling_mask]) - zero_threshold) / (1.0 - zero_threshold)
                # Blue channel increases from 0 to 255
                rgb[cooling_mask, 2] = (255 * neg_norm).astype(np.uint8)
                # Red and green stay at 0
                rgb[cooling_mask, 0] = 0
                rgb[cooling_mask, 1] = 0
            
            # Values between -threshold and +threshold remain black (already set)
            
            # Store scale values for the scale bar
            self.power_scale_min = min_power
            self.power_scale_max = max_power
            self.power_scale_type = "independent_log"
        else:
            # No significant power, show black (zero)
            rgb.fill(0)
            self.power_scale_min = 0
            self.power_scale_max = 0
            self.power_scale_type = "zero"
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (self.display_width, self.display_height)
        )
        self.screen.blit(scaled, (self.display_offset_x, self.display_offset_y))
        
        # Draw power scale bar
        self.draw_power_scale_bar()
        
    def draw_power_scale_bar(self):
        """Draw scale bar for power density with diverging scale."""
        # Position at bottom right (consistent with other scale bars)
        bar_width = int(200 * self.scale_factor)
        bar_height = int(20 * self.scale_factor)
        margin = int(20 * self.scale_factor)
        bar_x = self.screen.get_width() - self.toolbar_width - bar_width - margin
        bar_y = self.screen.get_height() - bar_height - margin - int(30 * self.scale_factor)
        
        if not hasattr(self, 'power_scale_type') or self.power_scale_type == "zero":
            # No power to display
            text = self.font.render("Power: 0 W/m³", True, (255, 255, 255))
            self.screen.blit(text, (bar_x, bar_y - 25))
            return
        
        # Draw gradient bar with proper blue-black-red colormap
        # Find where zero should be positioned based on independent scales
        range_total = abs(self.power_scale_min) + abs(self.power_scale_max)
        if range_total > 0:
            zero_position = abs(self.power_scale_min) / range_total
        else:
            zero_position = 0.5
        
        zero_pixel = int(bar_width * zero_position)
        
        for i in range(bar_width):
            if i < zero_pixel:  # Cooling side (blue)
                if zero_pixel > 0:
                    # Map to [0, 1] where 0 is full blue, 1 is black
                    t = i / zero_pixel
                    b = int(255 * (1 - t))  # Blue decreases toward zero
                    r = g = 0
                else:
                    r = g = b = 0
            elif i > zero_pixel:  # Heating side (red)
                if bar_width - zero_pixel > 0:
                    # Map to [0, 1] where 0 is black, 1 is full red
                    t = (i - zero_pixel) / (bar_width - zero_pixel - 1)
                    r = int(255 * t)  # Red increases from zero
                    g = b = 0
                else:
                    r = g = b = 0
            else:  # Zero position (black)
                r = g = b = 0
                
            color = (r, g, b)
            pygame.draw.line(self.screen, color, 
                            (bar_x + i, bar_y), 
                            (bar_x + i, bar_y + bar_height))
        
        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Draw labels
        def format_power(value):
            """Format power value with appropriate units."""
            abs_val = abs(value)
            if abs_val >= 1e9:
                return f"{value/1e9:.1f} GW/m³"
            elif abs_val >= 1e6:
                return f"{value/1e6:.1f} MW/m³"
            elif abs_val >= 1e3:
                return f"{value/1e3:.1f} kW/m³"
            elif abs_val >= 1:
                return f"{value:.1f} W/m³"
            elif abs_val >= 1e-3:
                return f"{value*1e3:.1f} mW/m³"
            else:
                return f"{value:.2e} W/m³"
        
        # Title
        title = self.font.render("Power Density", True, (255, 255, 255))
        self.screen.blit(title, (bar_x, bar_y - int(25 * self.scale_factor)))
        
        # Min (cooling) label
        min_text = format_power(self.power_scale_min)
        min_label = self.small_font.render(min_text, True, (150, 150, 255))
        self.screen.blit(min_label, (bar_x, bar_y + bar_height + 2))
        
        # Zero label (positioned at actual zero location)
        center_label = self.small_font.render("0", True, (255, 255, 255))
        # Use the same zero_position calculation from the color bar
        range_total = abs(self.power_scale_min) + abs(self.power_scale_max)
        if range_total > 0:
            zero_position = abs(self.power_scale_min) / range_total
        else:
            zero_position = 0.5
        zero_x = bar_x + int(bar_width * zero_position) - center_label.get_width() // 2
        self.screen.blit(center_label, (zero_x, bar_y + bar_height + 2))
        
        # Max (heating) label
        max_text = format_power(self.power_scale_max)
        max_label = self.small_font.render(max_text, True, (255, 150, 150))
        max_x = bar_x + bar_width - max_label.get_width()
        self.screen.blit(max_label, (max_x, bar_y + bar_height + 2))
        
    def render_info(self):
        """Render information overlay."""
        # Calculate FPS
        fps = self.clock.get_fps()
        
        # Get temperature stats
        non_space_mask = self.simulation.material_types != MaterialType.SPACE
        if np.any(non_space_mask):
            avg_temp = np.mean(self.simulation.temperature[non_space_mask])
        else:
            avg_temp = 2.7
            
        # Format time smartly
        time_str = self._format_time(self.simulation.time)
        step_num = getattr(self.simulation, 'step_count', 0)
        
        y = int(10 * self.scale_factor)
        lines = [
            f"FPS: {fps:.1f}",
            f"Step: {step_num:,}",
            f"Time: {time_str}",
            f"Mode: {self.display_mode.value}",
            f"Tool: {self.tools[self.current_tool]['name']}",
            f"Material: {self.selected_material.name}",
            f"Radius: {self.tool_radius}",
            f"Paused: {not hasattr(self.simulation, 'paused') or self.paused}",
            "",
            f"Avg T: {avg_temp:.1f} K",
            f"Sun angle: {self.simulation.solar_angle % 360:.0f}°",
        ]
        
        for line in lines:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (int(10 * self.scale_factor), y))
            y += int(20 * self.scale_factor)
            
    def render_help(self):
        """Render help overlay."""
        help_text = [
            "Controls:",
            "SPACE - Pause/Resume",
            "LEFT - Step backward",
            "RIGHT - Step forward (when paused)",
            "R - Reset simulation",
            "C - Clear board",
            "TAB/M - Cycle display mode", 
            "T - Cycle tools",
            "1-9 - Select material",
            "[/] - Rotate sun angle",
            "Left Click - Apply tool",
            "Right Click - Select cell",
            "Shift+Click - Reverse tool",
            "Scroll - Adjust tool radius",
            "Shift+Scroll - Adjust intensity",
            "S - Save screenshot",
            "H - Toggle this help",
            "I - Toggle info display",
            "L - Toggle performance logging",
            "ESC - Exit",
        ]
        
        # Semi-transparent background
        line_height = int(25 * self.scale_factor)
        overlay_width = int(300 * self.scale_factor)
        overlay = pygame.Surface((overlay_width, len(help_text) * line_height + int(20 * self.scale_factor)))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (self.screen.get_width() - overlay_width - int(10 * self.scale_factor), int(10 * self.scale_factor)))
        
        # Render text
        y = int(20 * self.scale_factor)
        for line in help_text:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.screen.get_width() - overlay_width + int(10 * self.scale_factor), y))
            y += line_height
            
    def render_performance(self):
        """Render performance timing overlay."""
        if not hasattr(self.simulation, 'step_timings') or not self.simulation.step_timings:
            return
            
        # Get timings
        timings = self.simulation.step_timings
        total_time = sum(timings.values())
        
        # Create timing lines
        lines = ["Performance Timing (ms):"]
        lines.append("-" * 30)
        
        # Sort by time descending
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        
        for name, time_sec in sorted_timings:
            time_ms = time_sec * 1000
            percentage = (time_sec / total_time * 100) if total_time > 0 else 0
            lines.append(f"{name:20} {time_ms:6.2f} ({percentage:4.1f}%)")
            
        lines.append("-" * 30)
        lines.append(f"{'Total':20} {total_time*1000:6.2f} (100.0%)")
        lines.append(f"{'FPS':20} {self.simulation.fps:6.1f}")
        
        # Calculate position (left side of screen)
        x = int(10 * self.scale_factor)
        y = self.screen.get_height() // 2
        line_height = int(18 * self.scale_factor)
        
        # Semi-transparent background
        overlay_width = int(320 * self.scale_factor)
        overlay_height = len(lines) * line_height + int(20 * self.scale_factor)
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (x - int(5 * self.scale_factor), y - int(10 * self.scale_factor)))
        
        # Render text
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i > 1 else (255, 255, 100)  # Yellow for headers
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (x, y + i * line_height))
            
        # Also print to console periodically
        if self.simulation.step_count % 60 == 0:  # Every 60 steps
            print("\nPerformance Report:")
            for line in lines:
                print(line)
                
    def render_toolbar(self):
        """Render the toolbar on the right side."""
        # Draw toolbar background
        toolbar_rect = pygame.Rect(self.sidebar_x, 0, self.toolbar_width, self.screen.get_height())
        pygame.draw.rect(self.screen, (40, 40, 40), toolbar_rect)
        pygame.draw.line(self.screen, (60, 60, 60), (self.sidebar_x, 0), (self.sidebar_x, self.screen.get_height()), 2)
        
        # Title
        title_text = self.font.render("TOOLS", True, (200, 200, 200))
        title_rect = title_text.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, int(25 * self.scale_factor)))
        self.screen.blit(title_text, title_rect)
        
        # Tool buttons
        button_y = int(50 * self.scale_factor)
        for i, tool in enumerate(self.tools):
            # Button background
            button_rect = pygame.Rect(
                self.sidebar_x + self.button_margin,
                button_y,
                self.toolbar_width - 2 * self.button_margin,
                self.button_height
            )
            
            # Highlight current tool
            if i == self.current_tool:
                pygame.draw.rect(self.screen, (80, 80, 120), button_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), button_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), button_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), button_rect, 1)
                
            # Tool icon and name
            icon_text = self.font.render(tool["icon"], True, (255, 255, 255))
            icon_rect = icon_text.get_rect(center=(button_rect.left + int(25 * self.scale_factor), button_rect.centery))
            self.screen.blit(icon_text, icon_rect)
            
            name_text = self.toolbar_font.render(tool["name"], True, (200, 200, 200))
            name_rect = name_text.get_rect(midleft=(button_rect.left + int(45 * self.scale_factor), button_rect.centery))
            self.screen.blit(name_text, name_rect)
            
            button_y += self.button_height + self.button_margin
            
        # Materials section
        mat_title = self.small_font.render("MATERIALS", True, (200, 200, 200))
        mat_title_rect = mat_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, button_y + int(20 * self.scale_factor)))
        self.screen.blit(mat_title, mat_title_rect)
        
        # Material buttons - 2 columns to fit all materials
        mat_button_y = button_y + int(40 * self.scale_factor)
        
        # Calculate button dimensions for 2 columns
        mat_button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        mat_button_height = int(25 * self.scale_factor)
        
        mat_index = 0
        for mat_type in MaterialType:
            # Skip SPACE - not a selectable material
            if mat_type == MaterialType.SPACE:
                continue
                
            # Calculate position - 2 columns
            col = mat_index % 2
            row = mat_index // 2
            
            # Button background
            mat_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (mat_button_width + self.button_margin),
                mat_button_y + row * (mat_button_height + int(3 * self.scale_factor)),
                mat_button_width,
                mat_button_height
            )
            
            # Highlight selected material
            if mat_type == self.selected_material:
                pygame.draw.rect(self.screen, (80, 80, 120), mat_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), mat_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), mat_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), mat_rect, 1)
                
            # Material color swatch (smaller for 2-column layout)
            try:
                props = self.material_db.get_properties(mat_type)
                color_rgb = props.color_rgb
            except KeyError:
                # Fallback color if material not found in database
                color_rgb = (128, 128, 128)  # Gray
            color_rect = pygame.Rect(mat_rect.left + int(3 * self.scale_factor), mat_rect.top + int(3 * self.scale_factor), int(18 * self.scale_factor), int(18 * self.scale_factor))
            pygame.draw.rect(self.screen, color_rgb, color_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
            
            # Material name (abbreviated if needed)
            mat_name = mat_type.name
            if len(mat_name) > 8:  # Abbreviate long names
                mat_name = mat_name[:8] + "."
            mat_text = self.toolbar_font.render(mat_name, True, (180, 180, 180))
            mat_text_rect = mat_text.get_rect(midleft=(mat_rect.left + int(25 * self.scale_factor), mat_rect.centery))
            self.screen.blit(mat_text, mat_text_rect)
            
            mat_index += 1
        
        # Calculate where material buttons end
        # Count materials excluding SPACE
        num_materials = len([m for m in MaterialType if m != MaterialType.SPACE])
        num_mat_rows = (num_materials + 1) // 2
        mat_button_y = button_y + int(40 * self.scale_factor) + num_mat_rows * (mat_button_height + int(3 * self.scale_factor))
            
        # Display modes section
        display_title = self.small_font.render("DISPLAY MODE", True, (200, 200, 200))
        display_title_rect = display_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, mat_button_y + int(15 * self.scale_factor)))
        self.screen.blit(display_title, display_title_rect)
        
        # Display mode buttons - 2 columns to save space
        display_button_y = mat_button_y + int(30 * self.scale_factor)
        display_modes = list(DisplayMode)
        mode_names = {
            DisplayMode.MATERIAL_DOMINANT: "Material",
            DisplayMode.TEMPERATURE: "Temperature",
            DisplayMode.GRAVITY: "Gravity",
            DisplayMode.POWER: "Power",
        }
        
        # Calculate button dimensions for 2 columns
        button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        button_height = int(22 * self.scale_factor)
        
        for i, mode in enumerate(display_modes):
            # Calculate position - 2 columns
            col = i % 2
            row = i // 2
            
            # Button background
            mode_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (button_width + self.button_margin),
                display_button_y + row * (button_height + int(3 * self.scale_factor)),
                button_width,
                button_height
            )
            
            # Highlight selected mode
            if mode == self.display_mode:
                pygame.draw.rect(self.screen, (80, 80, 120), mode_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), mode_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), mode_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), mode_rect, 1)
                
            # Mode name
            mode_text = self.toolbar_font.render(mode_names.get(mode, mode.value), True, (180, 180, 180))
            mode_text_rect = mode_text.get_rect(center=(mode_rect.centerx, mode_rect.centery))
            self.screen.blit(mode_text, mode_text_rect)
        
        # Calculate where display modes end (for next section)
        num_rows = (len(display_modes) + 1) // 2  # Round up
        display_section_end = display_button_y + num_rows * (button_height + int(3 * self.scale_factor))
            
        # Physics toggle section at bottom
        physics_section_y = display_section_end + int(30 * self.scale_factor)  # Position after display modes with margin
        
        # Divider line
        pygame.draw.line(self.screen, (60, 60, 60), 
                         (self.sidebar_x + int(10 * self.scale_factor), physics_section_y - int(10 * self.scale_factor)),
                         (self.sidebar_x + self.toolbar_width - int(10 * self.scale_factor), physics_section_y - int(10 * self.scale_factor)), 1)
        
        # Physics title
        physics_title = self.font.render("PHYSICS MODULES", True, (200, 200, 200))
        physics_title_rect = physics_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, physics_section_y + int(10 * self.scale_factor)))
        self.screen.blit(physics_title, physics_title_rect)
        
        # Physics module checkboxes - larger and more readable
        checkbox_y = physics_section_y + int(35 * self.scale_factor)
        checkbox_size = int(20 * self.scale_factor)  # Increased from 16
        checkbox_margin = int(10 * self.scale_factor)  # Increased from 5
        
        physics_modules = [
            ("Self Gravity", self.simulation.enable_self_gravity),
            ("Heat Diff.", self.simulation.enable_heat_diffusion),
            ("Internal Heat", self.simulation.enable_internal_heating),
            ("Solar Heat", self.simulation.enable_solar_heating),
            ("Rad. Cooling", self.simulation.enable_radiative_cooling),
            ("Mat. Process", self.simulation.enable_material_processes),
            ("Weathering", self.simulation.enable_weathering),
            ("Atmos. Proc.", self.simulation.enable_atmospheric_processes),
        ]
        
        for module_name, enabled in physics_modules:
            # Draw checkbox
            checkbox_rect = pygame.Rect(
                self.sidebar_x + checkbox_margin,
                checkbox_y,
                checkbox_size,
                checkbox_size
            )
            
            # Background with better contrast
            pygame.draw.rect(self.screen, (60, 60, 60), checkbox_rect)
            pygame.draw.rect(self.screen, (120, 120, 120), checkbox_rect, 2)
            
            # Check mark if enabled - larger and clearer
            if enabled:
                # Draw a thicker checkmark
                check_points = [
                    (checkbox_rect.left + int(4 * self.scale_factor), checkbox_rect.centery),
                    (checkbox_rect.left + checkbox_size // 3 + 1, checkbox_rect.bottom - int(4 * self.scale_factor)),
                    (checkbox_rect.right - int(4 * self.scale_factor), checkbox_rect.top + int(4 * self.scale_factor))
                ]
                pygame.draw.lines(self.screen, (0, 255, 0), False, check_points, int(3 * self.scale_factor))
            
            # Module name - larger font
            module_text = self.small_font.render(module_name, True, (220, 220, 220))
            module_rect = module_text.get_rect(midleft=(checkbox_rect.right + int(8 * self.scale_factor), checkbox_rect.centery))
            self.screen.blit(module_text, module_rect)
            
            checkbox_y += checkbox_size + int(8 * self.scale_factor)  # More spacing between items
            
    def render_selected_cell(self):
        """Render selected cell highlight and information."""
        if not self.selected_cell:
            return
            
        cx, cy = self.selected_cell
        
        # Draw white highlight box
        screen_x = cx * self.scale_x + self.display_offset_x
        screen_y = cy * self.scale_y + self.display_offset_y
        highlight_rect = pygame.Rect(
            screen_x, screen_y,
            self.scale_x, self.scale_y
        )
        pygame.draw.rect(self.screen, (255, 255, 255), highlight_rect, 2)
        
        # Gather cell information
        info_lines = [
            f"Cell: ({cx}, {cy})",
            f"Position: ({cx * self.simulation.cell_size:.1f}, {cy * self.simulation.cell_size:.1f}) m",
            ""
        ]
        
        # Material type
        mat_type = self.simulation.material_types[cy, cx]
        info_lines.append(f"Material: {mat_type.name}")
        info_lines.append("")
        
        # Calculate effective density
        effective_density = self.simulation.calculate_effective_density(self.simulation.temperature)
        eff_dens = effective_density[cy, cx]
        
        # Physical properties
        info_lines.extend([
            f"Temperature: {self.simulation.temperature[cy, cx]:.1f} K ({self.simulation.temperature[cy, cx] - 273.15:.1f}°C)",
            f"Base Density: {self.simulation.density[cy, cx]:.1f} kg/m³",
            f"Eff. Density: {eff_dens:.1f} kg/m³",
        ])
        
        # Gravity and power
        if hasattr(self.simulation, 'gravity_x') and hasattr(self.simulation, 'gravity_y'):
            gx = self.simulation.gravity_x[cy, cx]
            gy = self.simulation.gravity_y[cy, cx]
            g_mag = np.sqrt(gx**2 + gy**2)
            info_lines.append(f"Gravity: {g_mag:.2f} m/s² ({gx:.2f}, {gy:.2f})")
        
        if hasattr(self.simulation, 'power_density'):
            info_lines.append(f"Power: {self.simulation.power_density[cy, cx]:.2e} W/m³")
        
        # Draw info box in lower left corner
        info_x = 10
        bg_height = len(info_lines) * 18 + 10
        info_y = self.screen.get_height() - bg_height - 10  # Position at bottom
        
        # Create semi-transparent background
        max_width = max(self.small_font.size(line)[0] for line in info_lines)
        bg_surface = pygame.Surface((max_width + 20, bg_height))
        bg_surface.set_alpha(200)
        bg_surface.fill((0, 0, 0))
        self.screen.blit(bg_surface, (info_x - 5, info_y - 5))
        
        # Draw text
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y + i * 18))
            
    def render_scale_bar(self, min_val: float, max_val: float, unit: str, title: str):
        """Render a scale bar with smart formatting."""
        # Position at bottom right of main display area
        bar_width = 200
        bar_height = 20
        margin = 20
        # Position relative to the actual display area, not the whole window
        x = self.display_offset_x + self.display_width - bar_width - margin
        y = self.display_offset_y + self.display_height - bar_height - margin - 30
        
        # Create gradient
        gradient = pygame.Surface((bar_width, bar_height))
        for i in range(bar_width):
            # Map position to value
            t = i / (bar_width - 1)
            
            # Get color from appropriate colormap
            color = self._get_scale_bar_color(title, t, min_val, max_val)
            r, g, b = color
                
            pygame.draw.line(gradient, (r, g, b), (i, 0), (i, bar_height))
            
        # Draw gradient bar
        self.screen.blit(gradient, (x, y))
        
        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, bar_width, bar_height), 1)
        
        # Title
        title_text = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(title_text, (x, y - 25))
        
        # Min/Max labels with smart formatting
        min_label = self.format_smart(min_val, unit)
        max_label = self.format_smart(max_val, unit)
        
        min_text = self.small_font.render(min_label, True, (255, 255, 255))
        max_text = self.small_font.render(max_label, True, (255, 255, 255))
        
        self.screen.blit(min_text, (x, y + bar_height + 5))
        max_rect = max_text.get_rect(right=x + bar_width, top=y + bar_height + 5)
        self.screen.blit(max_text, max_rect)
        
    def _get_scale_bar_color(self, title: str, t: float, min_val: float, max_val: float) -> tuple:
        """Get color for scale bar position based on visualization type.
        
        Args:
            title: Name of the visualization (Temperature, Pressure, etc.)
            t: Normalized position (0-1) along the scale bar
            min_val: Minimum value of the scale
            max_val: Maximum value of the scale
            
        Returns:
            (r, g, b) tuple
        """
        # Map title to colormap
        colormap_mapping = {
            "Temperature": self.temp_colors,
            "Velocity": self.velocity_colors,
            "Gravity": self.gravity_colors,
        }
        
        # Simple colormaps - direct mapping
        if title in colormap_mapping:
            idx = int(t * 255)
            color = colormap_mapping[title][idx]
            return (color[0], color[1], color[2])
        
        
        # Default grayscale
        else:
            gray = int(t * 255)
            return (gray, gray, gray)
    
    def format_smart(self, value: float, unit: str) -> str:
        """Smart formatting with appropriate precision and SI prefixes."""
        if unit == "W/m³":
            # Power density
            return self.format_power_density(value)
        elif unit == "K":
            # Temperature
            return f"{value:.0f} K"
        elif unit == "m/s²":
            # Acceleration/gravity
            if abs(value) < 0.01:
                return f"{value*1000:.1f} mm/s²"
            elif abs(value) < 1:
                return f"{value:.3f} m/s²"
            else:
                return f"{value:.1f} m/s²"
        elif unit == "m/s":
            # Velocity
            if abs(value) < 0.01:
                return f"{value*1000:.1f} mm/s"
            elif abs(value) < 1:
                return f"{value:.3f} m/s"
            else:
                return f"{value:.1f} m/s"
        else:
            # Default
            return f"{value:.3g} {unit}"
            
    def format_power_density(self, power: float) -> str:
        """Format power density with SI prefixes."""
        abs_power = abs(power)
        
        if abs_power == 0:
            return "0 W/m³"
        elif abs_power < 1e-3:
            return f"{power*1e6:.1f} µW/m³"
        elif abs_power < 1:
            return f"{power*1e3:.1f} mW/m³"
        elif abs_power < 1e3:
            return f"{power:.1f} W/m³"
        elif abs_power < 1e6:
            return f"{power/1e3:.1f} kW/m³"
        elif abs_power < 1e9:
            return f"{power/1e6:.1f} MW/m³"
        else:
            return f"{power/1e9:.1f} GW/m³"
    
    def _format_time(self, seconds: float) -> str:
        """Format time in appropriate units."""
        if seconds < 60:
            return f"{seconds:.1f} s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} min"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hr"
        elif seconds < 86400 * 365:
            days = seconds / 86400
            return f"{days:.1f} days"
        else:
            years = seconds / (86400 * 365.25)
            if years < 1000:
                return f"{years:.1f} yr"
            elif years < 1e6:
                return f"{years/1e3:.1f} kyr"
            elif years < 1e9:
                return f"{years/1e6:.1f} Myr"
            else:
                return f"{years/1e9:.1f} Gyr"
