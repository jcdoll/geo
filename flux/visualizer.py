"""
Visualization for flux-based geological simulation.

Adapted to display volume fractions and continuous fields instead of
discrete materials per cell.
"""

import pygame
import numpy as np
from typing import Optional, Tuple, Dict, Any
from enum import Enum

from simulation import FluxSimulation
from materials import MaterialType, MaterialDatabase


class DisplayMode(Enum):
    """Available visualization modes."""
    MATERIAL_DOMINANT = "material_dominant"
    MATERIAL_COMPOSITE = "material_composite"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VELOCITY = "velocity"
    GRAVITY = "gravity"
    POWER = "power"


class FluxVisualizer:
    """Interactive visualizer for flux-based simulation."""
    
    def __init__(
        self,
        simulation: FluxSimulation,
        window_width: int = 800,
        window_height: int = 900,
    ):
        """
        Initialize visualizer.
        
        Args:
            simulation: FluxSimulation instance
            window_width: Window width in pixels
            window_height: Window height in pixels
        """
        self.simulation = simulation
        self.state = simulation.state
        self.material_db = simulation.material_db
        
        # Pygame setup
        pygame.init()
        
        # Add space for toolbar on the right
        self.toolbar_width = 200
        self.screen = pygame.display.set_mode((window_width + self.toolbar_width, window_height))
        pygame.display.set_caption("Flux-Based Geological Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 14)
        self.toolbar_font = pygame.font.Font(None, 12)
        
        # Display settings
        self.display_mode = DisplayMode.MATERIAL_COMPOSITE
        self.show_info = True
        self.show_help = False
        self.show_performance = False
        self.show_debug = False
        
        # Store window dimensions
        self.window_width = window_width
        self.window_height = window_height
        
        # Grid to screen mapping - ensure square pixels
        self.grid_surface = pygame.Surface((self.state.nx, self.state.ny))
        
        # Calculate scale to maintain square aspect ratio
        # Use the smaller scale to ensure the grid fits in the window
        scale = min(window_width / self.state.nx, window_height / self.state.ny)
        self.scale_x = scale
        self.scale_y = scale
        
        # Calculate actual display dimensions and center the grid
        self.display_width = int(self.state.nx * scale)
        self.display_height = int(self.state.ny * scale)
        self.display_offset_x = (window_width - self.display_width) // 2
        self.display_offset_y = (window_height - self.display_height) // 2
        
        # Interaction state
        self.running = True
        self.mouse_down = False
        self.selected_material = MaterialType.WATER
        self.tool_radius = 5
        self.tool_intensity = 0.1
        self.selected_cell = None  # (x, y) coordinates of selected cell for inspection
        
        # Tool types
        self.tools = [
            {"name": "Material", "desc": "Add/remove materials", "icon": "M"},
            {"name": "Heat", "desc": "Add/remove heat", "icon": "H"},
            {"name": "Pressure", "desc": "Apply pressure", "icon": "P"},
            {"name": "Velocity", "desc": "Set velocity", "icon": "V"},
        ]
        self.current_tool = 0  # Index into tools list
        
        # Sidebar dimensions
        self.sidebar_x = window_width
        self.button_height = 30
        self.button_margin = 3
        
        # Color maps
        self.init_colormaps()
        
        # Colormap registry for easy lookup
        self.colormap_registry = {
            DisplayMode.TEMPERATURE: self.temp_colors,
            DisplayMode.PRESSURE: self.pressure_colors,
            DisplayMode.VELOCITY: self.velocity_colors,
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
                
        # Pressure colormap (blue negative -> black zero -> red positive)
        self.pressure_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if 120 <= i <= 136:  # Near-zero pressure band (black)
                self.pressure_colors[i] = [0, 0, 0]
            elif i < 120:  # Negative pressure (blue)
                t = i / 120.0
                self.pressure_colors[i] = [0, 0, int(255 * (1 - t))]
            else:  # Positive pressure (red)
                t = (i - 136) / 120.0
                self.pressure_colors[i] = [int(255 * t), 0, 0]
                
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
            if not self.simulation.paused:
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
            self.simulation.paused = not self.simulation.paused
            
        elif event.key == pygame.K_r:
            if hasattr(self.simulation, 'reset'):
                self.simulation.reset()  # Will use last scenario
                # CRITICAL: Update our reference to the new state object!
                self.state = self.simulation.state
                self.simulation.paused = True  # Start paused after reset
                self.selected_cell = None  # Clear selected cell
                print("Simulation reset to initial state (paused)")
                # Force immediate render to show reset state
                self.render()
                pygame.display.flip()
            
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
            pygame.image.save(self.screen, "flux_simulation_screenshot.png")
            
        elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
            # Select material
            mat_idx = event.key - pygame.K_1
            if mat_idx < len(MaterialType):
                self.selected_material = list(MaterialType)[mat_idx]
                
        elif event.key == pygame.K_t:
            # Cycle through tools
            self.current_tool = (self.current_tool + 1) % len(self.tools)
                
        elif event.key == pygame.K_LEFT:
            # Step backward (requires state history)
            print("Step backward not implemented - requires state history")
            
        elif event.key == pygame.K_RIGHT:
            # Step forward one frame
            if self.simulation.paused:
                # Temporarily unpause to allow step
                self.simulation.paused = False
                self.simulation.step_forward()
                self.simulation.paused = True
                
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
        if 0 <= grid_x < self.state.nx and 0 <= grid_y < self.state.ny:
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
        button_y = 50
        for i, tool in enumerate(self.tools):
            if (x >= self.sidebar_x + self.button_margin and 
                x <= self.sidebar_x + self.toolbar_width - self.button_margin and
                y >= button_y and y <= button_y + self.button_height):
                self.current_tool = i
                return
            button_y += self.button_height + self.button_margin
            
        # Check material buttons - they start after the tools
        mat_button_y = button_y + 40  # Same as in render_toolbar
        for i, mat_type in enumerate(MaterialType):
            if (x >= self.sidebar_x + self.button_margin and
                x <= self.sidebar_x + self.toolbar_width - self.button_margin and
                y >= mat_button_y and y <= mat_button_y + 25):
                self.selected_material = mat_type
                return
            mat_button_y += 28
            
        # Check display mode buttons - 2 columns
        display_button_y = mat_button_y + 30  # After materials section
        display_modes = list(DisplayMode)
        button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        button_height = 22
        
        for i, mode in enumerate(display_modes):
            col = i % 2
            row = i // 2
            
            mode_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (button_width + self.button_margin),
                display_button_y + row * (button_height + 3),
                button_width,
                button_height
            )
            
            if mode_rect.collidepoint(x, y):
                self.display_mode = mode
                return
        
        # Calculate where display modes end
        num_rows = (len(display_modes) + 1) // 2
        display_section_end = display_button_y + num_rows * (button_height + 3)
            
        # Check physics module checkboxes - after display modes
        physics_section_y = display_section_end + 30  # Match render_toolbar
        checkbox_y = physics_section_y + 35
        checkbox_size = 20  # Matches render_toolbar
        checkbox_margin = 10
        
        physics_modules = [
            ("enable_gravity", self.simulation.enable_gravity),
            ("enable_momentum", self.simulation.enable_momentum),
            ("enable_advection", self.simulation.enable_advection),
            ("enable_heat_transfer", self.simulation.enable_heat_transfer),
            ("enable_uranium_heating", self.simulation.enable_uranium_heating),
            ("enable_solar_heating", self.simulation.enable_solar_heating),
            ("enable_phase_transitions", self.simulation.enable_phase_transitions),
            ("enable_atmospheric", self.simulation.enable_atmospheric),
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
                checkbox_rect.right + 8,
                checkbox_rect.top,
                self.toolbar_width - checkbox_margin - checkbox_size - 8,
                checkbox_size
            )
            
            if checkbox_rect.collidepoint(x, y) or label_rect.collidepoint(x, y):
                # Toggle the physics module
                setattr(self.simulation, attr_name, not current_value)
                return
                
            checkbox_y += checkbox_size + 8
    
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
        
        if 0 <= gx < self.state.nx and 0 <= gy < self.state.ny:
            # Create circular mask
            y_grid, x_grid = np.ogrid[:self.state.ny, :self.state.nx]
            dist = np.sqrt((x_grid - gx)**2 + (y_grid - gy)**2)
            mask = dist < self.tool_radius
            
            tool = self.tools[self.current_tool]
            
            if tool["name"] == "Material":
                if button == 1:  # Left click - add material
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        # Remove material (make space)
                        self.state.vol_frac[:, mask] = 0.0
                        self.state.vol_frac[MaterialType.SPACE.value, mask] = 1.0
                        
                        # Zero out velocities in removed regions
                        self.state.velocity_x[mask] = 0.0
                        self.state.velocity_y[mask] = 0.0
                    else:
                        # Add selected material
                        # First, reduce other materials proportionally
                        total = self.state.vol_frac.sum(axis=0)
                        for mat in range(self.state.n_materials):
                            if mat != self.selected_material.value:
                                self.state.vol_frac[mat, mask] *= (1 - self.tool_intensity)
                                
                        # Add selected material
                        self.state.vol_frac[self.selected_material.value, mask] += self.tool_intensity
                        
                        # Normalize
                        self.state.normalize_volume_fractions()
                        
                        # For cells that were previously empty (space), inherit properties from neighbors
                        space_mask = mask & (self.state.density < 1.0)  # Was mostly empty
                        if np.any(space_mask):
                            # Get average properties from non-empty neighbors
                            for j in range(self.state.ny):
                                for i in range(self.state.nx):
                                    if space_mask[j, i]:
                                        # Find non-empty neighbors
                                        neighbors = []
                                        for dj, di in [(-1,0), (1,0), (0,-1), (0,1)]:
                                            nj, ni = j + dj, i + di
                                            if (0 <= nj < self.state.ny and 0 <= ni < self.state.nx and
                                                self.state.density[nj, ni] > 10.0):  # Non-empty
                                                neighbors.append((nj, ni))
                                        
                                        if neighbors:
                                            # Average neighbor properties
                                            avg_temp = np.mean([self.state.temperature[nj, ni] for nj, ni in neighbors])
                                            avg_pressure = np.mean([self.state.pressure[nj, ni] for nj, ni in neighbors])
                                            avg_vx = np.mean([self.state.velocity_x[nj, ni] for nj, ni in neighbors])
                                            avg_vy = np.mean([self.state.velocity_y[nj, ni] for nj, ni in neighbors])
                                            
                                            self.state.temperature[j, i] = avg_temp
                                            self.state.pressure[j, i] = avg_pressure
                                            self.state.velocity_x[j, i] = avg_vx
                                            self.state.velocity_y[j, i] = avg_vy
                        
                    # Update properties
                    self.state.update_mixture_properties(self.material_db)
                    
            elif tool["name"] == "Heat":
                # Apply heat source/sink
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    # Cool down
                    self.state.temperature[mask] *= 0.95
                else:
                    # Heat up
                    self.state.temperature[mask] += 50.0
                    
            elif tool["name"] == "Pressure":
                # Apply pressure
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    # Negative pressure
                    self.state.pressure[mask] -= 1000.0
                else:
                    # Positive pressure
                    self.state.pressure[mask] += 1000.0
                    
            elif tool["name"] == "Velocity":
                # Set velocity toward/away from click point
                for j in range(self.state.ny):
                    for i in range(self.state.nx):
                        if mask[j, i]:
                            dx = i - gx
                            dy = j - gy
                            dist_sq = dx*dx + dy*dy + 1e-10
                            if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                                # Pull inward
                                self.state.velocity_x[j, i] = -dx / np.sqrt(dist_sq) * 10.0
                                self.state.velocity_y[j, i] = -dy / np.sqrt(dist_sq) * 10.0
                            else:
                                # Push outward
                                self.state.velocity_x[j, i] = dx / np.sqrt(dist_sq) * 10.0
                                self.state.velocity_y[j, i] = dy / np.sqrt(dist_sq) * 10.0
                
    def render(self):
        """Render the current simulation state."""
        self.screen.fill((0, 0, 0))
        
        # Render based on display mode
        if self.display_mode == DisplayMode.MATERIAL_DOMINANT:
            self.render_material_dominant()
        elif self.display_mode == DisplayMode.MATERIAL_COMPOSITE:
            self.render_material_composite()
        elif self.display_mode == DisplayMode.TEMPERATURE:
            self.render_temperature()
        elif self.display_mode == DisplayMode.PRESSURE:
            self.render_pressure()
        elif self.display_mode == DisplayMode.VELOCITY:
            self.render_velocity()
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
        margin = 40
        radius = 25
        center_x = self.display_offset_x + self.display_width - margin - radius
        center_y = self.display_offset_y + margin + radius
        
        # Get sun angle from simulation
        sun_angle = self.simulation.solar_angle
        
        # Draw thin circle showing planet's orbital path
        pygame.draw.circle(self.screen, (150, 150, 150), (center_x, center_y), radius, 1)
        
        # Draw sun in center (yellow)
        pygame.draw.circle(self.screen, (255, 255, 0), (center_x, center_y), 8)
        pygame.draw.circle(self.screen, (255, 200, 0), (center_x, center_y), 8, 2)
        
        # Calculate planet position on orbit
        # The sun angle tells us where the sun appears from the planet's perspective
        # If sun_angle = 0 (sun overhead), planet is at bottom of orbit
        # If sun_angle = π/2 (sun from east), planet is at west of orbit
        # If sun_angle = -π/2 (sun from west), planet is at east of orbit
        # Planet angle is opposite to sun angle
        planet_angle = sun_angle + np.pi
        planet_x = center_x + radius * np.sin(planet_angle)
        planet_y = center_y + radius * np.cos(planet_angle)
        
        # Draw planet (dark circle)
        pygame.draw.circle(self.screen, (50, 50, 50), (int(planet_x), int(planet_y)), 5)
        pygame.draw.circle(self.screen, (100, 100, 100), (int(planet_x), int(planet_y)), 5, 1)
        
        # Draw a small arrow showing sunlight direction on the planet
        # Arrow points from sun towards planet
        arrow_len = 12
        arrow_dir_x = planet_x - center_x
        arrow_dir_y = planet_y - center_y
        arrow_norm = np.sqrt(arrow_dir_x**2 + arrow_dir_y**2)
        if arrow_norm > 0:
            arrow_dir_x /= arrow_norm
            arrow_dir_y /= arrow_norm
            
            # Arrow start and end
            arrow_start_x = planet_x - arrow_dir_x * (5 + arrow_len)
            arrow_start_y = planet_y - arrow_dir_y * (5 + arrow_len)
            arrow_end_x = planet_x - arrow_dir_x * 5
            arrow_end_y = planet_y - arrow_dir_y * 5
            
            # Draw arrow
            pygame.draw.line(self.screen, (255, 200, 0), 
                           (int(arrow_start_x), int(arrow_start_y)),
                           (int(arrow_end_x), int(arrow_end_y)), 2)
        
        # Draw labels
        sun_text = self.small_font.render("SUN", True, (255, 255, 255))
        text_rect = sun_text.get_rect(center=(center_x, center_y + radius + 20))
        self.screen.blit(sun_text, text_rect)
        
        # Optional: Show angle in degrees for debugging
        if self.show_debug:
            angle_deg = (sun_angle * 180 / np.pi) % 360
            angle_text = self.small_font.render(f"{angle_deg:.0f}°", True, (200, 200, 200))
            angle_rect = angle_text.get_rect(center=(center_x, center_y + radius + 35))
            self.screen.blit(angle_text, angle_rect)
        
    def _render_field(self, rgb: np.ndarray, min_val: float, max_val: float, unit: str, title: str):
        """Common field rendering logic.
        
        Args:
            rgb: RGB array with shape (ny, nx, 3)
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
        """Render dominant material per cell."""
        # Find material with highest volume fraction at each cell
        dominant = np.argmax(self.state.vol_frac, axis=0)
        
        # Create RGB image
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.uint8)
        
        for mat_idx, mat_type in enumerate(MaterialType):
            props = self.material_db.get_properties(mat_type)
            mask = dominant == mat_idx
            rgb[mask] = props.color_rgb
            
        # Convert to surface and scale
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface, 
            (self.display_width, self.display_height)
        )
        self.screen.blit(scaled, (self.display_offset_x, self.display_offset_y))
        
    def render_material_composite(self):
        """Render composite material colors based on volume fractions."""
        # Compute weighted average of material colors
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.float32)
        
        for mat_idx, mat_type in enumerate(MaterialType):
            props = self.material_db.get_properties(mat_type)
            color = np.array(props.color_rgb, dtype=np.float32)
            
            # Add weighted contribution
            for c in range(3):
                rgb[:, :, c] += self.state.vol_frac[mat_idx] * color[c]
                
        # Convert to uint8 and render
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (self.display_width, self.display_height)
        )
        self.screen.blit(scaled, (self.display_offset_x, self.display_offset_y))
        
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
        
    def render_pressure(self):
        """Render pressure field."""
        P = self.state.pressure
        P_min, P_max = np.min(P), np.max(P)
        P_range = max(abs(P_min), abs(P_max))
        
        # Normalize for symmetric colormap
        if P_range > 0:
            P_norm = np.clip((P + P_range) / (2 * P_range) * 255, 0, 255).astype(np.uint8)
        else:
            P_norm = np.ones_like(P, dtype=np.uint8) * 128
        
        rgb = self.pressure_colors[P_norm]
        
        # Common rendering
        self._render_field(rgb, P_min, P_max, "Pa", "Pressure")
        
    def render_velocity(self):
        """Render velocity magnitude."""
        v_mag = np.sqrt(self.state.velocity_x**2 + self.state.velocity_y**2)
        v_min = 0
        v_max = np.max(v_mag)
        
        # Normalize
        if v_max > 0:
            v_norm = np.clip(v_mag / v_max * 255, 0, 255).astype(np.uint8)
        else:
            v_norm = np.zeros_like(v_mag, dtype=np.uint8)
        
        rgb = self.velocity_colors[v_norm]
        
        # Common rendering
        self._render_field(rgb, v_min, v_max, "m/s", "Velocity")
        
    def render_gravity(self):
        """Render gravitational field magnitude."""
        # Get gravity field from state or physics solver
        if hasattr(self.state, 'gravity_x') and hasattr(self.state, 'gravity_y'):
            gx = self.state.gravity_x
            gy = self.state.gravity_y
        elif hasattr(self.simulation, 'gravity_solver'):
            # Force gravity calculation if not done
            self.simulation.gravity_solver.solve_gravity()
            gx = self.state.gravity_x
            gy = self.state.gravity_y
        else:
            # Default gravity pointing down
            gx = np.zeros((self.state.ny, self.state.nx))
            gy = np.ones((self.state.ny, self.state.nx)) * 9.81
            
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
            uranium_idx = MaterialType.URANIUM.value
            if uranium_idx < self.state.n_materials:
                uranium_props = self.material_db.get_properties(MaterialType.URANIUM)
                if hasattr(uranium_props, 'heat_generation') and uranium_props.heat_generation > 0:
                    # Heat generation is in W/kg, multiply by density and volume fraction
                    uranium_mask = self.state.vol_frac[uranium_idx] > 0
                    if np.any(uranium_mask):
                        power[uranium_mask] = (self.state.vol_frac[uranium_idx][uranium_mask] * 
                                             self.state.density[uranium_mask] * 
                                             uranium_props.heat_generation)
        
        # Find the maximum absolute value for symmetric scaling
        max_abs_power = np.max(np.abs(power))
        
        # Create RGB array
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.uint8)
        
        if max_abs_power > 1e-10:
            # Use symmetric logarithmic scaling for better visualization
            # This handles both positive and negative values
            
            # Define a linear threshold below which we use linear scaling
            linear_threshold = max_abs_power * 0.001  # 0.1% of max
            
            # Create a symmetric log-like scaling
            def symlog_scale(values, threshold):
                """Symmetric log scaling that handles positive and negative values."""
                scaled = np.zeros_like(values)
                
                # Positive values
                pos_mask = values > threshold
                if np.any(pos_mask):
                    scaled[pos_mask] = 1 + np.log10(values[pos_mask] / threshold)
                    
                # Negative values
                neg_mask = values < -threshold
                if np.any(neg_mask):
                    scaled[neg_mask] = -1 - np.log10(-values[neg_mask] / threshold)
                    
                # Linear region near zero
                linear_mask = np.abs(values) <= threshold
                if np.any(linear_mask):
                    scaled[linear_mask] = values[linear_mask] / threshold
                    
                return scaled
            
            # Apply symmetric log scaling
            power_scaled = symlog_scale(power, linear_threshold)
            
            # Find scale limits
            scale_max = np.max(np.abs(power_scaled))
            if scale_max > 0:
                # Normalize to [-1, 1]
                power_norm = power_scaled / scale_max
            else:
                power_norm = np.zeros_like(power)
                
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
            self.power_scale_min = -max_abs_power
            self.power_scale_max = max_abs_power
            self.power_scale_type = "symmetric_log"
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
        bar_width = 200
        bar_height = 20
        margin = 20
        bar_x = self.screen.get_width() - self.toolbar_width - bar_width - margin
        bar_y = self.screen.get_height() - bar_height - margin - 30
        
        if not hasattr(self, 'power_scale_type') or self.power_scale_type == "zero":
            # No power to display
            text = self.font.render("Power: 0 W/m³", True, (255, 255, 255))
            self.screen.blit(text, (bar_x, bar_y - 25))
            return
        
        # Draw gradient bar with proper blue-black-red colormap
        for i in range(bar_width):
            # Map position to normalized value [-1, 1]
            norm_value = 2.0 * i / bar_width - 1.0
            
            # Define threshold for black center region
            zero_threshold = 0.05
            
            if norm_value > zero_threshold:  # Heating side (black to red)
                # Normalize from threshold to 1
                t = (norm_value - zero_threshold) / (1.0 - zero_threshold)
                r = int(255 * t)  # Red increases
                g = 0  # Green stays at 0
                b = 0  # Blue stays at 0
            elif norm_value < -zero_threshold:  # Cooling side (black to blue)
                # Normalize from -threshold to -1
                t = (abs(norm_value) - zero_threshold) / (1.0 - zero_threshold)
                r = 0  # Red stays at 0
                g = 0  # Green stays at 0
                b = int(255 * t)  # Blue increases
            else:  # Center region (black)
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
        self.screen.blit(title, (bar_x, bar_y - 25))
        
        # Min (cooling) label
        min_text = format_power(self.power_scale_min)
        min_label = self.small_font.render(min_text, True, (150, 150, 255))
        self.screen.blit(min_label, (bar_x, bar_y + bar_height + 2))
        
        # Center (zero) label
        center_label = self.small_font.render("0", True, (255, 255, 255))
        center_x = bar_x + bar_width // 2 - center_label.get_width() // 2
        self.screen.blit(center_label, (center_x, bar_y + bar_height + 2))
        
        # Max (heating) label
        max_text = format_power(self.power_scale_max)
        max_label = self.small_font.render(max_text, True, (255, 150, 150))
        max_x = bar_x + bar_width - max_label.get_width()
        self.screen.blit(max_label, (max_x, bar_y + bar_height + 2))
        
    def render_info(self):
        """Render information overlay."""
        info = self.simulation.get_info()
        
        y = 10
        lines = [
            f"FPS: {info['fps']:.1f}",
            f"Time: {info['time']:.1f} s",
            f"Step: {info['step_count']}",
            f"Mode: {self.display_mode.value}",
            f"Tool: {self.tools[self.current_tool]['name']}",
            f"Material: {self.selected_material.name}",
            f"Radius: {self.tool_radius}",
            f"Paused: {self.simulation.paused}",
            "",
            f"Avg T: {info['avg_temperature']:.1f} K",
            f"Mass: {info['total_mass']:.2e} kg",
        ]
        
        for line in lines:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 20
            
    def render_help(self):
        """Render help overlay."""
        help_text = [
            "Controls:",
            "SPACE - Pause/Resume",
            "LEFT - Step backward (not implemented)",
            "RIGHT - Step forward (when paused)",
            "R - Reset simulation (paused)",
            "TAB - Cycle display mode",
            "M - Cycle display mode", 
            "T - Cycle tools",
            "1-9 - Select material",
            "Left Click - Apply tool",
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
        overlay = pygame.Surface((300, len(help_text) * 25 + 20))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (self.screen.get_width() - 310, 10))
        
        # Render text
        y = 20
        for line in help_text:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.screen.get_width() - 300, y))
            y += 25
            
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
        x = 10
        y = self.screen.get_height() // 2
        line_height = 18
        
        # Semi-transparent background
        overlay_width = 320
        overlay_height = len(lines) * line_height + 20
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (x - 5, y - 10))
        
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
        title_rect = title_text.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, 25))
        self.screen.blit(title_text, title_rect)
        
        # Tool buttons
        button_y = 50
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
            icon_rect = icon_text.get_rect(center=(button_rect.left + 25, button_rect.centery))
            self.screen.blit(icon_text, icon_rect)
            
            name_text = self.toolbar_font.render(tool["name"], True, (200, 200, 200))
            name_rect = name_text.get_rect(midleft=(button_rect.left + 45, button_rect.centery))
            self.screen.blit(name_text, name_rect)
            
            button_y += self.button_height + self.button_margin
            
        # Materials section
        mat_title = self.small_font.render("MATERIALS", True, (200, 200, 200))
        mat_title_rect = mat_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, button_y + 20))
        self.screen.blit(mat_title, mat_title_rect)
        
        # Material buttons
        mat_button_y = button_y + 40
        for i, mat_type in enumerate(MaterialType):
            # Button background
            mat_rect = pygame.Rect(
                self.sidebar_x + self.button_margin,
                mat_button_y,
                self.toolbar_width - 2 * self.button_margin,
                25
            )
            
            # Highlight selected material
            if mat_type == self.selected_material:
                pygame.draw.rect(self.screen, (80, 80, 120), mat_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), mat_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), mat_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), mat_rect, 1)
                
            # Material color swatch
            props = self.material_db.get_properties(mat_type)
            color_rect = pygame.Rect(mat_rect.left + 5, mat_rect.top + 5, 20, 20)
            pygame.draw.rect(self.screen, props.color_rgb, color_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
            
            # Material name
            mat_text = self.toolbar_font.render(mat_type.name, True, (180, 180, 180))
            mat_text_rect = mat_text.get_rect(midleft=(mat_rect.left + 30, mat_rect.centery))
            self.screen.blit(mat_text, mat_text_rect)
            
            mat_button_y += 28
            
        # Display modes section
        display_title = self.small_font.render("DISPLAY MODE", True, (200, 200, 200))
        display_title_rect = display_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, mat_button_y + 15))
        self.screen.blit(display_title, display_title_rect)
        
        # Display mode buttons - 2 columns to save space
        display_button_y = mat_button_y + 30
        display_modes = list(DisplayMode)
        mode_names = {
            DisplayMode.MATERIAL_DOMINANT: "Material",
            DisplayMode.MATERIAL_COMPOSITE: "Composite",
            DisplayMode.TEMPERATURE: "Temperature",
            DisplayMode.PRESSURE: "Pressure",
            DisplayMode.VELOCITY: "Velocity",
            DisplayMode.GRAVITY: "Gravity",
            DisplayMode.POWER: "Power",
        }
        
        # Calculate button dimensions for 2 columns
        button_width = (self.toolbar_width - 3 * self.button_margin) // 2
        button_height = 22
        
        for i, mode in enumerate(display_modes):
            # Calculate position - 2 columns
            col = i % 2
            row = i // 2
            
            # Button background
            mode_rect = pygame.Rect(
                self.sidebar_x + self.button_margin + col * (button_width + self.button_margin),
                display_button_y + row * (button_height + 3),
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
        display_section_end = display_button_y + num_rows * (button_height + 3)
            
        # Physics toggle section at bottom
        physics_section_y = display_section_end + 30  # Position after display modes with margin
        
        # Divider line
        pygame.draw.line(self.screen, (60, 60, 60), 
                         (self.sidebar_x + 10, physics_section_y - 10),
                         (self.sidebar_x + self.toolbar_width - 10, physics_section_y - 10), 1)
        
        # Physics title
        physics_title = self.font.render("PHYSICS MODULES", True, (200, 200, 200))
        physics_title_rect = physics_title.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, physics_section_y + 10))
        self.screen.blit(physics_title, physics_title_rect)
        
        # Physics module checkboxes - larger and more readable
        checkbox_y = physics_section_y + 35
        checkbox_size = 20  # Increased from 16
        checkbox_margin = 10  # Increased from 5
        
        physics_modules = [
            ("Gravity", self.simulation.enable_gravity),
            ("Momentum", self.simulation.enable_momentum),
            ("Advection", self.simulation.enable_advection),
            ("Heat Transfer", self.simulation.enable_heat_transfer),
            ("Uranium Heat", self.simulation.enable_uranium_heating),
            ("Solar Heating", self.simulation.enable_solar_heating),
            ("Phase Trans.", self.simulation.enable_phase_transitions),
            ("Atmospheric", self.simulation.enable_atmospheric),
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
                    (checkbox_rect.left + 4, checkbox_rect.centery),
                    (checkbox_rect.left + checkbox_size // 3 + 1, checkbox_rect.bottom - 4),
                    (checkbox_rect.right - 4, checkbox_rect.top + 4)
                ]
                pygame.draw.lines(self.screen, (0, 255, 0), False, check_points, 3)
            
            # Module name - larger font
            module_text = self.small_font.render(module_name, True, (220, 220, 220))
            module_rect = module_text.get_rect(midleft=(checkbox_rect.right + 8, checkbox_rect.centery))
            self.screen.blit(module_text, module_rect)
            
            checkbox_y += checkbox_size + 8  # More spacing between items
            
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
            f"Position: ({cx * self.state.dx:.1f}, {cy * self.state.dx:.1f}) m",
            ""
        ]
        
        # Material fractions
        for mat_idx, mat_type in enumerate(MaterialType):
            frac = self.state.vol_frac[mat_idx, cy, cx]
            if frac > 0.01:  # Only show materials with > 1%
                info_lines.append(f"{mat_type.name}: {frac*100:.1f}%")
        
        info_lines.append("")
        
        # Physical properties
        info_lines.extend([
            f"Temperature: {self.state.temperature[cy, cx]:.1f} K ({self.state.temperature[cy, cx] - 273.15:.1f}°C)",
            f"Pressure: {self.state.pressure[cy, cx]/1e5:.2f} bar",
            f"Density: {self.state.density[cy, cx]:.1f} kg/m³",
            f"Velocity: ({self.state.velocity_x[cy, cx]:.2f}, {self.state.velocity_y[cy, cx]:.2f}) m/s",
        ])
        
        # Gravity and power
        if hasattr(self.state, 'gravity_x') and hasattr(self.state, 'gravity_y'):
            gx = self.state.gravity_x[cy, cx]
            gy = self.state.gravity_y[cy, cx]
            g_mag = np.sqrt(gx**2 + gy**2)
            info_lines.append(f"Gravity: {g_mag:.2f} m/s² ({gx:.2f}, {gy:.2f})")
        
        if hasattr(self.state, 'power_density'):
            info_lines.append(f"Power: {self.state.power_density[cy, cx]:.2e} W/m³")
        
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
        
        # Pressure needs special handling for negative/positive ranges
        elif title == "Pressure":
            if min_val < 0 and max_val > 0:
                # Full range - use entire colormap
                idx = int(t * 255)
            elif max_val <= 0:
                # All negative - use blue portion (0-120)
                idx = int(t * 120)
            else:
                # All positive - use red portion (136-255)
                idx = int(136 + t * 119)
            
            color = self.pressure_colors[idx]
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
        elif unit == "Pa":
            # Pressure
            return self.format_pressure(value)
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
            
    def format_pressure(self, pressure: float) -> str:
        """Format pressure with appropriate units."""
        abs_pressure = abs(pressure)
        
        if abs_pressure < 1000:
            return f"{pressure:.0f} Pa"
        elif abs_pressure < 1e5:
            return f"{pressure/1000:.1f} kPa"
        elif abs_pressure < 1e6:
            return f"{pressure/1e5:.2f} bar"
        else:
            return f"{pressure/1e6:.1f} MPa"