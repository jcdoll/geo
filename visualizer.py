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
    WATER_FRACTION = "water_fraction"


class FluxVisualizer:
    """Interactive visualizer for flux-based simulation."""
    
    def __init__(
        self,
        simulation: FluxSimulation,
        window_width: int = 800,
        window_height: int = 800,
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
        
        # Grid to screen mapping
        self.grid_surface = pygame.Surface((self.state.nx, self.state.ny))
        self.scale_x = window_width / self.state.nx
        self.scale_y = window_height / self.state.ny
        
        # Interaction state
        self.running = True
        self.mouse_down = False
        self.selected_material = MaterialType.WATER
        self.tool_radius = 5
        self.tool_intensity = 0.1
        
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
        
        # Custom event handler for test runner
        self.custom_event_handler = None
        
    def init_colormaps(self):
        """Initialize color maps for different display modes."""
        # Temperature colormap (blue -> cyan -> green -> yellow -> red)
        self.temp_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 64:
                # Blue to cyan
                self.temp_colors[i] = [0, i * 4, 255]
            elif i < 128:
                # Cyan to green
                self.temp_colors[i] = [0, 255, 255 - (i - 64) * 4]
            elif i < 192:
                # Green to yellow
                self.temp_colors[i] = [(i - 128) * 4, 255, 0]
            else:
                # Yellow to red
                self.temp_colors[i] = [255, 255 - (i - 192) * 4, 0]
                
        # Pressure colormap (purple to cyan gradient)
        self.pressure_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            self.pressure_colors[i] = [128 - i//2, i, 128 + i//2]
            
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
                self.simulation.paused = True  # Start paused after reset
                print("Simulation reset to initial state (paused)")
            
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
                self.simulation.step_forward()
                
        elif event.key == pygame.K_ESCAPE:
            self.running = False
            
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
            
        # Check display mode buttons
        display_button_y = mat_button_y + 30  # After materials section
        display_modes = list(DisplayMode)
        for i, mode in enumerate(display_modes):
            if (x >= self.sidebar_x + self.button_margin and
                x <= self.sidebar_x + self.toolbar_width - self.button_margin and
                y >= display_button_y and y <= display_button_y + 22):
                self.display_mode = mode
                return
            display_button_y += 25
    
    def apply_tool(self, mouse_pos: Tuple[int, int], button: int):
        """Apply interactive tool at mouse position."""
        # Convert mouse to grid coordinates
        gx = int(mouse_pos[0] / self.scale_x)
        gy = int(mouse_pos[1] / self.scale_y)
        
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
        elif self.display_mode == DisplayMode.WATER_FRACTION:
            self.render_water_fraction()
            
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
            
        pygame.display.flip()
        
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
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
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
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_temperature(self):
        """Render temperature field."""
        # Normalize temperature to 0-255 range
        T = self.state.temperature
        T_min, T_max = 0.0, 2000.0  # K
        T_norm = np.clip((T - T_min) / (T_max - T_min) * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        rgb = self.temp_colors[T_norm]
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_pressure(self):
        """Render pressure field."""
        # Normalize pressure
        P = self.state.pressure
        P_max = np.max(np.abs(P)) + 1e-10
        P_norm = np.clip(np.abs(P) / P_max * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        rgb = self.pressure_colors[P_norm]
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_velocity(self):
        """Render velocity magnitude."""
        # Compute velocity magnitude
        v_mag = np.sqrt(self.state.velocity_x**2 + self.state.velocity_y**2)
        v_max = np.max(v_mag) + 1e-10
        v_norm = np.clip(v_mag / v_max * 255, 0, 255).astype(np.uint8)
        
        # Simple grayscale for now
        rgb = np.stack([v_norm, v_norm, v_norm], axis=2)
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_water_fraction(self):
        """Render water volume fraction."""
        water_frac = self.state.vol_frac[MaterialType.WATER.value]
        water_norm = np.clip(water_frac * 255, 0, 255).astype(np.uint8)
        
        # Blue channel for water
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.uint8)
        rgb[:, :, 2] = water_norm
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_gravity(self):
        """Render gravitational field magnitude."""
        # Get gravity field from state or physics solver
        if hasattr(self.state, 'gravity_x') and hasattr(self.state, 'gravity_y'):
            gx = self.state.gravity_x
            gy = self.state.gravity_y
            g_mag = np.sqrt(gx**2 + gy**2)
        elif hasattr(self.simulation.physics, 'gx') and hasattr(self.simulation.physics, 'gy'):
            gx = self.simulation.physics.gx
            gy = self.simulation.physics.gy
            g_mag = np.sqrt(gx**2 + gy**2)
        else:
            # Default gravity pointing down
            g_mag = np.ones((self.state.ny, self.state.nx)) * 9.81
            
        # Normalize to 0-255
        g_max = np.max(g_mag) + 1e-10
        g_norm = np.clip(g_mag / g_max * 255, 0, 255).astype(np.uint8)
        
        # Purple gradient for gravity
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.uint8)
        rgb[:, :, 0] = g_norm // 2  # Red channel
        rgb[:, :, 1] = 0             # Green channel
        rgb[:, :, 2] = g_norm        # Blue channel
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
    def render_power(self):
        """Render power generation/dissipation."""
        # Calculate power from various sources
        power = np.zeros((self.state.ny, self.state.nx), dtype=np.float32)
        
        # Nuclear power from uranium
        uranium_frac = self.state.vol_frac[MaterialType.URANIUM.value]
        uranium_power = uranium_frac * 1e6  # W/m³ (simplified)
        power += uranium_power
        
        # Could add other power sources here (e.g., friction, compression)
        
        # Normalize to 0-255
        power_max = np.max(np.abs(power)) + 1e-10
        power_norm = np.clip(np.abs(power) / power_max * 255, 0, 255).astype(np.uint8)
        
        # Yellow-orange gradient for power
        rgb = np.zeros((self.state.ny, self.state.nx, 3), dtype=np.uint8)
        rgb[:, :, 0] = power_norm           # Red channel
        rgb[:, :, 1] = power_norm // 2      # Green channel
        rgb[:, :, 2] = 0                    # Blue channel
        
        # Render
        pygame.surfarray.blit_array(self.grid_surface, rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            self.grid_surface,
            (int(self.state.nx * self.scale_x), int(self.state.ny * self.scale_y))
        )
        self.screen.blit(scaled, (0, 0))
        
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
        
        # Display mode buttons
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
            DisplayMode.WATER_FRACTION: "Water",
        }
        
        for mode in display_modes:
            # Button background
            mode_rect = pygame.Rect(
                self.sidebar_x + self.button_margin,
                display_button_y,
                self.toolbar_width - 2 * self.button_margin,
                22
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
            
            display_button_y += 25
            
        # Info section at bottom
        info_y = self.screen.get_height() - 120
        
        # Divider line
        pygame.draw.line(self.screen, (60, 60, 60), 
                         (self.sidebar_x + 10, info_y - 10),
                         (self.sidebar_x + self.toolbar_width - 10, info_y - 10), 1)
        
        # Simulation info
        info_lines = [
            f"Grid: {self.state.nx}x{self.state.ny}",
            f"Cell Size: {self.state.dx}m",
            f"Time: {self.state.time:.1f}s",
            f"Step: {self.simulation.step_count}",
            f"FPS: {self.simulation.fps:.1f}",
        ]
        
        for i, line in enumerate(info_lines):
            text = self.toolbar_font.render(line, True, (160, 160, 160))
            self.screen.blit(text, (self.sidebar_x + 10, info_y + i * 20))