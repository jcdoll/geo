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
        window_height: int = 600,
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
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Flux-Based Geological Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 16)
        
        # Display settings
        self.display_mode = DisplayMode.MATERIAL_COMPOSITE
        self.show_info = True
        self.show_help = False
        
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
        
        # Color maps
        self.init_colormaps()
        
    def init_colormaps(self):
        """Initialize color maps for different display modes."""
        # Temperature colormap (blue to red)
        self.temp_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 128:
                self.temp_colors[i] = [0, 0, 255 * (1 - i/128)]  # Blue to black
            else:
                self.temp_colors[i] = [255 * (i-128)/128, 0, 0]  # Black to red
                
        # Pressure colormap (green gradient)
        self.pressure_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            self.pressure_colors[i] = [0, i, 0]
            
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
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
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
            self.simulation.reset()
            
        elif event.key == pygame.K_m:
            # Cycle display mode
            modes = list(DisplayMode)
            idx = modes.index(self.display_mode)
            self.display_mode = modes[(idx + 1) % len(modes)]
            
        elif event.key == pygame.K_h:
            self.show_help = not self.show_help
            
        elif event.key == pygame.K_i:
            self.show_info = not self.show_info
            
        elif event.key == pygame.K_s:
            # Save screenshot
            pygame.image.save(self.screen, "flux_simulation_screenshot.png")
            
        elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
            # Select material
            mat_idx = event.key - pygame.K_1
            if mat_idx < len(MaterialType):
                self.selected_material = list(MaterialType)[mat_idx]
                
        elif event.key == pygame.K_ESCAPE:
            self.running = False
            
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
            
            if button == 1:  # Left click - add material
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    # Remove material (make space)
                    self.state.vol_frac[:, mask] = 0.0
                    self.state.vol_frac[MaterialType.SPACE, mask] = 1.0
                else:
                    # Add selected material
                    # First, reduce other materials proportionally
                    total = self.state.vol_frac.sum(axis=0)
                    for mat in range(self.state.n_materials):
                        if mat != self.selected_material:
                            self.state.vol_frac[mat, mask] *= (1 - self.tool_intensity)
                            
                    # Add selected material
                    self.state.vol_frac[self.selected_material, mask] += self.tool_intensity
                    
                    # Normalize
                    self.state.normalize_volume_fractions()
                    
                # Update properties
                self.state.update_mixture_properties(self.material_db)
                
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
        elif self.display_mode == DisplayMode.WATER_FRACTION:
            self.render_water_fraction()
            
        # Draw info overlay
        if self.show_info:
            self.render_info()
            
        # Draw help overlay
        if self.show_help:
            self.render_help()
            
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
        water_frac = self.state.vol_frac[MaterialType.WATER]
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
        
    def render_info(self):
        """Render information overlay."""
        info = self.simulation.get_info()
        
        y = 10
        lines = [
            f"FPS: {info['fps']:.1f}",
            f"Time: {info['time']:.1f} s",
            f"Step: {info['step_count']}",
            f"Mode: {self.display_mode.value}",
            f"Tool: {self.selected_material.name} (r={self.tool_radius})",
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
            "R - Reset simulation",
            "M - Cycle display mode",
            "1-9 - Select material",
            "Left Click - Add material",
            "Shift+Click - Remove material",
            "Scroll - Adjust tool radius",
            "Shift+Scroll - Adjust intensity",
            "S - Save screenshot",
            "H - Toggle this help",
            "I - Toggle info display",
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