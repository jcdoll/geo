"""
Fast particle renderer using Pygame with optimizations.

A fallback option that works without GPU acceleration but still
provides good performance for moderate particle counts.
"""

import numpy as np
import pygame
from typing import Optional, Tuple
from ..core.particles import ParticleArrays


class PygameRenderer:
    """Optimized Pygame-based particle renderer.
    
    Features:
    - Efficient batch rendering
    - Multiple visualization modes
    - Color interpolation
    - Adaptive quality settings
    """
    
    def __init__(self, domain_size: Tuple[float, float] = (10.0, 6.0),
                 window_size: Tuple[int, int] = (1280, 768),
                 title: str = "SPH Simulation"):
        """Initialize Pygame renderer.
        
        Args:
            domain_size: Physical domain size (width, height)
            window_size: Window size in pixels
            title: Window title
        """
        self.domain_size = domain_size
        self.window_size = window_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        
        # Scaling from physical to pixel coordinates
        self.scale_x = window_size[0] / domain_size[0]
        self.scale_y = window_size[1] / domain_size[1]
        
        # Colors and modes
        self.color_mode = 'velocity'
        self.particle_size = 3
        self.show_stats = True
        self.is_running = True
        self.is_paused = False
        
        # Font for stats
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Color maps (simple versions)
        self.color_maps = {
            'velocity': self._velocity_colormap,
            'density': self._density_colormap,
            'temperature': self._temperature_colormap,
            'material': self._material_colormap,
            'pressure': self._pressure_colormap
        }
        
        # Background
        self.bg_color = (30, 30, 30)
        
        # Performance settings
        self.quality_mode = 'high'  # 'high', 'medium', 'low'
        self.max_particles_high = 10000
        self.max_particles_medium = 50000
        
        # Pre-create surface for batch rendering
        self.particle_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
        self._create_particle_sprite()
    
    def _create_particle_sprite(self):
        """Create a smooth particle sprite for better visuals."""
        size = 10
        center = size // 2
        
        for x in range(size):
            for y in range(size):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist <= center:
                    alpha = int(255 * (1 - dist/center)**2)
                    self.particle_surface.set_at((x, y), (255, 255, 255, alpha))
    
    def _velocity_colormap(self, values: np.ndarray) -> np.ndarray:
        """Simple velocity colormap (green to red)."""
        normalized = np.clip(values / 2.0, 0, 1)  # Assume max velocity 2 m/s
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        colors[:, 0] = (normalized * 255).astype(np.uint8)  # Red
        colors[:, 1] = ((1 - normalized) * 255).astype(np.uint8)  # Green
        colors[:, 2] = 0  # Blue
        return colors
    
    def _density_colormap(self, values: np.ndarray) -> np.ndarray:
        """Simple density colormap (blue gradient)."""
        normalized = np.clip((values - 900) / 200, 0, 1)  # 900-1100 kg/m³
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        colors[:, 0] = (normalized * 100).astype(np.uint8)
        colors[:, 1] = (normalized * 150).astype(np.uint8)
        colors[:, 2] = (normalized * 255).astype(np.uint8)
        return colors
    
    def _temperature_colormap(self, values: np.ndarray) -> np.ndarray:
        """Simple temperature colormap (blue to red)."""
        celsius = values - 273.15
        normalized = np.clip(celsius / 100, 0, 1)  # 0-100°C
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        colors[:, 0] = (normalized * 255).astype(np.uint8)
        colors[:, 1] = ((1 - abs(normalized - 0.5) * 2) * 255).astype(np.uint8)
        colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)
        return colors
    
    def _material_colormap(self, values: np.ndarray) -> np.ndarray:
        """Simple material colormap (distinct colors)."""
        material_colors = [
            (100, 150, 255),  # Water (blue)
            (150, 150, 150),  # Rock (gray)
            (255, 100, 50),   # Magma (orange)
            (100, 255, 100),  # Gas (green)
            (255, 255, 100),  # Sand (yellow)
        ]
        
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        for i, mat_id in enumerate(values.astype(int)):
            if 0 <= mat_id < len(material_colors):
                colors[i] = material_colors[mat_id]
            else:
                colors[i] = (255, 255, 255)
        
        return colors
    
    def _pressure_colormap(self, values: np.ndarray) -> np.ndarray:
        """Simple pressure colormap (diverging blue-white-red)."""
        normalized = np.clip(values / 1e5 + 0.5, 0, 1)  # Center at 0
        colors = np.zeros((len(values), 3), dtype=np.uint8)
        
        # Blue for negative, red for positive
        mask_neg = normalized < 0.5
        mask_pos = normalized >= 0.5
        
        # Negative (blue)
        colors[mask_neg, 2] = 255
        colors[mask_neg, 0] = (normalized[mask_neg] * 2 * 255).astype(np.uint8)
        colors[mask_neg, 1] = (normalized[mask_neg] * 2 * 255).astype(np.uint8)
        
        # Positive (red)
        colors[mask_pos, 0] = 255
        colors[mask_pos, 1] = ((1 - (normalized[mask_pos] - 0.5) * 2) * 255).astype(np.uint8)
        colors[mask_pos, 2] = ((1 - (normalized[mask_pos] - 0.5) * 2) * 255).astype(np.uint8)
        
        return colors
    
    def update_particles(self, particle_arrays: ParticleArrays, n_active: int,
                        time: float = 0.0):
        """Update and render particles.
        
        Args:
            particle_arrays: Particle data
            n_active: Number of active particles
            time: Current simulation time
        """
        if self.is_paused:
            return
        
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Determine quality based on particle count
        if n_active < self.max_particles_high:
            render_mode = 'sprites'
            step = 1
        elif n_active < self.max_particles_medium:
            render_mode = 'circles'
            step = 1
        else:
            render_mode = 'pixels'
            step = max(1, n_active // 50000)  # Subsample if too many
        
        # Get particle data
        positions_x = particle_arrays.position_x[:n_active:step]
        positions_y = particle_arrays.position_y[:n_active:step]
        
        # Convert to screen coordinates
        screen_x = (positions_x * self.scale_x).astype(int)
        screen_y = (self.window_size[1] - positions_y * self.scale_y).astype(int)
        
        # Get colors based on mode
        if self.color_mode == 'velocity':
            v_mag = np.sqrt(
                particle_arrays.velocity_x[:n_active:step]**2 + 
                particle_arrays.velocity_y[:n_active:step]**2
            )
            colors = self.color_maps['velocity'](v_mag)
        elif self.color_mode == 'density':
            colors = self.color_maps['density'](particle_arrays.density[:n_active:step])
        elif self.color_mode == 'temperature':
            colors = self.color_maps['temperature'](particle_arrays.temperature[:n_active:step])
        elif self.color_mode == 'material':
            colors = self.color_maps['material'](particle_arrays.material_id[:n_active:step])
        elif self.color_mode == 'pressure':
            colors = self.color_maps['pressure'](particle_arrays.pressure[:n_active:step])
        else:
            colors = np.full((len(positions_x), 3), 255, dtype=np.uint8)
        
        # Render particles
        if render_mode == 'sprites':
            # High quality with sprites
            for i in range(len(screen_x)):
                if 0 <= screen_x[i] < self.window_size[0] and 0 <= screen_y[i] < self.window_size[1]:
                    tinted = self.particle_surface.copy()
                    tinted.fill((*colors[i], 255), special_flags=pygame.BLEND_MULT)
                    self.screen.blit(tinted, (screen_x[i] - 5, screen_y[i] - 5))
        
        elif render_mode == 'circles':
            # Medium quality with circles
            for i in range(len(screen_x)):
                if 0 <= screen_x[i] < self.window_size[0] and 0 <= screen_y[i] < self.window_size[1]:
                    pygame.draw.circle(self.screen, colors[i], 
                                     (screen_x[i], screen_y[i]), self.particle_size)
        
        else:  # pixels
            # Low quality but fast
            pixel_array = pygame.surfarray.pixels3d(self.screen)
            for i in range(len(screen_x)):
                x, y = screen_x[i], screen_y[i]
                if 0 <= x < self.window_size[0] and 0 <= y < self.window_size[1]:
                    pixel_array[x, y] = colors[i]
            del pixel_array  # Release the surface
        
        # Draw stats
        if self.show_stats:
            self._draw_stats(particle_arrays, n_active, time, step)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # Cap at 60 FPS
    
    def _draw_stats(self, particles: ParticleArrays, n_active: int, 
                   time: float, step: int):
        """Draw statistics overlay."""
        stats = [
            f"Time: {time:.2f}s",
            f"Particles: {n_active:,}" + (f" (showing 1/{step})" if step > 1 else ""),
            f"FPS: {self.clock.get_fps():.1f}",
            f"Mode: {self.color_mode.title()}",
        ]
        
        y_offset = 10
        for stat in stats:
            text = self.font.render(stat, True, (255, 255, 255))
            text_rect = text.get_rect()
            text_rect.topleft = (10, y_offset)
            
            # Add background for readability
            bg_rect = text_rect.inflate(10, 4)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            self.screen.blit(text, text_rect)
            y_offset += 30
        
        # Controls help
        help_text = "Space: Pause | D/V/T/M/P: Change mode | Q: Quality | S: Stats | ESC: Quit"
        text = self.small_font.render(help_text, True, (200, 200, 200))
        text_rect = text.get_rect()
        text_rect.bottomleft = (10, self.window_size[1] - 10)
        self.screen.blit(text, text_rect)
    
    def handle_events(self) -> bool:
        """Handle pygame events.
        
        Returns:
            False if window should close
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_d:
                    self.color_mode = 'density'
                elif event.key == pygame.K_v:
                    self.color_mode = 'velocity'
                elif event.key == pygame.K_t:
                    self.color_mode = 'temperature'
                elif event.key == pygame.K_m:
                    self.color_mode = 'material'
                elif event.key == pygame.K_p:
                    self.color_mode = 'pressure'
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
                elif event.key == pygame.K_q:
                    # Cycle quality modes
                    modes = ['high', 'medium', 'low']
                    idx = modes.index(self.quality_mode)
                    self.quality_mode = modes[(idx + 1) % len(modes)]
                    print(f"Quality mode: {self.quality_mode}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.particle_size = min(10, self.particle_size + 1)
                elif event.key == pygame.K_MINUS:
                    self.particle_size = max(1, self.particle_size - 1)
        
        return True
    
    def close(self):
        """Clean up and close the renderer."""
        pygame.quit()