"""
Fast GPU-accelerated SPH particle renderer using Vispy.

Features:
- Handles millions of particles at interactive framerates
- Multiple visualization modes (density, velocity, temperature, pressure)
- Smooth particle rendering with GPU shaders
- Interactive camera controls
"""

import numpy as np
from vispy import app, scene
from vispy.color import get_colormap
from typing import Optional, Callable
from ..core.particles import ParticleArrays


class SPHRenderer:
    """GPU-accelerated particle renderer using Vispy.
    
    Provides real-time visualization of SPH simulations with:
    - Multiple coloring modes
    - Adaptive particle sizes
    - Smooth GPU-based rendering
    - Interactive controls
    """
    
    def __init__(self, domain_size: tuple = (10.0, 6.0), 
                 max_particles: int = 100_000,
                 bgcolor: str = '#1e1e1e',
                 title: str = 'SPH Simulation'):
        """Initialize the renderer.
        
        Args:
            domain_size: (width, height) of simulation domain
            max_particles: Maximum particles to pre-allocate
            bgcolor: Background color
            title: Window title
        """
        self.domain_size = domain_size
        self.max_particles = max_particles
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            bgcolor=bgcolor,
            size=(1280, 720),
            title=title,
            show=True
        )
        
        # Add grid for central widget
        self.grid = self.canvas.central_widget.add_grid()
        
        # Add 2D view
        self.view = self.grid.add_view(row=0, col=0, row_span=2)
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(
            x=(0, domain_size[0]),
            y=(0, domain_size[1])
        )
        
        # Create particle visual with GPU markers
        self.particles = scene.visuals.Markers(
            pos=np.zeros((max_particles, 2), dtype=np.float32),
            size=10,
            face_color='blue',
            edge_color=None,
            symbol='disc',  # GPU-accelerated circular sprites
            scaling=True,
            antialias=1.0
        )
        self.view.add(self.particles)
        
        # Add colorbar
        self.colorbar = scene.widgets.ColorBarWidget(
            label='',
            clim=(0, 1),
            cmap='viridis',
            orientation='right',
            border_width=1
        )
        self.grid.add_widget(self.colorbar, row=0, col=1)
        
        # Add text overlay for stats
        self.stats_text = scene.visuals.Text(
            '',
            color='white',
            anchor_x='left',
            anchor_y='top',
            font_size=12,
            pos=(10, 10),
            parent=self.view.scene
        )
        
        # Colormaps
        self.colormaps = {
            'density': get_colormap('blues'),
            'velocity': get_colormap('viridis'),
            'temperature': get_colormap('hot'),
            'pressure': get_colormap('RdBu_r'),
            'material': get_colormap('tab10')
        }
        
        # Current visualization mode
        self.color_mode = 'velocity'
        self.size_mode = 'adaptive'  # 'adaptive', 'fixed', 'density'
        
        # Animation state
        self.is_running = True
        self.frame_count = 0
        self.fps_tracker = []
        
        # Keyboard controls
        @self.canvas.connect
        def on_key_press(event):
            if event.key == ' ':
                self.is_running = not self.is_running
            elif event.key == 'D':
                self.color_mode = 'density'
                self.update_colorbar_label()
            elif event.key == 'V':
                self.color_mode = 'velocity'
                self.update_colorbar_label()
            elif event.key == 'T':
                self.color_mode = 'temperature'
                self.update_colorbar_label()
            elif event.key == 'P':
                self.color_mode = 'pressure'
                self.update_colorbar_label()
            elif event.key == 'M':
                self.color_mode = 'material'
                self.update_colorbar_label()
            elif event.key == 'R':
                self.reset_camera()
            elif event.key == 'Escape':
                self.canvas.close()
                app.quit()
    
    def update_particles(self, particle_arrays: ParticleArrays, n_active: int,
                        time: float = 0.0):
        """Update particle visualization.
        
        Args:
            particle_arrays: Particle data in SoA format
            n_active: Number of active particles
            time: Current simulation time
        """
        if not self.is_running:
            return
        
        # Get positions (2D)
        positions = np.column_stack([
            particle_arrays.position_x[:n_active],
            particle_arrays.position_y[:n_active]
        ])
        
        # Compute colors based on mode
        colors, clim = self.compute_colors(particle_arrays, n_active)
        
        # Compute sizes based on mode
        sizes = self.compute_sizes(particle_arrays, n_active)
        
        # Update particle visual
        self.particles.set_data(
            pos=positions,
            face_color=colors,
            size=sizes
        )
        
        # Update colorbar
        self.colorbar.clim = clim
        self.colorbar.cmap = self.colormaps[self.color_mode]
        
        # Update stats
        self.update_stats(particle_arrays, n_active, time)
        
        # Track FPS
        self.frame_count += 1
    
    def compute_colors(self, particles: ParticleArrays, n_active: int):
        """Compute particle colors based on current mode.
        
        Returns:
            (colors, clim) - color array and colorbar limits
        """
        if self.color_mode == 'density':
            values = particles.density[:n_active]
            clim = (900, 1100)  # kg/m³
        
        elif self.color_mode == 'velocity':
            values = np.sqrt(
                particles.velocity_x[:n_active]**2 + 
                particles.velocity_y[:n_active]**2
            )
            clim = (0, 2.0)  # m/s
        
        elif self.color_mode == 'temperature':
            values = particles.temperature[:n_active] - 273.15  # Convert to Celsius
            clim = (0, 100)  # °C
        
        elif self.color_mode == 'pressure':
            values = particles.pressure[:n_active] / 1e5  # Convert to bar
            clim = (-1, 1)  # bar
        
        elif self.color_mode == 'material':
            values = particles.material_id[:n_active].astype(float)
            clim = (0, 9)  # 10 material types
        
        else:
            values = np.zeros(n_active)
            clim = (0, 1)
        
        # Normalize and apply colormap
        norm_values = np.clip((values - clim[0]) / (clim[1] - clim[0]), 0, 1)
        colors = self.colormaps[self.color_mode].map(norm_values)
        
        # Add alpha channel based on density for fluid effects
        if self.color_mode != 'material':
            alpha = np.clip(particles.density[:n_active] / 1200, 0.3, 1.0)
            colors[:, 3] = alpha
        
        return colors, clim
    
    def compute_sizes(self, particles: ParticleArrays, n_active: int):
        """Compute particle sizes based on mode.
        
        Returns:
            Size array in pixels
        """
        if self.size_mode == 'adaptive':
            # Size based on smoothing length
            base_size = particles.smoothing_h[:n_active] * 200
        
        elif self.size_mode == 'density':
            # Size inversely proportional to density (larger when expanded)
            base_size = 20 * (1000 / particles.density[:n_active])**0.5
        
        else:  # fixed
            base_size = np.full(n_active, 10)
        
        # Clamp to reasonable range
        return np.clip(base_size, 5, 50)
    
    def update_stats(self, particles: ParticleArrays, n_active: int, time: float):
        """Update statistics overlay."""
        # Compute stats
        avg_density = np.mean(particles.density[:n_active])
        max_velocity = np.max(np.sqrt(
            particles.velocity_x[:n_active]**2 + 
            particles.velocity_y[:n_active]**2
        ))
        avg_temp = np.mean(particles.temperature[:n_active]) - 273.15
        
        # FPS calculation
        import time as pytime
        current_time = pytime.time()
        self.fps_tracker.append(current_time)
        self.fps_tracker = [t for t in self.fps_tracker if current_time - t < 1.0]
        fps = len(self.fps_tracker)
        
        # Build stats text
        stats = (
            f"Time: {time:.3f} s\n"
            f"Particles: {n_active:,}\n"
            f"FPS: {fps}\n"
            f"Avg Density: {avg_density:.0f} kg/m³\n"
            f"Max Velocity: {max_velocity:.2f} m/s\n"
            f"Avg Temp: {avg_temp:.1f} °C\n"
            f"\n"
            f"Color Mode: {self.color_mode.title()} [{self.get_mode_key()}]\n"
            f"Controls: Space=Pause, R=Reset, ESC=Quit"
        )
        
        self.stats_text.text = stats
    
    def get_mode_key(self):
        """Get the keyboard shortcut for current mode."""
        keys = {'density': 'D', 'velocity': 'V', 'temperature': 'T', 
                'pressure': 'P', 'material': 'M'}
        return keys.get(self.color_mode, '?')
    
    def update_colorbar_label(self):
        """Update colorbar label based on mode."""
        labels = {
            'density': 'Density (kg/m³)',
            'velocity': 'Velocity (m/s)',
            'temperature': 'Temperature (°C)',
            'pressure': 'Pressure (bar)',
            'material': 'Material Type'
        }
        self.colorbar.label.text = labels.get(self.color_mode, '')
    
    def reset_camera(self):
        """Reset camera to default view."""
        self.view.camera.set_range(
            x=(0, self.domain_size[0]),
            y=(0, self.domain_size[1])
        )
    
    def run(self, update_callback: Optional[Callable] = None):
        """Run the visualization loop.
        
        Args:
            update_callback: Function called each frame to update physics
        """
        if update_callback:
            timer = app.Timer(interval=0.0, connect=update_callback, start=True)
        
        app.run()
    
    def screenshot(self, filename: str = 'sph_screenshot.png'):
        """Save current view to file."""
        img = self.canvas.render()
        import vispy.io
        vispy.io.write_png(filename, img)
        print(f"Screenshot saved to {filename}")


class SPHRenderer3D(SPHRenderer):
    """3D version of the SPH renderer for future extension."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Switch to 3D camera
        self.view.camera = scene.TurntableCamera(
            elevation=30,
            azimuth=30,
            distance=10,
            fov=60
        )
        
        # Add axes
        self.axes = scene.visuals.XYZAxis(parent=self.view.scene)
        
    def update_particles(self, particle_arrays: ParticleArrays, n_active: int,
                        time: float = 0.0):
        """Update for 3D (adds z=0 for 2D simulations)."""
        if not self.is_running:
            return
        
        # Get positions (add z=0 for 2D)
        positions = np.column_stack([
            particle_arrays.position_x[:n_active],
            particle_arrays.position_y[:n_active],
            np.zeros(n_active)  # z = 0
        ])
        
        # Rest is same as 2D
        colors, clim = self.compute_colors(particle_arrays, n_active)
        sizes = self.compute_sizes(particle_arrays, n_active)
        
        self.particles.set_data(
            pos=positions,
            face_color=colors,
            size=sizes
        )
        
        self.colorbar.clim = clim
        self.colorbar.cmap = self.colormaps[self.color_mode]
        self.update_stats(particle_arrays, n_active, time)
        self.frame_count += 1