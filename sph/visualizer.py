"""
Interactive visualizer for SPH simulation with toolbar UI.

Adapted from flux/visualizer.py with particle-based rendering.
"""

import pygame
import pygame.gfxdraw
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
import time

import sph
from sph.core.particles import ParticleArrays
from sph.core.kernel_vectorized import CubicSplineKernel
from sph.core.integrator_vectorized import integrate_leapfrog_vectorized
from sph.physics import MaterialType, MaterialDatabase, handle_phase_transitions
from sph.physics.thermal_vectorized import compute_heat_conduction_vectorized
from sph.physics.gravity_vectorized import compute_gravity_direct_batched


class DisplayMode(Enum):
    """Available visualization modes."""
    MATERIAL = "material"
    TEMPERATURE = "temperature" 
    PRESSURE = "pressure"
    VELOCITY = "velocity"
    DENSITY = "density"
    PHASE = "phase"
    FORCE = "force"


class SPHVisualizer:
    """Interactive visualizer for SPH simulation."""
    
    def __init__(
        self,
        particles: ParticleArrays,
        n_active: int,
        domain_size: Tuple[float, float] = (100.0, 100.0),
        window_size: Tuple[int, int] = (800, 900),
        target_fps: int = 60,
        ui_scale: float = 1.5,
        search_radius_factor: float = 2.0,
    ):
        """
        Initialize visualizer.
        
        Args:
            particles: Particle arrays
            n_active: Number of active particles
            domain_size: Physical domain size in meters
            window_size: Window size in pixels
            target_fps: Target frames per second
            ui_scale: Scale factor for entire UI (default 1.5)
            search_radius_factor: Neighbor search radius as multiple of h (default 2.0)
        """
        self.particles = particles
        self.n_active = n_active
        self.domain_size = domain_size
        self.target_fps = target_fps
        self.ui_scale = ui_scale
        self.search_radius_factor = search_radius_factor
        
        # Physics
        self.kernel = CubicSplineKernel(dim=2)
        self.material_db = MaterialDatabase()
        
        # Try to use GPU backend if available
        import sph
        try:
            sph.set_backend('gpu')
            print(f"Using GPU backend: {sph.get_backend()}")
        except:
            pass  # GPU not available, will use default
        # Create spatial hash with proper bounds for centered domain
        # Domain extends from -size/2 to +size/2
        domain_min = (-domain_size[0] / 2, -domain_size[1] / 2)
        # Cell size should be ~2h for efficiency
        # Estimate h from domain size (typical spacing is ~1% of domain)
        typical_h = domain_size[0] * 0.01 * 1.3  # spacing * 1.3
        cell_size = 2.0 * typical_h
        self.spatial_hash = sph.create_spatial_hash(domain_size, cell_size, domain_min=domain_min)
        
        # Pygame setup
        pygame.init()
        
        # Scale all UI dimensions
        # Add space for toolbar on the right
        self.toolbar_width = int(200 * ui_scale)
        self.window_width = int(window_size[0] * ui_scale + self.toolbar_width)
        self.window_height = int(window_size[1] * ui_scale)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("SPH Geological Simulation")
        
        # Fonts - scale font sizes
        self.font = pygame.font.Font(None, int(20 * ui_scale))
        self.small_font = pygame.font.Font(None, int(14 * ui_scale))
        self.toolbar_font = pygame.font.Font(None, int(12 * ui_scale))
        
        # Display settings
        self.display_mode = DisplayMode.MATERIAL
        self.show_info = True
        self.show_help = False
        self.show_performance = False
        
        # Display area (excluding toolbar)
        self.display_width = int(window_size[0] * ui_scale)
        self.display_height = int(window_size[1] * ui_scale)
        
        # Coordinate transformation
        # Map simulation space to screen with (0,0) at center
        # The domain extends from -domain_size/2 to +domain_size/2
        # +x right, +y up in simulation -> +x right, +y down in screen (flipped)
        # Force square aspect ratio: use the smaller scale for both axes
        scale = min(self.display_width / domain_size[0], 
                   self.display_height / domain_size[1])
        self.scale_x = scale
        self.scale_y = scale
        self.center_x = self.display_width / 2
        self.center_y = self.display_height / 2
        
        # Simulation state
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = True  # Start paused
        self.step_count = 0
        self.sim_time = 0.0
        self.dt = 0.1  # Start with much larger timestep
        self.use_adaptive_dt = True  # Enable adaptive timestepping
        self.dt_history = []  # Track timestep history
        
        # Physics modules - all enabled by default
        self.enable_external_gravity = False  # External uniform gravity field
        self.enable_self_gravity = True  # N-body self-gravity
        self.enable_pressure = True
        self.enable_viscosity = True
        self.enable_heat_transfer = True
        self.enable_phase_transitions = True
        
        # Don't auto-configure gravity - let user control via checkboxes
        # self.configure_gravity_for_scenario()
        
        # Interaction state
        self.mouse_down = False
        self.selected_material = MaterialType.WATER
        self.tool_radius = 5.0  # meters
        self.tool_intensity = 0.1
        self.selected_particle = None
        
        # Tools
        self.tools = [
            {"name": "Material", "desc": "Add/change material", "icon": "M"},
            {"name": "Heat", "desc": "Add/remove heat", "icon": "H"},
            {"name": "Velocity", "desc": "Set velocity", "icon": "V"},
            {"name": "Delete", "desc": "Remove particles", "icon": "D"},
        ]
        self.current_tool = 0
        
        # Sidebar - scale all dimensions
        self.sidebar_x = int(window_size[0] * ui_scale)
        self.button_height = int(30 * ui_scale)
        self.button_margin = int(3 * ui_scale)
        
        # Performance tracking
        self.fps = 0
        self.frame_times = []
        self.step_timings = {}
        
        # Reset message display
        self.show_reset_message = False
        self.reset_message_timer = 0
        self.show_clear_message = False
        self.clear_message_timer = 0
        
        # Store initial state for reset
        self.store_initial_state()
        
        # Color maps
        self.init_colormaps()
        
    def init_colormaps(self):
        """Initialize color maps for different display modes."""
        # Temperature colormap (black -> red -> yellow -> white)
        self.temp_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 64:
                # Black to red
                self.temp_colors[i] = [i * 4, 0, 0]
            elif i < 128:
                # Red to orange
                t = (i - 64) / 64.0
                self.temp_colors[i] = [255, int(165 * t), 0]
            elif i < 192:
                # Orange to yellow
                t = (i - 128) / 64.0
                self.temp_colors[i] = [255, 165 + int(90 * t), 0]
            else:
                # Yellow to white
                t = (i - 192) / 64.0
                self.temp_colors[i] = [255, 255, int(255 * t)]
                
        # Pressure colormap (blue negative -> black zero -> red positive)
        self.pressure_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 128:
                # Negative pressure (blue)
                t = i / 128.0
                self.pressure_colors[i] = [0, 0, int(255 * (1 - t))]
            else:
                # Positive pressure (red)
                t = (i - 128) / 128.0
                self.pressure_colors[i] = [int(255 * t), 0, 0]
                
        # Velocity colormap (black -> green -> yellow)
        self.velocity_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 128:
                # Black to green
                t = i / 128.0
                self.velocity_colors[i] = [0, int(255 * t), 0]
            else:
                # Green to yellow
                t = (i - 128) / 128.0
                self.velocity_colors[i] = [int(255 * t), 255, 0]
                
        # Density colormap (black -> blue -> cyan -> white)
        self.density_colors = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                # Black to blue
                t = i / 85.0
                self.density_colors[i] = [0, 0, int(255 * t)]
            elif i < 170:
                # Blue to cyan
                t = (i - 85) / 85.0
                self.density_colors[i] = [0, int(255 * t), 255]
            else:
                # Cyan to white
                t = (i - 170) / 86.0
                self.density_colors[i] = [int(255 * t), 255, 255]
    
    def configure_gravity_for_scenario(self):
        """Configure gravity settings based on scenario type (initial setup only)."""
        # Only configure if we have particles
        if self.n_active == 0:
            return
            
        # Detect scenario type based on particle distribution
        # If particles are concentrated in a sphere at origin, it's a planet scenario
        center_x = np.mean(self.particles.position_x[:self.n_active])
        center_y = np.mean(self.particles.position_y[:self.n_active])
        radius = np.max(np.sqrt(
            (self.particles.position_x[:self.n_active] - center_x)**2 +
            (self.particles.position_y[:self.n_active] - center_y)**2
        ))
        
        # Check if particles form a compact sphere near origin
        is_planet = abs(center_x) < 5.0 and abs(center_y) < 5.0 and radius < self.domain_size[0] * 0.45
        
        if is_planet:
            # Planet scenario: self-gravity only
            self.enable_external_gravity = False
            self.enable_self_gravity = True
        else:
            # Surface/other scenario: external gravity only
            self.enable_external_gravity = True
            self.enable_self_gravity = False
    
    def run(self):
        """Main visualization loop."""
        while self.running:
            frame_start = time.perf_counter()
            
            self.handle_events()
            
            # Step simulation if not paused
            if not self.paused:
                self.step_physics()
                
            self.render()
            
            # FPS tracking
            frame_time = time.perf_counter() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)
            self.fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
            
            self.clock.tick(self.target_fps)
            
        pygame.quit()
        
    def step_physics(self):
        """Perform one physics step."""
        # Skip physics if no particles
        if self.n_active == 0:
            return
            
        t0 = time.perf_counter()
        
        # 1. Spatial hash
        t_hash = time.perf_counter()
        self.spatial_hash.build_vectorized(self.particles, self.n_active)
        # Use search_radius_factor * h as search radius
        # Get typical smoothing length from first particle
        h_typical = self.particles.smoothing_h[0] if self.n_active > 0 else 2.0
        search_radius = self.search_radius_factor * h_typical
        self.spatial_hash.query_neighbors_vectorized(self.particles, self.n_active, search_radius)
        self.step_timings['spatial_hash'] = time.perf_counter() - t_hash
        
        # 2. Density
        t_density = time.perf_counter()
        sph.compute_density(self.particles, self.kernel, self.n_active)
        self.step_timings['density'] = time.perf_counter() - t_density
        
        # 3. Pressure (Stable calculation)
        if self.enable_pressure:
            t_pressure = time.perf_counter()
            
            # Use improved pressure calculation with overlap prevention
            from sph.physics.pressure_overlap_prevention import (
                compute_pressure_simple_repulsive, get_stable_bulk_modulus_improved
            )
            
            # Get improved bulk moduli
            bulk_modulus = np.zeros(self.n_active, dtype=np.float32)
            for i in range(self.n_active):
                bulk_modulus[i] = get_stable_bulk_modulus_improved(self.particles.material_id[i])
            
            density_ref = self.material_db.get_density_ref_array(
                self.particles.material_id[:self.n_active]
            )
            
            self.particles.pressure[:self.n_active] = compute_pressure_simple_repulsive(
                self.particles.density[:self.n_active],
                density_ref,
                bulk_modulus,
                gamma=7.0
            )
            
            self.step_timings['pressure'] = time.perf_counter() - t_pressure
        
        # 4. Forces (pressure + viscosity)
        t_forces = time.perf_counter()
        gravity = np.array([0, -9.81]) if self.enable_external_gravity else None
        alpha_visc = 0.1 if self.enable_viscosity else 0.0
        sph.compute_forces(self.particles, self.kernel, self.n_active, gravity, alpha_visc)
        
        # Add overlap prevention forces
        from sph.physics.overlap_forces import add_overlap_prevention_forces
        add_overlap_prevention_forces(self.particles, self.n_active, 
                                    overlap_distance=0.4, repulsion_strength=100.0)
        
        # Add water-specific behavior
        from sph.physics.water_behavior import add_water_cohesion_forces
        add_water_cohesion_forces(self.particles, self.n_active, cohesion_strength=0.3)
        
        # Add cohesion for solid materials (disabled for now due to stability issues)
        # TODO: Re-enable after fixing pressure calculation
        # from sph.physics.cohesion_vectorized import compute_cohesive_forces_vectorized
        # compute_cohesive_forces_vectorized(self.particles, self.kernel, self.n_active, 
        #                                   self.material_db, cutoff_factor=1.5,
        #                                   temperature_softening=True)
        
        self.step_timings['forces'] = time.perf_counter() - t_forces
        
        # 5. Self-gravity (optional, expensive)
        if self.enable_self_gravity and self.n_active < 5000:
            t_gravity = time.perf_counter()
            compute_gravity_direct_batched(self.particles, self.n_active, 
                                         G=6.67430e-11 * 1e6, softening=0.1)
            self.step_timings['self_gravity'] = time.perf_counter() - t_gravity
        
        # 6. Heat transfer
        if self.enable_heat_transfer:
            t_heat = time.perf_counter()
            dT_dt = compute_heat_conduction_vectorized(self.particles, self.kernel,
                                                      self.material_db, self.n_active)
            # Apply temperature change
            self.particles.temperature[:self.n_active] += dT_dt * self.dt
            self.step_timings['heat_transfer'] = time.perf_counter() - t_heat
        
        # 7. Phase transitions
        if self.enable_phase_transitions:
            t_phase = time.perf_counter()
            handle_phase_transitions(self.particles, self.material_db, self.n_active, self.dt)
            self.step_timings['phase_transitions'] = time.perf_counter() - t_phase
        
        # 8. Adaptive timestep (if enabled)
        if self.use_adaptive_dt:
            from sph.core.timestep import compute_adaptive_timestep
            new_dt = compute_adaptive_timestep(self.particles, self.n_active, 
                                             self.material_db, max_dt=1.0)  # Much larger max
            # Less smoothing for faster response
            self.dt = 0.5 * self.dt + 0.5 * new_dt
            self.dt_history.append(self.dt)
            if len(self.dt_history) > 100:
                self.dt_history.pop(0)
        
        # 9. Integration
        t_integrate = time.perf_counter()
        # Use centered domain bounds
        domain_bounds = (-self.domain_size[0]/2, self.domain_size[0]/2, 
                        -self.domain_size[1]/2, self.domain_size[1]/2)
        integrate_leapfrog_vectorized(self.particles, self.n_active, self.dt, domain_bounds)
        self.step_timings['integration'] = time.perf_counter() - t_integrate
        
        # Update counters
        self.step_count += 1
        self.sim_time += self.dt
        self.step_timings['total'] = time.perf_counter() - t0
        
    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if event.pos[0] >= self.sidebar_x:
                        self.handle_toolbar_click(event.pos)
                    else:
                        self.mouse_down = True
                        self.apply_tool(event.pos)
                elif event.button == 3:  # Right click
                    if event.pos[0] < self.sidebar_x:
                        self.handle_particle_selection(event.pos)
                        
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    self.apply_tool(event.pos)
                    
            elif event.type == pygame.MOUSEWHEEL:
                # Adjust tool radius/intensity
                if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                    self.tool_intensity = np.clip(
                        self.tool_intensity + event.y * 0.01, 0.01, 1.0
                    )
                else:
                    self.tool_radius = np.clip(
                        self.tool_radius + event.y, 1.0, 20.0
                    )
                    
    def handle_keydown(self, event):
        """Handle keyboard input."""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            
        elif event.key == pygame.K_r:
            # Reset simulation to initial state
            self.reset_simulation()
            
        elif event.key == pygame.K_c:
            # Clear the board (remove all particles)
            self.clear_board()
            
        elif event.key == pygame.K_m or event.key == pygame.K_TAB:
            # Cycle display mode
            modes = list(DisplayMode)
            idx = modes.index(self.display_mode)
            self.display_mode = modes[(idx + 1) % len(modes)]
            
        elif event.key == pygame.K_h:
            self.show_help = not self.show_help
            
        elif event.key == pygame.K_i:
            self.show_info = not self.show_info
            
        elif event.key == pygame.K_l:
            self.show_performance = not self.show_performance
            
        elif event.key == pygame.K_t:
            # Toggle adaptive timestep
            self.use_adaptive_dt = not self.use_adaptive_dt
            print(f"Adaptive timestep: {'ON' if self.use_adaptive_dt else 'OFF'}")
            
        elif event.key == pygame.K_s:
            # Save screenshot
            pygame.image.save(self.screen, "sph_screenshot.png")
            print("Screenshot saved to sph_screenshot.png")
            
        elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
            # Select material
            mat_idx = event.key - pygame.K_1
            if mat_idx < len(MaterialType):
                self.selected_material = list(MaterialType)[mat_idx]
                
        elif event.key == pygame.K_t:
            # Cycle through tools
            self.current_tool = (self.current_tool + 1) % len(self.tools)
            
        elif event.key == pygame.K_RIGHT:
            # Step forward one frame when paused
            if self.paused:
                self.step_physics()
                
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
            
        # Check material buttons
        mat_button_y = button_y + 40
        for i, mat_type in enumerate(MaterialType):
            if (x >= self.sidebar_x + self.button_margin and
                x <= self.sidebar_x + self.toolbar_width - self.button_margin and
                y >= mat_button_y and y <= mat_button_y + 25):
                self.selected_material = mat_type
                return
            mat_button_y += 28
            
        # Check display mode buttons
        display_button_y = mat_button_y + 30
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
                
        # Check physics module checkboxes
        num_rows = (len(display_modes) + 1) // 2
        physics_section_y = display_button_y + num_rows * (button_height + 3) + 30
        checkbox_y = physics_section_y + 35
        checkbox_size = 20
        checkbox_margin = 10
        
        physics_modules = [
            ("enable_external_gravity", self.enable_external_gravity),
            ("enable_self_gravity", self.enable_self_gravity),
            ("enable_pressure", self.enable_pressure),
            ("enable_viscosity", self.enable_viscosity),
            ("enable_heat_transfer", self.enable_heat_transfer),
            ("enable_phase_transitions", self.enable_phase_transitions),
        ]
        
        for attr_name, current_value in physics_modules:
            checkbox_rect = pygame.Rect(
                self.sidebar_x + checkbox_margin,
                checkbox_y,
                checkbox_size,
                checkbox_size
            )
            
            label_rect = pygame.Rect(
                checkbox_rect.right + 8,
                checkbox_rect.top,
                self.toolbar_width - checkbox_margin - checkbox_size - 8,
                checkbox_size
            )
            
            if checkbox_rect.collidepoint(x, y) or label_rect.collidepoint(x, y):
                setattr(self, attr_name, not current_value)
                return
                
            checkbox_y += checkbox_size + 8
            
    def add_particles_at_position(self, world_x: float, world_y: float):
        """Add new particles at the given position."""
        # Check if we have room for more particles
        max_particles = len(self.particles.position_x)
        if self.n_active >= max_particles - 10:
            print(f"Warning: Near particle limit ({self.n_active}/{max_particles})")
            return
            
        # Material properties
        props = self.material_db.get_properties(self.selected_material)
        
        # Determine particle spacing based on material and tool size
        base_spacing = 1.3  # Base spacing matching smoothing length
        
        # Adjust spacing based on tool radius to avoid too many particles
        if self.tool_radius < 5:
            spacing_multiplier = 1.0
        elif self.tool_radius < 10:
            spacing_multiplier = 1.5
        else:
            spacing_multiplier = 2.0
            
        spacing = base_spacing * spacing_multiplier
        
        # Liquids can be slightly tighter
        if self.selected_material in [MaterialType.WATER, MaterialType.MAGMA]:
            spacing *= 0.9
            
        # Calculate how many particles fit in the tool radius
        # Use ceil to ensure we include all particles within the radius
        n_radial = int(np.ceil(self.tool_radius / spacing))
        
        # Add particles in a hexagonal pattern
        particles_added = 0
        max_particles_per_click = 50  # Limit to prevent explosion
        
        for r in range(n_radial + 1):
            if r == 0:
                # Center particle
                if (self.n_active < max_particles and
                    -self.domain_size[0]/2 < world_x < self.domain_size[0]/2 and 
                    -self.domain_size[1]/2 < world_y < self.domain_size[1]/2):
                    self.add_single_particle(world_x, world_y, props)
                    particles_added += 1
            else:
                # Particles in a ring
                circumference = 2 * np.pi * r * spacing
                n_in_ring = max(6, int(circumference / spacing))
                
                for i in range(n_in_ring):
                    if self.n_active >= max_particles or particles_added >= max_particles_per_click:
                        break
                        
                    angle = 2 * np.pi * i / n_in_ring
                    px = world_x + r * spacing * np.cos(angle)
                    py = world_y + r * spacing * np.sin(angle)
                    
                    # Check if position is within tool radius AND domain bounds
                    dist_from_center = np.sqrt((px - world_x)**2 + (py - world_y)**2)
                    if (dist_from_center <= self.tool_radius and
                        -self.domain_size[0]/2 < px < self.domain_size[0]/2 and 
                        -self.domain_size[1]/2 < py < self.domain_size[1]/2):
                        self.add_single_particle(px, py, props)
                        particles_added += 1
                        
        if particles_added > 0:
            print(f"Added {particles_added} {self.selected_material.name} particles")
            
    def add_single_particle(self, x: float, y: float, props):
        """Add a single particle at the given position."""
        i = self.n_active
        
        # Position and velocity
        self.particles.position_x[i] = x
        self.particles.position_y[i] = y
        self.particles.velocity_x[i] = 0.0
        self.particles.velocity_y[i] = 0.0
        
        # Material properties
        self.particles.material_id[i] = self.selected_material.value
        self.particles.smoothing_h[i] = 1.3  # Standard smoothing length
        
        # Mass based on material density
        volume = np.pi * self.particles.smoothing_h[i]**2  # 2D volume
        self.particles.mass[i] = props.density_ref * volume
        
        # Initial conditions
        self.particles.density[i] = props.density_ref
        self.particles.pressure[i] = 0.0
        self.particles.temperature[i] = props.default_temperature
        
        # Forces
        self.particles.force_x[i] = 0.0
        self.particles.force_y[i] = 0.0
        
        # Clear neighbor data
        self.particles.neighbor_count[i] = 0
        self.particles.neighbor_ids[i, :] = -1
        
        # Increment active count
        self.n_active += 1
    
    def store_initial_state(self):
        """Store the initial state of all particles for reset functionality."""
        self.initial_state = {
            'n_active': self.n_active,
            'position_x': self.particles.position_x[:self.n_active].copy(),
            'position_y': self.particles.position_y[:self.n_active].copy(),
            'velocity_x': self.particles.velocity_x[:self.n_active].copy(),
            'velocity_y': self.particles.velocity_y[:self.n_active].copy(),
            'material_id': self.particles.material_id[:self.n_active].copy(),
            'temperature': self.particles.temperature[:self.n_active].copy(),
            'mass': self.particles.mass[:self.n_active].copy(),
            'density': self.particles.density[:self.n_active].copy(),
            'smoothing_h': self.particles.smoothing_h[:self.n_active].copy(),
        }
        print("Initial state stored for reset")
        
    def reset_simulation(self):
        """Reset the simulation to its initial state."""
        print("Resetting simulation...")
        
        # Restore particle count
        self.n_active = self.initial_state['n_active']
        
        # Restore all particle properties
        self.particles.position_x[:self.n_active] = self.initial_state['position_x']
        self.particles.position_y[:self.n_active] = self.initial_state['position_y']
        self.particles.velocity_x[:self.n_active] = self.initial_state['velocity_x']
        self.particles.velocity_y[:self.n_active] = self.initial_state['velocity_y']
        self.particles.material_id[:self.n_active] = self.initial_state['material_id']
        self.particles.temperature[:self.n_active] = self.initial_state['temperature']
        self.particles.mass[:self.n_active] = self.initial_state['mass']
        self.particles.density[:self.n_active] = self.initial_state['density']
        self.particles.smoothing_h[:self.n_active] = self.initial_state['smoothing_h']
        
        # Reset forces and pressure
        self.particles.force_x[:self.n_active] = 0.0
        self.particles.force_y[:self.n_active] = 0.0
        self.particles.pressure[:self.n_active] = 0.0
        
        # Reset neighbor data
        self.particles.neighbor_count[:self.n_active] = 0
        self.particles.neighbor_ids[:self.n_active, :] = -1
        
        # Reset simulation counters
        self.step_count = 0
        self.sim_time = 0.0
        self.paused = True  # Pause after reset
        
        # Clear performance tracking
        self.frame_times = []
        self.step_timings = {}
        self.dt_history = []
        
        # Show reset message
        self.show_reset_message = True
        self.reset_message_timer = 120  # Show for 2 seconds at 60 FPS
        
    def clear_board(self):
        """Clear all particles from the simulation."""
        print("Clearing all particles...")
        
        # Set active particle count to 0
        self.n_active = 0
        
        # Reset simulation counters
        self.step_count = 0
        self.sim_time = 0.0
        self.paused = True  # Pause after clearing
        
        # Clear performance tracking
        self.frame_times = []
        self.step_timings = {}
        self.dt_history = []
        
        # Clear selected particle
        self.selected_particle = None
        
        # Show clear message
        self.show_clear_message = True
        self.clear_message_timer = 120  # Show for 2 seconds at 60 FPS
        
        print(f"Board cleared. Ready to add new particles.")
    
    def handle_particle_selection(self, pos):
        """Handle right-click particle selection."""
        # Convert screen to world coordinates
        world_x = (pos[0] - self.center_x) / self.scale_x
        world_y = (self.center_y - pos[1]) / self.scale_y
        
        # Find nearest particle
        min_dist = float('inf')
        selected = None
        
        for i in range(self.n_active):
            dx = self.particles.position_x[i] - world_x
            dy = self.particles.position_y[i] - world_y
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist and dist < 2.0:  # Within 2m
                min_dist = dist
                selected = i
                
        self.selected_particle = selected
        
    def apply_tool(self, mouse_pos: Tuple[int, int]):
        """Apply current tool at mouse position."""
        # Convert to world coordinates
        # Screen space: (0,0) top-left, +x right, +y down
        # World space: (0,0) center, +x right, +y up
        world_x = (mouse_pos[0] - self.center_x) / self.scale_x
        world_y = (self.center_y - mouse_pos[1]) / self.scale_y
        
        tool = self.tools[self.current_tool]
        
        if tool["name"] == "Material":
            # Add new particles in a circular pattern
            if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                # Shift+click: Convert existing particles
                for i in range(self.n_active):
                    dx = self.particles.position_x[i] - world_x
                    dy = self.particles.position_y[i] - world_y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < self.tool_radius:
                        self.particles.material_id[i] = self.selected_material.value
                        # Update mass based on new material
                        props = self.material_db.get_properties(self.selected_material)
                        volume = np.pi * self.particles.smoothing_h[i]**2  # 2D volume
                        self.particles.mass[i] = props.density_ref * volume
            else:
                # Normal click: Add new particles
                self.add_particles_at_position(world_x, world_y)
                    
        elif tool["name"] == "Heat":
            # Add/remove heat
            for i in range(self.n_active):
                dx = self.particles.position_x[i] - world_x
                dy = self.particles.position_y[i] - world_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < self.tool_radius:
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        # Cool down
                        self.particles.temperature[i] *= 0.95
                    else:
                        # Heat up
                        self.particles.temperature[i] += 50.0
                        
        elif tool["name"] == "Velocity":
            # Set velocity
            for i in range(self.n_active):
                dx = self.particles.position_x[i] - world_x
                dy = self.particles.position_y[i] - world_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < self.tool_radius:
                    if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                        # Pull inward
                        if dist > 0.1:
                            self.particles.velocity_x[i] = -dx / dist * 10.0
                            self.particles.velocity_y[i] = -dy / dist * 10.0
                    else:
                        # Push outward
                        if dist > 0.1:
                            self.particles.velocity_x[i] = dx / dist * 10.0
                            self.particles.velocity_y[i] = dy / dist * 10.0
                            
        elif tool["name"] == "Delete":
            # Delete particles by swapping with particles at the end
            i = 0
            while i < self.n_active:
                dx = self.particles.position_x[i] - world_x
                dy = self.particles.position_y[i] - world_y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < self.tool_radius:
                    # Swap with last active particle
                    self.n_active -= 1
                    if i < self.n_active:
                        # Swap all properties
                        for attr in ['position_x', 'position_y', 'velocity_x', 'velocity_y',
                                    'mass', 'density', 'pressure', 'temperature', 
                                    'material_id', 'smoothing_h', 'force_x', 'force_y']:
                            arr = getattr(self.particles, attr)
                            arr[i] = arr[self.n_active]
                        # Don't increment i since we need to check the swapped particle
                        continue
                i += 1
                    
    def render(self):
        """Render the current state."""
        self.screen.fill((0, 0, 0))
        
        # Render particles (all modes now use particle rendering)
        self.render_particles()
        
        # Draw colorbar/legend for current display mode
        self.render_scale_bar()
        
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
        
        # Draw selected particle info
        if self.selected_particle is not None:
            self.render_selected_particle()
            
        # Draw reset message
        if self.show_reset_message:
            self.render_reset_message()
            
        # Draw clear message
        if self.show_clear_message:
            self.render_clear_message()
            
        # Draw tool cursor outline
        self.render_tool_cursor()
            
        pygame.display.flip()
        
    def render_particles(self):
        """Render particles based on current display mode."""
        # Get particle positions in screen coordinates
        # Transform from simulation space (0,0 at center, +y up) to screen space
        screen_x = (self.particles.position_x[:self.n_active] * self.scale_x + self.center_x).astype(int)
        # Flip y-axis: simulation +y up -> screen +y down
        screen_y = (self.center_y - self.particles.position_y[:self.n_active] * self.scale_y).astype(int)
        
        # Determine colors based on display mode
        if self.display_mode == DisplayMode.MATERIAL:
            colors = self.get_material_colors()
        elif self.display_mode == DisplayMode.TEMPERATURE:
            colors = self.get_temperature_colors()
        elif self.display_mode == DisplayMode.PRESSURE:
            colors = self.get_pressure_colors()
        elif self.display_mode == DisplayMode.VELOCITY:
            colors = self.get_velocity_colors()
        elif self.display_mode == DisplayMode.DENSITY:
            colors = self.get_density_colors()
        elif self.display_mode == DisplayMode.PHASE:
            colors = self.get_phase_colors()
        elif self.display_mode == DisplayMode.FORCE:
            colors = self.get_force_colors()
        else:
            colors = [(255, 255, 255)] * self.n_active
            
        # Draw particles
        # Calculate particle radius based on smoothing length
        # Use a minimum radius of 2 pixels for visibility
        particle_radius = max(2, int(self.particles.smoothing_h[0] * self.scale_x * 0.3))
        
        for i in range(self.n_active):
            if 0 <= screen_x[i] < self.display_width and 0 <= screen_y[i] < self.display_height:
                if particle_radius >= 2:
                    # Draw anti-aliased filled circle for smoother appearance
                    pygame.gfxdraw.filled_circle(self.screen, screen_x[i], screen_y[i], 
                                                particle_radius, colors[i])
                    # Add anti-aliased outline
                    pygame.gfxdraw.aacircle(self.screen, screen_x[i], screen_y[i], 
                                          particle_radius, colors[i])
                else:
                    # For very small particles, use regular circle
                    pygame.draw.circle(self.screen, colors[i], 
                                     (screen_x[i], screen_y[i]), particle_radius)
                
    def get_material_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on material type."""
        colors = []
        for i in range(self.n_active):
            mat_type = MaterialType(self.particles.material_id[i])
            props = self.material_db.get_properties(mat_type)
            colors.append(props.color_rgb)
        return colors
        
    def get_temperature_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on temperature."""
        if self.n_active == 0:
            return []
            
        temps = self.particles.temperature[:self.n_active]
        t_min, t_max = 0, 2000  # Fixed range for consistency
        
        # Normalize
        t_norm = np.clip((temps - t_min) / (t_max - t_min) * 255, 0, 255).astype(int)
        
        colors = []
        for i in range(self.n_active):
            color = self.temp_colors[t_norm[i]]
            colors.append((color[0], color[1], color[2]))
        return colors
        
    def get_pressure_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on pressure."""
        if self.n_active == 0:
            return []
            
        pressures = self.particles.pressure[:self.n_active]
        p_max = max(abs(pressures.min()), abs(pressures.max()))
        
        if p_max > 0:
            # Normalize to [-1, 1] then to [0, 255]
            p_norm = np.clip((pressures / p_max + 1) * 127.5, 0, 255).astype(int)
        else:
            p_norm = np.ones(self.n_active, dtype=int) * 128
            
        colors = []
        for i in range(self.n_active):
            color = self.pressure_colors[p_norm[i]]
            colors.append((color[0], color[1], color[2]))
        return colors
        
    def get_velocity_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on velocity magnitude."""
        if self.n_active == 0:
            return []
            
        vel_mag = np.sqrt(self.particles.velocity_x[:self.n_active]**2 + 
                         self.particles.velocity_y[:self.n_active]**2)
        v_max = max(vel_mag.max(), 10.0)  # Min scale of 10 m/s
        
        # Normalize
        v_norm = np.clip(vel_mag / v_max * 255, 0, 255).astype(int)
        
        colors = []
        for i in range(self.n_active):
            color = self.velocity_colors[v_norm[i]]
            colors.append((color[0], color[1], color[2]))
        return colors
        
    def get_density_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on density."""
        if self.n_active == 0:
            return []
            
        densities = self.particles.density[:self.n_active]
        d_min, d_max = 0, 5000  # Fixed range
        
        # Normalize
        d_norm = np.clip((densities - d_min) / (d_max - d_min) * 255, 0, 255).astype(int)
        
        colors = []
        for i in range(self.n_active):
            color = self.density_colors[d_norm[i]]
            colors.append((color[0], color[1], color[2]))
        return colors
        
    def get_phase_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors showing phase state (solid/liquid/gas)."""
        colors = []
        for i in range(self.n_active):
            mat_type = MaterialType(self.particles.material_id[i])
            temp = self.particles.temperature[i]
            props = self.material_db.get_properties(mat_type)
            
            # Determine phase based on material type
            if mat_type in [MaterialType.SPACE, MaterialType.AIR, MaterialType.WATER_VAPOR]:
                # Gas phase
                colors.append((100, 100, 255))  # Light blue
            elif mat_type == MaterialType.WATER:
                # Liquid
                colors.append((0, 0, 255))  # Blue
            elif mat_type == MaterialType.MAGMA:
                # Molten
                colors.append((255, 165, 0))  # Orange
            elif mat_type == MaterialType.ICE:
                # Frozen
                colors.append((200, 200, 255))  # Light gray-blue
            else:
                # Solid rocks/minerals - color by temperature
                if temp < 500:
                    colors.append((100, 100, 100))  # Gray (cold)
                elif temp < 1000:
                    colors.append((150, 100, 100))  # Warm gray
                elif temp < 1500:
                    colors.append((200, 100, 50))   # Hot (reddish)
                else:
                    colors.append((255, 100, 0))     # Very hot (orange)
                
        return colors
        
    def get_force_colors(self) -> List[Tuple[int, int, int]]:
        """Get colors based on total force/acceleration magnitude."""
        if self.n_active == 0:
            return []
            
        colors = []
        
        # Use the actual force magnitudes from the physics simulation
        # Convert force to acceleration: a = F/m
        accel_x = self.particles.force_x[:self.n_active] / self.particles.mass[:self.n_active]
        accel_y = self.particles.force_y[:self.n_active] / self.particles.mass[:self.n_active]
        
        # Calculate acceleration magnitude
        accel_magnitudes = np.sqrt(accel_x**2 + accel_y**2)
        
        # Handle case where no forces have been computed yet
        if np.all(accel_magnitudes == 0):
            # Return default color
            return [(50, 50, 50)] * self.n_active
        
        # Normalize and map to colors
        a_min = np.min(accel_magnitudes)
        a_max = np.max(accel_magnitudes)
        
        if a_max > a_min:
            for i in range(self.n_active):
                # Normalize to 0-1
                t = (accel_magnitudes[i] - a_min) / (a_max - a_min)
                # Map to color: black -> purple -> cyan -> white
                if t < 0.33:
                    # Black to purple
                    r = int(128 * (t / 0.33))
                    g = 0
                    b = int(128 * (t / 0.33))
                elif t < 0.67:
                    # Purple to cyan
                    s = (t - 0.33) / 0.34
                    r = int(128 * (1 - s))
                    g = int(128 * s)
                    b = 128 + int(127 * s)
                else:
                    # Cyan to white
                    s = (t - 0.67) / 0.33
                    r = int(128 + 127 * s)
                    g = int(128 + 127 * s)
                    b = 255
                colors.append((r, g, b))
        else:
            # Uniform field
            colors = [(128, 0, 128)] * self.n_active
            
        return colors
        
    def render_gravity_vectors(self):
        """Render gravity vector visualization."""
        # First render particles with subdued colors
        colors = [(50, 50, 50)] * self.n_active  # Dark gray for all particles
        screen_x = (self.particles.position_x[:self.n_active] * self.scale_x + self.center_x).astype(int)
        screen_y = (self.center_y - self.particles.position_y[:self.n_active] * self.scale_y).astype(int)
        particle_radius = max(1, int(self.particles.smoothing_h[0] * self.scale_x * 0.3))
        
        for i in range(self.n_active):
            if 0 <= screen_x[i] < self.display_width and 0 <= screen_y[i] < self.display_height:
                pygame.draw.circle(self.screen, colors[i], 
                                 (screen_x[i], screen_y[i]), particle_radius)
        
        # Grid of arrows showing gravity field
        grid_spacing = 40  # pixels
        arrow_scale = 10.0
        
        # External gravity vector
        if self.enable_external_gravity:
            g_vec = np.array([0, -9.81])
            
            for x in range(grid_spacing, self.display_width, grid_spacing):
                for y in range(grid_spacing, self.display_height, grid_spacing):
                    # Arrow start point
                    start_x = x
                    start_y = y
                    
                    # Arrow end point (scaled and flipped for screen)
                    end_x = start_x + int(g_vec[0] * arrow_scale)
                    end_y = start_y - int(g_vec[1] * arrow_scale)  # Flip y for screen
                    
                    # Draw arrow
                    pygame.draw.line(self.screen, (150, 150, 150), 
                                   (start_x, start_y), (end_x, end_y), 2)
                    
                    # Arrowhead
                    angle = np.arctan2(end_y - start_y, end_x - start_x)
                    head_size = 5
                    head_angle = 0.5
                    
                    head_x1 = end_x - head_size * np.cos(angle - head_angle)
                    head_y1 = end_y - head_size * np.sin(angle - head_angle)
                    head_x2 = end_x - head_size * np.cos(angle + head_angle)
                    head_y2 = end_y - head_size * np.sin(angle + head_angle)
                    
                    pygame.draw.polygon(self.screen, (150, 150, 150),
                                      [(end_x, end_y), (head_x1, head_y1), (head_x2, head_y2)])
        
        # Self-gravity visualization (show acceleration vectors at particle positions)
        if self.enable_self_gravity and self.n_active < 1000:
            # Calculate center of mass
            total_mass = np.sum(self.particles.mass[:self.n_active])
            com_x = np.sum(self.particles.mass[:self.n_active] * self.particles.position_x[:self.n_active]) / total_mass
            com_y = np.sum(self.particles.mass[:self.n_active] * self.particles.position_y[:self.n_active]) / total_mass
            
            # Convert to screen coordinates
            com_screen_x = int(com_x * self.scale_x + self.center_x)
            com_screen_y = int(self.center_y - com_y * self.scale_y)
            
            # Draw center of mass
            pygame.draw.circle(self.screen, (255, 255, 0), (com_screen_x, com_screen_y), 5)
            pygame.draw.circle(self.screen, (255, 255, 0), (com_screen_x, com_screen_y), 10, 2)
            
            # Sample particles for gravity vectors
            step = max(1, self.n_active // 50)  # Show up to 50 vectors
            for i in range(0, self.n_active, step):
                # Direction to center of mass
                dx = com_x - self.particles.position_x[i]
                dy = com_y - self.particles.position_y[i]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0.1:
                    # Normalize and scale
                    dx /= dist
                    dy /= dist
                    
                    # Screen position
                    screen_x = int(self.particles.position_x[i] * self.scale_x + self.center_x)
                    screen_y = int(self.center_y - self.particles.position_y[i] * self.scale_y)
                    
                    # Arrow end point
                    arrow_len = min(30, int(1000.0 / dist))  # Longer arrows closer to COM
                    end_x = screen_x + int(dx * arrow_len)
                    end_y = screen_y - int(dy * arrow_len)  # Flip y for screen
                    
                    # Draw arrow
                    pygame.draw.line(self.screen, (100, 100, 255), 
                                   (screen_x, screen_y), (end_x, end_y), 2)
        
    def render_scale_bar(self):
        """Render appropriate scale bar or legend for current display mode."""
        if self.display_mode == DisplayMode.MATERIAL:
            self.render_material_legend()
        elif self.display_mode == DisplayMode.PHASE:
            self.render_phase_legend()
        elif self.display_mode == DisplayMode.TEMPERATURE:
            self.render_temperature_colorbar()
        elif self.display_mode == DisplayMode.PRESSURE:
            self.render_pressure_colorbar()
        elif self.display_mode == DisplayMode.VELOCITY:
            self.render_velocity_colorbar()
        elif self.display_mode == DisplayMode.DENSITY:
            self.render_density_colorbar()
        elif self.display_mode == DisplayMode.FORCE:
            self.render_force_colorbar()
    
    def render_colorbar(self, title: str, min_val: float, max_val: float, 
                       unit: str, colormap: np.ndarray, symmetric: bool = False):
        """Render a colorbar with gradient and labels."""
        # Position in bottom right of main display area - scale dimensions
        bar_width = int(180 * self.ui_scale)
        bar_height = int(20 * self.ui_scale)
        margin = int(20 * self.ui_scale)
        x = self.display_width - bar_width - margin
        y = self.display_height - bar_height - margin - int(40 * self.ui_scale)
        
        # Create gradient surface
        gradient = pygame.Surface((bar_width, bar_height))
        for i in range(bar_width):
            t = i / (bar_width - 1)
            
            if symmetric and min_val < 0 and max_val > 0:
                # Map to full colormap range for symmetric data
                idx = int(t * 255)
            else:
                # Map to appropriate portion
                idx = int(t * 255)
            
            color = colormap[idx] if isinstance(colormap, np.ndarray) else colormap(t)
            pygame.draw.line(gradient, color, (i, 0), (i, bar_height))
        
        # Draw gradient
        self.screen.blit(gradient, (x, y))
        
        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, bar_width, bar_height), 1)
        
        # Title
        title_text = self.font.render(title, True, (255, 255, 255))
        self.screen.blit(title_text, (x, y - 25))
        
        # Labels
        min_text = self.small_font.render(f"{min_val:.1f} {unit}", True, (255, 255, 255))
        max_text = self.small_font.render(f"{max_val:.1f} {unit}", True, (255, 255, 255))
        
        self.screen.blit(min_text, (x, y + bar_height + 5))
        max_rect = max_text.get_rect(right=x + bar_width, top=y + bar_height + 5)
        self.screen.blit(max_text, max_rect)
        
        # Zero marker for symmetric colorbars
        if symmetric and min_val < 0 and max_val > 0:
            zero_x = x + int(bar_width * (-min_val / (max_val - min_val)))
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (zero_x, y), (zero_x, y + bar_height), 2)
            zero_text = self.small_font.render("0", True, (255, 255, 255))
            zero_rect = zero_text.get_rect(centerx=zero_x, top=y + bar_height + 5)
            self.screen.blit(zero_text, zero_rect)
    
    def render_material_legend(self):
        """Render legend for material types."""
        # Place in bottom right of main display area
        margin = 20
        x = self.display_width - 180
        y_start = self.display_height - 250
        
        # Count materials to size background
        present_materials = set()
        for i in range(self.n_active):
            mat_id = self.particles.material_id[i]
            present_materials.add(MaterialType(mat_id))
        
        # Semi-transparent background
        bg_height = 25 + len(present_materials) * 18 + 10
        background = pygame.Surface((190, bg_height))
        background.set_alpha(180)
        background.fill((0, 0, 0))
        self.screen.blit(background, (x - 5, y_start - 5))
        
        # Title
        y = y_start
        title = self.font.render("Materials", True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        y += 25
        
        for mat_type in sorted(present_materials, key=lambda m: m.value):
            props = self.material_db.get_properties(mat_type)
            
            # Color swatch
            pygame.draw.rect(self.screen, props.color_rgb, (x, y, 20, 15))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, 20, 15), 1)
            
            # Label
            label = self.small_font.render(mat_type.name.replace('_', ' ').title(), 
                                         True, (255, 255, 255))
            self.screen.blit(label, (x + 25, y))
            y += 18
    
    def render_phase_legend(self):
        """Render legend for phase states."""
        # Place in bottom right of main display area
        margin = 20
        x = self.display_width - 180
        y_start = self.display_height - 200
        
        phases = [
            ((100, 100, 255), "Gas"),
            ((0, 0, 255), "Liquid"),
            ((255, 165, 0), "Molten"),
            ((200, 200, 255), "Frozen"),
            ((100, 100, 100), "Solid (Cold)"),
            ((150, 100, 100), "Solid (Warm)"),
            ((200, 100, 50), "Solid (Hot)"),
            ((255, 100, 0), "Solid (Very Hot)")
        ]
        
        # Semi-transparent background
        bg_height = 25 + len(phases) * 18 + 10
        background = pygame.Surface((190, bg_height))
        background.set_alpha(180)
        background.fill((0, 0, 0))
        self.screen.blit(background, (x - 5, y_start - 5))
        
        # Title
        y = y_start
        title = self.font.render("Phase States", True, (255, 255, 255))
        self.screen.blit(title, (x, y))
        y += 25
        
        for color, name in phases:
            # Color swatch
            pygame.draw.rect(self.screen, color, (x, y, 20, 15))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y, 20, 15), 1)
            
            # Label
            label = self.small_font.render(name, True, (255, 255, 255))
            self.screen.blit(label, (x + 25, y))
            y += 18
    
    def render_temperature_colorbar(self):
        """Render temperature colorbar."""
        temps = self.particles.temperature[:self.n_active]
        t_min = np.min(temps) if self.n_active > 0 else 0
        t_max = np.max(temps) if self.n_active > 0 else 2000
        
        # Create temperature colormap function
        def temp_colormap(t):
            # Black -> red -> yellow -> white
            if t < 0.33:
                # Black to red
                r = int(255 * (t / 0.33))
                return (r, 0, 0)
            elif t < 0.67:
                # Red to yellow
                g = int(255 * ((t - 0.33) / 0.34))
                return (255, g, 0)
            else:
                # Yellow to white
                b = int(255 * ((t - 0.67) / 0.33))
                return (255, 255, b)
        
        self.render_colorbar("Temperature", t_min, t_max, "K", temp_colormap)
    
    def render_pressure_colorbar(self):
        """Render pressure colorbar."""
        pressures = self.particles.pressure[:self.n_active]
        p_min = np.min(pressures) if self.n_active > 0 else -1000
        p_max = np.max(pressures) if self.n_active > 0 else 1000
        
        # Pressure colormap: blue (negative) -> black (zero) -> red (positive)
        self.render_colorbar("Pressure", p_min, p_max, "Pa", self.pressure_colors, symmetric=True)
    
    def render_velocity_colorbar(self):
        """Render velocity colorbar."""
        if self.n_active > 0:
            vel_x = self.particles.velocity_x[:self.n_active]
            vel_y = self.particles.velocity_y[:self.n_active]
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            v_min = 0
            v_max = np.max(vel_mag)
        else:
            v_min = 0
            v_max = 10
        
        self.render_colorbar("Velocity", v_min, v_max, "m/s", self.velocity_colors)
    
    def render_density_colorbar(self):
        """Render density colorbar."""
        densities = self.particles.density[:self.n_active]
        d_min = np.min(densities) if self.n_active > 0 else 0
        d_max = np.max(densities) if self.n_active > 0 else 5000
        
        self.render_colorbar("Density", d_min, d_max, "kg/m³", self.density_colors)
    
    def render_force_colorbar(self):
        """Render force/acceleration magnitude colorbar."""
        # Use the same acceleration values as the particle coloring
        if self.n_active > 0:
            accel_x = self.particles.force_x[:self.n_active] / self.particles.mass[:self.n_active]
            accel_y = self.particles.force_y[:self.n_active] / self.particles.mass[:self.n_active]
            accel_magnitudes = np.sqrt(accel_x**2 + accel_y**2)
            
            # Handle case where no forces computed yet
            if np.all(accel_magnitudes == 0):
                g_min = 0
                g_max = 10  # Default range
            else:
                g_min = np.min(accel_magnitudes)
                g_max = np.max(accel_magnitudes)
        else:
            g_min = 0
            g_max = 10
        
        # Create gravity colormap function
        def gravity_colormap(t):
            # Black -> purple -> cyan -> white
            if t < 0.33:
                r = int(128 * (t / 0.33))
                g = 0
                b = int(128 * (t / 0.33))
            elif t < 0.67:
                s = (t - 0.33) / 0.34
                r = int(128 * (1 - s))
                g = int(128 * s)
                b = 128 + int(127 * s)
            else:
                s = (t - 0.67) / 0.33
                r = int(128 + 127 * s)
                g = int(128 + 127 * s)
                b = 255
            return (r, g, b)
        
        self.render_colorbar("Total Acceleration", g_min, g_max, "m/s²", gravity_colormap)
        
        # Add info about which gravity types are active
        # Get colorbar position
        bar_width = 180
        bar_height = 20
        margin = 20
        info_x = self.display_width - bar_width - margin
        info_y = self.display_height - bar_height - margin - 40 + bar_height + 30
        if self.enable_external_gravity:
            text = self.small_font.render("External: ON", True, (200, 200, 200))
            self.screen.blit(text, (info_x, info_y))
            info_y += 15
        if self.enable_self_gravity:
            text = self.small_font.render("Self-gravity: ON", True, (200, 200, 200))
            self.screen.blit(text, (info_x, info_y))
            
            # Draw center of mass marker if self-gravity is on
            if self.n_active > 0:
                total_mass = np.sum(self.particles.mass[:self.n_active])
                if total_mass > 0:
                    com_x = np.sum(self.particles.mass[:self.n_active] * self.particles.position_x[:self.n_active]) / total_mass
                    com_y = np.sum(self.particles.mass[:self.n_active] * self.particles.position_y[:self.n_active]) / total_mass
                    
                    # Convert to screen coordinates
                    com_screen_x = int(com_x * self.scale_x + self.center_x)
                    com_screen_y = int(self.center_y - com_y * self.scale_y)
                    
                    # Draw center of mass marker
                    if 0 <= com_screen_x < self.display_width and 0 <= com_screen_y < self.display_height:
                        pygame.draw.circle(self.screen, (255, 255, 0), (com_screen_x, com_screen_y), 5)
                        pygame.draw.circle(self.screen, (255, 255, 0), (com_screen_x, com_screen_y), 10, 2)
        
    def render_info(self):
        """Render information overlay."""
        y = 10
        lines = [
            f"FPS: {self.fps:.1f}",
            f"Time: {self.sim_time:.3f} s",
            f"Step: {self.step_count}",
            f"dt: {self.dt:.4f} s" + (" (adaptive)" if self.use_adaptive_dt else ""),
            f"Particles: {self.n_active}",
            f"Mode: {self.display_mode.value}",
            f"Tool: {self.tools[self.current_tool]['name']}",
            f"Material: {self.selected_material.name}",
            f"Radius: {self.tool_radius:.1f} m",
            f"Paused: {self.paused}",
            "",
            f"Backend: {sph.get_backend()}",
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
            "C - Clear board",
            "RIGHT - Step forward (when paused)",
            "TAB/M - Cycle display mode",
            "T - Toggle adaptive timestep",
            "1-9 - Select material",
            "Left Click - Apply tool (add particles)",
            "Shift+Click - Replace existing particles",
            "Right Click - Select particle",
            "Scroll - Adjust tool radius",
            "Shift+Scroll - Adjust intensity",
            "S - Save screenshot",
            "H - Toggle this help",
            "I - Toggle info display",
            "L - Toggle performance",
            "ESC - Exit",
        ]
        
        # Semi-transparent background
        overlay = pygame.Surface((300, len(help_text) * 20 + 20))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (self.window_width - 510, 10))
        
        # Render text
        y = 20
        for line in help_text:
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.window_width - 500, y))
            y += 20
            
    def render_reset_message(self):
        """Render reset confirmation message."""
        # Create semi-transparent overlay
        message = "SIMULATION RESET"
        text = self.font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.display_width // 2, self.display_height // 2))
        
        # Background box
        padding = 20
        box_rect = text_rect.inflate(padding * 2, padding)
        overlay = pygame.Surface((box_rect.width, box_rect.height))
        overlay.set_alpha(200)
        overlay.fill((0, 100, 0))  # Green background
        
        self.screen.blit(overlay, box_rect)
        self.screen.blit(text, text_rect)
        
        # Decrement timer
        self.reset_message_timer -= 1
        if self.reset_message_timer <= 0:
            self.show_reset_message = False
    
    def render_clear_message(self):
        """Render clear confirmation message."""
        # Create semi-transparent overlay
        message = "BOARD CLEARED"
        text = self.font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.display_width // 2, self.display_height // 2))
        
        # Background box
        padding = 20
        box_rect = text_rect.inflate(padding * 2, padding)
        overlay = pygame.Surface((box_rect.width, box_rect.height))
        overlay.set_alpha(200)
        overlay.fill((100, 0, 0))  # Red background
        
        self.screen.blit(overlay, box_rect)
        self.screen.blit(text, text_rect)
        
        # Decrement timer
        self.clear_message_timer -= 1
        if self.clear_message_timer <= 0:
            self.show_clear_message = False
    
    def render_tool_cursor(self):
        """Render tool cursor outline at mouse position."""
        mouse_pos = pygame.mouse.get_pos()
        
        # Only show if mouse is in the display area (not in toolbar)
        if mouse_pos[0] >= self.sidebar_x:
            return
            
        # Get current tool
        tool = self.tools[self.current_tool]
        
        # Only show for tools that use radius (Material, Heat, Delete)
        if tool["name"] not in ["Material", "Heat", "Delete"]:
            return
            
        # Convert mouse position to world coordinates
        world_x = (mouse_pos[0] - self.center_x) / self.scale_x
        world_y = (self.center_y - mouse_pos[1]) / self.scale_y
        
        # Check if in bounds
        if not (-self.domain_size[0]/2 < world_x < self.domain_size[0]/2 and 
                -self.domain_size[1]/2 < world_y < self.domain_size[1]/2):
            return
            
        # Draw circle outline at mouse position
        radius_pixels = int(self.tool_radius * self.scale_x)
        
        # Use different colors for different tools
        if tool["name"] == "Material":
            color = (255, 255, 255)  # White for material
        elif tool["name"] == "Heat":
            if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                color = (100, 100, 255)  # Blue for cooling
            else:
                color = (255, 100, 100)  # Red for heating
        elif tool["name"] == "Delete":
            color = (255, 100, 100)  # Red for delete
        else:
            color = (255, 255, 255)  # Default white
            
        # Draw the circle outline
        pygame.draw.circle(self.screen, color, mouse_pos, radius_pixels, 2)
    
    def render_performance(self):
        """Render performance timing overlay."""
        if not self.step_timings:
            return
            
        # Create timing lines
        lines = ["Performance (ms):"]
        lines.append("-" * 25)
        
        # Sort by time descending
        sorted_timings = sorted(self.step_timings.items(), 
                               key=lambda x: x[1], reverse=True)
        
        total_time = self.step_timings.get('total', 0)
        
        for name, time_sec in sorted_timings:
            if name == 'total':
                continue
            time_ms = time_sec * 1000
            percentage = (time_sec / total_time * 100) if total_time > 0 else 0
            lines.append(f"{name[:15]:15} {time_ms:5.1f} ({percentage:3.0f}%)")
            
        lines.append("-" * 25)
        lines.append(f"{'Total':15} {total_time*1000:5.1f}")
        
        # Position
        x = 10
        y = self.window_height // 2
        
        # Background
        overlay = pygame.Surface((250, len(lines) * 18 + 10))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (x - 5, y - 5))
        
        # Text
        for i, line in enumerate(lines):
            color = (255, 255, 255) if i > 1 else (255, 255, 100)
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (x, y + i * 18))
            
    def render_toolbar(self):
        """Render the toolbar on the right side."""
        # Background
        toolbar_rect = pygame.Rect(self.sidebar_x, 0, self.toolbar_width, self.window_height)
        pygame.draw.rect(self.screen, (40, 40, 40), toolbar_rect)
        pygame.draw.line(self.screen, (60, 60, 60), 
                        (self.sidebar_x, 0), (self.sidebar_x, self.window_height), 2)
        
        # Title
        title_text = self.font.render("TOOLS", True, (200, 200, 200))
        title_rect = title_text.get_rect(center=(self.sidebar_x + self.toolbar_width // 2, 25))
        self.screen.blit(title_text, title_rect)
        
        # Tool buttons
        button_y = 50
        for i, tool in enumerate(self.tools):
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
                
            # Icon and name
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
            mat_rect = pygame.Rect(
                self.sidebar_x + self.button_margin,
                mat_button_y,
                self.toolbar_width - 2 * self.button_margin,
                25
            )
            
            # Highlight selected
            if mat_type == self.selected_material:
                pygame.draw.rect(self.screen, (80, 80, 120), mat_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), mat_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), mat_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), mat_rect, 1)
                
            # Color swatch
            props = self.material_db.get_properties(mat_type)
            color_rect = pygame.Rect(mat_rect.left + 5, mat_rect.top + 5, 15, 15)
            pygame.draw.rect(self.screen, props.color_rgb, color_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), color_rect, 1)
            
            # Name
            mat_text = self.toolbar_font.render(mat_type.name, True, (180, 180, 180))
            mat_text_rect = mat_text.get_rect(midleft=(mat_rect.left + 25, mat_rect.centery))
            self.screen.blit(mat_text, mat_text_rect)
            
            mat_button_y += 28
            
        # Display modes section
        display_title = self.small_font.render("DISPLAY MODE", True, (200, 200, 200))
        display_title_rect = display_title.get_rect(
            center=(self.sidebar_x + self.toolbar_width // 2, mat_button_y + 15)
        )
        self.screen.blit(display_title, display_title_rect)
        
        # Display mode buttons - 2 columns
        display_button_y = mat_button_y + 30
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
            
            if mode == self.display_mode:
                pygame.draw.rect(self.screen, (80, 80, 120), mode_rect)
                pygame.draw.rect(self.screen, (120, 120, 180), mode_rect, 2)
            else:
                pygame.draw.rect(self.screen, (50, 50, 50), mode_rect)
                pygame.draw.rect(self.screen, (70, 70, 70), mode_rect, 1)
                
            mode_text = self.toolbar_font.render(mode.value.title(), True, (180, 180, 180))
            mode_text_rect = mode_text.get_rect(center=(mode_rect.centerx, mode_rect.centery))
            self.screen.blit(mode_text, mode_text_rect)
            
        # Physics toggles
        num_rows = (len(display_modes) + 1) // 2
        physics_section_y = display_button_y + num_rows * (button_height + 3) + 30
        
        physics_title = self.font.render("PHYSICS", True, (200, 200, 200))
        physics_title_rect = physics_title.get_rect(
            center=(self.sidebar_x + self.toolbar_width // 2, physics_section_y + 10)
        )
        self.screen.blit(physics_title, physics_title_rect)
        
        # Checkboxes
        checkbox_y = physics_section_y + 35
        checkbox_size = 20
        checkbox_margin = 10
        
        physics_modules = [
            ("Ext. Gravity", self.enable_external_gravity),
            ("Self Gravity", self.enable_self_gravity),
            ("Pressure", self.enable_pressure),
            ("Viscosity", self.enable_viscosity),
            ("Heat Transfer", self.enable_heat_transfer),
            ("Phase Trans.", self.enable_phase_transitions),
        ]
        
        for module_name, enabled in physics_modules:
            # Checkbox
            checkbox_rect = pygame.Rect(
                self.sidebar_x + checkbox_margin,
                checkbox_y,
                checkbox_size,
                checkbox_size
            )
            
            pygame.draw.rect(self.screen, (60, 60, 60), checkbox_rect)
            pygame.draw.rect(self.screen, (120, 120, 120), checkbox_rect, 2)
            
            if enabled:
                # Check mark
                check_points = [
                    (checkbox_rect.left + 4, checkbox_rect.centery),
                    (checkbox_rect.left + checkbox_size // 3 + 1, checkbox_rect.bottom - 4),
                    (checkbox_rect.right - 4, checkbox_rect.top + 4)
                ]
                pygame.draw.lines(self.screen, (0, 255, 0), False, check_points, 3)
            
            # Label
            module_text = self.small_font.render(module_name, True, (220, 220, 220))
            module_rect = module_text.get_rect(midleft=(checkbox_rect.right + 8, checkbox_rect.centery))
            self.screen.blit(module_text, module_rect)
            
            checkbox_y += checkbox_size + 8
            
    def render_selected_particle(self):
        """Render info about selected particle."""
        if self.selected_particle >= self.n_active:
            self.selected_particle = None
            return
            
        i = self.selected_particle
        
        # Highlight particle (use same transformation as particle rendering)
        screen_x = int(self.particles.position_x[i] * self.scale_x + self.center_x)
        screen_y = int(self.center_y - self.particles.position_y[i] * self.scale_y)
        pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), 10, 2)
        
        # Info box
        mat_type = MaterialType(self.particles.material_id[i])
        # Calculate force and acceleration
        fx = self.particles.force_x[i]
        fy = self.particles.force_y[i]
        force_mag = np.sqrt(fx**2 + fy**2)
        mass = self.particles.mass[i]
        ax = fx / mass if mass > 0 else 0
        ay = fy / mass if mass > 0 else 0
        accel_mag = np.sqrt(ax**2 + ay**2)
        info_lines = [
            f"Particle: {i}",
            f"Position: ({self.particles.position_x[i]:.1f}, {self.particles.position_y[i]:.1f})",
            f"Material: {mat_type.name}",
            f"Temperature: {self.particles.temperature[i]:.1f} K",
            f"Pressure: {self.particles.pressure[i]:.1e} Pa",
            f"Density: {self.particles.density[i]:.1f} kg/m³",
            f"Velocity: ({self.particles.velocity_x[i]:.2f}, {self.particles.velocity_y[i]:.2f})",
            f"Force: ({fx:.1f}, {fy:.1f}) N, |F|={force_mag:.1f} N",
            f"Accel: ({ax:.2f}, {ay:.2f}) m/s², |a|={accel_mag:.2f} m/s²",
            f"Neighbors: {self.particles.neighbor_count[i]}",
        ]
        
        # Position at bottom left
        info_x = 10
        info_y = self.window_height - len(info_lines) * 18 - 20
        
        # Background
        max_width = max(self.small_font.size(line)[0] for line in info_lines)
        bg_surface = pygame.Surface((max_width + 20, len(info_lines) * 18 + 10))
        bg_surface.set_alpha(200)
        bg_surface.fill((0, 0, 0))
        self.screen.blit(bg_surface, (info_x - 5, info_y - 5))
        
        # Text
        for i, line in enumerate(info_lines):
            text = self.small_font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (info_x, info_y + i * 18))