"""core_tools.py – Convenience mix-in with editing & I/O helpers.

This light-weight mix-in bundles a handful of user-facing helper
functions (heat source painting, material editing, logging toggle, …)
that are shared by the visualiser and potential debug scripts.  The code
is factored out from the legacy *simulation_engine_original.py* so that
both the new modular engine and any minimalist sand-box wrappers can
reuse the mature, well-tested implementations without copying 1000+ lines
of unrelated physics.

The mix-in assumes the host class provides the **public attributes**
listed below – these are already satisfied by :class:`geo.core_state.CoreState`.

Required attributes
-------------------
    width, height                – grid dimensions (int)
    material_types               – 2-D ``MaterialType`` array
    temperature                  – 2-D ``float64`` array (K)
    pressure, pressure_offset    – 2-D ``float64`` arrays (MPa)
    cell_size                    – scalar metres per cell (float)
    material_db                  – :class:`MaterialDatabase` instance
    _get_distances_from_center   – helper returning distance field
    _update_material_properties  – refreshes derived density/k/cp arrays
    _save_state                  – pushes current snapshot to ``history``
    history, max_history         – undo buffer (list + capacity)

The actual physics (thermal diffusion, pressure solve, etc.) live in
other modules – this mix-in is purely *UI/interaction glue*.
"""
from __future__ import annotations

from typing import Tuple
import logging

import numpy as np

try:
    from .materials import MaterialType
except ImportError:  # fallback for unit tests executed outside package context
    from materials import MaterialType  # type: ignore


class CoreToolsMixin:
    """Convenience helper methods usable by the GUI and notebooks."""

    # ------------------------------------------------------------------
    # Logging / diagnostic helpers
    # ------------------------------------------------------------------
    def toggle_logging(self):  # type: ignore[override] – runtime mixed into CoreState
        """Toggle verbose DEBUG logging output.

        The visualiser binds this to the **L** key so users can quickly
        inspect performance and stability information.
        """
        if not hasattr(self, "logger"):
            # Late import paranoia – construct a dummy logger so we never crash
            self.logger = logging.getLogger(f"GeoGame_{id(self)}")
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())
        self.logging_enabled = not getattr(self, "logging_enabled", False)
        new_level = logging.DEBUG if self.logging_enabled else logging.INFO
        self.logger.setLevel(new_level)
        self.logger.info(
            "Logging %s – level set to %s",
            "ENABLED" if self.logging_enabled else "DISABLED",
            logging.getLevelName(new_level),
        )

    # ------------------------------------------------------------------
    # Inline intensity helper (Gaussian roll-off)
    # ------------------------------------------------------------------
    def _create_gaussian_intensity_field(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        *,
        effective_radius_multiplier: float = 2.0,
    ) -> np.ndarray:
        """Return a smooth Gaussian-like intensity distribution (0–1)."""
        effective_radius = radius * effective_radius_multiplier
        distances = self._get_distances_from_center(center_x, center_y)
        falloff_mask = distances <= effective_radius
        norm_dist = np.where(falloff_mask, distances / effective_radius, 1.0)
        return np.where(falloff_mask, np.exp(-2.0 * norm_dist ** 2), 0.0)

    # ------------------------------------------------------------------
    # Editing helpers (heat, pressure, material painting)
    # ------------------------------------------------------------------
    def add_heat_source(self, x: int, y: int, radius: int, temperature: float):
        """Raise temperature in a soft-edged circular patch."""
        intensity = self._create_gaussian_intensity_field(x, y, radius)
        temp_addition = intensity * temperature
        self.temperature = np.maximum(self.temperature, self.temperature + temp_addition)

    def apply_tectonic_stress(self, x: int, y: int, radius: int, pressure_increase: float):
        """Persistently increase pressure in a circular region (MPa)."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.pressure_offset[ny, nx] += pressure_increase

        # Recompute pressure with new offset if a solve method exists
        if hasattr(self, "fluid_dynamics"):
            self.fluid_dynamics.calculate_planetary_pressure()

    def delete_material_blob(self, x: int, y: int, radius: int = 1):
        """Replace a circular blob with **SPACE**."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.material_types[ny, nx] = MaterialType.SPACE
        self._properties_dirty = True

    def add_material_blob(self, x: int, y: int, radius: int, material_type):
        """Paint *material_type* into a circular blob; set sensible T defaults."""
        if not isinstance(material_type, MaterialType):
            try:
                material_type = MaterialType(material_type)
            except ValueError as exc:
                raise ValueError(f"Invalid material_type: {material_type}") from exc

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        self.material_types[ny, nx] = material_type
                        if material_type == MaterialType.MAGMA:
                            self.temperature[ny, nx] = max(
                                self.temperature[ny, nx],
                                getattr(self, "melting_temperature", 1200 + 273.15) + 100.0,
                            )
                        else:
                            # Default to ~room temperature so new rock neither melts nor freezes.
                            self.temperature[ny, nx] = 300.0  # K
        self._properties_dirty = True
        self._update_material_properties()

    # ------------------------------------------------------------------
    # Colour-mapped data extraction for the visualiser
    # ------------------------------------------------------------------
    def get_visualization_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb, T, P, power_density) arrays ready for display."""
        # Build RGB array by mapping material enum → colour
        colours = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        unique_mats = set(self.material_types.flatten())
        for mat in unique_mats:
            mask = self.material_types == mat
            if np.any(mask):
                props = self.material_db.get_properties(mat)
                colours[mask] = props.color_rgb
        return colours, self.temperature, self.pressure, getattr(self, "power_density", np.zeros_like(self.temperature))

    # ------------------------------------------------------------------
    # Placeholder for kinematics toggle (used by visualiser ‑ key *M*)
    # ------------------------------------------------------------------
    def toggle_kinematics_mode(self):  # noqa: D401 – simple toggle
        """Switch between discrete and unified kinematics (stub).

        The full unified-flow solver is not yet implemented.  For now we just
        flip a boolean and return its new state so the GUI can update its
        status text.
        """
        self.unified_kinematics = not getattr(self, "unified_kinematics", False)
        return "UNIFIED" if self.unified_kinematics else "DISCRETE"

    # ------------------------------------------------------------------
    # Statistics summary (used by visualiser status bar + stats tab)
    # ------------------------------------------------------------------
    def get_stats(self) -> dict:  # type: ignore[override]
        """Return key simulation metrics in a stable dict layout."""

        # Material composition as percentages (by cell count)
        flat = self.material_types.flatten()
        material_strings = np.array([m.value if hasattr(m, "value") else str(m) for m in flat])
        unique, counts = np.unique(material_strings, return_counts=True)
        material_percentages = {u: 100.0 * c / len(flat) for u, c in zip(unique, counts)}
        sorted_materials = dict(sorted(material_percentages.items(), key=lambda kv: kv[1], reverse=True))

        stats = {
            'time': getattr(self, 'time', 0.0),
            'dt': getattr(self, 'dt', 1.0),
            'effective_dt': getattr(self, '_actual_effective_dt', getattr(self, 'dt', 1.0)),
            'stability_factor': getattr(self, '_last_stability_factor', 1.0),
            'substeps': getattr(self, '_actual_substeps', 1),
            'max_thermal_diffusivity': getattr(self, '_max_thermal_diffusivity', 0.0),
            'avg_temperature': float(np.mean(self.temperature) - 273.15),  # °C
            'max_temperature': float(np.max(self.temperature) - 273.15),   # °C
            'avg_pressure': float(np.mean(self.pressure)),  # MPa
            'max_pressure': float(np.max(self.pressure)),
            'material_composition': sorted_materials,
            'history_length': len(getattr(self, 'history', [])),
        }
        return stats 