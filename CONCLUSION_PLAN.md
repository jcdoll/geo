We are trying to conclude this project.


Goals:
1) Three separate implementations can be run independently without errors: ca, flux, sph.

2) Each has limitations that are noted in the primary readme file (geo/README.md).

3) Each folder should have its own main.py instance that is used as a main entry point
    - python ca/main.py
    - python flux/main.py
    - python sph/main.py

4) Each folder should have its own test suite, all of which should be passing
    - python ca/tests/
    - python flux/tests/
    - python sph/tests/

5) Each folder has its own physics writeup
    - ca/PHYSICS.md
    - flux/PHYSICS.md
    - sph/PHYSICS.md

6) The only files in the geo base directory should be documentation applicable to all three versions or setup scripts. Clean things up.
    
7) Overview of each version
    - ca: simple while still having some physics, including self-gravity, thermal diffusion, solar heating, radiative cooling. Cell swaps are purely density based.
    - flux: tries to address the cell swapping criteria by solving for pressure using velocity projection. Problem is that velocity projection with large density variation is unstable so can't simulate everything.
    - sph: alternative approach to avoid needing to model space explicitly, but the problem with this approach is that modeling materials with high cohesive strength (rock, etc) is a problem, requires short timesteps.




Next steps:
1) Revisit ca implementation and clean up.
- Take flux visualizer improvements and port to ca
    - sun/planet position indicator in top right
    - C to clear board
    - Right click to select a tile to view, right click same tile again to deselect
    - Unified color bar scales, blue/black/red color scheme (blue = negative, black = zero, red = positive)
    - Anything else that you notice from reviewing flux/visualizer
- Clean up heat generation
    - DO NOT use planet shaped heating in the core and the crust
    - Instead add Uranium as a material type, like we do in the flux version
- Make solar heating more obvious
    - Add a slight color difference that indicates cells that are illuminated by the sun
    - The dark side should be slightly darker and the light side should be slightly lighter
- Add sun rotation
    - Add a sun angle parameter (like we do in the flux vesion) so that the solar heating direction varies over time.
- Remove quality setting
    - Use the high quality setting everywhere, remove the other downgraded versions

2) Simplify the ca/ buoyancy and cell swapping behavior
- Do not solve for pressure and velocity
- Use a simple swapping rule based on density differences and the gravity vector (swap away from the gravity vector)
- Add a probabilty to swap to each material (always <1 so that swapping takes some time - for a pair the probability is either the maximum of the two or the product of the two?)
- Remove all rigid body mechanics code

3) Attempt one minor ca/ improvement re: lateral buoyancy.
- Right now if we have a pillar of water it will not spread laterally
- We should allow FLUIDS to move laterally (perpendicular to gravity vector)
- This needs to be a special rule for fluids only so that the entire planet is not unstable
- Please take a stab at this and try to implement a very simple rule that allows water to spread and fill in holes, etc.

4) Restore stability to flux implementation.
- We recently enabled the pressure solver in low density regions (space). This lead to enormous instability problems.
    - Do not solve for pressure via velocity projection in space regions - set velocity and pressure to zero
    - TODO: Determine a hack so that material can move through the space region, without needing to solve for pressure in that region

5) Wrap up SPH
- I have concluded that SPH is not a great fit for our planetary model
- Adding sufficient binding force to capture the behavior of solids causes the simulation to be WAY too slow
- Instead we can just clean things to a "normal" SPH implementation that runs reasonably fast and has decent behavior

6) Clean up all docs and writeups

7) Remove any test scripts that are not in the test directories. All tests should be */tests.




Future work (add remaining items here):
  1. GPU acceleration improvements - Fix RTX 5080 compatibility issues and optimize GPU kernels for SPH
  2. Barnes-Hut tree for SPH gravity - Implement O(N log N) gravity calculation instead of current O(N²)
  3. Adaptive timestepping - Implement CFL-based timestep adjustment for all three implementations
  4. Parallel processing - Add multi-core support for CA and flux implementations

  Physics Completeness:

  5. Atmospheric circulation - Add wind patterns and moist convection to CA/flux
  6. Phase transitions - Implement proper melting/freezing with latent heat for all materials
  7. Erosion and sedimentation - Add surface processes for geological realism
  8. Variable material properties - Temperature-dependent viscosity, thermal conductivity

  Code Quality & Architecture:

  9. Unified physics interface - Create common API across CA, flux, and SPH for easier comparison
  10. Remove deprecated code - Clean up old CA implementation files marked as deprecated
  11. Consolidate duplicate functionality - Many similar calculations exist across implementations
  12. Numerical stability monitoring - Add automatic detection and recovery from instabilities

  Testing & Validation:

  13. Conservation law tests - Automated verification of mass, momentum, and energy conservation
  14. Physical validation suite - Compare against analytical solutions and experimental data
  15. Long-term stability tests - Extended runs to verify geological timescale behavior
  16. Cross-implementation comparison - Systematic comparison of CA vs flux vs SPH results

  Documentation:

  17. Parameter tuning guide - Document how to adjust parameters for different scenarios
  18. Material property reference - Comprehensive list with sources and validation
  19. Troubleshooting guide - Common issues and solutions for each implementation
  20. Implementation guides - Step-by-step instructions for adding new physics features

  Visualization & UI:

  21. 3D visualization option - Current 2D view limits understanding of some phenomena
  22. Real-time data plotting - Graphs of temperature, pressure, velocity profiles
  23. Interactive parameter adjustment - Modify simulation parameters during runtime
  24. Export capabilities - Save simulation data for external analysis

  Future Research Directions:

  25. Machine learning integration - Use ML for parameter optimization or surrogate modeling
  26. Multi-scale modeling - Couple different spatial/temporal scales
  27. Chemical reactions - Add mineralogy and chemical transformations
  28. Planetary formation scenarios - Initial conditions for accretion disk simulations
