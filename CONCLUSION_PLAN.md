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

2) Simplify the ca/ buoyancy and cell swapping behavior
- Do not solve for pressure and velocity
- Use a simple swapping rule based on density differences and the gravity vector (swap away from the gravity vector)
- Add a probabilty to swap to each material
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
1) TODO