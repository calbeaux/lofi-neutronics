**Building SyTH Library** `libsyTH.so`

cd ~/syth/build && rm -rf * && source ~/syth_tpls/load_env.sh && ~/syth/build_scripts/build_with_syth_env.sh ~/syth/build_scripts/build_petsc_release_omp_all.sh ~/syth -DSYTH_ENABLE_Fortran=OFF -DBUILD_SHARED_LIBS=ON && make -j


## Temperature-dependent MPACT inputs

Add this block to the SyTH input JSON's `neutronics` configuration before
generating its simulation `.h5` input:

```json
"neutronics": {
  "type": "point_kinetics",
  "initial_power": 1000000.0,
  "substeps": 1,
  "mpact": {
    "files": ["mpact/800.h5", "mpact/900.h5", "mpact/1000.h5"],
    "state": "STATE_0001"
  }
}
```

Change `type` to `diffusion` to use the same library with diffusion. File paths
are relative to the **simulation HDF5 file's directory**. Run the simulation as
usual with `python driver.py input_files/my_simulation.h5`. The simulation input
and MPACT output files serve different purposes; MPACT files do not replace the
SyTH mesh, fluid configuration, or state schedule.

Coupled runs require `"neutron_precursor_solve": true` in the SyTH input JSON's
`options` block. Regenerate the simulation HDF5 after changing this setting;
the native backend reads `/options/neutron_precursor_solve` from that file.
Lofi rejects missing or disabled precursor transport before initializing SyTH,
because the native delayed-neutron emissions getter requires its precursor manager.

To load a sweep directly without regenerating the simulation input, use:

```bash
python driver.py input_files/PSAR.h5 \
  --mpact-dir ~/Desktop/temperature_sweep \
  --mpact-temperature-source filename
```

This overrides the embedded `mpact` configuration. Effective beta is read from
the sum of `STATE_0001/CORE_XS/Beta` delayed groups and interpolated at each
diffusion node's temperature. A configured `beta_effective` is unnecessary when
an MPACT library is loaded. Without a library, diffusion requires an explicit
`beta_effective`, including an explicit zero for prompt-only calculations.

You can also load a library once and pass it directly:

```python
from Neutronics.mpact_library import MPACTLibrary
from Coupling.simulation import Simulation

library = MPACTLibrary.from_files(["800.h5", "900.h5", "1000.h5"])
xs = library.cross_sections(850.0)  # xs.fast and xs.th; SI units
pk = library.kinetics(850.0)
print(pk.beta_effective, pk.generation_time, pk.neutron_lifetime, pk.alpha_T)
print(library.profiles(850.0).power.relative_residual)

simulation = Simulation("my_simulation.h5", library=library)
simulation.set_up_data_container()
try:
    for index, state in enumerate(simulation.states):
        simulation.set_state(index, state)
        simulation.run()
finally:
    simulation._syth.close()
```

The library requires at least two distinct `CORE_XS/FUELTEMP` values in kelvin
for the same core by default. If that metadata is incorrect, set
`"temperature_source": "filename"` inside the `mpact` configuration. Filenames
must contain exactly one temperature token such as `msrr_T0700.h5` (700 K).
The reader uses the filename value without modifying the HDF5 file, and records
both the selected source and original metadata in `library.metadata`.
The equivalent Python call for the temperature sweep is:

```python
from pathlib import Path

library = MPACTLibrary.from_files(
    sorted(Path("~/Desktop/temperature_sweep").expanduser().glob("*.h5")),
    temperature_source="filename",
)
```

Cross sections and kinetics are interpolated linearly;
queries outside the supplied temperature range raise `ValueError`. `rho` is
interpolated from `1 - 1/keff` at each sample, and `alpha_T` is its interval slope
in reactivity/K (multiply by `1e5` for pcm/K). Interior knots use the right-hand
slope. Generation time and neutron lifetime are separate quantities, with
`neutron_lifetime = keff * generation_time`; PK uses generation time.

Required core edits include transport, removal, scattering, nu-fission,
fission and kappa-fission cross sections, velocities, prompt/delayed spectra,
flux, energy grid, `Beta`, and `Delayed Neutron Precursor Decay Constants`.
The reader supports two groups, negligible upscatter, and spectra `[1, 0]`.
It converts removal to absorption by subtracting intergroup outscatter using
MPACT's source-group/destination-group scattering layout. Geometry, energy
groups, and boundary metadata must agree across the files. The current diffusion
solver requires vacuum axial boundaries and omits radial leakage; it does
not reproduce a general 3D MPACT model's keff or temperature feedback.

`NODAL_XS` must also contain populated forward/adjoint flux, volume, axial mesh,
velocity, prompt/delayed spectra, and fission-production edits. Zero-volume
padding is excluded. The timing estimate is
`sum(V * adjoint * flux / velocity) /
sum(V * (adjoint dot chi) * (nu_sigma_f dot flux))`.
This is an estimate from spatially collapsed edits, not a transport timing tally.
Missing or zero adjoint data are rejected. Source paths and calculation provenance
are available in `library.metadata`.

PK refits nonnegative axial power and adjoint profiles for every sample using
sine harmonics 1, 3, and 5 plus an offset. Power comes from nodal kappa-fission
times flux; adjoint importance uses the delayed spectrum. Fits use axial cell
averages and volume weights. `profiles(T)` exposes both fits and their relative
residuals. The outermost planes with positive fission power define the fueled
extent, which maps by normalized axial position to the center channel's
bottom-to-top nodes (supplied order for legacy nodes without bounds).
The fit family is symmetric; asymmetric MPACT profiles can have larger
residuals, which should be checked for the intended core.

Initial thermal data select the profiles using volume-average temperature;
the profiles then remain fixed. Subsequent PK temperature averages use volume
and the frozen adjoint importance. Power integrates to `initial_power` in steady
state. Temperature feedback uses `rho(T) - rho(T_initial)` plus the precursor
balance offset, and reference neutron normalization remains fixed.

Diffusion uses node temperatures and kappa-fission power directly. The existing
SyTH global power-to-fission factor is updated from MPACT data: diffusion uses
the current flux-weighted total fission/power ratio, while PK uses the library's
center-channel spectrum at its current temperature. A single global factor cannot preserve
different nodewise energy-per-fission ratios. SyTH still owns precursor generation,
decay, and transport; the library exposes `beta_groups` and `decay_constants` but
does not overwrite SyTH's precursor model. Ensure those models are consistent
before interpreting coupled transients.

Inputs without `mpact` retain the existing formulas and constant PK parameters.
Run the portable validation suite with `PYTHONPATH=. pytest -q`; it uses synthetic
HDF5 inputs and mocks SyTH, so no native SyTH library or Desktop files are needed.

## Multichannel cores

Both neutronics strategies accept a `Coupling.core_geometry.CoreGeometry` or an
existing list of `Node` objects. Geometry groups coupled nodes into vertical
channels using their bounding-box transverse centers, normalizes reversed box
corners, and orders cells bottom-to-top. The center channel is nearest the
midpoint of the transverse channel extents; ties use the lowest node index.
Nodes without bounds retain their supplied order as one legacy channel. Bounds
must be supplied for every node or none. Channels must have contiguous, matching
axial boundaries; diffusion additionally requires at least three uniformly
spaced axial cells. Unequal axial grids are rejected rather than interpolated.

Diffusion solves only the center channel's two-group axial problem. For PSAR,
this is 16 cells (32 unknowns), using nodes 1189–1204 in the current generated
mesh. Cross sections, beta, temperatures, and delayed-neutron sources come from
that channel. PK similarly builds its existing fixed axial power and adjoint
profiles on the center channel and uses only its temperature and precursor
feedback. Changes in outer-channel temperatures or emissions do not directly
affect either neutronics solver in this approximation.

The axial power density in W/m³ is repeated in every channel. Total power is
the sum of power density times actual liquid volume over all coupled nodes;
channels with different volumes therefore receive different integrated powers.
Diffusion normalizes this full-core sum to `initial_power` in steady state and
lets it evolve during transients. PK distributes `amplitude * initial_power`
over the full core at every update. Its reference neutron population uses the
center channel's share of initial power, calculated from the axial shape and
volumes (1/148 for PSAR). That reference remains fixed during transients, so
center-channel precursor emissions and population use the same normalization.

`core_volume` reports the full coupled volume; solver temperature diagnostics
report the representative center-channel temperature. SyTH still transports
precursors and receives power at every coupled node's original index. The
diffusion power-to-fission factor integrates the representative spectrum using
full-core axial volumes. No radial leakage or radial power variation is modeled.

`set_core_nodes(...)` rebuilds geometry and resets solver histories and PK
reference profiles/population. Select a solver method again after replacement,
and rewire data arrays if the new node indices exceed their allocated size.
Initial SyTH temperatures and emissions are received before the first
neutronics update in both legacy and MPACT modes.

PSAR's existing input still needs an MPACT library (for example the documented
`--mpact-dir` option) or valid legacy neutronics parameters. Geometry support
does not supply missing cross sections or kinetics parameters.

## Coupled steady-state iterations

Each steady state now repeats the thermal-hydraulic solve, precursor solve, and
neutronics update until the coupled fields settle. At least two outer passes
are required. The default limit is 50 passes, with a relative tolerance of
`1e-6` for power density, temperature, and delayed-neutron source over the
coupled nodes. Each change is measured using the L2 norm, scaled by the larger
of the old and new field norms; unchanged zero fields have zero error.

These settings can be overridden under `neutronics` before generating HDF5:

```json
"steady_state": {
  "max_iterations": 50,
  "tolerance": 1e-6
}
```

The defaults also apply to existing HDF5 inputs without this block. These are
outer coupling iterations; each pass still performs SyTH's own fluid and
precursor convergence solves. The final power is sent to SyTH on every pass,
including legacy runs. The solution is written once after outer convergence;
reaching the limit raises an error before advancing to the transient.

This loop addresses thermal/precursor consistency with the updated power shape.
It does not change SyTH's state-captured precursor conversion factor: the native
state must still be created with the correct factor, or the native setter must
update the active source, to prevent a factor jump at the transient transition.

### MPACT-calibrated radial buckling

Set `"radial_buckling": true` in the `neutronics` object of a diffusion input
(the PSAR diffusion JSON enables this). The option defaults to `false` and
requires an MPACT temperature library. Regenerate the input HDF5 from the JSON
before running; existing HDF5 results retain their embedded configuration.

Diffusion fits one nonnegative radial buckling squared, in m⁻², at each MPACT
sample temperature so its static total-fission eigenvalue matches MPACT's
`keff`. Calibration uses the current axial mesh and vacuum boundaries, without
transported precursor inventory. One B² applies to both energy groups; each
cell adds `D_g(T) * B²(T)` as a separate diagonal leakage loss (m⁻¹), preserving
physical absorption and axial boundary conditions. The solver linearly
interpolates B² at the local cell temperature and rejects temperatures outside
the library range. Calibration fails if a sample requires negative buckling.

The geometry-specific table is cached by `Diffusion` and invalidated when core
nodes change. `diffusion.buckling_table` returns a copy of its `[T, B²]` rows;
`diffusion.static_eigenvalue(temperatures)` evaluates the calibrated static
model. Passing `buckling_squared=0` explicitly evaluates the uncorrected model.
Sample eigenvalues match exactly within solver tolerance; interpolating B²
need not match interpolated MPACT reactivity exactly between samples. This
correction represents radial leakage, not a radial power shape.

Run `python scripts/compare_temperature_feedback.py` to export the buckling
table, sample matching errors, and corrected temperature-feedback plots under
`results/temperature_feedback/`. This analysis does not alter saved HDF5 runs.
