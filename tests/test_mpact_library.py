"""MPACT input and solver checks independent of the native SyTH library."""

import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from Coupling.data_container import DataContainer
from Coupling.node import Node
from Coupling.state import State
from Neutronics.diffusion import Diffusion
from Neutronics.mpact_library import MPACTLibrary, _basis
from Neutronics.point_kinetics import PointKineticsStrategy, PointKineticsTransientStrategy


def make_sample(path, temperature, scale=1.0, flux_scale=1.0, adjoint_scale=1.0):
    """Known two-group material and positive profiles, with invalid padding."""
    with h5py.File(path, "w") as f:
        for boundary in ("bot", "top", "rad"):
            f[f"CORE/bc_{boundary}"] = np.bytes_("vacuum")
        s = f.create_group("STATE_0001")
        s["keff"] = 1.05 - (temperature - 800)*1e-4
        c = s.create_group("CORE_XS")
        values = dict(FUELTEMP=temperature, XSTR=[.3, .4], XSRM=[.025, .03],
                      NXSF=[.005, .04], XSF=[.002, .016], KXSF=[6e-14, 4.8e-13],
                      VELOCITY=[3e7, 6e5], CHI=[1., 0.], CHID=[1., 0.],
                      XSS=[[.2, .02], [0., .3]], Beta=[.002, .004],
                      energy_grid=[2e7, 5.], FLUX=[2., 1.])
        for key, value in values.items():
            a = np.asarray(value)
            if key in {"XSTR", "XSRM", "NXSF", "XSF", "KXSF", "XSS", "Beta"}:
                a = a*scale
            c[key] = a
        c["Delayed Neutron Precursor Decay Constants"] = [.01, .1]
        n = s.create_group("NODAL_XS")
        planes = 8
        n["AXIALMESH"] = np.linspace(0, 160, planes+1)
        volume = np.zeros((2, planes, 1))
        volume[0, :, 0] = 100
        n["VOL"] = volume
        centers = (np.arange(planes)+.5)/planes
        power_shape = 1 + .5*np.sin(np.pi*centers)
        adj_shape = 1 + .25*np.sin(np.pi*centers)
        for key in ("FLUX", "ADJOINT FLUX", "VELOCITY", "NXSF", "XSF", "KXSF", "CHI", "CHID"):
            a = np.full((2, 2, planes, 1), np.nan)
            for g in range(2):
                if key == "FLUX":
                    row = values[key][g] * power_shape * flux_scale
                elif key == "ADJOINT FLUX":
                    row = (g+1) * adj_shape * adjoint_scale
                else:
                    row = np.full(planes, c[key][()][g])
                a[g, 0, :, 0] = row
            n[key] = a
    return path


@pytest.fixture
def samples(tmp_path):
    return [make_sample(tmp_path/"cold.h5", 800),
            make_sample(tmp_path/"hot.h5", 1000, scale=1.1)]


@pytest.fixture
def library(samples):
    return MPACTLibrary.from_files(samples[::-1])


def nodes():
    return [Node(.2, .01, 1), Node(.4, .03, 3), Node(.4, .02, 6)]


def diffusion_nodes():
    # The axial diffusion stencil requires uniform spacing; PK also tests
    # nonuniform cells through nodes().
    return [Node(.4, n.volume, n.index) for n in nodes()]


def wire(strategy, temperature=900):
    container = DataContainer(7)
    container.register("temperature", temperature)
    container.register("precursor_source", 0.)
    strategy.register_data_container_variables(container)
    strategy.set_up_data_container_interactions(container)
    return container


def test_units_scattering_and_interpolation(library):
    xs = library.cross_sections(900)
    assert xs.fast.tr == pytest.approx(.3*1.05*100)
    assert xs.fast.a == pytest.approx((.025-.02)*1.05*100)
    assert xs.fast.s_out == pytest.approx(.02*1.05*100)
    assert xs.th.a == pytest.approx(.03*1.05*100)
    assert xs.th.s_out == 0
    assert xs.fast.velocity == 3e5
    assert xs.fast.kappa_f == pytest.approx(6e-12*1.05)
    assert xs.th.f == pytest.approx(.016*100*1.05)
    assert xs.fast.d_coef == pytest.approx(1/(3*xs.fast.tr))
    np.testing.assert_array_equal(library.temperatures, [800, 1000])
    assert library.metadata["state"] == "STATE_0001"


def test_kinetics_and_right_hand_slopes(samples, tmp_path):
    third = make_sample(tmp_path/"third.h5", 1100)
    library = MPACTLibrary.from_files(samples + [third])
    cold, middle, hot = [library.kinetics(t) for t in [800, 1000, 1100]]
    assert cold.rho == pytest.approx(1-1/1.05)
    assert library.kinetics(900).rho == pytest.approx((cold.rho+middle.rho)/2)
    assert cold.alpha_T == pytest.approx((middle.rho-cold.rho)/200)
    assert middle.alpha_T == pytest.approx((hot.rho-middle.rho)/100)
    assert hot.alpha_T == middle.alpha_T
    k = library.kinetics(900)
    assert k.beta_effective == pytest.approx(.006*1.05)
    np.testing.assert_allclose(k.decay_constants, [.01, .1])
    assert k.neutron_lifetime == pytest.approx(k.keff*k.generation_time)
    assert k.fissions_per_joule == pytest.approx(1/3e-11)


def test_spatial_timing_and_normalization_invariance(tmp_path):
    a = make_sample(tmp_path/"a.h5", 800)
    b = make_sample(tmp_path/"b.h5", 1000, flux_scale=7, adjoint_scale=13)
    lib = MPACTLibrary.from_files([a, b])
    # Nodal adjoint has energy weights [1,2], scalar flux [2,1].
    expected = (2/3e7 + 2/6e5) / (.005*2+.04)
    assert lib.kinetics(800).generation_time == pytest.approx(expected)
    assert lib.kinetics(1000).generation_time == pytest.approx(expected)
    assert lib.kinetics(800).neutrons_per_joule == pytest.approx(2.5/3e-11)


@pytest.mark.parametrize("temperature", [799, 1001, np.nan, np.inf])
def test_out_of_range(library, temperature):
    for method in (library.cross_sections, library.kinetics, library.profiles):
        with pytest.raises(ValueError, match="outside MPACT range"):
            method(temperature)


@pytest.mark.parametrize("change,match", [
    ("duplicate", "Duplicate"), ("grid", "energy grids"),
    ("mesh", "core meshes"), ("boundary", "boundaries"),
    ("upscatter", "upscatter"), ("absorption", "Negative absorption"),
    ("adjoint", "normalization"), ("velocity", "velocities"),
    ("missing", "Missing MPACT dataset"), ("spectrum", "CHI="),
    ("groups", "delayed-neutron"),
])
def test_bad_libraries(samples, change, match):
    with h5py.File(samples[1], "r+") as f:
        c = f["STATE_0001/CORE_XS"]
        n = f["STATE_0001/NODAL_XS"]
        if change == "duplicate": c["FUELTEMP"][...] = 800
        elif change == "grid": c["energy_grid"][...] = [2e7, 6]
        elif change == "mesh": n["AXIALMESH"][...] = np.linspace(0, 170, 9)
        elif change == "boundary": f["CORE/bc_rad"][...] = np.bytes_("reflecting")
        elif change == "upscatter": c["XSS"][1, 0] = .001
        elif change == "absorption": c["XSRM"][0] = .001
        elif change == "adjoint": n["ADJOINT FLUX"][...] = 0
        elif change == "velocity": n["VELOCITY"][0, 0, 0, 0] = 0
        elif change == "missing": del c["XSF"]
        elif change == "spectrum": c["CHI"][...] = [.9, .1]
        elif change == "groups": c["Beta"][...] = [.5, .5]
    with pytest.raises(ValueError, match=match):
        MPACTLibrary.from_files(samples)


def test_single_sample_rejected(samples):
    with pytest.raises(ValueError, match="at least two"):
        MPACTLibrary.from_files(samples[:1])


def test_filename_temperatures_override_wrong_metadata(tmp_path):
    cold = make_sample(tmp_path/"msrr_T0700.h5", 500)
    hot = make_sample(tmp_path/"msrr_T1200.h5", 850)
    lib = MPACTLibrary.from_files([hot, cold], temperature_source="filename")
    np.testing.assert_array_equal(lib.temperatures, [700, 1200])
    assert lib.metadata["temperature_source"] == "filename"
    assert lib.metadata["recorded_temperatures"] == [500, 850]
    assert lib.metadata["sources"] == [str(cold), str(hot)]
    cold_rho, hot_rho = [lib.kinetics(t).rho for t in (700, 1200)]
    assert lib.kinetics(950).rho == pytest.approx((cold_rho+hot_rho)/2)
    assert lib.kinetics(950).alpha_T == pytest.approx((hot_rho-cold_rho)/500)
    # The explicit option does not alter the file or the default reader behavior.
    assert MPACTLibrary.from_files([cold, hot]).temperatures.tolist() == [500, 850]
    with h5py.File(cold) as f:
        assert f["STATE_0001/CORE_XS/FUELTEMP"][()] == 500


@pytest.mark.parametrize("name", ["msrr_700.h5", "msrr_T0000.h5", "msrr_T0700_T0800.h5"])
def test_filename_temperature_errors(tmp_path, name):
    path = make_sample(tmp_path/name, 800)
    with pytest.raises(ValueError, match="filename"):
        MPACTLibrary.from_files([path], temperature_source="filename")


def test_filename_source_ignores_missing_temperature_metadata(tmp_path):
    paths = [make_sample(tmp_path/"msrr_T0700.h5", 800),
             make_sample(tmp_path/"msrr_T0756.h5", 800)]
    with h5py.File(paths[0], "r+") as f:
        del f["STATE_0001/CORE_XS/FUELTEMP"]
    lib = MPACTLibrary.from_files(paths, temperature_source="filename")
    assert lib.temperatures.tolist() == [700, 756]
    assert lib.metadata["recorded_temperatures"] == [None, 800]
    with pytest.raises(ValueError, match="temperature_source"):
        MPACTLibrary.from_files(paths, temperature_source="auto")


def test_diffusion_rejects_unsupported_mesh_and_axial_boundaries(samples, library):
    with pytest.raises(ValueError, match="three core nodes"):
        Diffusion(diffusion_nodes()[:2], 1e6, library=library)
    for path in samples:
        with h5py.File(path, "r+") as f:
            del f["CORE/bc_top"]
            f["CORE/bc_top"] = np.bytes_("reflecting")
    reflective = MPACTLibrary.from_files(samples)
    with pytest.raises(ValueError, match="vacuum axial"):
        Diffusion(diffusion_nodes(), 1e6, library=reflective)
    # PK estimates can still be queried for a reflecting-axial MPACT model.
    assert reflective.kinetics(900).generation_time > 0
    json.dumps(reflective.metadata)


def test_profiles_nonnegative_cell_averages_and_conservation(library):
    profiles = library.profiles(900)
    for fit in [profiles.power, profiles.adjoint]:
        assert np.min(_basis(np.linspace(0, 1, 10001)) @ fit.coefficients) >= -1e-12
        assert fit.relative_residual < .02
        fine = fit.cell_averages(np.linspace(0, 1, 101))
        assert fine.mean() == pytest.approx(fit.cell_averages([0, 1])[0])
    power, adjoint = profiles.map_nodes(nodes())
    volumes = np.array([n.volume for n in nodes()])
    assert power @ volumes == pytest.approx(1)
    assert adjoint @ (power*volumes) == pytest.approx(1)


def test_profiles_are_refitted_for_each_run(samples):
    with h5py.File(samples[1], "r+") as f:
        g = f["STATE_0001/NODAL_XS"]
        modulation = 1 + .4*np.sin(3*np.pi*(np.arange(8)+.5)/8)
        for key in ("FLUX", "ADJOINT FLUX"):
            values = g[key][()]
            values[:, 0, :, 0] *= modulation
            g[key][...] = values
    lib = MPACTLibrary.from_files(samples)
    for name in ("power", "adjoint"):
        cold = getattr(lib.profiles(800), name).coefficients
        hot = getattr(lib.profiles(1000), name).coefficients
        assert not np.allclose(cold, hot)
        np.testing.assert_allclose(getattr(lib.profiles(900), name).coefficients, (cold+hot)/2)


def test_queries_do_not_reopen_files(library, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("A query attempted to reopen an MPACT file")
    monkeypatch.setattr(h5py, "File", fail)
    library.cross_sections(850)
    library.kinetics(850)
    library.profiles(850)


def test_pk_updates_feedback_freezes_profiles_and_population(library):
    pk = PointKineticsStrategy(nodes(), initial_power=1e6, library=library)
    data = wire(pk)
    pk.set_solver_method(State("steadystate"))
    pk.update_power()
    shape, adj, population = pk._flux_shape.copy(), pk._adjoint.copy(), pk._n_0
    assert sum(data["power"][n.index]*n.volume for n in nodes()) == pytest.approx(1e6)
    rho0 = pk.get_reactivity_feedback()
    pk.set_solver_method(SimpleNamespace(state_type="transient", time_step=.001))
    data["temperature"][:] = 950
    pk.update_power()
    assert pk.beta_effective == pytest.approx(library.kinetics(950).beta_effective)
    assert pk.generation_time == pytest.approx(library.kinetics(950).generation_time)
    assert pk.mean_neutron_lifetime == pytest.approx(library.kinetics(950).neutron_lifetime)
    assert pk.get_reactivity_feedback()-rho0 == pytest.approx(
        library.kinetics(950).rho-library.kinetics(900).rho)
    np.testing.assert_array_equal(pk._flux_shape, shape)
    np.testing.assert_array_equal(pk._adjoint, adj)
    assert pk._n_0 == population


def test_pk_temperature_weights_and_equilibrium(library):
    pk = PointKineticsStrategy(nodes(), initial_power=1e6, library=library)
    data = wire(pk)
    data["temperature"][[1, 3, 6]] = [820, 900, 980]
    pk.initialize_reference()
    expected = np.average([820, 900, 980], weights=pk._adjoint*np.array([.01, .03, .02]))
    assert pk.get_core_temperature() == pytest.approx(expected)
    data["precursor_source"][:] = 1e10
    pk.set_solver_method(State("steadystate"))
    pk.update_power()
    pk.set_solver_method(SimpleNamespace(state_type="transient", time_step=.01))
    pk.update_power()
    assert pk.amplitude == pytest.approx(1)


def test_pk_balances_after_steady_state_temperature_changes(library):
    pk = PointKineticsStrategy(nodes(), initial_power=1e6, library=library)
    data = wire(pk)
    pk.initialize_reference()
    reference_rho = pk._reference_rho
    population = pk._n_0
    shape = pk._flux_shape.copy()
    adjoint = pk._adjoint.copy()

    # Thermal convergence and precursor transport change both fields after
    # the reference was initialized. Repeated calibration must not accumulate.
    for temperature, source in [(950, 1e10), (970, 2e10)]:
        data["temperature"][:] = temperature
        data["precursor_source"][:] = source
        pk.set_solver_method(State("steadystate"))
        pk.update_power()
        derivative = ((pk.get_reactivity_feedback() - pk.beta_effective)
                      / pk.generation_time + pk.get_core_precursor_sum())
        assert derivative == pytest.approx(0, abs=1e-10)
        pk.set_solver_method(SimpleNamespace(state_type="transient", time_step=.05))
        for _ in range(3):
            pk.update_power()
            assert pk.amplitude == pytest.approx(1)

    # Calibration stays fixed during the actual transient: feedback survives.
    rho = pk.get_reactivity_feedback()
    data["temperature"][:] = 980
    pk.update_power()
    assert pk.get_reactivity_feedback() - rho == pytest.approx(
        library.kinetics(980).rho - library.kinetics(970).rho)
    assert pk.amplitude < 1
    assert pk._reference_rho == reference_rho
    assert pk._n_0 == population
    np.testing.assert_array_equal(pk._flux_shape, shape)
    np.testing.assert_array_equal(pk._adjoint, adjoint)


def test_zero_growth_rate(library):
    pk = PointKineticsStrategy(nodes(), initial_power=1e6, library=library)
    data = wire(pk)
    pk.initialize_reference()
    pk.set_initial_reactivity(pk.beta_effective)
    data["precursor_source"][:] = 1e10
    solver = PointKineticsTransientStrategy(pk, .1)
    assert solver.get_new_amplitude() == pytest.approx(1+.1*pk.get_core_precursor_sum())


def test_diffusion_local_beta_and_energy_power(library):
    diffusion = Diffusion(diffusion_nodes(), 1e6, library=library)
    data = wire(diffusion)
    data["temperature"][[1, 3, 6]] = [800, 900, 1000]
    _, _, fission = diffusion.set_up_matrix()
    for i, node in enumerate(diffusion_nodes()):
        t = data["temperature"][node.index]
        xs, k = library.cross_sections(t), library.kinetics(t)
        assert fission[i] == pytest.approx(xs.fast.nu_f*node.length*(1-k.beta_effective))
    diffusion.set_solver_method(State("steadystate"))
    diffusion.update_power()
    assert sum(data["power"][n.index]*n.volume for n in diffusion_nodes()) == pytest.approx(1e6)
    assert diffusion.fission_power_factor() == pytest.approx(1/3e-11)
    for i, node in enumerate(diffusion_nodes()):
        xs = diffusion.get_xs(node)
        assert data["power"][node.index] == pytest.approx(
            xs.fast.kappa_f*diffusion.flux[i] + xs.th.kappa_f*diffusion.flux[i+3])
    data["temperature"][1] = 1001
    with pytest.raises(ValueError, match="outside"):
        diffusion.update_power()


@pytest.mark.parametrize("beta", [None, np.nan, -0.01, 1.0])
def test_diffusion_requires_valid_beta_without_library(beta):
    with pytest.raises(ValueError, match="beta_effective"):
        Diffusion(diffusion_nodes(), 1e6, beta=beta)


@pytest.mark.parametrize("source", ["directory", "preloaded"])
def test_driver_passes_mpact_library_to_simulation(samples, monkeypatch, source):
    import driver

    received = {}
    class FakeSimulation:
        def __init__(self, input_file, verbose, library):
            received["library"] = library
            self.states = []
            self._syth = SimpleNamespace(close=lambda: received.update(closed=True))
        def set_up_data_container(self): pass

    monkeypatch.setattr(driver, "Simulation", FakeSimulation)
    if source == "directory":
        # Use the filename override, including when both stored temperatures agree.
        cold = make_sample(samples[0].parent/"msrr_T0800.h5", 750)
        hot = make_sample(samples[0].parent/"msrr_T1000.h5", 750)
        for path in samples:
            path.unlink()
        driver._run_simulation("unused.h5", verbose=False, mpact_dir=cold.parent,
                               temperature_source="filename")
        assert received["library"].temperatures.tolist() == [800, 1000]
    else:
        library = MPACTLibrary.from_files(samples)
        driver._run_simulation("unused.h5", verbose=False, library=library)
        assert received["library"] is library
    assert received["library"].kinetics(900).beta_effective > 0
    assert received["closed"]


def test_diffusion_unchanged_temperature_equilibrium(library):
    diffusion = Diffusion(diffusion_nodes(), 1e6, library=library)
    data = wire(diffusion)
    diffusion.set_solver_method(State("steadystate"))
    diffusion.update_power()
    initial = diffusion.flux.copy()
    diffusion.set_solver_method(SimpleNamespace(state_type="transient", time_step=.001, scheme="BDF1"))
    diffusion.update_power()
    np.testing.assert_allclose(diffusion.flux, initial, rtol=1e-7)
    assert sum(data["power"][n.index]*n.volume for n in nodes()) == pytest.approx(1e6)


def test_legacy_pk_keeps_constant_parameters_and_profiles():
    pk = PointKineticsStrategy(nodes(), 2e-4, .006, -1e-5, 1e6, 1)
    data = wire(pk)
    pk.set_solver_method(State("steadystate"))
    pk.update_power()
    rho0 = pk.get_reactivity_feedback()
    data["temperature"][:] = 950
    assert pk.get_reactivity_feedback()-rho0 == pytest.approx(-1e-5*50)
    assert pk.generation_time == 2e-4
    assert pk.beta_effective == .006


@pytest.mark.parametrize("strategy", ["pk", "diffusion"])
@pytest.mark.parametrize("temperature_source", ["dataset", "filename"])
def test_simulation_config_and_initial_thermal_data(samples, monkeypatch, tmp_path, strategy, temperature_source):
    from Coupling.simulation import Simulation

    class FakeSyth:
        def __init__(self, filename): self.events = []
        def set_state(self, index): self.events.append(("state", index))
        def get_fluid_node_value(self, name, index):
            self.events.append(("receive", index))
            return 900.
        def get_delayed_neutron_emissions(self, index): return 0.
        def set_precursor_source_per_power(self, value):
            self.events.append(("factor", value))
        def set_node_power(self, index, value): self.events.append(("power", index, value))
        def solve_fluid_flow(self, **kwargs): self.events.append(("flow",))
        def solve_mass_transport(self, **kwargs): self.events.append(("transport",))
        def write_solution_to_input_file(self): self.events.append(("write",))
        def get_node_count(self): return 12

    monkeypatch.setitem(sys.modules, "SyTH.syth", SimpleNamespace(Syth=FakeSyth))
    if temperature_source == "filename":
        samples = [make_sample(tmp_path/"msrr_T0800.h5", 750),
                   make_sample(tmp_path/"msrr_T1000.h5", 950)]
    config = {"neutronics": {"type": strategy, "initial_power": 1e6,
                              "mpact": {"files": [p.name for p in samples],
                                        "temperature_source": temperature_source}}}
    path = tmp_path/"simulation.h5"
    with h5py.File(path, "w") as f:
        f["metadata/input_file"] = np.bytes_(json.dumps(config))
        f["options/neutron_precursor_solve"] = True
        # Deliberately use indices whose lexical order differs from numeric order.
        for index in [2, 10, 11]:
            g = f.create_group(f"fluid_solve/fluid_mesh/nodes/node_{index}")
            g["tag"] = np.bytes_("core")
            g["is_coupled"] = True
            g["length"], g["volume"] = .2, .01
    sim = Simulation(path)
    assert [n.index for n in sim._core_nodes] == [2, 10, 11]
    sim.set_up_data_container()
    assert sim._temperature[2] == 900
    assert sim._syth.events[-1][0] == "factor"
    assert sim._syth.events[-1][1] == pytest.approx(1/3e-11)
    sim.set_state(0, State("steadystate"))
    sim.run()
    assert sum(sim._power[n.index]*n.volume for n in sim._core_nodes) == pytest.approx(1e6)
    if strategy == "pk":
        assert sim._neutronics._t0 == 900
    sim.set_state(1, State({"time_duration": .03, "time_step": .01, "scheme": "BDF1"}))
    sim.run()
    assert sum(sim._power[n.index]*n.volume for n in sim._core_nodes) == pytest.approx(1e6)
    assert ("transport",) in sim._syth.events
    # A preloaded library takes precedence over configured filenames.
    config["neutronics"]["mpact"]["files"] = ["does-not-exist.h5"]
    with h5py.File(path, "r+") as f:
        del f["metadata/input_file"]
        f["metadata/input_file"] = np.bytes_(json.dumps(config))
    preloaded = Simulation(path, library=sim.library)
    assert preloaded.library is sim.library


@pytest.mark.parametrize("enabled", [None, False])
def test_simulation_rejects_disabled_precursors_before_native_initialization(tmp_path, monkeypatch, enabled):
    from Coupling.simulation import Simulation

    def unexpected_native_init(*args, **kwargs):
        pytest.fail("Native SyTH must not initialize with disabled precursor transport")
    monkeypatch.setitem(sys.modules, "SyTH.syth", SimpleNamespace(Syth=unexpected_native_init))
    path = tmp_path/"disabled_precursors.h5"
    with h5py.File(path, "w") as f:
        # The actual HDF5 option takes precedence over stale embedded JSON.
        f["metadata/input_file"] = np.bytes_(json.dumps({"options": {"neutron_precursor_solve": True}}))
        f.create_group("fluid_solve/fluid_mesh/nodes")
        if enabled is not None:
            f["options/neutron_precursor_solve"] = enabled
    with pytest.raises(ValueError, match="neutron_precursor_solve=true"):
        Simulation(path)


@pytest.fixture
def buckling_library(library):
    """Samples with known, independently prescribed radial leakage."""
    raw = Diffusion(diffusion_nodes(), 1e6, library=library)
    for sample, b2 in zip(library._samples, [8.0, 12.0]):
        k = raw.static_eigenvalue(sample['temperature'], buckling_squared=b2)
        sample['keff'] = k
        sample['rho'] = 1 - 1/k
    return library


def test_buckling_fit_and_interpolation(buckling_library):
    lib = buckling_library
    d = Diffusion(diffusion_nodes(), 1e6, library=lib, radial_buckling=True)
    np.testing.assert_allclose(d.buckling_table[:, 1], [8, 12], atol=1e-8)
    for temperature in lib.temperatures:
        assert d.static_eigenvalue(temperature) == pytest.approx(
            lib.kinetics(temperature).keff, abs=1e-8)
    assert d.buckling_squared(900) == pytest.approx(10)
    with pytest.raises(ValueError, match='outside MPACT range'):
        d.buckling_squared(799)
    table = d.buckling_table
    table[:] = 0
    assert d.buckling_squared(900) == pytest.approx(10)


def test_buckling_changes_only_local_diagonal_losses(buckling_library):
    lib = buckling_library
    raw = Diffusion(diffusion_nodes(), 1e6, library=lib)
    fitted = Diffusion(diffusion_nodes(), 1e6, library=lib, radial_buckling=True)
    a, b = wire(raw), wire(fitted)
    a['temperature'][[1, 3, 6]] = [820, 900, 980]
    b['temperature'][:] = a['temperature']
    a['precursor_source'][:] = 1e10
    b['precursor_source'][:] = a['precursor_source']
    A0, source0, f0 = raw.set_up_matrix()
    A1, source1, f1 = fitted.set_up_matrix()
    expected = []
    for group in ['fast', 'th']:
        for node in fitted._axial_nodes:
            t = a['temperature'][node.index]
            xs = getattr(lib.cross_sections(t), group)
            expected.append(node.length * xs.d_coef * (8 + (t-800)*.02))
    np.testing.assert_allclose(A1-A0, np.diag(expected), atol=1e-14)
    np.testing.assert_array_equal(source0, source1)
    np.testing.assert_array_equal(f0, f1)
    for node in fitted._axial_nodes:
        x = fitted.get_xs(node)
        y = raw.get_xs(node)
        for group in ['fast', 'th']:
            assert getattr(x, group).a == getattr(y, group).a
            assert getattr(x, group).extrapolation_distance == getattr(y, group).extrapolation_distance


def test_buckling_cache_and_geometry_invalidation(buckling_library, monkeypatch):
    d = Diffusion(diffusion_nodes(), 1e6, library=buckling_library, radial_buckling=True)
    old = d.buckling_table
    with monkeypatch.context() as m:
        m.setattr(d, '_calibrate_buckling', lambda: pytest.fail('recalibrated cached table'))
        d.buckling_squared(850)
        d.static_eigenvalue(950)
    d.set_core_nodes([Node(.2, .01, i) for i in range(6)])
    assert d._buckling_table is None
    assert not np.allclose(d.buckling_table[:, 1], old[:, 1])


def test_buckling_disabled_and_unattainable_target(library):
    raw = Diffusion(diffusion_nodes(), 1e6, library=library)
    assert raw.buckling_table is None
    assert raw.buckling_squared(900) == 0
    assert raw.static_eigenvalue(900) == raw.static_eigenvalue(900, buckling_squared=0)
    for sample in library._samples:
        k = raw.static_eigenvalue(sample['temperature'], buckling_squared=0) * 1.1
        sample['keff'] = k
        sample['rho'] = 1-1/k
    fitted = Diffusion(diffusion_nodes(), 1e6, library=library, radial_buckling=True)
    with pytest.raises(ValueError, match='800 K, target keff='):
        fitted.buckling_table


def test_buckling_requires_library_and_boolean():
    with pytest.raises(ValueError, match='requires an MPACT library'):
        Diffusion(diffusion_nodes(), 1e6, beta=.006, radial_buckling=True)
    with pytest.raises(ValueError, match='must be a boolean'):
        Diffusion(diffusion_nodes(), 1e6, beta=.006, radial_buckling='true')


def test_buckling_input_option(buckling_library):
    from Coupling.simulation import Simulation
    config = {'neutronics': {'type': 'diffusion', 'initial_power': 1e6,
                             'radial_buckling': True}}
    d = Simulation._build_neutronics_strategy(config, diffusion_nodes(), library=buckling_library)
    assert d.radial_buckling is True
    assert d.buckling_squared(900) == pytest.approx(10)


def test_zero_fitted_buckling_preserves_runtime_matrix(library):
    raw = Diffusion(diffusion_nodes(), 1e6, library=library)
    for sample in library._samples:
        k = raw.static_eigenvalue(sample['temperature'], buckling_squared=0)
        sample['keff'] = k
        sample['rho'] = 1-1/k
    fitted = Diffusion(diffusion_nodes(), 1e6, library=library, radial_buckling=True)
    a, b = wire(raw), wire(fitted)
    np.testing.assert_array_equal(fitted.buckling_table[:, 1], 0)
    for x, y in zip(raw.set_up_matrix(), fitted.set_up_matrix()):
        np.testing.assert_array_equal(x, y)


def test_buckling_static_calibration_does_not_change_transient_fields(buckling_library):
    d = Diffusion(diffusion_nodes(), 1e6, library=buckling_library, radial_buckling=True)
    data = wire(d)
    data['temperature'][[1, 3, 6]] = [820, 900, 980]
    data['precursor_source'][:] = 1e10
    temperature = data['temperature'].copy()
    flux = d.flux.copy()
    k = d.static_eigenvalue(900)
    np.testing.assert_array_equal(data['temperature'], temperature)
    np.testing.assert_array_equal(d.flux, flux)
    data['precursor_source'][:] = 0
    assert d.static_eigenvalue(900) == k
