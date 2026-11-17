"""Representative-channel neutronics, independent of the native SyTH backend."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from Coupling.core_geometry import CoreGeometry
from Coupling.data_container import DataContainer
from Coupling.node import Node
from Coupling.simulation import Simulation
from Coupling.state import State
from Neutronics.diffusion import Diffusion
from Neutronics.mpact_library import MPACTLibrary
from Neutronics.point_kinetics import PointKineticsStrategy
from test_mpact_library import make_sample


@pytest.fixture
def library(tmp_path):
    return MPACTLibrary.from_files([
        make_sample(tmp_path / 'cold.h5', 800),
        make_sample(tmp_path / 'hot.h5', 1000, scale=1.1)])


def mesh(volumes=(1., 1., 1.), offset=(0., 0., 0.)):
    nodes = []
    for channel, (x, volume) in enumerate(zip((-1., 0., 1.), volumes)):
        for axial in range(4):
            lo = np.array([x - .1, -.1, axial * .2]) + offset
            hi = lo + [.2, .2, .2]
            bounds = np.r_[lo, hi] if axial % 2 else np.r_[hi, lo]
            nodes.append(Node(.2, .01 * volume, 3 + channel * 30 + axial * 2, tuple(bounds)))
    return nodes


def wire(solver):
    data = DataContainer(max(n.index for n in solver.core.nodes) + 1)
    data.register('temperature', 900.)
    data.register('precursor_source', 0.)
    solver.register_data_container_variables(data)
    solver.set_up_data_container_interactions(data)
    return data


def build(kind, nodes, power, library):
    if kind == 'diffusion':
        return Diffusion(nodes, power, beta=.006, library=library)
    return PointKineticsStrategy(nodes, 2e-4, .006, -1e-5, power, library=library)


def integrated(solver, data):
    return sum(data['power'][n.index] * n.volume for n in solver.core.nodes)


def test_geometry_order_bounds_translation_and_ties():
    nodes = mesh(offset=(12., -5., 2.))
    core = CoreGeometry(nodes[::-1])
    assert len(core.channels) == 3
    assert [n.index for n in core.center_channel] == [33, 35, 37, 39]
    assert core.volume == pytest.approx(.12)
    for channel in core.channels:
        assert [core.axial_index[n.index] for n in channel] == [0, 1, 2, 3]
    # Removing the middle channel makes the two remaining channels equidistant.
    tied = CoreGeometry([n for n in nodes if n.index not in {33, 35, 37, 39}])
    assert tied.center_channel[0].index == 3
    legacy = [Node(.2, .01, i) for i in (8, 2, 6)]
    assert CoreGeometry(legacy).center_channel == tuple(legacy)


@pytest.mark.parametrize('failure', ['missing', 'misaligned', 'gap', 'nonuniform', 'duplicate'])
def test_invalid_geometry(failure):
    nodes = mesh()
    match = 'matching axial boundaries'
    if failure == 'missing':
        nodes[0].bounds = None
        match = 'every node or none'
    elif failure == 'misaligned':
        nodes.pop(0)
    elif failure == 'gap':
        nodes.pop(1)
        match = 'contiguous'
    elif failure == 'nonuniform':
        nodes = [Node(length, .01, i) for i, length in enumerate((.2, .3, .2))]
        match = 'uniform axial mesh'
    else:
        nodes[0].index = nodes[1].index
        match = 'unique'
    with pytest.raises(ValueError, match=match):
        Diffusion(nodes, 1e6, beta=.006)


def test_psar_geometry():
    path = Path(__file__).resolve().parents[1] / 'input_files/PSAR.h5'
    if not path.exists():
        pytest.skip('Generated PSAR HDF5 input is not present')
    core = CoreGeometry(Simulation._get_core(path))
    assert len(core.nodes) == 2368
    assert len(core.channels) == 148
    assert all(len(channel) == 16 for channel in core.channels)
    assert [n.index for n in core.center_channel] == list(range(1189, 1205))
    solver = Diffusion(core, 1e7, beta=.006)
    assert solver.flux.shape == (32,)
    assert solver.core_volume == pytest.approx(sum(n.volume for n in core.nodes))


@pytest.mark.parametrize('kind', ['diffusion', 'pk'])
@pytest.mark.parametrize('mpact', [False, True])
def test_broadcast_conserves_power_with_different_volumes(kind, mpact, library):
    solver = build(kind, mesh(volumes=(2., 1., 4.)), 7e6, library if mpact else None)
    data = wire(solver)
    solver.set_solver_method(State('steadystate'))
    solver.update_power()
    assert integrated(solver, data) == pytest.approx(7e6)
    center = [data['power'][n.index] for n in solver.core.center_channel]
    for channel in solver.core.channels:
        np.testing.assert_allclose([data['power'][n.index] for n in channel], center)
    if kind == 'pk':
        assert solver._center_power_fraction() == pytest.approx(1/7)


@pytest.mark.parametrize('kind', ['diffusion', 'pk'])
@pytest.mark.parametrize('mpact', [False, True])
def test_replication_feedback_and_transient_equivalence(kind, mpact, library):
    lib = library if mpact else None
    multi = build(kind, mesh(), 3e6, lib)
    single = build(kind, multi.core.center_channel, 1e6, lib)
    dm, ds = wire(multi), wire(single)
    for solver, data in ((multi, dm), (single, ds)):
        # Nonzero source tests center-population normalization in PK.
        for n in solver.core.nodes:
            data['precursor_source'][n.index] = 1e10
        solver.set_solver_method(State('steadystate'))
        solver.update_power()
        solver.set_solver_method(SimpleNamespace(state_type='transient', time_step=1e-4, scheme='BDF1'))
    if kind == 'diffusion':
        np.testing.assert_allclose(multi.flux, single.flux, rtol=1e-10)
        before = multi.set_up_matrix()
    else:
        assert multi._n_0 == pytest.approx(single._n_0)
        before = (multi.get_reactivity_feedback(), multi.get_core_precursor_sum())
    center_ids = {n.index for n in multi.core.center_channel}
    for n in multi.core.nodes:
        if n.index not in center_ids:
            dm['temperature'][n.index] = 1500.  # Outside the MPACT range, deliberately.
            dm['precursor_source'][n.index] = 1e15
    assert multi.get_core_temperature() == pytest.approx(single.get_core_temperature())
    if kind == 'diffusion':
        for original, current in zip(before, multi.set_up_matrix()):
            np.testing.assert_array_equal(original, current)
    else:
        assert before == pytest.approx((multi.get_reactivity_feedback(), multi.get_core_precursor_sum()))
    multi.update_power()
    single.update_power()
    if kind == 'pk':
        assert multi.amplitude == pytest.approx(1.)
        assert multi.amplitude == pytest.approx(single.amplitude)
        population = multi._n_0
    # Apply identical center feedback to both simulations.
    for solver, data in ((multi, dm), (single, ds)):
        for n in solver.core.center_channel:
            data['temperature'][n.index] = 920.
            data['precursor_source'][n.index] *= 1.1
    multi.update_power()
    single.update_power()
    if kind == 'diffusion':
        np.testing.assert_allclose(multi.flux, single.flux, rtol=1e-9)
        assert not np.isclose(integrated(multi, dm), 3e6, rtol=1e-6)
    else:
        assert multi.amplitude == pytest.approx(single.amplitude)
        assert multi.amplitude != pytest.approx(1.)
        assert multi._n_0 == population
    assert integrated(multi, dm) == pytest.approx(3 * integrated(single, ds))


@pytest.mark.parametrize('kind', ['diffusion', 'pk'])
def test_geometry_replacement_resets_solver_and_preserves_data(kind, library):
    solver = build(kind, mesh(), 3e6, library)
    data = wire(solver)
    solver.set_solver_method(State('steadystate'))
    solver.update_power()
    solver.set_core_nodes(solver.core.channels[0])
    assert len(solver.core.nodes) == 4
    assert solver._power is data['power']
    solver.set_solver_method(State('steadystate'))
    solver.update_power()
    assert integrated(solver, data) == pytest.approx(3e6)


@pytest.mark.parametrize('kind', ['diffusion', 'pk'])
@pytest.mark.parametrize('mpact', [False, True])
def test_simulation_receives_initial_data_and_sends_every_channel(tmp_path, monkeypatch, kind, mpact, library):
    class FakeSyth:
        def __init__(self, filename): self.sent = {}
        def set_precursor_source_per_power(self, value): pass
        def set_state(self, index): pass
        def get_fluid_node_value(self, name, index): return 900.
        def get_delayed_neutron_emissions(self, index): return 1e10
        def set_node_power(self, index, value): self.sent[index] = value
    monkeypatch.setitem(sys.modules, 'SyTH.syth', SimpleNamespace(Syth=FakeSyth))
    path = tmp_path / 'simulation.h5'
    config = {'neutronics': {'type': kind, 'initial_power': 3e6,
                            'beta_effective': .006, 'mean_neutron_lifetime': 2e-4, 'alpha_T': -1e-5}}
    with h5py.File(path, 'w') as f:
        f['metadata/input_file'] = np.bytes_(json.dumps(config))
        f['options/neutron_precursor_solve'] = True
        for n in mesh():
            g = f.create_group(f'fluid_solve/fluid_mesh/nodes/node_{n.index}')
            g['is_coupled'] = True
            g['length'], g['volume'], g['bounding_box'] = n.length, n.volume, n.bounds
    sim = Simulation(path, library=library if mpact else None)
    sim.set_up_data_container()
    assert all(sim._temperature[n.index] == 900. for n in sim._core_nodes)
    sim.set_state(0, State('steadystate'))
    sim._neutronics.update_power()
    sim.send_data()
    assert set(sim._syth.sent) == {n.index for n in sim._core_nodes}
    assert sum(sim._syth.sent[n.index] * n.volume for n in sim._core_nodes) == pytest.approx(3e6)


@pytest.mark.parametrize('mpact', [False, True])
def test_pk_reference_uses_shape_weighted_center_fraction(mpact, library):
    nodes = mesh()
    # Change only the outer channels' end-cell volumes. A volume-only fraction
    # would misrepresent their share of this nonuniform axial power shape.
    for n in nodes:
        if n.index in {3, 9, 63, 69}:
            n.volume *= 5
    solver = build('pk', nodes, 3e6, library if mpact else None)
    data = wire(solver)
    solver.set_solver_method(State('steadystate'))
    solver.update_power()
    center_power = sum(data['power'][n.index] * n.volume for n in solver.core.center_channel)
    fraction = center_power / integrated(solver, data)
    assert fraction == pytest.approx(solver._center_power_fraction())
    assert not np.isclose(fraction, sum(n.volume for n in solver.core.center_channel) / solver.core_volume)
    if mpact:
        k = library.kinetics(solver.get_core_temperature())
        expected = k.generation_time * k.neutrons_per_joule * center_power
    else:
        expected = 2.43 * solver.generation_time * center_power / 3.24e-11
    assert solver._n_0 == pytest.approx(expected)
