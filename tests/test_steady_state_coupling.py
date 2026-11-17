"""Outer steady-state coupling checks without the native SyTH backend."""
from types import SimpleNamespace

import numpy as np
import pytest

from Coupling.node import Node
from Coupling.simulation import Simulation
from Coupling.state import State


def make_simulation(gain=10., max_iterations=30, tolerance=1e-6, slow_field=None):
    sim = Simulation.__new__(Simulation)
    sim.library = None
    sim._kappa = 3.24e-11
    sim._state = State('steadystate')
    sim._core_nodes = [Node(1., 1., 1), Node(1., 1., 4)]
    sim._steady_state_max_iterations = max_iterations
    sim._steady_state_tolerance = tolerance
    sim._temperature = np.full(5, 910.)
    sim._precursor_source = np.zeros(5)
    sim._power = np.zeros(5)
    indices = [1, 4]

    class Backend:
        def __init__(self):
            self.applied = np.zeros(5)
            self.temperature = sim._temperature.copy()
            self.source = np.zeros(5)
            self.fluid_calls = self.mass_calls = self.writes = 0

        def set_node_power(self, index, value): self.applied[index] = value

        def solve_fluid_flow(self, **kwargs):
            assert kwargs['delta_time'] == 0.
            self.fluid_calls += 1
            if slow_field == 'temperature':
                self.temperature[indices] = 900. + .5 * (self.temperature[indices] - 900.)
            else:
                self.temperature[indices] = 900. + gain * (self.applied[indices] - 1.)

        def solve_mass_transport(self, **kwargs):
            assert kwargs['delta_time'] == 0.
            assert kwargs['fluid_solve_tol'] == 1e-12
            self.mass_calls += 1
            if slow_field == 'precursor_source':
                self.source[indices] = 10. * (1. - .5**self.mass_calls)
            else:
                self.source[indices] = self.applied[indices] * 10.

        def get_fluid_node_value(self, name, index): return self.temperature[index]
        def get_delayed_neutron_emissions(self, index): return self.source[index]
        def get_node_count(self): return 5
        def write_solution_to_input_file(self): self.writes += 1

    def update_power():
        deviation = 0. if slow_field else .01 * (sim._temperature[1] - 900.)
        sim._power[indices] = [1. + deviation, 1. - deviation]

    sim._syth = Backend()
    sim._neutronics = SimpleNamespace(update_power=update_power)
    return sim


def test_outer_loop_converges_the_fields_used_by_the_backend():
    sim = make_simulation()
    sim._run_steady_state()
    backend = sim._syth
    assert 2 < sim._steady_state_iterations < sim._steady_state_max_iterations
    assert backend.fluid_calls == backend.mass_calls == sim._steady_state_iterations
    assert backend.writes == 1
    np.testing.assert_allclose(backend.temperature[[1, 4]], 900., atol=1e-4)
    np.testing.assert_allclose(backend.source[[1, 4]], backend.applied[[1, 4]] * 10., rtol=1e-6)
    np.testing.assert_allclose(sim._power, backend.applied)
    assert sim._power[[1, 4]].sum() == pytest.approx(2.)


@pytest.mark.parametrize('field', ['temperature', 'precursor_source'])
def test_unchanged_power_does_not_stop_other_fields_from_converging(field):
    sim = make_simulation(slow_field=field)
    sim._run_steady_state()
    assert sim._steady_state_iterations > 2
    assert sim._syth.writes == 1
    np.testing.assert_array_equal(sim._power[[1, 4]], [1., 1.])


def test_already_balanced_fields_still_get_two_coupling_passes():
    sim = make_simulation()
    sim._temperature[:] = sim._syth.temperature[:] = 900.
    sim._precursor_source[[1, 4]] = sim._syth.source[[1, 4]] = 10.
    sim._run_steady_state()
    assert sim._steady_state_iterations == 2
    assert sim._syth.writes == 1


def test_iteration_limit_raises_without_writing_a_false_steady_solution():
    sim = make_simulation(gain=-100., max_iterations=4)
    with pytest.raises(RuntimeError, match='did not converge after 4 iterations'):
        sim._run_steady_state()
    assert sim._steady_state_iterations == 4
    assert sim._syth.fluid_calls == sim._syth.mass_calls == 4
    assert sim._syth.writes == 0


@pytest.mark.parametrize('settings', [
    None, {'max_iterations': 1}, {'max_iterations': True}, {'max_iterations': 2.5},
    {'tolerance': 0}, {'tolerance': float('nan')}, {'tolerance': True},
])
def test_invalid_steady_state_settings(settings):
    with pytest.raises(ValueError, match='steady_state'):
        Simulation._get_steady_state_settings({'neutronics': {'steady_state': settings}})


def test_steady_state_settings_defaults_and_overrides():
    assert Simulation._get_steady_state_settings({}) == (50, 1e-6)
    config = {'neutronics': {'steady_state': {'max_iterations': 8, 'tolerance': 1e-4}}}
    assert Simulation._get_steady_state_settings(config) == (8, 1e-4)
