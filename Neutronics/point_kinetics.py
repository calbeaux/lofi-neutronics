from abc import ABC, abstractmethod

import numpy as np

from Neutronics.neutronics_strategy import NeutronicsStrategy


def _sine_harmonics(x, A1, A3, A5, L, D):
    return A1*np.sin(np.pi*x/L) + A3*np.sin(3*np.pi*x/L) + A5*np.sin(5*np.pi*x/L) + D


def _build_profile(params, n_nodes):
    A1, A3, A5, L, D = params
    x = np.linspace(0, L, n_nodes)
    return _sine_harmonics(x, A1, A3, A5, L, D)


class PointKineticsStrategy(NeutronicsStrategy):
    _ADJOINT_PARAMETERS = np.array([0.9466757, 0.0804915, 0.02652392, 1.00031018, 0.41353545])
    _FLUX_PARAMETERS = np.array([1.39165538e+00, 1.01009672e-03, -2.56510424e-03, 9.99997599e-01, 1.65738299e-01])

    def __init__(
        self,
        core_nodes,
        mean_neutron_lifetime,
        beta_effective,
        alpha_t,
        initial_power,
        substep,
    ):
        self._core_nodes = core_nodes
        self._mean_neutron_lifetime = mean_neutron_lifetime
        self._beta_effective = beta_effective
        self._modified_beta_effective = beta_effective
        self._alpha_t = alpha_t
        self._initial_power = initial_power
        self._total_power = initial_power
        self._substeps = substep
        self._t0 = None
        self._precursor_source = None
        self._temperature = None
        self._power = None
        self._solver = None
        self._amplitude = 1.0
        self._flux_shape = _build_profile(self._FLUX_PARAMETERS, len(self._core_nodes))
        self._adjoint = _build_profile(self._ADJOINT_PARAMETERS, len(self._core_nodes))
        self._n_0 = 2.43 * self._mean_neutron_lifetime * initial_power / (3.24e-11)
        self._rho_0 = 0.0

    @property
    def total_power(self):
        return self._total_power

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def beta_effective(self):
        return self._beta_effective

    @property
    def mean_neutron_lifetime(self):
        return self._mean_neutron_lifetime

    def set_initial_reactivity(self, rho):
        self._rho_0 = rho

    def update_initial_temperature(self):
        self._t0 = self.get_core_average(self._temperature)

    def set_solver_method(self, state):
        solver_type = state.state_type
        print("Setting neutronics solver method to", solver_type)
        if solver_type == "steady_state":
            self._solver = PointKineticsSteadyStateStrategy(self)
        elif solver_type == "transient":
            if self._substeps is not None:
                self._solver = PointKineticsTransientStrategy(
                    self, state.time_step / self._substeps
                )
            else:
                self._solver = PointKineticsTransientStrategy(self, state.time_step)
        return self._solver

    def set_up_data_container_interactions(self, data_container):
        self._precursor_source = data_container["precursor_source"]
        self._temperature = data_container["temperature"]

    def register_data_container_variables(self, data_container):
        self._power = data_container.register("power", 0.0)

    def get_core_average(self, data):
        average = 0.0
        for index, node in enumerate(self._core_nodes):
            average += data[node.index] * self._adjoint[index] / len(self._adjoint)
        return average

    def get_reactivity_feedback(self):
        core_temp = self.get_core_average(self._temperature)
        return self._alpha_t * (core_temp - self._t0) + self._rho_0

    def update_power(self):
        self._amplitude = self._solver.get_new_amplitude()
        self._total_power = self._amplitude * self._initial_power

        node_volumes = np.array([node.volume for node in self._core_nodes])
        normalization = np.sum(self._flux_shape * node_volumes)

        for index, node in enumerate(self._core_nodes):
            self._power[node.index] = self._total_power * self._flux_shape[index] / normalization

        print(f"Total power: {self._total_power}")
        delivered = sum(
            self._power[node.index] * node.volume for node in self._core_nodes
        )
        assert (
            abs(delivered - self._total_power) < 1e-6 * self._total_power
        ), f"Power not conserved: {delivered} vs {self._total_power}"

    def get_core_precursor_sum(self):
        total = 0.0
        for index, node in enumerate(self._core_nodes):
            total += self._precursor_source[node.index] * self._adjoint[index]
        return total / self._n_0

    def calculate_solution(self):
        total = 0.0
        for node in self._core_nodes:
            total += (
                node.volume
                * self._power[node.index]
                * self._beta_effective
                / (self.mean_neutron_lifetime * self._initial_power)
            )
        return total


class PointKineticsSolverStrategy(ABC):
    @abstractmethod
    def get_new_amplitude(self):
        pass


class PointKineticsSteadyStateStrategy(PointKineticsSolverStrategy):
    def __init__(self, point_kinetics_strategy):
        self._pk = point_kinetics_strategy

    def get_new_amplitude(self):
        self._pk.update_initial_temperature()
        self._pk._beta_effective = (
            self._pk.get_core_precursor_sum() * self._pk.mean_neutron_lifetime
        )
        print(f"calculate solution: {self._pk.calculate_solution()}")
        return 1.0


class PointKineticsTransientStrategy(PointKineticsSolverStrategy):
    def __init__(self, point_kinetics_strategy, dt):
        self._pk = point_kinetics_strategy
        self._dt = dt

    def get_new_amplitude(self):
        gamma = (
            self._pk.get_reactivity_feedback() - self._pk.beta_effective
        ) / self._pk.mean_neutron_lifetime
        pc_sum = self._pk.get_core_precursor_sum()
        new_amplitude = (self._pk.amplitude + pc_sum / gamma) * np.exp(
            gamma * self._dt
        ) - pc_sum / gamma
        print(f"rho: {self._pk.get_reactivity_feedback()}")
        print(
            "Core temperature:",
            self._pk.get_core_average(self._pk._temperature),
            "Initial temperature:",
            self._pk._t0,
        )
        print(f"mean neutron lifetime: {self._pk.mean_neutron_lifetime}")
        print(f"beta effective: {self._pk.beta_effective}")
        print(f"Gamma: {gamma}")
        print(f"Core precursor sum: {self._pk.get_core_precursor_sum()}")
        print(f"New amplitude: {new_amplitude}")
        return new_amplitude
