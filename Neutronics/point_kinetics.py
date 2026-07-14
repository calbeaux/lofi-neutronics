from abc import ABC, abstractmethod

import numpy as np

from Neutronics.neutronics_strategy import NeutronicsStrategy


class PointKineticsStrategy(NeutronicsStrategy):
    def __init__(
        self,
        core_nodes,
        prompt_neutron_lifetime,
        beta_effective,
        alpha_t,
        initial_power,
        substep,
    ):
        self._core_nodes = core_nodes
        self._prompt_neutron_lifetime = prompt_neutron_lifetime
        self._beta_effective = beta_effective
        self._modified_beta_effective = beta_effective
        self._alpha_t = alpha_t
        self._initial_power = initial_power
        self._total_power = initial_power
        self._substeps = substep
        self._adjoint = np.ones(len(self._core_nodes))
        self._t0 = None
        self._precursor_source = None
        self._temperature = None
        self._power = None
        self._solver = None
        self._amplitude = 1.0
        self._weights = self.get_weights()
        self._n_0 = 2.43 * self._prompt_neutron_lifetime * initial_power / (3.24e-11)
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
    def prompt_neutron_lifetime(self):
        return self._prompt_neutron_lifetime

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

    def get_weights(self):
        cell_positions = []
        for index, node in enumerate(self._core_nodes):
            length = node.length
            if index == 0:
                cell_positions.append(length / 2)
            else:
                cell_positions.append(cell_positions[-1] + length)

        total_length = cell_positions[-1] + self._core_nodes[-1].length / 2
        return [np.sin(np.pi * x / total_length) for x in cell_positions]

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

        weights = np.array(self._weights)
        node_volumes = np.array([node.volume for node in self._core_nodes])
        normalization = np.sum(weights * node_volumes)

        for index, node in enumerate(self._core_nodes):
            self._power[node.index] = self._total_power * weights[index] / normalization

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
                / (self.prompt_neutron_lifetime * self._initial_power)
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
            self._pk.get_core_precursor_sum() * self._pk.prompt_neutron_lifetime
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
        ) / self._pk.prompt_neutron_lifetime
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
        print(f"prompt neutron lifetime: {self._pk.prompt_neutron_lifetime}")
        print(f"beta effective: {self._pk.beta_effective}")
        print(f"Gamma: {gamma}")
        print(f"Core precursor sum: {self._pk.get_core_precursor_sum()}")
        print(f"New amplitude: {new_amplitude}")
        return new_amplitude
