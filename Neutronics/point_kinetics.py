from abc import ABC, abstractmethod

import numpy as np

from Neutronics.radial_shape import RadialShape
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
        mean_neutron_lifetime=None,
        beta_effective=None,
        alpha_t=None,
        initial_power=1.0,
        substep=1,
        library=None,
        radial_shape=None,
    ):
        self._radial_config = radial_shape
        self.library = library
        super().set_core_nodes(core_nodes)
        self._mean_neutron_lifetime = mean_neutron_lifetime
        self._beta_effective = beta_effective
        self._modified_beta_effective = beta_effective
        self._alpha_t = alpha_t
        self._initial_power = initial_power
        self._total_power = initial_power
        self._substeps = substep if substep is not None and substep > 0 else 1
        self._precursor_source = None
        self._temperature = None
        self._power = None
        self._reset_core_state()

    def _reset_core_state(self):
        self.radial = (RadialShape(self.core, self.library, self._radial_config)
                       if self._radial_config is not None else None)
        self._t0 = None
        self._solver = None
        self._amplitude = 1.0
        self._total_power = self._initial_power
        self._flux_shape = _build_profile(self._FLUX_PARAMETERS, len(self._axial_nodes))
        self._adjoint = _build_profile(self._ADJOINT_PARAMETERS, len(self._axial_nodes))
        self._adjoint /= np.sum(self._adjoint) / len(self._adjoint)
        self._generation_time = self._mean_neutron_lifetime  # legacy convention
        self._n_0 = (2.43 * self._mean_neutron_lifetime * self._initial_power
                     * self._center_power_fraction() / 3.24e-11
                     if self.library is None else None)
        self._rho_0 = 0.0
        self._reference_rho = None
        self._reference_initialized = False

    def _center_power_fraction(self):
        if self.radial is not None:
            return 1.0
        volumes = np.array([n.volume for n in self._axial_nodes])
        return (self._flux_shape @ volumes) / (self._flux_shape @ self.core.axial_volumes)

    def set_core_nodes(self, core_nodes):
        super().set_core_nodes(core_nodes)
        self._reset_core_state()

    def initialize_reference(self):
        if self.library is None or self._reference_initialized:
            return
        volumes = np.array([n.volume for n in self._axial_nodes])
        temperatures = np.array([self._temperature[n.index] for n in self._axial_nodes])
        initial_temperature = (self.radial.reference_temperature if self.radial is not None
                               else np.average(temperatures, weights=volumes))
        self._flux_shape, self._adjoint = self.library.profiles(initial_temperature).map_nodes(self._axial_nodes)
        self._t0 = self.get_core_temperature()
        kinetics = self.library.kinetics(self._t0)
        self._reference_rho = kinetics.rho
        self._n_0 = kinetics.generation_time * kinetics.neutrons_per_joule * self._initial_power * self._center_power_fraction()
        self._reference_initialized = True
        self._refresh_kinetics()
        self.set_initial_reactivity(
            self.beta_effective - self.get_core_precursor_sum() * self.generation_time)

    def _refresh_kinetics(self):
        if self.library is not None:
            kinetics = self.library.kinetics(self.get_core_temperature())
            self._generation_time = kinetics.generation_time
            self._mean_neutron_lifetime = kinetics.neutron_lifetime
            self._beta_effective = kinetics.beta_effective
            self._alpha_t = kinetics.alpha_T

    @property
    def generation_time(self):
        return self._generation_time

    def fission_power_factor(self):
        """Fissions/J for SyTH's existing global power-to-source interface."""
        return self.library.kinetics(self.get_core_temperature()).fissions_per_joule

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
        if self.library is None:
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

    def get_core_temperature(self):
        return self.get_core_average(self._temperature)

    def _full_importance(self):
        importance = self.radial.importance * self._adjoint[None,:]
        heating = self.radial.weights * self._flux_shape[None,:] * self.radial.volumes
        return importance / (np.sum(importance*heating)/heating.sum())

    def get_core_average(self, data):
        if self.radial is not None:
            return np.average(data[self.radial.indices],
                              weights=self._full_importance()*self.radial.volumes)
        if self.library is not None:
            weights = self._adjoint * np.array([n.volume for n in self._axial_nodes])
            return np.average([data[n.index] for n in self._axial_nodes], weights=weights)
        average = 0.0
        for index, node in enumerate(self._axial_nodes):
            average += data[node.index] * self._adjoint[index] / sum(self._adjoint)
        return average

    def get_reactivity_feedback(self):
        core_temp = self.get_core_average(self._temperature)
        if self.library is not None:
            return self.library.kinetics(core_temp).rho - self._reference_rho + self._rho_0
        return self._alpha_t * (core_temp - self._t0) + self._rho_0

    def update_power(self):
        self.initialize_reference()
        self._refresh_kinetics()
        self._amplitude = self._solver.get_new_amplitude()
        self._total_power = self._amplitude * self._initial_power

        normalization = self._flux_shape @ self.core.axial_volumes
        self.core.broadcast(self._total_power * self._flux_shape / normalization, self._power, self.radial)

        print(f"Total power: {self._total_power}")
        delivered = sum(self._power[n.index] * n.volume for n in self._core_nodes)
        assert np.isclose(delivered, self._total_power, rtol=1e-6, atol=1e-12), (
            f"Power not conserved: {delivered} vs {self._total_power}")

    def get_core_precursor_sum(self):
        if self.radial is not None:
            return np.sum(self._precursor_source[self.radial.indices]*self._full_importance())/self._n_0
        total = 0.0
        for index, node in enumerate(self._axial_nodes):
            total += self._precursor_source[node.index] * self._adjoint[index]
        return total / self._n_0

    def calculate_solution(self):
        total = 0.0
        nodes = self._core_nodes if self.radial is not None else self._axial_nodes
        for node in nodes:
            total += (
                node.volume
                * self._power[node.index]
                * self._beta_effective
                / (self.generation_time * self._initial_power * self._center_power_fraction())
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
        # Balance the full transient equation at amplitude=1. The library
        # reference predates thermal convergence, so its accumulated feedback
        # must be included in the steady-state reactivity offset.
        feedback = (self._pk.library.kinetics(self._pk.get_core_temperature()).rho
                    - self._pk._reference_rho
                    if self._pk.library is not None else 0.0)
        self._pk.set_initial_reactivity(
            - self._pk.get_core_precursor_sum() * self._pk.generation_time + self._pk.beta_effective
            - feedback
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
        ) / self._pk.generation_time
        pc_sum = self._pk.get_core_precursor_sum()
        z = gamma * self._dt
        source_integral = self._dt if z == 0 else self._dt * np.expm1(z) / z
        new_amplitude = self._pk.amplitude * np.exp(z) + pc_sum * source_integral
        print(f"rho: {self._pk.get_reactivity_feedback()}")
        print(
            "Core temperature:",
            self._pk.get_core_average(self._pk._temperature),
            "Initial temperature:",
            self._pk._t0,
        )
        core_avg_temp = 0.0
        total_core_precursors = 0.0
        for index, node in enumerate(self._pk._axial_nodes):
            core_avg_temp += self._pk._temperature[node.index]
            total_core_precursors += self._pk._precursor_source[node.index]
        print(f"core average temperature: {core_avg_temp/len(self._pk._axial_nodes)}")
        print(f"mean neutron lifetime: {self._pk.mean_neutron_lifetime}")
        print(f"beta effective: {self._pk.beta_effective}")
        print(f"Gamma: {gamma}")
        print(f"Core precursor sum: {self._pk.get_core_precursor_sum()}")
        print(f"mean of adjoint: {sum(self._pk._adjoint)/len(self._pk._adjoint)}")
        print(f"Total core precursors: {total_core_precursors/ self._pk._n_0}")
        print(f"New amplitude: {new_amplitude}")
        print(f"computed dp/dt: {(new_amplitude - self._pk.amplitude) / self._dt}")
        print(f"calculated dp/dt: {new_amplitude * gamma + pc_sum}")

        return new_amplitude


class PointKineticsNumericalTransientStrategy(PointKineticsSolverStrategy):
    """
    Theta-method (generalized trapezoidal) solver for the same frozen-coefficient
    amplitude equation used by ``PointKineticsTransientStrategy``:

        dn/dt = gamma * n + pc_sum,  gamma and pc_sum held constant over dt

    theta=0.0 -> forward Euler (explicit)
    theta=0.5 -> Crank-Nicolson (default)
    theta=1.0 -> backward Euler (implicit)

    Since the ODE is linear and gamma/pc_sum are frozen over the step, the
    theta update has a closed-form algebraic solution (no Newton iteration
    needed), making it a direct numerical counterpart to compare against the
    exact exponential solution used by PointKineticsTransientStrategy.
    """

    def __init__(self, point_kinetics_strategy, dt, theta=0.5):
        self._pk = point_kinetics_strategy
        self._dt = dt
        self._theta = theta

    def get_new_amplitude(self):
        gamma = (
            self._pk.get_reactivity_feedback() - self._pk.beta_effective
        ) / self._pk.generation_time
        pc_sum = self._pk.get_core_precursor_sum()
        n_old = self._pk.amplitude
        theta = self._theta
        dt = self._dt

        numerator = n_old * (1.0 + (1.0 - theta) * dt * gamma) + dt * pc_sum
        denominator = 1.0 - theta * dt * gamma
        new_amplitude = numerator / denominator

        print(f"rho: {self._pk.get_reactivity_feedback()}")
        print(f"Gamma: {gamma}")
        print(f"Core precursor sum: {pc_sum * self._pk.mean_neutron_lifetime}")
        print(f"theta: {theta}")
        print(f"New amplitude: {new_amplitude}")
        print(f"computed dp/dt: {(new_amplitude - n_old) / dt}")

        return new_amplitude
