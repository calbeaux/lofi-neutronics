from abc import ABC, abstractmethod

import numpy as np

from Coupling import node
from Neutronics.cross_sections import XSSet
from Neutronics.neutronics_strategy import NeutronicsStrategy


class Diffusion(NeutronicsStrategy):
    def __init__(self, core_nodes, initial_power, substep, beta):
        self._core_nodes = core_nodes
        self._ncell = len(core_nodes)
        self._initial_power = initial_power
        self._substeps = substep
        self._beta = beta
        self._steady_state = True
        self.k_eff = 1.0
        self._temperature = None
        self._precursor_source = None
        self._power = None
        self.solver_strategy = None
        self.normalization_factor = None
        self.flux = np.ones(2 * self._ncell) * 1e8
        self.tolerance = 1e-8
        self.max_iterations = 500

    def set_solver_method(self, state):
        solver_type = state.state_type
        if solver_type == "steady_state":
            self._steady_state = True
            self.solver_strategy = DiffusionSteadyStateStrategy(self)
        elif solver_type == "transient":
            self.solver_strategy = DiffusionTransientStrategy(
                self, state.time_step, state.scheme
            )
            self._steady_state = False
        return self.solver_strategy

    def set_up_data_container_interactions(self, data_container):
        self._temperature = data_container["temperature"]
        self._precursor_source = data_container["precursor_source"]

    def register_data_container_variables(self, data_container):
        self._power = data_container.register("power", 0.0)

    def get_xs(self, node):
        return XSSet(self._temperature[node.index])

    def total_precursor_sum(self):
        return sum(
            self._precursor_source[node.index] for node in self._core_nodes
        )

    def calibrate_beta(self):
        total_fission_source = 0.0
        for index, node in enumerate(self._core_nodes):
            xs = self.get_xs(node)
            row_fast = index
            row_thermal = index + self._ncell
            total_fission_source += (
                xs.fast.nu_f * self.flux[row_fast] + xs.th.nu_f * self.flux[row_thermal]
            ) * node.length
        total_precursor_source = self.total_precursor_sum()
        self._beta = total_precursor_source / (
            total_fission_source + total_precursor_source
        )

    @staticmethod
    def harmonic_mean(x, y):
        return 2 * x * y / (x + y)

    def set_up_matrix(self, k=1.0):
        matrix_a = np.zeros((2 * self._ncell, 2 * self._ncell))
        vector_b = np.zeros(2 * self._ncell)
        vector_f = np.zeros(2 * self._ncell)

        previous_xs = None
        current_xs = self.get_xs(self._core_nodes[0])
        next_xs = self.get_xs(self._core_nodes[1])

        node = self._core_nodes[0]
        d_up_th = self.harmonic_mean(current_xs.th.d_coef, next_xs.th.d_coef)
        d_up_fast = self.harmonic_mean(current_xs.fast.d_coef, next_xs.fast.d_coef)
        diagonal_fast = (
            current_xs.fast.a * node.length
            + d_up_fast / node.length
            + current_xs.fast.s_out * node.length
        )
        diagonal_thermal = current_xs.th.a * node.length + d_up_th / node.length
        matrix_a[0, 0] = diagonal_fast + (
            current_xs.fast.d_coef
            / ((node.length / 2) + current_xs.fast.extrapolation_distance)
        )
        matrix_a[0, 1] = -d_up_fast / node.length
        matrix_a[self._ncell, self._ncell] = diagonal_thermal + (
            current_xs.th.d_coef
            / ((node.length / 2) + current_xs.th.extrapolation_distance)
        )
        matrix_a[self._ncell, self._ncell + 1] = -d_up_th / node.length
        matrix_a[self._ncell, 0] = -current_xs.fast.s_out * node.length
        vector_b[0] = (
            (1 - self._beta) * current_xs.fast.nu_f * self.flux[0] * node.length / k
        )
        vector_b[0] += (
            (1 - self._beta)
            * current_xs.th.nu_f
            * self.flux[self._ncell]
            * node.length
            / k
        )
        vector_f[0] = current_xs.fast.nu_f * node.length * (1 - self._beta)
        vector_f[self._ncell] = current_xs.th.nu_f * node.length * (1 - self._beta)
        vector_b[0] += self._precursor_source[node.index] *  (node.volume/node.length) / k

        for index, node in enumerate(self._core_nodes[1:-1], start=1):
            previous_xs = current_xs
            current_xs = next_xs
            next_xs = self.get_xs(self._core_nodes[index + 1])
            row_fast = index
            row_thermal = row_fast + self._ncell
            cell_index_fast = index
            cell_index_thermal = index + self._ncell
            d_down_th = self.harmonic_mean(previous_xs.th.d_coef, current_xs.th.d_coef)
            d_up_th = self.harmonic_mean(current_xs.th.d_coef, next_xs.th.d_coef)
            d_down_fast = self.harmonic_mean(
                previous_xs.fast.d_coef, current_xs.fast.d_coef
            )
            d_up_fast = self.harmonic_mean(current_xs.fast.d_coef, next_xs.fast.d_coef)

            diagonal_fast = (
                current_xs.fast.a * node.length
                + (d_down_fast + d_up_fast) / node.length
                + current_xs.fast.s_out * node.length
            )
            diagonal_thermal = (
                current_xs.th.a * node.length + (d_down_th + d_up_th) / node.length
            )

            matrix_a[row_fast, cell_index_fast] = diagonal_fast
            matrix_a[row_thermal, cell_index_thermal] = diagonal_thermal
            matrix_a[row_fast, cell_index_fast - 1] = -d_down_fast / node.length
            matrix_a[row_thermal, cell_index_thermal - 1] = -d_down_th / node.length
            matrix_a[row_fast, cell_index_fast + 1] = -d_up_fast / node.length
            matrix_a[row_thermal, cell_index_thermal + 1] = -d_up_th / node.length
            matrix_a[row_thermal, cell_index_fast] = (
                -current_xs.fast.s_out * node.length
            )

            vector_b[row_fast] = (
                (1 - self._beta)
                * current_xs.fast.nu_f
                * self.flux[row_fast]
                * node.length
                / k
            )
            vector_b[row_fast] += (
                (1 - self._beta)
                * current_xs.th.nu_f
                * self.flux[row_thermal]
                * node.length
                / k
            )
            vector_b[row_fast] += self._precursor_source[node.index] * (node.volume/node.length)/ k
            vector_f[row_fast] = current_xs.fast.nu_f * node.length * (1 - self._beta)
            vector_f[row_thermal] = current_xs.th.nu_f * node.length * (1 - self._beta)

        row_fast += 1
        row_thermal += 1
        cell_index_fast += 1
        cell_index_thermal += 1
        previous_xs = current_xs
        current_xs = next_xs
        node = self._core_nodes[-1]
        d_down_th = self.harmonic_mean(previous_xs.th.d_coef, current_xs.th.d_coef)
        d_down_fast = self.harmonic_mean(
            previous_xs.fast.d_coef, current_xs.fast.d_coef
        )
        diagonal_fast = (
            current_xs.fast.a * node.length
            + d_down_fast / node.length
            + current_xs.fast.s_out * node.length
        )
        diagonal_thermal = current_xs.th.a * node.length + d_down_th / node.length
        matrix_a[row_fast, cell_index_fast] = diagonal_fast + (
            current_xs.fast.d_coef
            / ((node.length / 2) + current_xs.fast.extrapolation_distance)
        )
        matrix_a[row_fast, cell_index_fast - 1] = -d_down_fast / node.length
        matrix_a[row_thermal, cell_index_thermal] = diagonal_thermal + (
            current_xs.th.d_coef
            / ((node.length / 2) + current_xs.th.extrapolation_distance)
        )
        matrix_a[row_thermal, cell_index_thermal - 1] = -d_down_th / node.length
        matrix_a[row_thermal, cell_index_fast] = -current_xs.fast.s_out * node.length
        vector_b[cell_index_fast] = (
            (1 - self._beta)
            * current_xs.fast.nu_f
            * self.flux[cell_index_fast]
            * node.length
            / k
        )
        vector_b[cell_index_fast] += (
            (1 - self._beta)
            * current_xs.th.nu_f
            * self.flux[cell_index_thermal]
            * node.length
            / k
        )
        vector_b[cell_index_fast] += self._precursor_source[node.index] * (node.volume/node.length) / k
        vector_f[cell_index_fast] = (
            current_xs.fast.nu_f * node.length * (1 - self._beta)
        )
        vector_f[cell_index_thermal] = (
            current_xs.th.nu_f * node.length * (1 - self._beta)
        )
        return matrix_a, vector_b, vector_f

    def update_power(self):
        self.solver_strategy.solve()
        total_power = 0.0
        for index, node in enumerate(self._core_nodes):
            xs = self.get_xs(node)
            row_fast = index
            row_thermal = row_fast + self._ncell
            self._power[node.index] = (
                xs.fast.kappa_f * self.flux[row_fast]
                + xs.th.kappa_f * self.flux[row_thermal]
            )
            total_power += self._power[node.index] * node.volume
        if self._steady_state:
            self.normalization_factor = self._initial_power / total_power
            total_power *= self.normalization_factor
            self._power *= self.normalization_factor
            self.flux *= self.normalization_factor

        print(f"Total power: {total_power}")



class DiffusionSolverStrategy(ABC):
    @abstractmethod
    def solve(self):
        pass


class DiffusionSteadyStateStrategy(DiffusionSolverStrategy):
    def __init__(self, diffusion_strategy):
        self._diffusion = diffusion_strategy
        self._relax = 1.0

    def solve(self):
        matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix()
        self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
        f_old = vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
        for i in range(self._diffusion.max_iterations):
            # self._diffusion.calibrate_beta()
            matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix(self._diffusion.k_eff)
            self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
            f_new = (
                vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
            )
            k_eff_new = self._diffusion.k_eff * f_new / f_old
            print(f"Iteration {i}: k-eff = {k_eff_new}")
            if abs(k_eff_new - self._diffusion.k_eff) / self._diffusion.k_eff < 1e-12:
                self._diffusion.k_eff = k_eff_new
                break
            self._diffusion.k_eff = k_eff_new
            f_old = f_new

class DiffusionTransientStrategy(DiffusionSolverStrategy):
    coefficient_matrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [4 / 3, -1 / 3, 0, 0, 0, 0],
            [18 / 11, -9 / 11, 2 / 11, 0, 0, 0],
            [48 / 25, -36 / 25, 16 / 25, -3 / 25, 0, 0],
            [300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137, 0],
            [360 / 147, -450 / 147, 400 / 147, -225 / 147, 72 / 147, -10 / 147],
        ]
    )
    gamma_vector = np.array([1, 3 / 2, 11 / 6, 25 / 12, 137 / 60, 147 / 60])

    def __init__(self, diffusion_strategy, dt, scheme):
        self._diffusion = diffusion_strategy
        self._relax = 0.55
        self._dt = dt
        # self._order = int(scheme[-1])
        self._order = 2
        self._old_flux = np.zeros((self._order, 2 * self._diffusion._ncell))
        self.set_up_current_data_container_interactions()

    def set_up_current_data_container_interactions(self):
        for i in range(self._order):
            self._old_flux[i] = self._diffusion.flux

    def update_old_variable_array(self):
        self._old_flux = np.roll(self._old_flux, 1, axis=0)
        self._old_flux[0] = self._diffusion.flux

    def solve(self):
        self.update_old_variable_array()
        gamma = self.gamma_vector[self._order - 1] / self._dt
        coefficient_vector = (
            self.coefficient_matrix[self._order - 1][: self._order] * gamma
        )

        for i in range(self._diffusion.max_iterations):
            matrix_a, vector_b, _ = self._diffusion.set_up_matrix(
                k=self._diffusion.k_eff
            )

            for index, node in enumerate(self._diffusion._core_nodes):
                current_xs = self._diffusion.get_xs(node)
                vector_b[index] += (
                    node.length
                    * (self._old_flux.T[index] @ coefficient_vector)
                    / current_xs.fast.velocity
                )
                matrix_a[index, index] += (
                    node.length * gamma / current_xs.fast.velocity
                )
                vector_b[index + self._diffusion._ncell] += (
                    node.length
                    * (
                        self._old_flux.T[index + self._diffusion._ncell]
                        @ coefficient_vector
                    )
                    / current_xs.th.velocity
                )
                matrix_a[
                    index + self._diffusion._ncell, index + self._diffusion._ncell
                ] += (node.length * gamma / current_xs.th.velocity)

            new = np.linalg.solve(matrix_a, vector_b)
            diff = np.linalg.norm(self._diffusion.flux - new)/np.linalg.norm(new)
            self._diffusion.flux = self._relax * self._diffusion.flux + (1 - self._relax) * new
            if diff < self._diffusion.tolerance:
                print(f"Converged after {i} iterations with diff = {diff}")
                print(f"Flux difference: ({(self._diffusion.flux - self._old_flux[0])/self._old_flux[0]})")
                break
            elif i == self._diffusion.max_iterations - 1:
                print(f"Did not converge after {i} iterations with diff = {diff}")
