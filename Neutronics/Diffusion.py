from abc import ABC, abstractmethod

import numpy as np

from Neutronics.neutronics_strategy import NeutronicsStrategy


class Diffusion(NeutronicsStrategy):
    def __init__(self, core_nodes, initial_power, substep, beta):
        print("Initializing Diffusion Neutronics Strategy")
        self._core_nodes = core_nodes
        self._ncell = len(core_nodes)
        self._initial_power = initial_power
        self._substeps = substep
        self._beta = beta
        self._steady_state = True
        self._enthalpy = None
        self._neutron_population = None
        self._precursor_groups = None
        self._density = None
        self._power = None
        self._reaction_rate = None
        self.sigma_fission = None
        self.solver_strategy = None
        self.normalization_factor = None
        self.total_power = None
        self.flux = np.ones(2 * self._ncell) * 1e8

    def set_solver_method(self, state):
        solver_type = state.state_type
        print("Setting neutronics solver method to", solver_type)
        if solver_type == "steady_state":
            self._steady_state = True
            self.solver_strategy = DiffusionSteadyStateStrategy(self)
        elif solver_type == "transient":
            self.solver_strategy = DiffusionTransientStrategy(
                self, state.time_duration, state.scheme
            )
            self._steady_state = False
        return self.solver_strategy

    def set_up_data_container_interactions(self, data_container):
        self._precursor_groups = [
            data_container[f"{self._data_manager.variable_name}_{i}"]
            for i in range(self._data_manager.n_groups)
        ]
        self._enthalpy = data_container["enthalpy"]
        self._density = data_container["density"]

    def register_data_container_variables(self, data_container):
        surfaces = []
        nodes = []
        for node in self._core_nodes:
            surfaces.append(node.in_surfaces[0].index)
            surfaces.append(node.out_surfaces[0].index)
            nodes.append(node.index)
        self._neutron_population = data_container.register("neutron_population", 1e8)
        self._reaction_rate = data_container.register("reaction_rate", 1e8)
        self._power = data_container.register("power", 1e8)
        for i in range(self._mesh.n_connections):
            if i not in surfaces:
                self._neutron_population[i] = 0.0
        for i in range(self._mesh.n_nodes):
            if i not in nodes:
                self._reaction_rate[i] = 0.0
                self._power[i] = 0.0

    def get_xs(self, node):
        class CrossSections:
            def __init__(
                self, sigma_tr, sigma_a, nu_sigma_f, in_scatter, out_scatter, kappa_f
            ):
                self.tr = sigma_tr
                self.a = sigma_a
                self.nu_f = nu_sigma_f
                self.s_in = in_scatter
                self.s_out = out_scatter
                self.d_coef = 1 / (3 * self.tr)
                self.extrapolation_distance = 0.7104 / self.tr
                self.kappa_f = kappa_f

        class XSSet:
            def __init__(self, fast: CrossSections, th: CrossSections):
                self.fast = fast
                self.th = th

        def fast_sigma_transport(temperature):
            return 0.2 - 1e-4 * (temperature - 300)

        def fast_sigma_absorption(temperature):
            return 0.01 + 5e-5 * (temperature - 300)

        def fast_nu_sigma_fission(temperature):
            return 0.03 - 1e-5 * (temperature - 300)

        def fast_in_scatter(temperature):
            return 0.005 + 2e-5 * (temperature - 300)

        def fast_out_scatter(temperature):
            return 0.02 + 3e-5 * (temperature - 300)

        def fast_kappa_fission(temperature):
            return 200 - 0.05 * (temperature - 300)

        def th_sigma_transport(temperature):
            return 0.1 - 5e-5 * (temperature - 300)

        def th_sigma_absorption(temperature):
            return 0.02 + 1e-5 * (temperature - 300)

        def th_nu_sigma_fission(temperature):
            return 0.01 - 5e-6 * (temperature - 300)

        def th_in_scatter(temperature):
            return 0.004 + 1e-5 * (temperature - 300)

        def th_out_scatter(temperature):
            return 0.01 + 2e-5 * (temperature - 300)

        def th_kappa_fission(temperature):
            return 180 - 0.03 * (temperature - 300)

        fluid_temp = self._fluid_property.temperature(node.cell_average(self._enthalpy))
        fast_xs = CrossSections(
            fast_sigma_transport(fluid_temp),
            fast_sigma_absorption(fluid_temp),
            fast_nu_sigma_fission(fluid_temp),
            fast_in_scatter(fluid_temp),
            fast_out_scatter(fluid_temp),
            fast_kappa_fission(fluid_temp),
        )
        th_xs = CrossSections(
            th_sigma_transport(fluid_temp),
            th_sigma_absorption(fluid_temp),
            th_nu_sigma_fission(fluid_temp),
            th_in_scatter(fluid_temp),
            th_out_scatter(fluid_temp),
            th_kappa_fission(fluid_temp),
        )
        return XSSet(fast_xs, th_xs)

    def precursor_sum(self, node):
        precursor_sum = 0.0
        for index, group in enumerate(self._precursor_groups):
            precursor_sum += (
                node.cell_average(group)
                * node.cell_average(self._density)
                * self._data_manager.decay_constants[index]
            )
        return precursor_sum * node.volume

    def total_precursor_sum(self):
        precursor_sum = 0.0
        for node in self._core_nodes:
            for index, group in enumerate(self._precursor_groups):
                precursor_sum += (
                    node.cell_average(group)
                    * node.cell_average(self._density)
                    * self._data_manager.decay_constants[index]
                    * node.volume
                )
        return precursor_sum

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
            current_xs.fast.a * node.volume
            + d_up_fast / node.volume
            + current_xs.fast.s_out * node.volume
        )
        diagonal_thermal = current_xs.th.a * node.volume + d_up_th / node.volume
        matrix_a[0, 0] = diagonal_fast + (
            current_xs.fast.d_coef
            / ((node.volume / 2) + current_xs.fast.extrapolation_distance)
        )
        matrix_a[0, 1] = -d_up_fast / node.volume
        matrix_a[self._ncell, self._ncell] = diagonal_thermal + (
            current_xs.th.d_coef
            / ((node.volume / 2) + current_xs.th.extrapolation_distance)
        )
        matrix_a[self._ncell, self._ncell + 1] = -d_up_th / node.volume
        matrix_a[self._ncell, 0] = -current_xs.fast.s_out * node.volume
        vector_b[0] = (
            (1 - self._beta) * current_xs.fast.nu_f * self.flux[0] * node.volume / k
        )
        vector_b[0] += (
            (1 - self._beta)
            * current_xs.th.nu_f
            * self.flux[self._ncell]
            * node.volume
            / k
        )
        vector_f[0] = current_xs.fast.nu_f * node.volume * (1 - self._beta)
        vector_f[self._ncell] = current_xs.th.nu_f * node.volume * (1 - self._beta)
        vector_b[0] += self.precursor_sum(node) / k

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
                current_xs.fast.a * node.volume
                + (d_down_fast + d_up_fast) / node.volume
                + current_xs.fast.s_out * node.volume
            )
            diagonal_thermal = (
                current_xs.th.a * node.volume + (d_down_th + d_up_th) / node.volume
            )

            matrix_a[row_fast, cell_index_fast] = diagonal_fast
            matrix_a[row_thermal, cell_index_thermal] = diagonal_thermal
            matrix_a[row_fast, cell_index_fast - 1] = -d_down_fast / node.volume
            matrix_a[row_thermal, cell_index_thermal - 1] = -d_down_th / node.volume
            matrix_a[row_fast, cell_index_fast + 1] = -d_up_fast / node.volume
            matrix_a[row_thermal, cell_index_thermal + 1] = -d_up_th / node.volume
            matrix_a[row_thermal, cell_index_fast] = (
                -current_xs.fast.s_out * node.volume
            )

            vector_b[row_fast] = (
                (1 - self._beta)
                * current_xs.fast.nu_f
                * self.flux[row_fast]
                * node.volume
                / k
            )
            vector_b[row_fast] += (
                (1 - self._beta)
                * current_xs.th.nu_f
                * self.flux[row_thermal]
                * node.volume
                / k
            )
            vector_b[row_fast] += self.precursor_sum(node) / k
            vector_f[row_fast] = current_xs.fast.nu_f * node.volume * (1 - self._beta)
            vector_f[row_thermal] = current_xs.th.nu_f * node.volume * (1 - self._beta)

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
            current_xs.fast.a * node.volume
            + d_down_fast / node.volume
            + current_xs.fast.s_out * node.volume
        )
        diagonal_thermal = current_xs.th.a * node.volume + d_down_th / node.volume
        matrix_a[row_fast, cell_index_fast] = diagonal_fast + (
            current_xs.fast.d_coef
            / ((node.volume / 2) + current_xs.fast.extrapolation_distance)
        )
        matrix_a[row_fast, cell_index_fast - 1] = -d_down_fast / node.volume
        matrix_a[row_thermal, cell_index_thermal] = diagonal_thermal + (
            current_xs.th.d_coef
            / ((node.volume / 2) + current_xs.th.extrapolation_distance)
        )
        matrix_a[row_thermal, cell_index_thermal - 1] = -d_down_th / node.volume
        matrix_a[row_thermal, cell_index_fast] = -current_xs.fast.s_out * node.volume
        vector_b[cell_index_fast] = (
            (1 - self._beta)
            * current_xs.fast.nu_f
            * self.flux[cell_index_fast]
            * node.volume
            / k
        )
        vector_b[cell_index_fast] += (
            (1 - self._beta)
            * current_xs.th.nu_f
            * self.flux[cell_index_thermal]
            * node.volume
            / k
        )
        vector_b[cell_index_fast] += self.precursor_sum(node) / k
        vector_f[cell_index_fast] = (
            current_xs.fast.nu_f * node.volume * (1 - self._beta)
        )
        vector_f[cell_index_thermal] = (
            current_xs.th.nu_f * node.volume * (1 - self._beta)
        )
        return matrix_a, vector_b, vector_f

    def update_power(self):
        self.solver_strategy.solve()
        absolute_power = 0.0
        for index, node in enumerate(self._core_nodes):
            xs = self.get_xs(node)
            row_fast = index
            row_thermal = row_fast + self._ncell
            self._reaction_rate[node.index] = (
                xs.fast.nu_f * self.flux[row_fast] + xs.th.nu_f * self.flux[row_thermal]
            )
            self._power[node.index] = (
                xs.fast.kappa_f * self.flux[row_fast]
                + xs.th.kappa_f * self.flux[row_thermal]
            )
            absolute_power += self._power[node.index] * node.volume

        if self._steady_state:
            self.normalization_factor = self._initial_power / absolute_power
            print(f"Steady state power normalization: {self.normalization_factor}")
        self.total_power = absolute_power * self.normalization_factor
        self._power *= self.normalization_factor
        self._reaction_rate *= self.normalization_factor
        self.flux *= self.normalization_factor


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
        k_eff = 1.0
        for i in range(20):
            matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix(k_eff)
            self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
            f_new = (
                vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
            )
            k_eff = k_eff * f_new / f_old
            f_old = f_new
            print(f"Iteration {i}: k-eff = {k_eff}")
            if abs(f_new - f_old) / f_new < 1e-6:
                break


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
        self._relax = 0.75
        self._dt = dt
        self._order = order = int(scheme[-1])
        self._old_flux = np.zeros((order, 2 * self._diffusion._ncell))
        self._thermal_velocity = 2200
        self._fast_velocity = 1e7

    def set_up_current_data_container_interactions(self, data_container):
        for i in range(self._order):
            self._old_flux[i] = self._diffusion.flux

    def update_old_variable_array(self, data_container):
        self._old_flux = np.roll(self._old_flux, 1)
        self._old_flux[0] = self._diffusion.flux
        velocity = np.mean(data_container["velocity"])
        min_velocity = np.min(data_container["velocity"])
        power = self._diffusion.total_power

        def fmt(value):
            return f"{value:.8e}" if value is not None else "None"

        print(
            f"Avg Velocity: {fmt(velocity)}, Min Velocity: {fmt(min_velocity)}, Core Power: {fmt(power)}"
        )

    def solve(self):
        matrix_a, vector_b, _ = self._diffusion.set_up_matrix(k=1.0)
        gamma = self.gamma_vector[self._order - 1] / self._dt
        coefficient_vector = (
            self.coefficient_matrix[self._order - 1][: self._order] * gamma
        )

        for index, node in enumerate(self._diffusion._core_nodes):
            vector_b[index] += (
                node.volume
                * (self._old_flux.T[index] @ coefficient_vector)
                / self._fast_velocity
            )
            matrix_a[index, index] += node.volume * gamma / self._fast_velocity
            vector_b[index + self._diffusion._ncell] += (
                node.volume
                * (
                    self._old_flux.T[index + self._diffusion._ncell]
                    @ coefficient_vector
                )
                / self._thermal_velocity
            )
            matrix_a[
                index + self._diffusion._ncell, index + self._diffusion._ncell
            ] += (node.volume * gamma / self._thermal_velocity)

        self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
