from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import brentq

from Coupling.core_geometry import CoreGeometry
from Neutronics.radial_shape import RadialShape
from Neutronics.cross_sections import XSSet
from Neutronics.neutronics_strategy import NeutronicsStrategy


class Diffusion(NeutronicsStrategy):
    def __init__(self, core_nodes, initial_power, substep=1, beta=None, library=None, *, radial_buckling=False, radial_shape=None):
        if library is None and beta is None:
            raise ValueError(
                "Diffusion requires an MPACT library or beta_effective. "
                "Configure neutronics.mpact, pass library=..., or use driver.py --mpact-dir. "
                "MPACT beta is read from STATE_*/CORE_XS/Beta.")
        if library is None and (not np.isfinite(beta) or not 0 <= beta < 1):
            raise ValueError("beta_effective must be finite and in [0, 1).")
        if not isinstance(radial_buckling, bool):
            raise ValueError("radial_buckling must be a boolean.")
        if radial_buckling and library is None:
            raise ValueError("radial_buckling requires an MPACT library.")
        self._radial_config = radial_shape
        self.radial_buckling = radial_buckling
        self.library = library
        self.set_core_nodes(core_nodes)
        if library is not None:
            boundaries = library.metadata["boundaries"]
            if any(boundaries[b] != "vacuum" for b in ("bot", "top")):
                raise ValueError("The diffusion solver requires vacuum axial boundaries.")
        self._initial_power = initial_power
        self._substeps = substep if substep is not None and substep > 0 else 1
        self._beta = beta
        self._steady_state = True
        self.k_eff = 1.0
        self._temperature = None
        self._precursor_source = None
        self._power = None
        self.solver_strategy = None
        self.normalization_factor = 1.0
        self.flux = np.ones(2 * self._ncell) * 1e8
        self.tolerance = 1e-8
        self.max_iterations = 500

    def set_core_nodes(self, core_nodes):
        core = CoreGeometry.from_nodes(core_nodes)
        core.validate_diffusion()
        self._buckling_table = None
        self._radial_xs_cache = {}
        self._radial_beta_cache = {}
        super().set_core_nodes(core)
        self.radial = (RadialShape(self.core, self.library, self._radial_config)
                       if self._radial_config is not None else None)
        self._ncell = len(self._axial_nodes)
        self.flux = np.ones(2 * self._ncell) * 1e8
        self.k_eff = 1.0
        self.normalization_factor = 1.0
        # Geometry changes invalidate the solver and its time history.
        self.solver_strategy = None

    def set_solver_method(self, state):
        solver_type = state.state_type
        if solver_type == "steady_state":
            self._steady_state = True
            self.solver_strategy = DiffusionSteadyStateStrategy(self)
        elif solver_type == "transient":
            self.solver_strategy = DiffusionTransientStrategy(
                self, state.time_step/self._substeps, state.scheme
            )
            self._steady_state = False
        return self.solver_strategy

    def set_up_data_container_interactions(self, data_container):
        self._temperature = data_container["temperature"]
        self._precursor_source = data_container["precursor_source"]

    def register_data_container_variables(self, data_container):
        self._power = data_container.register("power", 0.0)

    def get_xs(self, node):
        if self.radial is not None:
            i = self.core.axial_index[node.index]
            temperatures = tuple(self._temperature[self.radial.indices[:,i]])
            cached = self._radial_xs_cache.get(i)
            if cached is None or cached[0] != temperatures:
                value = self.radial.collapse_xs(i, self._temperature, self.library,
                                               self.buckling_squared)
                self._radial_xs_cache[i] = (temperatures, value)
            return self._radial_xs_cache[i][1]
        if self.library is not None:
            temperature = self._temperature[node.index]
            return self.library.cross_sections(
                temperature, radial_buckling_squared=self.buckling_squared(temperature))
        return XSSet(self._temperature[node.index])

    @property
    def buckling_table(self):
        """Return a copy of the geometry-specific fitted [T (K), B² (1/m²)] table."""
        if not self.radial_buckling:
            return None
        if self._buckling_table is None:
            self._calibrate_buckling()
        return self._buckling_table.copy()

    def buckling_squared(self, temperature):
        if not self.radial_buckling:
            return 0.0
        # Use the same bounds policy as the cross-section library.
        self.library._interval(temperature)
        table = self.buckling_table
        return float(np.interp(temperature, table[:, 0], table[:, 1]))

    def static_eigenvalue(self, temperatures, *, buckling_squared=None):
        """Total-fission k for a prescribed axial temperature field, no precursors.

        A supplied scalar B² overrides calibration, including when it is zero.
        Otherwise use the configured local-temperature buckling table.
        """
        if self.library is None:
            raise ValueError("Static temperature eigenvalues require an MPACT library.")
        temperatures = np.broadcast_to(temperatures, (self._ncell,))
        if buckling_squared is not None and (
                not np.isfinite(buckling_squared) or buckling_squared < 0):
            raise ValueError("Buckling squared must be finite and nonnegative.")
        xs_values = [self.library.cross_sections(
            float(t), radial_buckling_squared=(self.buckling_squared(t)
                        if buckling_squared is None else buckling_squared))
                     for t in temperatures]
        A, _, vf = self._assemble_matrix(xs_values=xs_values, total_fission=True)
        F = np.zeros_like(A)
        for i in range(self._ncell):
            F[i, i] = vf[i]
            F[i, i + self._ncell] = vf[i + self._ncell]
        values = eigvals(F, A)
        physical = values.real[(np.abs(values.imag) < 1e-9)
                               & np.isfinite(values.real) & (values.real > 0)]
        if not len(physical):
            raise RuntimeError("No positive real diffusion eigenvalue.")
        return float(physical.max())

    def _calibrate_buckling(self):
        fitted = []
        for temperature in self.library.temperatures:
            target = self.library.kinetics(temperature).keff

            def residual(b2):
                return self.static_eigenvalue(temperature, buckling_squared=b2) - target

            at_zero = residual(0.0)
            context = f"at {temperature:g} K, target keff={target:.10g}"
            if abs(at_zero) <= 1e-10:
                root = 0.0
            elif at_zero < 0:
                raise ValueError("No nonnegative radial buckling can match MPACT " + context)
            else:
                upper = 1.0
                for _ in range(60):
                    if residual(upper) <= 0:
                        break
                    upper *= 2
                else:
                    raise ValueError("Cannot bracket radial buckling " + context)
                root = brentq(residual, 0.0, upper, xtol=1e-10)
            if abs(residual(root)) > 1e-8:
                raise RuntimeError("Radial buckling calibration failed " + context)
            fitted.append((temperature, root))
        self._buckling_table = np.array(fitted)

    def get_beta(self, node):
        if self.radial is not None:
            i = self.core.axial_index[node.index]
            weights = self.radial.weights[:,i]*self.radial.volumes[:,i]
            temperatures = tuple(self._temperature[self.radial.indices[:,i]])
            cached = self._radial_beta_cache.get(i)
            if cached is None or cached[0] != temperatures:
                value = np.average([self.library.kinetics(t).beta_effective
                                    for t in temperatures], weights=weights)
                self._radial_beta_cache[i] = (temperatures, value)
            return self._radial_beta_cache[i][1]
        if self.library is not None:
            return self.library.kinetics(self._temperature[node.index]).beta_effective
        return self._beta

    def fission_power_factor(self):
        fissions = power = 0.0
        for i, node in enumerate(self._axial_nodes):
            xs = self.get_xs(node)
            fast, thermal = self.flux[i], self.flux[i + self._ncell]
            volume = self.core.axial_volumes[i]
            fissions += volume * (xs.fast.f * fast + xs.th.f * thermal)
            power += volume * (xs.fast.kappa_f * fast + xs.th.kappa_f * thermal)
        if power <= 0:
            raise ValueError("Cannot normalize a nonpositive fission power.")
        return fissions / power

    def projected_precursor_source(self, node):
        if self.radial is None:
            return self._precursor_source[node.index]
        i = self.core.axial_index[node.index]
        return (self._precursor_source[self.radial.indices[:,i]].sum()
                / self.core.axial_volumes[i] * node.volume)

    def total_precursor_sum(self):
        return sum(
            self.projected_precursor_source(node) * (node.length / node.volume) for node in self._axial_nodes
        )
    def get_core_temperature(self):
        if self.radial is not None:
            return np.average(self._temperature[self.radial.indices],
                              weights=self.radial.volumes)
        return sum(
            self._temperature[node.index] * node.volume for node in self._axial_nodes
        ) / sum(n.volume for n in self._axial_nodes)

    @staticmethod
    def harmonic_mean(x, y):
        return 2 * x * y / (x + y)

    def set_up_matrix(self, k=1.0):
        return self._assemble_matrix(k)

    def _assemble_matrix(self, k=1.0, *, xs_values=None, total_fission=False):
        # Calibration and runtime share the complete spatial discretization.
        def get_xs(node):
            return (xs_values[self.core.axial_index[node.index]]
                    if xs_values is not None else self.get_xs(node))

        def get_beta(node):
            return 0.0 if total_fission else self.get_beta(node)

        def get_source(node):
            return 0.0 if total_fission else self.projected_precursor_source(node)

        matrix_a = np.zeros((2 * self._ncell, 2 * self._ncell))
        vector_b = np.zeros(2 * self._ncell)
        vector_f = np.zeros(2 * self._ncell)

        previous_xs = None
        current_xs = get_xs(self._axial_nodes[0])
        next_xs = get_xs(self._axial_nodes[1])

        node = self._axial_nodes[0]
        beta = get_beta(node)
        d_up_th = self.harmonic_mean(current_xs.th.d_coef, next_xs.th.d_coef)
        d_up_fast = self.harmonic_mean(current_xs.fast.d_coef, next_xs.fast.d_coef)
        diagonal_fast = (
            (current_xs.fast.a + getattr(current_xs.fast, "radial_leakage", 0.0)) * node.length
            + d_up_fast / node.length
            + current_xs.fast.s_out * node.length
        )
        diagonal_thermal = (current_xs.th.a + getattr(current_xs.th, "radial_leakage", 0.0)) * node.length + d_up_th / node.length
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
            (1 - beta) * current_xs.fast.nu_f * self.flux[0] * node.length / k
        )
        vector_b[0] += (
            (1 - beta)
            * current_xs.th.nu_f
            * self.flux[self._ncell]
            * node.length
            / k
        )
        vector_f[0] = current_xs.fast.nu_f * node.length * (1 - beta)
        vector_f[self._ncell] = current_xs.th.nu_f * node.length * (1 - beta)
        vector_b[0] += get_source(node) *  (node.length/node.volume) / k

        for index, node in enumerate(self._axial_nodes[1:-1], start=1):
            beta = get_beta(node)
            previous_xs = current_xs
            current_xs = next_xs
            next_xs = get_xs(self._axial_nodes[index + 1])
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
                (current_xs.fast.a + getattr(current_xs.fast, "radial_leakage", 0.0)) * node.length
                + (d_down_fast + d_up_fast) / node.length
                + current_xs.fast.s_out * node.length
            )
            diagonal_thermal = (
                (current_xs.th.a + getattr(current_xs.th, "radial_leakage", 0.0)) * node.length + (d_down_th + d_up_th) / node.length
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
                (1 - beta)
                * current_xs.fast.nu_f
                * self.flux[row_fast]
                * node.length
                / k
            )
            vector_b[row_fast] += (
                (1 - beta)
                * current_xs.th.nu_f
                * self.flux[row_thermal]
                * node.length
                / k
            )
            vector_b[row_fast] += get_source(node) * (node.length/node.volume) / k
            vector_f[row_fast] = current_xs.fast.nu_f * node.length * (1 - beta)
            vector_f[row_thermal] = current_xs.th.nu_f * node.length * (1 - beta)

        row_fast += 1
        row_thermal += 1
        cell_index_fast += 1
        cell_index_thermal += 1
        previous_xs = current_xs
        current_xs = next_xs
        node = self._axial_nodes[-1]
        beta = get_beta(node)
        d_down_th = self.harmonic_mean(previous_xs.th.d_coef, current_xs.th.d_coef)
        d_down_fast = self.harmonic_mean(
            previous_xs.fast.d_coef, current_xs.fast.d_coef
        )
        diagonal_fast = (
            (current_xs.fast.a + getattr(current_xs.fast, "radial_leakage", 0.0)) * node.length
            + d_down_fast / node.length
            + current_xs.fast.s_out * node.length
        )
        diagonal_thermal = (current_xs.th.a + getattr(current_xs.th, "radial_leakage", 0.0)) * node.length + d_down_th / node.length
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
            (1 - beta)
            * current_xs.fast.nu_f
            * self.flux[cell_index_fast]
            * node.length
            / k
        )
        vector_b[cell_index_fast] += (
            (1 - beta)
            * current_xs.th.nu_f
            * self.flux[cell_index_thermal]
            * node.length
            / k
        )
        vector_b[cell_index_fast] += get_source(node) * (node.length/node.volume) / k
        vector_f[cell_index_fast] = (
            current_xs.fast.nu_f * node.length * (1 - beta)
        )
        vector_f[cell_index_thermal] = (
            current_xs.th.nu_f * node.length * (1 - beta)
        )
        return matrix_a, vector_b, vector_f

    def update_power(self):
        self.solver_strategy.solve()
        axial_power = np.zeros(self._ncell)
        for index, node in enumerate(self._axial_nodes):
            xs = self.get_xs(node)
            row_fast = index
            row_thermal = row_fast + self._ncell
            if self.library is not None:
                axial_power[index] = (xs.fast.kappa_f * self.flux[row_fast]
                                          + xs.th.kappa_f * self.flux[row_thermal])
            else:
                axial_power[index] = (
                    3.24e-11 * xs.fast.nu_f * self.flux[row_fast] / 2.43
                    + 3.24e-11 * xs.th.nu_f * self.flux[row_thermal] / 2.43)
        total_power = axial_power @ self.core.axial_volumes
        if not np.isfinite(total_power) or total_power <= 0:
            raise ValueError("Cannot normalize a nonpositive fission power.")
        if self._steady_state:
            self.normalization_factor = self._initial_power / total_power
            self.flux *= self.normalization_factor
            total_power *= self.normalization_factor
            axial_power *= self.normalization_factor
        self.core.broadcast(axial_power, self._power, self.radial)
        self.total_power = total_power

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
        matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix(self._diffusion.k_eff)
        self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
        f_old = vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
        for i in range(self._diffusion.max_iterations):
            matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix(self._diffusion.k_eff)
            self._diffusion.flux = np.linalg.solve(matrix_a, vector_b)
            f_new = (
                vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
            )
            print(f"Delayed Neutron fraction: {self._diffusion.total_precursor_sum() / f_new}")
            print(f"Iteration {i}: f_new = {f_new}, f_old = {f_old}, k_eff = {self._diffusion.k_eff}")
            k_eff_new = self._diffusion.k_eff * f_new / f_old
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
        self._order = int(scheme[-1])
        #TODO: order should maybe be set to 2 or possibly set to theta method
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
            matrix_a, vector_b, vector_f = self._diffusion.set_up_matrix(
                k=self._diffusion.k_eff
            )

            for index, node in enumerate(self._diffusion._axial_nodes):
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
                f_new = (
                                vector_f @ self._diffusion.flux + self._diffusion.total_precursor_sum()
                            )
                print(f"Delayed Neutron fraction: {self._diffusion.total_precursor_sum() / f_new}")
                break
            elif i == self._diffusion.max_iterations - 1:
                print(f"Did not converge after {i} iterations with diff = {diff}")

