import json
from pathlib import Path
import h5py
import numpy as np
from Coupling.node import Node
from Coupling.core_geometry import CoreGeometry
from Coupling.state import State
from Coupling.data_container import DataContainer
from Neutronics.point_kinetics import PointKineticsStrategy
from Neutronics.diffusion import Diffusion
from Neutronics.mpact_library import MPACTLibrary


class Simulation:
    """
    A class to manage the simulation of a coupled neutronics and thermal-hydraulic system.
    """

    def __init__(self, input_file, verbose=True, library=None):
        self._core_nodes = Simulation._get_core(input_file)
        self._input_dict = Simulation._get_input_json(input_file)
        self._validate_precursor_transport(input_file)
        self._steady_state_max_iterations, self._steady_state_tolerance = (
            self._get_steady_state_settings(self._input_dict)
        )
        self._steady_state_iterations = 0
        self.core = CoreGeometry(self._core_nodes)
        self.library = library
        if self.library is None:
            config = self._input_dict.get("neutronics", {}).get("mpact")
            if config is not None:
                base = Path(input_file).expanduser().resolve().parent
                paths = [base / Path(p).expanduser() for p in config["files"]]
                self.library = MPACTLibrary.from_files(
                    paths, state=config.get("state", "STATE_0001"),
                    temperature_source=config.get("temperature_source", "dataset"))
        self._run_null = bool(
            self._input_dict.get("neutronics", {}).get("null_transient", False)
        )
        self._initial_power = self._input_dict.get("neutronics", {}).get("initial_power", 1e6)
        self._neutronics = Simulation._build_neutronics_strategy(
            self._input_dict, self.core, library=self.library
        )
        self._substeps = self._neutronics._substeps
        self.states = Simulation._get_states(self._input_dict)
        self._data_container = DataContainer(self._core_nodes[-1].index + 1)
        self._verbose = verbose
        self._temperature = None
        self._precursor_source = None
        self._power = None
        self._kappa = 3.24e-11  # joules per fission
        self._state = None
        # Import the native backend only when a simulation is actually created.
        from SyTH.syth import Syth
        self._syth = Syth(str(input_file))
        if self.library is None:
            self._syth.set_precursor_source_per_power(1 / self._kappa)
            self._syth.set_state(1)
        self._syth.set_state(0)

    @staticmethod
    def _get_steady_state_settings(input_dict):
        settings = input_dict.get("neutronics", {}).get("steady_state", {})
        if not isinstance(settings, dict):
            raise ValueError("neutronics.steady_state must be an object.")
        max_iterations = settings.get("max_iterations", 50)
        tolerance = settings.get("tolerance", 1e-6)
        if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or max_iterations < 2):
            raise ValueError("steady_state.max_iterations must be an integer >= 2.")
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not np.isfinite(tolerance) or not 0 < tolerance < 1):
            raise ValueError("steady_state.tolerance must be finite and between 0 and 1.")
        return max_iterations, tolerance

    @staticmethod
    def _validate_precursor_transport(input_file):
        # The native emissions getter dereferences the precursor manager. SyTH
        # creates that manager only when this equation is enabled in the HDF5.
        with h5py.File(input_file, "r") as f:
            key = "options/neutron_precursor_solve"
            enabled = key in f and bool(f[key][()])
        if not enabled:
            raise ValueError(
                "Coupled neutronics requires /options/neutron_precursor_solve=true "
                "in the SyTH HDF5 input. Add 'neutron_precursor_solve': true to "
                "the input JSON options and regenerate the HDF5 file. "
                "Otherwise SyTH cannot safely return delayed neutron emissions.")

    @staticmethod
    def _get_core(input_file):
        """
        Return the list of core nodes stored in an HDF5 input file.

        The file is expected to contain groups under ``fluid_solve/fluid_mesh/nodes``.
        Each node group is inspected for a ``tag`` dataset. Nodes tagged ``core`` are
        converted into ``Node`` instances and returned in a list.
        """
        core_nodes = []

        with h5py.File(input_file, "r") as h5_file:
            nodes_group = h5_file["fluid_solve/fluid_mesh/nodes"]

            for node_name, node_obj in nodes_group.items():
                if not isinstance(node_obj, h5py.Group):
                    continue

                node_group = node_obj
                is_coupled = bool(node_group["is_coupled"][()])
                if not is_coupled:
                    continue

                bounds_dataset = node_group.get("bounding_box")
                bounds = (
                    tuple(bounds_dataset[()])
                    if isinstance(bounds_dataset, h5py.Dataset)
                    else None
                )

                length_dataset = node_group.get("length")
                volume_dataset = node_group.get("volume")
                if not isinstance(length_dataset, h5py.Dataset) or not isinstance(
                    volume_dataset, h5py.Dataset
                ):
                    continue

                length = float(length_dataset[()])
                volume = float(volume_dataset[()])
                index = int(node_name.split("_")[-1])

                node = Node(length=length, volume=volume, index=index, bounds=bounds)
                node.tag = "core"
                core_nodes.append(node)

        return sorted(core_nodes, key=lambda node: node.index)

    @staticmethod
    def _get_input_json(input_file):
        """
        Extract the input JSON from the HDF5 file.
        """
        with h5py.File(input_file, "r") as h5f:
            metadata_group = h5f["metadata"]
            input_file_dataset = metadata_group["input_file"]
            input_payload = input_file_dataset[()]
            return json.loads(input_payload.decode("utf-8"))

    @staticmethod
    def _build_neutronics_strategy(input_dict, core_nodes, library=None):
        """
        Build a concrete neutronics strategy from a config dictionary. Should be in the syth json input file.
        """
        neutronics_dict = input_dict.get("neutronics", {})
        assert neutronics_dict, "No neutronics configuration found in input file."
        strategy_type = str(neutronics_dict.get("type", "")).strip().lower()

        if strategy_type in {"point_kinetics", "point kinetics", "pointkinetics", "pk"}:
            mean_neutron_lifetime = neutronics_dict.get(
                "mean_neutron_lifetime", neutronics_dict.get("meanNeutronLifetime")
            )
            beta_effective = neutronics_dict.get(
                "beta_effective", neutronics_dict.get("betaEffective")
            )
            alpha_t = neutronics_dict.get("alpha_T", neutronics_dict.get("alphaT"))
            initial_power = neutronics_dict.get(
                "initial_power", neutronics_dict.get("Power_0")
            )
            substeps = neutronics_dict.get("substeps", neutronics_dict.get("substep"))

            return PointKineticsStrategy(
                core_nodes,
                mean_neutron_lifetime,
                beta_effective,
                alpha_t,
                initial_power,
                substeps,
                library=library,
                radial_shape=neutronics_dict.get("radial_shape"),
            )

        if strategy_type == "diffusion":
            beta_effective = neutronics_dict.get(
                "beta_effective", neutronics_dict.get("betaEffective")
            )
            initial_power = neutronics_dict.get(
                "initial_power", neutronics_dict.get("Power_0")
            )
            substeps = neutronics_dict.get("substeps", neutronics_dict.get("substep"))

            return Diffusion(core_nodes, initial_power, substeps, beta_effective, library=library,
                             radial_buckling=neutronics_dict.get("radial_buckling", False),
                             radial_shape=neutronics_dict.get("radial_shape"))

        raise ValueError(f"Unsupported neutronics strategy type: {strategy_type!r}")

    @staticmethod
    def _get_states(input_dict):
        states = []
        if "states" not in input_dict:
            return [State("steadystate")]
        for dictionary in input_dict.get("states", {}):
            if "transient" in dictionary:
                states.append(State(dictionary["transient"]))
            else:
                states.append(State("steadystate"))
        return states

    def _register_data(self):
        self._temperature = self._data_container.register("temperature")
        self._precursor_source = self._data_container.register("precursor_source")
        self._neutronics.register_data_container_variables(self._data_container)

    def _set_up_data_container_interactions(self):
        self._power = self._data_container["power"]
        self._neutronics.set_up_data_container_interactions(self._data_container)

    def set_up_data_container(self):
        self._register_data()
        self._set_up_data_container_interactions()
        self.receive_data()
        if self.library is not None:
            if isinstance(self._neutronics, PointKineticsStrategy):
                self._neutronics.initialize_reference()
            self._update_fission_power_factor()

    def receive_data(self):
        for node in self._core_nodes:
            self._temperature[node.index] = self._syth.get_fluid_node_value(
                "temperature", node.index
            )

            precursor = self._syth.get_delayed_neutron_emissions(node.index)
            self._precursor_source[node.index] = precursor

    def send_data(self):
        self._update_fission_power_factor()
        for node in self._core_nodes:
            self._syth.set_node_power(node.index, self._power[node.index])

    def _update_fission_power_factor(self):
        if self.library is not None:
            self._syth.set_precursor_source_per_power(
                self._neutronics.fission_power_factor())

    def set_state(self, index: int, state: State):
        self._state = state
        self._neutronics.set_solver_method(state)
        self._syth.set_state(index)

    def run(self):
        if self._state.state_type == "steady_state":
            self._run_steady_state()
        elif self._state.state_type == "transient":
            if self._run_null:
                self._run_null_transient()
                self._run_null = False
            else:
                self._run_transient()
        else:
            raise ValueError(f"Unsupported state type: {self._state.state_type!r}")

    def _run_null_transient(self):
        current_time = self._state.time_step
        self._neutronics.set_solver_method(State("steadystate"))
        while current_time < self._state.time_duration:
            self._syth.solve_fluid_flow(delta_time=self._state.time_step, fluid_solve_tol=1e-6)
            self._syth.solve_mass_transport(delta_time=self._state.time_step, fluid_solve_tol=1e-12)
            self.receive_data()
            self._neutronics.update_power()
            self.send_data() # doesn't need to be here
            current_time += self._state.time_step
            print(f"Time: {current_time:.3f}")
            print(f"core temperature : {self._neutronics.get_core_temperature()}")
            print(self._precursor_diagnostic())

    def _run_transient(self):
        original_temp = self._neutronics.get_core_temperature()
        current_time = self._state.time_step
        while current_time < self._state.time_duration:
            print(f"Current time: {current_time}")
            print("solving fluid flow")
            self._syth.solve_fluid_flow(delta_time=self._state.time_step, fluid_solve_tol=1e-6)
            print("solving mass transport")
            self._syth.solve_mass_transport(delta_time=self._state.time_step, fluid_solve_tol=1e-12)
            for substep in range(self._substeps):
                print(f"Substep {substep + 1}/{self._substeps}")
                self.receive_data()
                self._neutronics.update_power()
                self.send_data()

            current_time += self._state.time_step
            print(self._precursor_diagnostic())
            print(f"core temperature : {self._neutronics.get_core_temperature()}.  diff : {self._neutronics.get_core_temperature() - original_temp}")

    def _run_steady_state(self):
        indices = [node.index for node in self._core_nodes]
        self._steady_state_iterations = 0
        self._neutronics.update_power()
        self.send_data()

        fields = {"power": self._power, "temperature": self._temperature,
                  "precursor_source": self._precursor_source}
        for iteration in range(1, self._steady_state_max_iterations + 1):
            previous = {name: values[indices].copy() for name, values in fields.items()}
            print(f"Coupled steady-state iteration {iteration}/{self._steady_state_max_iterations}")
            self._syth.solve_fluid_flow(delta_time=0.0, fluid_solve_tol=1e-6)
            self._syth.solve_mass_transport(delta_time=0.0, fluid_solve_tol=1e-12)
            self.receive_data()
            self._neutronics.update_power()
            self.send_data()
            self._steady_state_iterations = iteration

            errors = {}
            for name, values in fields.items():
                current = values[indices]
                if not np.all(np.isfinite(current)) or not np.all(np.isfinite(previous[name])):
                    raise RuntimeError(f"Non-finite {name} during coupled steady state.")
                scale = max(np.linalg.norm(current), np.linalg.norm(previous[name]))
                errors[name] = (float(np.linalg.norm(current - previous[name]) / scale)
                                if scale > 0 else 0.0)
            print("Coupled relative changes: " + ", ".join(
                f"{name}={error:.3e}" for name, error in errors.items()))
            if iteration >= 2 and max(errors.values()) <= self._steady_state_tolerance:
                print(f"Coupled steady state converged after {iteration} iterations")
                self._syth.write_solution_to_input_file()
                print(self._precursor_diagnostic())
                return

        raise RuntimeError(
            f"Coupled steady state did not converge after {self._steady_state_max_iterations} "
            f"iterations (tolerance={self._steady_state_tolerance:g}, changes={errors})."
        )

    def _precursor_diagnostic(self):
        source = self.get_total_precursors()
        if self.library is not None:
            return f"Total delayed neutron source: {source} neutrons/s"
        return f"Total precursors: {source / (2.43 * 1e6 / self._kappa)}"

    def get_total_precursors(self):
        total = 0
        for i in range(self._syth.get_node_count()):
            total += self._syth.get_delayed_neutron_emissions(i)
        return total
