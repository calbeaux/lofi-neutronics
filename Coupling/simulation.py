import json
import h5py
from Coupling.node import Node
from Coupling.state import State
from Coupling.data_container import DataContainer
from Neutronics.point_kinetics import PointKineticsStrategy
from Neutronics.diffusion import Diffusion
from SyTH.syth import Syth


class Simulation:
    """
    A class to manage the simulation of a coupled neutronics and thermal-hydraulic system.
    """

    def __init__(self, input_file, verbose=True):
        self._syth = Syth(input_file)
        self._core_nodes = Simulation._get_core(input_file)
        self._input_dict = Simulation._get_input_json(input_file)
        self._run_null = bool(
            self._input_dict.get("neutronics", {}).get("null_transient", False)
        )
        self._neutronics = Simulation._build_neutronics_strategy(
            self._input_dict, self._core_nodes
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
        self._syth.set_neutron_precursor_source_coupled_to_power_factor(1 / self._kappa)
        self._syth.set_state(1)
        self._syth.set_state(0)

    @staticmethod
    def _get_core(input_file):
        """
        Return the list of core nodes stored in an HDF5 input file.

        The file is expected to contain groups under ``fluid_solve/fluid_mesh/nodes``.
        Each node group is inspected for a ``tag`` dataset. Nodes tagged ``core`` are
        converted into ``Node`` instances and returned in a list.
        """
        core_nodes = []

        def _decode_tag(value):
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if hasattr(value, "item"):
                value = value.item()
                if isinstance(value, bytes):
                    return value.decode("utf-8")
            return value

        with h5py.File(input_file, "r") as h5_file:
            nodes_group = h5_file["fluid_solve/fluid_mesh/nodes"]

            for node_name, node_obj in nodes_group.items():
                if not isinstance(node_obj, h5py.Group):
                    continue

                node_group = node_obj
                tag_dataset = node_group.get("tag")
                tag = tag_dataset[()] if isinstance(tag_dataset, h5py.Dataset) else None
                if _decode_tag(tag) != "core":
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

        return core_nodes

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
    def _build_neutronics_strategy(input_dict, core_nodes):
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
            )

        if strategy_type == "diffusion":
            beta_effective = neutronics_dict.get(
                "beta_effective", neutronics_dict.get("betaEffective")
            )
            initial_power = neutronics_dict.get(
                "initial_power", neutronics_dict.get("Power_0")
            )
            substeps = neutronics_dict.get("substeps", neutronics_dict.get("substep"))

            return Diffusion(core_nodes, initial_power, substeps, beta_effective)

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

    def receive_data(self):
        for node in self._core_nodes:
            self._temperature[node.index] = self._syth.get_fluid_node_value(
                "temperature", node.index
            )
            # self._temperature[node.index] = 1200.0
            precursor = self._syth.get_total_delayed_neutron_sources(node.index)
            self._precursor_source[node.index] = precursor

    def send_data(self):
        for node in self._core_nodes:
            self._syth.set_node_power(node.index, self._power[node.index])

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
            print(
                            f"Total precursors: {self.get_total_precursors() / (2.43 * 1e6 / self._kappa)}"
                        )

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
            print(
                f"Total precursors: {self.get_total_precursors() / (2.43 * 1e6 / self._kappa)}"
            )
            print(f"core temperature : {self._neutronics.get_core_temperature()}.  diff : {self._neutronics.get_core_temperature() - original_temp}")

    def _run_steady_state(self):
        self._neutronics.update_power()
        self.send_data()
        print("solving fluid flow")
        self._syth.solve_fluid_flow(delta_time=self._state.time_step)
        print("solving mass transport")
        self._syth.solve_mass_transport(
            delta_time=self._state.time_step, fluid_solve_tol=1e-12
        )
        self.receive_data()
        self._neutronics.update_power()
        self._syth.write_solution_to_input_file()
        print(
            f"Total precursors: {self.get_total_precursors() / (2.43 * 1e6 / self._kappa)}"
        )

    def get_total_precursors(self):
        total = 0
        for i in range(self._syth.get_node_count()):
            total += self._syth.get_total_delayed_neutron_sources(i)
        return total
