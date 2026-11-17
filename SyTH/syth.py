from __future__ import annotations

import ctypes

from . import syth_raw as _raw


class Syth:
    def __init__(self, filename: str):
        self.filename = filename
        self._initialize(filename)

    @staticmethod
    def _encode_text(value: str) -> bytes:
        return value.encode("utf-8")

    @classmethod
    def _initialize(cls, filename: str) -> None:
        encoded = cls._encode_text(filename)
        _raw.syTH_initialize(encoded, len(encoded), _raw.SYTH_API_VERSION)

    @staticmethod
    def _call_with_output(function, *args, output_type):
        output = output_type()
        function(*args, ctypes.byref(output))
        return output.value

    def close(self) -> None:
        _raw.syTH_finalize()

    def finalize(self) -> None:
        self.close()

    def __enter__(self) -> "Syth":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_solution(self, filename: str) -> None:
        encoded = self._encode_text(filename)
        _raw.syTH_writeSolution(encoded, len(encoded))

    def write_solution_to_input_file(self) -> None:
        _raw.syTH_writeSolutionToInputFile()

    def solve_fluid_flow(
        self, fluid_solve_tol: float = 1e-6, delta_time: float = 0.0
    ) -> None:
        _raw.syTH_solveFluidFlow(fluid_solve_tol, delta_time)

    def solve_mass_transport(
        self, fluid_solve_tol: float = 1e-6, delta_time: float = 0.0
    ) -> None:
        _raw.syTH_solveMassTransport(fluid_solve_tol, delta_time)

    def set_state(self, state_index: int) -> None:
        _raw.syTH_setState(state_index)

    def set_node_power(self, node_index: int, node_power: float) -> None:
        """
        Sets the power of a node.

        Args:
            - node_index (int): The index of the node.
            - node_power (float): Power in that volume [W/m^3]
        """
        _raw.syTH_setNodePower(node_index, node_power)

    def set_precursor_source_per_power(
        self, factor: float
    ) -> None:
        """
        Sets the precursor source per unit power.

        Args:
            - factor (float): The coupling factor.
        """
        _raw.syTH_setPrecursorSourcePerPower(factor)

    def stage_previous_concentration(
        self, species_id: int, node_index: int, concentration: float
    ) -> None:
        """Stage a previous cell concentration (#/m³) for a 1-based group ID."""
        _raw.syTH_stagePreviousConcentration(species_id, node_index, concentration)

    def get_delayed_neutron_emissions(self, node_index: int) -> float:
        """
        Gets the delayed neutron emissions for a node.

        Args:
            - node_index (int): The index of the node.

        Returns:
            float: The delayed neutron emissions.
        """
        return self._call_with_output(
            _raw.syTH_getDelayedNeutronEmissions,
            node_index,
            output_type=_raw.SRK,
        )

    def get_fluid_node_value(self, variable_name: str, node_index: int) -> float:
        encoded = self._encode_text(variable_name)
        return self._call_with_output(
            _raw.syTH_getFluidNodeValue,
            encoded,
            len(encoded),
            node_index,
            output_type=_raw.SRK,
        )

    def get_iterations(self) -> int:
        return self._call_with_output(_raw.syTH_getIterations, output_type=_raw.SIK)

    def get_node_count(self) -> int:
        """Return the full fluid-mesh count; the C API reports coupled nodes only."""
        import h5py

        with h5py.File(self.filename, "r") as input_file:
            return int(input_file["fluid_solve/fluid_mesh/nodes/number_of_nodes"][()])

    def get_coupled_node_count(self) -> int:
        return self._call_with_output(
            _raw.syTH_getCoupledNodeCount, output_type=_raw.SIK
        )

    def get_first_coupled_node(self) -> int:
        """Return the first coupled node's index; requires a nonempty coupled core."""
        return self._call_with_output(
            _raw.syTH_getFirstCoupledNode, output_type=_raw.SIK
        )
