from __future__ import annotations

import ctypes

from . import syth_raw as _raw


class SyTH:
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

    def __enter__(self) -> "SyTH":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write_solution(self, filename: str) -> None:
        encoded = self._encode_text(filename)
        _raw.syTH_writeSolution(encoded, len(encoded))

    def write_solution_to_input_file(self) -> None:
        _raw.syTH_writeSolutionToInputFile()

    def solve_fluid_flow(self, fluid_solve_tol: float = 1e-6, delta_time: float = 0.0) -> None:
        _raw.syTH_solveFluidFlow(fluid_solve_tol, delta_time)

    def solve_mass_transport(self, fluid_solve_tol: float = 1e-6, delta_time: float = 0.0) -> None:
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

    def set_neutron_precursor_source_coupled_to_power_factor(self, factor: float) -> None:
        """
        Sets the neutron precursor source coupled to power factor.

        Args:
            - factor (float): The coupling factor.
        """
        _raw.syTH_setNeutronPrecursorSourceCoupledToPowerFactor(factor)

    def get_total_delayed_neutron_sources(self, node_index: int) -> float:
        """
        Gets the total delayed neutron sources for a node.

        Args:
            - node_index (int): The index of the node.

        Returns:
            float: The total delayed neutron sources.
        """
        return self._call_with_output(
            _raw.syTH_getTotalDelayedNeutronSources,
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
        return self._call_with_output(_raw.syTH_getNodeCount, output_type=_raw.SIK)
