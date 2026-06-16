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

    def specify_current_core_power(self, core_power: float) -> None:
        _raw.syTH_specifyCurrentCorePower(core_power)

    def set_depletion_time(self, current_time: float) -> None:
        _raw.syTH_setDepletionTime(current_time)

    def set_fluid_surface_value(
        self, value: float, variable_name: str, surface_index: int
    ) -> None:
        encoded = self._encode_text(variable_name)
        _raw.syTH_setFluidSurfaceValue(value, encoded, len(encoded), surface_index)

    def set_inlet_value(self, value: float, variable_name: str) -> None:
        encoded = self._encode_text(variable_name)
        _raw.syTH_setInletValue(value, encoded, len(encoded))

    def set_outlet_value(self, value: float, variable_name: str) -> None:
        encoded = self._encode_text(variable_name)
        _raw.syTH_setOutletValue(value, encoded, len(encoded))

    def set_node_power(self, node_index: int, node_power: float) -> None:
        _raw.syTH_setNodePower(node_index, node_power)

    def set_node_fission_rx_rate(self, iso_zzaaai: int, node_index: int, rx_rate: float) -> None:
        _raw.syTH_setNodeFissionRxRate(iso_zzaaai, node_index, rx_rate)

    def add_node_power(self, node_index: int, node_power: float) -> None:
        _raw.syTH_addNodePower(node_index, node_power)

    def set_species_source(self, species_id: int, node_index: int, volumetric_concentration: float) -> None:
        _raw.syTH_setSpeciesSource(species_id, node_index, volumetric_concentration)

    def set_species_decay_constant(self, species_id: int, decay_constant: float) -> None:
        _raw.syTH_setSpeciesDecayConstant(species_id, decay_constant)

    def set_burnup(self, node_index: int, burnup: float) -> None:
        _raw.syTH_setBurnup(node_index, burnup)

    def set_state(self, state_index: int) -> None:
        _raw.syTH_setState(state_index)

    def get_node_count(self) -> int:
        return self._call_with_output(_raw.syTH_getNodeCount, output_type=_raw.SIK)

    def get_node_bounds(self, node_index: int) -> tuple[float, float, float, float, float, float]:
        bounds = (_raw.SRK * 6)()
        _raw.syTH_getNodeBounds(node_index, bounds)
        return tuple(bounds)

    def get_node_volume(self, node_index: int) -> float:
        return self._call_with_output(_raw.syTH_getNodeVolume, node_index, output_type=_raw.SRK)

    def get_fluid_surface_value(self, variable_name: str, surface_index: int) -> float:
        encoded = self._encode_text(variable_name)
        return self._call_with_output(
            _raw.syTH_getFluidSurfaceValue,
            encoded,
            len(encoded),
            surface_index,
            output_type=_raw.SRK,
        )

    def get_fluid_inlet_value(self, variable_name: str) -> float:
        encoded = self._encode_text(variable_name)
        return self._call_with_output(
            _raw.syTH_getFluidInletValue, encoded, len(encoded), output_type=_raw.SRK
        )

    def get_fluid_outlet_value(self, variable_name: str) -> float:
        encoded = self._encode_text(variable_name)
        return self._call_with_output(
            _raw.syTH_getFluidOutletValue, encoded, len(encoded), output_type=_raw.SRK
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

    def add_species(self, species_name: str) -> int:
        encoded = self._encode_text(species_name)
        return self._call_with_output(_raw.syTH_addSpecies, encoded, len(encoded), output_type=_raw.SIK)

    def get_mod_node_avg_temp(self, node_index: int) -> float:
        return self._call_with_output(
            _raw.syTH_Get_Mod_Node_Avg_Temp, node_index, output_type=_raw.SRK
        )

    def get_iterations(self) -> int:
        return self._call_with_output(_raw.syTH_getIterations, output_type=_raw.SIK)

    def check_init(self) -> bool:
        value = ctypes.c_bool()
        _raw.syTH_checkInit(ctypes.byref(value))
        return value.value

    def get_num_states(self) -> int:
        return self._call_with_output(_raw.syTH_getNumStates, output_type=_raw.SIK)

    def check_current_core_power(self) -> float:
        return self._call_with_output(_raw.syTH_checkCurrentCorePower, output_type=_raw.SRK)

    def get_depletion_time(self) -> float:
        return self._call_with_output(_raw.syTH_getDepletionTime, output_type=_raw.SRK)

    def get_burnup(self, node_index: int) -> float:
        return self._call_with_output(_raw.syTH_getBurnup, node_index, output_type=_raw.SRK)

    def get_pressure_differential(self) -> float:
        return self._call_with_output(_raw.syTH_getPressureDifferential, output_type=_raw.SRK)

    def get_species(self, species_id: int, node_index: int) -> float:
        return self._call_with_output(
            _raw.syTH_getSpecies,
            species_id,
            node_index,
            output_type=_raw.SRK,
        )
