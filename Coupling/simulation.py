import json
import h5py
from Coupling.node import Node
from DataContainer import DataContainer
from Neutronics.NeutronicsStrategy import NeutronicsStrategy
from Neutronics.PointKinetics import PointKineticsStrategy
from Neutronics.Diffusion import Diffusion
from SyTH.syth import SyTH

class Simulation:
    """
    A class to manage the simulation of a coupled neutronics and thermal-hydraulic system.
    """

    def __init__(self, input_file, verbose=True):
        self._core_nodes = Simulation._getCore(input_file)
        self._neutronics = Simulation._buildNeutronicsStrategy(input_file, self._core_nodes)
        self._syth = SyTH(input_file)
        self._datacontainer = DataContainer(self._core_nodes[-1].index)
        self._verbose = verbose
        self._temperature = None
        self._precursorConcentration = None
        self._power = None
        self._flux = None

    @staticmethod
    def _getCore(input_file):
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

            for node_name, node_group in nodes_group.items():
                tag = node_group["tag"][()] if "tag" in node_group else None
                if _decode_tag(tag) != "core":
                    continue

                bounds = tuple(node_group["bounding_box"][()]) if "bounding_box" in node_group else None
                length = float(node_group["length"][()])
                volume = float(node_group["volume"][()])
                index = int(node_name.split("_")[-1])

                node = Node(length=length, volume=volume, index=index, bounds=bounds)
                node.tag = "core"
                core_nodes.append(node)

        return core_nodes

    @staticmethod
    def _buildNeutronicsStrategy(input_file, core_nodes):
        """
        Build a concrete neutronics strategy from a config dictionary. Should be in the syth json input file.
        """
        with h5py.File(input_file, "r") as h5f:
            metadata_group = h5f["metadata"]
            input_file_dataset = metadata_group["input_file"]
            input_payload = input_file_dataset[()]
            input_dict = json.loads(input_payload.decode("utf-8"))

        neutronics_dict = input_dict.get("neutronics", {})
        assert neutronics_dict, "No neutronics configuration found in input file."
        strategy_type = str(neutronics_dict.get("type", "")).strip().lower()

        if strategy_type in {"point_kinetics", "point kinetics", "pointkinetics", "pk"}:
            prompt_neutron_lifetime = neutronics_dict.get("prompt_neutron_lifetime", neutronics_dict.get("promptNeutronLifetime"))
            beta_effective = neutronics_dict.get("beta_effective", neutronics_dict.get("betaEffective"))
            sigma_fission = neutronics_dict.get("sigma_fission", neutronics_dict.get("sigmaFission"))
            alpha_t = neutronics_dict.get("alpha_T", neutronics_dict.get("alphaT"))
            initial_power = neutronics_dict.get("initial_power", neutronics_dict.get("Power_0"))
            substeps = neutronics_dict.get("substeps", neutronics_dict.get("substep"))

            return PointKineticsStrategy(
                core_nodes,
                prompt_neutron_lifetime,
                beta_effective,
                sigma_fission,
                alpha_t,
                initial_power,
                substeps
            )

        if strategy_type == "diffusion":
            beta_effective = neutronics_dict.get("beta_effective", neutronics_dict.get("betaEffective"))
            initial_power = neutronics_dict.get("initial_power", neutronics_dict.get("Power_0"))
            substeps = neutronics_dict.get("substeps", neutronics_dict.get("substep"))

            return Diffusion(
                core_nodes,
                initial_power,
                substeps,
                beta_effective
            )

        raise ValueError(f"Unsupported neutronics strategy type: {strategy_type!r}")

    def _registerData(self):
        self._temperature = self._datacontainer.register("temperature")
        self._precursorConcentration = self._datacontainer.register("precursor_concentration")
        self._neutronics.registerDataContainerVariables(self._datacontainer)

    def _setUpDataContainerInteractions(self):
        self._power = self._datacontainer["power"]
        self._flux = self._datacontainer["flux"]
        self._neutronics.setUpDataContainerInteractions(self._datacontainer)

    def setUpDataContainer(self):
        self._registerData()
        self._setUpDataContainerInteractions()

    def receiveData(self):
        for node in self._core_nodes:
            self._temperature[node.index] = self._syth.get_fluid_node_value(node.index, "temperature")
            self._precursorConcentration[node.index] = self._syth.get_fluid_node_value(node.index, "precursor_concentration")

    def sendData(self):
        for node in self._core_nodes:
            self._syth.set_node_power(node.index, self._power[node.index])
            self._syth.set_node_fission_rx_rate(node.index, "precursor_concentration", self._flux[node.index])



