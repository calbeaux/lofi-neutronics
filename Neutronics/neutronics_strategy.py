from abc import ABC, abstractmethod


class NeutronicsStrategy(ABC):
    """
    Abstract base class for neutronics strategies.
    """

    @abstractmethod
    def set_solver_method(self, state):
        """
        Sets the solver method for the neutronics strategy.
        """
        pass

    @abstractmethod
    def set_up_data_container_interactions(self, data_container):
        """
        Sets up interactions with required variables in data container.
        """
        pass

    @abstractmethod
    def register_data_container_variables(self, data_container):
        """
        Registers the variables required for the neutronics strategy.
        """
        pass

    @abstractmethod
    def update_power(self):
        """
        Updates the power in the strategy.
        """
        pass

    def set_core_nodes(self, core_nodes):
        """
        Sets the core nodes for the neutronics strategy.
        """
        self._core_nodes = core_nodes

    @property
    def core_volume(self):
        """
        Returns the total volume of the core.
        """
        return sum(node.volume for node in self._core_nodes)
