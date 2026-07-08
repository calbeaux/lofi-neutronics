from abc import ABC, abstractmethod

class NeutronicsStrategy(ABC):
    """
    Abstract base class for neutronics strategies.
    """

    @abstractmethod
    def setSolverMethod(self, state):
        """
        Sets the solver method for the neutronics strategy.
        """
        pass

    @abstractmethod
    def setUpDataContainerInteractions(self, datacontainer):
        """
        Sets up interactions with required variables in data container.
        """
        pass

    @abstractmethod
    def registerDataContainerVariables(self, datacontainer):
        """
        Registers the variables required for the neutronics strategy.
        """
        pass

    @abstractmethod
    def updatePower(self):
        """
        Updates the power in the strategy.
        """
        pass

    def setCoreNodes(self, core_nodes):
        """
        Sets the core nodes for the neutronics strategy.
        """
        self._coreNodes = core_nodes

    @property
    def core_volume(self):
        """
        Returns the total volume of the core.
        """
        return sum(node.volume for node in self._coreNodes)
