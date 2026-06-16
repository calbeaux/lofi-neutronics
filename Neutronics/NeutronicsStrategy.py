from abc import ABC, abstractmethod

class NeutronicsStrategy(ABC):
    """
    Abstract base class for neutronics strategies.
    """

    @abstractmethod
    def setSolverMethod(self, simStrat):
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
    def updateNeutronPopulation(self):
        """
        Updates the neutron population in the strategy.
        """
        pass

    def setCoreNodes(self, core_nodes):
        """
        Sets the core nodes for the neutronics strategy.
        """
        self.core_nodes = core_nodes
