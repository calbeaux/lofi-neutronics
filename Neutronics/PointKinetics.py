from abc import ABC, abstractmethod
import numpy as np
from Neutronics.NeutronicsStrategy import NeutronicsStrategy

class PointKineticsStrategy(NeutronicsStrategy):

    def __init__(self, coreNodes, promptNeutronLifetime, betaEffective, sigmaFission, alphaT, Power_0, substep):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - promptNeutronLifetime (float): Prompt neutron lifetime (s)
            - betaEffective (float): Effective delayed neutron fraction (unitless)
            - sigmaFission (float): Macroscopic fission cross-section (1/m)
            - alphaT (float): Thermal expansion coefficient (1/K)
            - Power_0 (float): Initial power (W)
            - substep (int): Number of substeps for the transient simulation
        """
        self._coreNodes         = coreNodes
        self._Lambda            = promptNeutronLifetime
        self._betaEffective     = betaEffective
        self._modifiedBetaEffective = betaEffective
        self._alphaT            = alphaT
        self._P_0               = Power_0
        self._total_power       = Power_0
        self._substeps          = substep
        self._adjoint           = np.ones(len(self._coreNodes)) # should be replaced with actual adjoint values
        self._T0                = None
        self._precursorSource   = None
        self._temperature       = None
        self._power             = None
        self._solver            = None
        self._amplitude         = 1.0
        self._weights           = self.getWeights()
        self.coreVolume         = 0
        self.coreVolume         = sum(node.volume for node in self._coreNodes)
        self._N_0               = 2.43 * self._Lambda * Power_0/(3.24e-11)
        self._rho_0             = 0.0


    @property
    def totalPower(self):
        """
        Returns the total power
        """
        return self._total_power

    @property
    def amplitude(self):
        """
        Returns the amplitude
        """
        return self._amplitude

    @property
    def betaEffective(self):
        """
        Returns the effective delayed neutron fraction
        """
        return self._betaEffective

    @property
    def Lambda(self):
        """
        Returns the prompt neutron lifetime
        """
        return self._Lambda

    def setInitialReactivity(self, rho):
        self._rho_0 = rho

    def updateInitialTemperature(self):
        """
        Sets the initial temperature for the neutronics strategy

        Args:
            - T0 (float): Initial temperature
        """
        self._T0 = self.getCoreAverage(self._temperature)

    def setSolverMethod(self, state):
        """
        Sets the solver method for the neutronics strategy

        Args:
            - solver: Solver method to be used
        """
        solver_type = state.type
        print("Setting neutronics solver method to", solver_type)
        if solver_type == "SteadyState":
            self._solver = PointKineticsSteadyStateStrategy(self)
        elif solver_type == "Transient":
            if self._substeps is not None:
                self._solver = PointKineticsTransientStrategy(self, state.time_step / self._substeps)
            else:
                self._solver = PointKineticsTransientStrategy(self, state.time_step)
        return self._solver


    def setUpDataContainerInteractions(self, datacontainer):
        """
        Sets up interactions with required variables in data container like enthalpy used for material properties

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        self._precursorSource = datacontainer['precursor_source']
        self._temperature = datacontainer['temperature']

    def registerDataContainerVariables(self, datacontainer):
        """
        Registers the variables required for the neutronics strategy in the data container

        Args:
            - datacontainer (DataContainer): Data container to register variables in
        """
        self._power = datacontainer.register('power', 0.0)

    def getWeights(self):
        """
        Returns an *unnormalized* shape function (peak at center, zero at ends).
        Power normalization is handled separately in updatePower().
        """
        cell_positions = []
        for index, node in enumerate(self._coreNodes):
            L = node.length
            if index == 0:
                cell_positions.append(L / 2)
            else:
                cell_positions.append(cell_positions[-1] + L)

        total_length = cell_positions[-1] + self._coreNodes[-1].length / 2
        cell_weights = [np.sin(np.pi * x / total_length) for x in cell_positions]
        return cell_weights

    def getCoreAverage(self, data):
        """
        Returns the core average of a given data array

        Args:
            - data (np.array): Data array to get the core average of

        Returns:
            - float: Core average of the data array
        """
        average = 0
        for index, node in enumerate(self._coreNodes):
            average += data[node.index] * self._adjoint[index]/len(self._adjoint)
        return average


    def getReactivityFeedback(self):
        """
        Calculates the reactivity feedback coefficient alpha based on temperature

        Returns:
            - float: Reactivity feedback coefficient alpha
        """
        core_temp =  self.getCoreAverage(self._temperature)
        return self._alphaT * (core_temp - self._T0) + self._rho_0

    def updatePower(self):
        self._amplitude = self._solver.getNewAmplitude()
        self._total_power = self._amplitude * self._P_0

        weights = np.array(self._weights)
        node_volumes = np.array([node.volume for node in self._coreNodes])  # m^3

        # This denominator makes sum(power_i * V_i) == total_power exactly,
        # regardless of whether node volumes are uniform.
        normalization = np.sum(weights * node_volumes)

        for index, node in enumerate(self._coreNodes):
            self._power[node.index] = self._total_power * weights[index] / normalization

        print(f"Total power: {self._total_power}")
        delivered = sum(self._power[node.index] * node.volume for node in self._coreNodes)
        assert abs(delivered - self._total_power) < 1e-6 * self._total_power, \
        f"Power not conserved: {delivered} vs {self._total_power}"

    def getCorePrecursorSum(self):
        """
        Returns the sum of all precursor concentrations in the core

        Returns:
            - float: Sum of all precursor concentrations
        """

        total = 0
        for index, node in enumerate(self._coreNodes):
            total += self._precursorSource[node.index] * self._adjoint[index]
        return total/self._N_0

    def calculateSolution(self):
        total = 0
        for node in self._coreNodes:
            total += node.volume * self._power[node.index] * self._betaEffective /(self.Lambda * self._P_0)
        return total

class PointKineticsSolverStrategy(ABC):
    """
    A class representing a neutronics solver strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """
    @abstractmethod
    def getNewAmplitude(self):
        """
        Updates the amplitude in the strategy

        Args:
            - amplitude (float): New amplitude
        """
        pass

class PointKineticsSteadyStateStrategy(PointKineticsSolverStrategy):

    def __init__(self, PointKineticsStrategy):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._pk = PointKineticsStrategy


    def getNewAmplitude(self):
        """
        Updates the amplitude in the strategy

        Args:
            - amplitude (float): New amplitude
        """
        self._pk.updateInitialTemperature()
        # rho_0 = self._pk.betaEffective - self._pk.getCorePrecursorSum()*self._pk.Lambda
        # self._pk.setInitialReactivity(rho_0)
        self._pk._betaEffective = self._pk.getCorePrecursorSum()*self._pk.Lambda
        print(f"calculate solution: {self._pk.calculateSolution()}")
        return 1.0


class PointKineticsTransientStrategy(PointKineticsSolverStrategy):
    """
    Analytic solution to dp/dt = ((alpha - beta_eff)/Lambda) * p + sum(lambda_i * C_i)
    p(t + dt) = (p(t) + pc_sum/gamma)*np.exp(gamma * self._dt) - pc_sum/gamma
    """
    def __init__(self, PointKineticsStrategy, dt):
        self._pk = PointKineticsStrategy
        self._dt = dt

    def getNewAmplitude(self):
        """
        Updates the amplitude in the strategy

        Args:
            - amplitude (float): New amplitude
        """
        gamma = (self._pk.getReactivityFeedback() - self._pk.betaEffective) / self._pk.Lambda
        pc_sum = self._pk.getCorePrecursorSum()
        new_amplitude = (self._pk.amplitude + pc_sum/gamma)*np.exp(gamma * self._dt) - pc_sum/gamma
        print(f"rho: {self._pk.getReactivityFeedback()}")
        print(f"Core temperature: {self._pk.getCoreAverage(self._pk._temperature)} Initial temperature: {self._pk._T0}")
        print(f"Lambda: {self._pk.Lambda}")
        print(f"beta Effective: {self._pk.betaEffective}")
        print(f"Gamma: {gamma}")
        print(f"Core precursor sum: {self._pk.getCorePrecursorSum()}")
        print(f"New amplitude: {new_amplitude}")
        return new_amplitude
        # return 1.0

"""
class PointKineticsRK45TransientStrategy(PointKineticsSolverStrategy):

    # Transient RK45: dp/dt = ((alpha - beta_eff)/Lambda) * p + sum(lambda_i * C_i)

    def __init__(self, pk: PointKineticsStrategy, dt: float, rtol=1e-6, atol=1e-9, max_step=None):
        self._pk = pk
        self._dt = dt
        self._rtol = rtol
        self._atol = atol
        self._max_step = dt/2 if max_step is None else max_step
        self._relax = 1.0
        self._old_amplitude = self._pk.amplitude
        self._beta_effective = self._pk.betaEffective

    def updateOldVariableArray(self, datacontainer):

        # Updates the old variable array with the current neutron population

        # Args:
        #     - datacontainer (datacontainer): Set up data container with all registered variables


        self._old_amplitude = self._pk.amplitude
        self._beta_effective = self._pk.betaEffective

    def _rk45_step(self, n, dt):
        # Single RK45 step for neutron amplitude.

        def rhs(n_val):
            alpha = self._pk.getReactivityFeedback()
            Lambda = self._pk.Lambda
            pc_sum = self._pk.getCorePrecursorSum()/ Lambda
            gamma = (alpha - self._beta_effective) / Lambda
            print(f"rho: {alpha}")
            print(f"Lambda: {Lambda}")
            print(f"beta Effective: {self._beta_effective}")
            print(f"Core precursor sum: {pc_sum}")
            print(f"Gamma: {gamma}")
            print(f"RHS: {gamma * n_val + pc_sum}")
            return gamma * n_val + pc_sum

        # Classic RK4 coefficients
        k1 = rhs(n)
        k2 = rhs(n + 0.5 * dt * k1)
        k3 = rhs(n + 0.5 * dt * k2)
        k4 = rhs(n +       dt * k3)

        return n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def getNewAmplitude(self):
        print(f"Old amplitude: {self._old_amplitude}")
        new_amplitude = self._rk45_step(self._old_amplitude, self._dt)
        print(f"New amplitude: {new_amplitude}")
        return new_amplitude

    def getModifiedBetaEffective(self):

        coreTime = 0
        excoreTime = 0
        tank_node_indices = self._mesh.transverse_outlet_nodes
        for node in self._mesh.Nodes():
            if node.index not in tank_node_indices:
                if node not in self._coreNodes:
                    excoreTime += node.length / node.cellAverage(self._velocity)
                else:
                    coreTime += node.length / node.cellAverage(self._velocity)
        rho_0 = 0
        for index, beta_i in enumerate(self._dataManager.yieldTerms):
            lambda_i = self._dataManager.lossTerms[index]
            rho_0 += beta_i * (1 - lambda_i/(lambda_i + (1/coreTime) * (1 - np.exp(-lambda_i * excoreTime))))
        print(f"Rho_0: {rho_0} Original Beta: {self._betaEffective}  Modified Beta: {self._betaEffective - rho_0} Core Time: {coreTime}, Excore Time: {excoreTime}")
        # return self._betaEffective - rho_0
        return self._betaEffective

    def getCorePrecursorSum(self):

        precursor_sum = 0
        for index, group in enumerate(self._precursorGroups):
            for n in self._coreNodes:
                precursor_sum += n.cellAverage(group) * n.cellAverage(self._density) * self._dataManager.decayConstants[index] * n.volume
            # precursor_sum += self.getCoreAverage(group * self._density) * self._dataManager.decayConstants[index]
        return precursor_sum/self.coreVolume
"""
