from NeutronicsStrategy import NeutronicsStrategy

class PointKineticsStrategy(NeutronicsStrategy):
    """
    A class representing a neutronics strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """

    _joulesPerMeV = 1.6021773e-13

    def __init__(self, coreNodes, promptNeutronLifetime, betaEffective, sigmaFission, alphaT, Power_0, substep):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - promptNeutronLifetime (float): Prompt neutron lifetime (s)
            - betaEffective (float): Effective delayed neutron fraction (unitless)
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._coreNodes         = coreNodes
        self._Lambda            = promptNeutronLifetime
        self._betaEffective     = betaEffective
        self._sigmaFission      = sigmaFission
        self._alphaT            = alphaT
        self._P_0               = Power_0
        self._substeps          = substep
        self._steady_solve      = PointKineticsSteadyStateStrategy(self)
        self._T0                = None
        self._neutronPopulation = None
        self._precursorGroups   = None
        self._enthalpy          = None
        self._density           = None
        self._velocity          = None
        self._solver            = None
        self._n_t               = 0
        self._surface_indices, self._weights, self._cell_weights = self.getSurfacesAndPositions()
        self.coreVolume        = 0
        for node in self._coreNodes:
            self.coreVolume += node.volume

    @property
    def sigmaFission(self):
        """
        Returns the macroscopic fission cross-section [1/m]
        """
        return self._sigmaFission

    @property
    def neutronPopulation(self):
        """
        Returns the neutron population
        """
        return self._n_t

    def updateInitialTemperature(self):
        """
        Sets the initial temperature for the neutronics strategy

        Args:
            - T0 (float): Initial temperature
        """
        self._T0 = self._fluidProperty.temperature(self.getCoreAverage(self._enthalpy))

    def setSolverMethod(self, simStrat):
        """
        Sets the solver method for the neutronics strategy

        Args:
            - solver: Solver method to be used
        """
        solver_type = simStrat.solveType()
        print("Setting neutronics solver method to", solver_type)
        if solver_type == "SteadyState":
            self._solver = self._steady_solve
        elif solver_type == "Transient":
            if self._substeps is not None:
                # self._solver = PointKineticsTransientStrategy(self, simStrat._dt / self._substeps , simStrat._ode_scheme)
                self._solver = PointKineticsRK45TransientStrategy(self, simStrat._dt/self._substeps)
            else:
                # self._solver = PointKineticsTransientStrategy(self, simStrat._dt , simStrat._ode_scheme)
                self._solver = PointKineticsRK45TransientStrategy(self, simStrat._dt)
        return self._solver

    def setMultiScalarTransportManager(self, mstManager: MultipleScalarTransportManager.MultipleScalarTransportManager):
        """
        Sets the multiple scalar transport manager for the neutronics strategy

        Args:
            - mstManager (MultipleScalarTransportManager): Multiple scalar transport manager
        """
        self._dataManager = mstManager

    def setUpDataContainerInteractions(self, datacontainer: DataContainer.DataContainer):
        """
        Sets up interactions with required variables in data container like enthalpy used for material properties

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        self._precursorGroups = [
            datacontainer[f"{self._dataManager.variableName}_{i}"]
            for i in range(self._dataManager.nGroups)
        ]
        self._enthalpy = datacontainer['enthalpy']
        self._density  = datacontainer['density']
        self._velocity = datacontainer['velocity']

    def registerDataContainerVariables(self, datacontainer):
        """
        Registers the variables required for the neutronics strategy in the data container

        Args:
            - datacontainer (DataContainer): Data container to register variables in
        """
        self._neutronPopulation = datacontainer.register('neutronPopulation', 1e8)
        for i in range(self._mesh.nConnections):
            if i not in self._surface_indices:
                self._neutronPopulation[i] = 0.0

    def checkNeutronPopulationConsistency(self):
        """
        Checks the consistency of the neutron population in the strategy
        """
        total_population = 0
        for node in self._coreNodes:
            total_population += node.volume * node.cellAverage(self._neutronPopulation)
        expected_population = self.coreVolume * self._n_t
        if abs(total_population - expected_population) / expected_population > 1e-6:
            print(f"Warning: Neutron population inconsistency detected! Total: {total_population:.6e}, Expected: {expected_population:.6e}")

    def getSurfacesAndPositions(self):
        """
        Returns the surfaces and positions of the core nodes

        Returns:
            - List[Surface]: List of surfaces in the core
            - List[float]: List of positions of the core nodes
        """
        surfaces = []
        cell_positions = []
        surface_weights = [0]
        cell_weights = []
        for index, node in enumerate(self._coreNodes):
            L = node.length
            if index == 0:
                surfaces.append(node.inSurfaces[0].index)
                cell_positions.append(L/2)
            else:
                cell_positions.append(cell_positions[-1] + L)
            surfaces.append(node.outSurfaces[0].index)
        for x in cell_positions:
            cell_weights.append(np.sin(np.pi * x / cell_positions[-1]))
        total_weight = sum(cell_weights)
        cell_weights = [w * len(self._coreNodes) / total_weight for w in cell_weights]
        for index, w in enumerate(cell_weights):
            surface_weights.append(2 * w - surface_weights[-1])
        surface_weights[-1] = 0.0  # Ensure last weight is exactly zero
        print(f"Surface Weights: {surface_weights}")
        print(f"Cell Weights: {cell_weights}")
        print(f"cell positions: {cell_positions}")
        print(f"Total Weight: {sum(cell_weights)}")
        return surfaces, surface_weights, cell_weights

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
            average += node.cellAverage(data) * node.volume * self._cell_weights[index]
        return average/self.coreVolume

    def getModifiedBetaEffective(self):
        """
        Returns the modified effective delayed neutron fraction

        Returns:
            - float: Modified effective delayed neutron fraction
        """
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
        """
        Returns the core average of the precursor concentration times decay constant

        Returns:
            - float: Core average of the precursor concentration times decay constant
        """
        precursor_sum = 0
        for index, group in enumerate(self._precursorGroups):
            for n in self._coreNodes:
                precursor_sum += n.cellAverage(group) * n.cellAverage(self._density) * self._dataManager.decayConstants[index] * n.volume
            # precursor_sum += self.getCoreAverage(group * self._density) * self._dataManager.decayConstants[index]
        return precursor_sum/self.coreVolume

    def calculateSteadyStatePrecursors(self, neutron_pop):
        """
        Calculates the steady-state precursor concentrations
        C_i = (beta_i / (Lambda * lambda_i)) * n
        gets sum of lambda_i * C_i
        """
        steady_state = 0
        for index, group in enumerate(self._precursorGroups):
            steady_state +=  self._dataManager.yieldTerms[index] / self._Lambda
        return steady_state * neutron_pop

    def _alpha(self):
        """
        Calculates the reactivity feedback coefficient alpha based on temperature

        Returns:
            - float: Reactivity feedback coefficient alpha
        """
        core_temp =  self._fluidProperty.temperature(self.getCoreAverage(self._enthalpy))
        # print(f"Core Temperature: {core_temp}, T0: {self._T0}, Alpha: {self._alphaT * (core_temp - self._T0)}")
        return self._alphaT * (core_temp - self._T0)

    def updateBetaEffective(self, new_neutron_pop = None):
        """
        Updates the effective delayed neutron fraction based on reactivity feedback

        Returns:
            - float: Updated effective delayed neutron fraction
        """
        if new_neutron_pop is None:
            new_neutron_pop = self._n_t
        print(f"Old Beta Effective: {self._betaEffective}")
        self._betaEffective = self._Lambda * self.getCorePrecursorSum()/new_neutron_pop
        print(f"New Beta Effective: {self._betaEffective}")

    def updateNeutronPopulation(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """

        new_neutron_pop, relax = self._solver.getNewNeutronPopulation()
        self._n_t = new_neutron_pop
        for index in self._surface_indices:
            self._neutronPopulation[index] = new_neutron_pop * self._weights[index] * relax + self._neutronPopulation[index] * (1 - relax)
        # self.checkNeutronPopulationConsistency()


class PointKineticsSolverStrategy(ABC):
    """
    A class representing a neutronics solver strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """
    @abstractmethod
    def getNewNeutronPopulation(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """
        pass

class PointKineticsSteadyStateStrategy(PointKineticsSolverStrategy):
    """
    A class representing a steady-state neutronics strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """

    _joulesPerMeV = 1.6021773e-13
    def __init__(self, PointKineticsStrategy):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._pk = PointKineticsStrategy
        self._relax = 1.0


    def getNewNeutronPopulation(self):
        """
        Updates the neutron population in the strategy
        in units of neutrons/m^3

        Args:
            - neutronPopulation (float): New neutron population
        """
        new_neutron_pop =  self._pk._P_0/ (self._pk._sigmaFission * 100 * 200 * 2200 * self._pk._joulesPerMeV * self._pk.coreVolume)
        power = new_neutron_pop * self._pk._sigmaFission * 100 * 200 * 2200 * self._pk._joulesPerMeV * self._pk.coreVolume
        print(f"Steady State Neutron Population: {new_neutron_pop:.8e} Precursor Sum : {self._pk.getCorePrecursorSum():.8e} Precursor Sum Should be {self._pk.calculateSteadyStatePrecursors(new_neutron_pop):.8e}  Core Power: {power:.8e}")
        # print(f"Steady State Neutron Population: {new_neutron_pop} Precursor Sum : {self._pk.getCorePrecursorSum()} Precursor Sum Should be {self._pk.calculateSteadyStatePrecursors(new_neutron_pop)}  Core Power: {power}")
        self._pk.updateInitialTemperature()
        self._pk.updateBetaEffective(new_neutron_pop)
        return new_neutron_pop, self._relax

class PointKineticsTransientStrategy(PointKineticsSolverStrategy):
    """
    A class representing a transient neutronics strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """

    _joulesPerMeV = 1.6021773e-13
    coefficent_matrix = np.array( [ [      1,        0,       0,        0,      0,       0],
                                    [    4/3,     -1/3,       0,        0,      0,       0],
                                    [  18/11,    -9/11,    2/11,        0,      0,       0],
                                    [  48/25,   -36/25,   16/25,    -3/25,      0,       0],
                                    [300/137, -300/137, 200/137,  -75/137, 12/137,       0],
                                    [360/147, -450/147, 400/147, -225/147, 72/147, -10/147] ] )

    gamma_vector = np.array([1, 3/2, 11/6, 25/12, 137/60, 147/60])

    def __init__(self, PointKineticsStrategy, dt, scheme):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._pk = PointKineticsStrategy
        self._relax = 0.75
        self._dt = dt
        self._order = order = int(scheme[-1])
        self._gamma = self.gamma_vector[self._order - 1]/ self._dt
        self._coefficent_vector = self.coefficent_matrix[order - 1][:order] * self._gamma
        self._beta_effective = self._pk.getModifiedBetaEffective()
        self._oldNeutronPopulation = np.zeros(order)
        self._oldNeutronPopulationSubSteps = np.zeros(order)  # assuming max 10 substeps


    def setUpCurrentDataContainerInteractions(self, datacontainer):
        """
        Sets up interactions with required variables in data container like enthalpy used for material properties

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        for i in range(self._order):
            self._oldNeutronPopulation[i] = self._pk.neutronPopulation

    def updateOldVariableArray(self, datacontainer):
        """
        Updates the old variable array with the current neutron population

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        self._oldNeutronPopulation = np.roll(self._oldNeutronPopulationSubSteps, 1)
        self._oldNeutronPopulation[0] = self._pk.neutronPopulation
        self._beta_effective = self._pk.getModifiedBetaEffective()
        nt = self._oldNeutronPopulation[0]
        alpha = self._pk._alpha()
        core_temp = self._pk._fluidProperty.temperature(self._pk.getCoreAverage(self._pk._enthalpy))
        T0 = self._pk._T0
        velocity = np.mean(datacontainer['velocity'])
        avg_temp = np.mean(self._pk._fluidProperty.temperature(datacontainer['enthalpy']))
        n_coef = (self._pk._alpha() - self._pk._betaEffective) / self._pk._Lambda
        core_inlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[0]])
        core_outlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[-1]])

        power = nt * self._pk._sigmaFission * 100 * 200 * 2200 * self._pk._joulesPerMeV * self._pk.coreVolume
        self._pk.checkNeutronPopulationConsistency()
        def fmt(x):
            return f"{x:.8e}" if (x is not None) else "None"
        print(f"Neutron Population: {fmt(nt)}, N Coef: {fmt(n_coef)}, Precursor Sum: {fmt(self._pk.getCorePrecursorSum())}, Reactivity Feedback Alpha: {fmt(alpha)} Core Power: {fmt(power)}, Initial Temperature: {fmt(T0)}, Avg Velocity: {fmt(velocity)}, Avg Temperature: {fmt(avg_temp)} Inlet Temp: {fmt(core_inlet_temp)}, Outlet Temp: {fmt(core_outlet_temp)} ")

    def getNewNeutronPopulation(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """
        # substeps = 20
        # self._oldNeutronPopulationSubSteps = deepcopy(self._oldNeutronPopulation)
        n_coef = (self._pk._alpha() - self._beta_effective) / self._pk._Lambda
        # pc_sum = self._pk.getCorePrecursorSum()
        # for step in range(substeps):
        #     new_neutron_pop = self._oldNeutronPopulationSubSteps @ self._coefficent_vector * substeps + pc_sum
        #     new_neutron_pop *= 1/(self._gamma * substeps - n_coef)
        #     self._oldNeutronPopulationSubSteps = np.roll(self._oldNeutronPopulationSubSteps, 1)
        #     self._oldNeutronPopulationSubSteps[0] = float(new_neutron_pop)
        new_neutron_pop = self._oldNeutronPopulation @ self._coefficent_vector + self._pk.getCorePrecursorSum()
        new_neutron_pop *= 1/(self._gamma - n_coef)
        return new_neutron_pop, self._relax


class PointKineticsRK45TransientStrategy(PointKineticsSolverStrategy):
    """
    Transient RK45: dn/dt = ((alpha - beta_eff)/Lambda) * n + sum(lambda_i * C_i)
    """
    def __init__(self, pk: PointKineticsStrategy, dt: float, rtol=1e-6, atol=1e-9, max_step=None):
        self._pk = pk
        self._dt = dt
        self._rtol = rtol
        self._atol = atol
        self._max_step = dt/2 if max_step is None else max_step
        self._relax = 1.0
        self._old_neutron_population = None
        self._beta_effective = self._pk.getModifiedBetaEffective()

    def setUpCurrentDataContainerInteractions(self, datacontainer):
        """
        Sets up interactions with required variables in data container like enthalpy used for material properties

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        pass

    def updateOldVariableArray(self, datacontainer):
        """
        Updates the old variable array with the current neutron population

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        self._old_neutron_population = self._pk.neutronPopulation
        self._beta_effective = self._pk.getModifiedBetaEffective()
        alpha = self._pk._alpha()
        core_temp = self._pk._fluidProperty.temperature(self._pk.getCoreAverage(self._pk._enthalpy))
        T0 = self._pk._T0
        velocity = np.mean(datacontainer['velocity'])
        avg_temp = np.mean(self._pk._fluidProperty.temperature(datacontainer['enthalpy']))
        n_coef = (self._pk._alpha() - self._pk._betaEffective) / self._pk._Lambda
        core_inlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[0]])
        core_outlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[-1]])
        max_core_temp = np.max(self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices]))
        power = self._old_neutron_population * self._pk._sigmaFission * 100 * 200 * 2200 * self._pk._joulesPerMeV * self._pk.coreVolume
        def fmt(x):
            return f"{x:.8e}" if (x is not None) else "None"
        print(f"Neutron Population: {fmt(self._old_neutron_population)}, N Coef: {fmt(n_coef)}, Precursor Sum: {fmt(self._pk.getCorePrecursorSum())}, Reactivity Feedback Alpha: {fmt(alpha)} Core Power: {fmt(power)}\nCore Temperature: {fmt(core_temp)}, Initial Temperature: {fmt(T0)}, Avg Velocity: {fmt(velocity)}, Avg Temperature: {fmt(avg_temp)} Inlet Temp: {fmt(core_inlet_temp)}, Outlet Temp: {fmt(core_outlet_temp)} Max Core Temp: {fmt(max_core_temp)} ")


    def getNewNeutronPopulation(self):
        # initial state (core-averaged)
        y0 = np.array([self._old_neutron_population], dtype=float)

        # optionally freeze alpha over the fluid step
        alpha0 = self._pk._alpha()
        # print(f"old neutron population: {self._old_neutron_population:.8e}, alpha0: {alpha0:.8e}, beta_effective: {self._beta_effective:.8e}")
        def rhs(t, y):
            alpha = alpha0 if alpha0 is not None else self._pk._alpha()
            Lambda = self._pk._Lambda
            n_coef = (alpha - self._beta_effective) / Lambda
            return n_coef * y + self._pk.getCorePrecursorSum()

        sol = solve_ivp(
            rhs, (0.0, self._dt), y0, t_eval=[self._dt]
        )
        if not sol.success:
            raise RuntimeError(f"RK45 failed: {sol.message}")

        n_new = float(sol.y[0, -1])
        return n_new, self._relax