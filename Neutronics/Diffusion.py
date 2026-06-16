from Neutronics import NeutronicsStrategy


class Diffusion(NeutronicsStrategy):
    """
    A class representing a diffusion neutronics strategy for fluid simulations.
    This will be 1D Diffusion.
    """

    def __init__(self, coreNodes, Power_0, substep,  beta):

        """
        Initializes a given solver object
        Args:
            - coreNodes (List[node]): List of core node objects
            - Power_0 (float): Initial power
            - substep (float): Substep size
            - beta (float): Beta value
        """
        print("Initializing Diffusion Neutronics Strategy")
        self._coreNodes         = coreNodes
        self._ncell             = len(coreNodes)
        self._P_0               = Power_0
        self._substeps          = substep
        self._beta              = beta
        self._steadyState       = True
        self._enthalpy          = None
        self._neutronPopulation = None
        self._precursorGroups   = None
        self._density           = None
        self._power             = None
        self._reactionRate      = None
        self.sigmaFission       = None
        self.solverStrategy     = None
        self.normalizationFactor  = None
        self.totalPower = None
        self.flux               = np.ones(2 * self._ncell) * 1e8  # 2 energy groups


    def setSolverMethod(self, simStrat):
        """
        Sets the solver method for the neutronics strategy.
        """
        solver_type = simStrat.solveType()
        print("Setting neutronics solver method to", solver_type)
        if solver_type == "SteadyState":
            self._steadyState = True
            self.solverStrategy = DiffusionSteadyStateStrategy(self)
        elif solver_type == "Transient":
            self.solverStrategy = DiffusionTransientStrategy(self, simStrat._dt, simStrat._ode_scheme)
            self._steadyState = False
        return self.solverStrategy

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

    def registerDataContainerVariables(self, datacontainer):
        """
        Registers the variables required for the neutronics strategy.
        """
        surfaces = []
        nodes = []
        for node in self._coreNodes:
            surfaces.append(node.inSurfaces[0].index)
            surfaces.append(node.outSurfaces[0].index)
            nodes.append(node.index)
        self._neutronPopulation = datacontainer.register('neutronPopulation', 1e8)
        self._reactionRate = datacontainer.register('reactionRate', 1e8)
        self._power = datacontainer.register('power', 1e8)
        for i in range(self._mesh.nConnections):
            if i not in surfaces:
                self._neutronPopulation[i] = 0.0
        for i in range(self._mesh.nNodes):
            if i not in nodes:
                self._reactionRate[i] = 0.0
                self._power[i] = 0.0

    def getXS(self, node):
        """
        Returns the cross-sections for the given node based on fluid temperature.
        Args:
            - node (Node): Node to get cross-sections for
        Returns:
            - XSSet: Cross-sections for the given node
        """
        class CrossSections:
            """
            A class representing the cross-sections for a given energy group.
            """
            def __init__(self, sigma_tr, sigma_a, nu_sigma_f, in_scatter, out_scatter, kappa_f):

                self.tr = sigma_tr
                self.a = sigma_a
                self.nu_f = nu_sigma_f
                self.s_in = in_scatter
                self.s_out = out_scatter
                self.D = 1/(3*self.tr)
                self.d = 0.7104/self.tr
                self.kappa_f = kappa_f

        class XSSet:
            """
            A class representing the cross-sections for both fast and thermal energy groups.
            """
            def __init__(self, fast: CrossSections, th: CrossSections):
                self.fast = fast
                self.th = th

        def fast_sigma_transport(temperature):
            return 0.2 - 1e-4 * (temperature - 300)

        def fast_sigma_absorption(temperature):
            return 0.01 + 5e-5 * (temperature - 300)

        def fast_nu_sigma_fission(temperature):
            return 0.03 - 1e-5 * (temperature - 300)

        def fast_in_scatter(temperature):
            return 0.005 + 2e-5 * (temperature - 300)

        def fast_out_scatter(temperature):
            return 0.02 + 3e-5 * (temperature - 300)

        def fast_kappa_fission(temperature):
            return 200 - 0.05 * (temperature - 300)

        def th_sigma_transport(temperature):
            return 0.1 - 5e-5 * (temperature - 300)

        def th_sigma_absorption(temperature):
            return 0.02 + 1e-5 * (temperature - 300)

        def th_nu_sigma_fission(temperature):
            return 0.01 - 5e-6 * (temperature - 300)

        def th_in_scatter(temperature):
            return 0.004 + 1e-5 * (temperature - 300)

        def th_out_scatter(temperature):
            return 0.01 + 2e-5 * (temperature - 300)

        def th_kappa_fission(temperature):
            return 180 - 0.03 * (temperature - 300)


        fluid_temp = self._fluidProperty.temperature(node.cellAverage(self._enthalpy))
        fast_xs = CrossSections(
            fast_sigma_transport(fluid_temp),
            fast_sigma_absorption(fluid_temp),
            fast_nu_sigma_fission(fluid_temp),
            fast_in_scatter(fluid_temp),
            fast_out_scatter(fluid_temp),
            fast_kappa_fission(fluid_temp)
        )
        th_xs = CrossSections(
            th_sigma_transport(fluid_temp),
            th_sigma_absorption(fluid_temp),
            th_nu_sigma_fission(fluid_temp),
            th_in_scatter(fluid_temp),
            th_out_scatter(fluid_temp),
            th_kappa_fission(fluid_temp)
        )
        return XSSet(fast_xs, th_xs)

    def precursorSum(self, node):
        precursor_sum = 0
        for index, group in enumerate(self._precursorGroups):
            precursor_sum += node.cellAverage(group) * node.cellAverage(self._density) * self._dataManager.decayConstants[index]
        return precursor_sum * node.volume

    def totalPrecursorSum(self):
        precursor_sum = 0
        for node in self._coreNodes:
            for index, group in enumerate(self._precursorGroups):
                precursor_sum += node.cellAverage(group) * node.cellAverage(self._density) * self._dataManager.decayConstants[index] * node.volume
        return precursor_sum

    def harmonicMean(self, x, y):
        return 2 * x * y / (x + y)

    def setUpMatrix(self, k = 1):
        A = np.zeros((2 * self._ncell , 2 * self._ncell ))
        b = np.zeros(2 * self._ncell )
        F = np.zeros(2 * self._ncell )
        previous_xs = None
        current_xs = self.getXS(self._coreNodes[0])
        next_xs = self.getXS(self._coreNodes[1])
        # BOTTOM VACUUM BC
        node = self._coreNodes[0]
        D_up_th = self.harmonicMean(current_xs.th.D, next_xs.th.D)
        D_up_fast = self.harmonicMean(current_xs.fast.D, next_xs.fast.D)
        diagonal_fast = current_xs.fast.a * node.volume + + D_up_fast/ node.volume + current_xs.fast.s_out * node.volume #outscatter
        diagonal_thermal = current_xs.th.a * node.volume +  D_up_th/ node.volume #+ current_xs.th.s_out * node.volume no upscatter
        A[0,0] = diagonal_fast + (current_xs.fast.D/((node.volume/2)+current_xs.fast.d))
        A[0,1] = -D_up_fast/node.volume
        A[self._ncell, self._ncell] =  diagonal_thermal + (current_xs.th.D/((node.volume/2)+current_xs.th.d))
        A[self._ncell, self._ncell + 1] = -D_up_th/node.volume
        A[self._ncell, 0] = -current_xs.fast.s_out *node.volume #downscatter
        b[0] = (1 - self._beta) * current_xs.fast.nu_f * self.flux[0] * node.volume / k
        b[0] += (1 - self._beta) * current_xs.th.nu_f * self.flux[self._ncell] * node.volume / k
        F[0] = current_xs.fast.nu_f * node.volume * (1 - self._beta)
        F[self._ncell] = current_xs.th.nu_f * node.volume * (1 - self._beta)
        b[0] += self.precursorSum(node)/ k

        for index, node in enumerate(self._coreNodes[1:-1], start=1):
            previous_xs = current_xs
            current_xs = next_xs
            next_xs = self.getXS(self._coreNodes[index + 1])
            row_fast = index
            row_thermal = row_fast + self._ncell
            cell_index_fast = index
            cell_index_thermal = index + self._ncell
            D_down_th = self.harmonicMean(previous_xs.th.D, current_xs.th.D)
            D_up_th = self.harmonicMean(current_xs.th.D, next_xs.th.D)
            D_down_fast = self.harmonicMean(previous_xs.fast.D, current_xs.fast.D)
            D_up_fast = self.harmonicMean(current_xs.fast.D, next_xs.fast.D)

            diagonal_fast = current_xs.fast.a * node.volume + (D_down_fast + D_up_fast)/ node.volume + current_xs.fast.s_out * node.volume #outscatter
            diagonal_thermal = current_xs.th.a * node.volume + (D_down_th + D_up_th)/ node.volume #+ current_xs.th.s_out * node.volume no upscatter


            A[row_fast, cell_index_fast] = diagonal_fast
            A[row_thermal, cell_index_thermal] = diagonal_thermal
            A[row_fast,cell_index_fast -1] = -D_down_fast / node.volume
            A[row_thermal, cell_index_thermal -1] = -D_down_th / node.volume
            A[row_fast, cell_index_fast +1] = - D_up_fast/node.volume
            A[row_thermal, cell_index_thermal +1] = - D_up_th/node.volume
            A[row_thermal, cell_index_fast] = -current_xs.fast.s_out *node.volume #downscatter

            # Fission Source
            b[row_fast] = (1 - self._beta) * current_xs.fast.nu_f * self.flux[row_fast] * node.volume / k
            b[row_fast] += (1 - self._beta) * current_xs.th.nu_f * self.flux[row_thermal] * node.volume / k
            b[row_fast] += self.precursorSum(node)/k
            F[row_fast] = current_xs.fast.nu_f * node.volume * (1 - self._beta)
            F[row_thermal] = current_xs.th.nu_f * node.volume * (1 - self._beta)

        # Top VACUUM BC
        row_fast +=1
        row_thermal +=1
        cell_index_fast +=1
        cell_index_thermal +=1
        previous_xs = current_xs
        current_xs = next_xs
        node = self._coreNodes[-1]
        D_down_th = self.harmonicMean(previous_xs.th.D, current_xs.th.D)
        D_down_fast = self.harmonicMean(previous_xs.fast.D, current_xs.fast.D)
        diagonal_fast = current_xs.fast.a * node.volume + + D_down_fast/ node.volume + current_xs.fast.s_out * node.volume #outscatter
        diagonal_thermal = current_xs.th.a * node.volume +  D_down_th/ node.volume #+ current_xs.th.s_out * node.volume no upscatter
        A[row_fast,cell_index_fast] = diagonal_fast + (current_xs.fast.D/((node.volume/2)+current_xs.fast.d))
        A[row_fast,cell_index_fast -1] = -D_down_fast/node.volume
        A[row_thermal,cell_index_thermal] =  diagonal_thermal + (current_xs.th.D/((node.volume/2)+current_xs.th.d))
        A[row_thermal,cell_index_thermal -1] = -D_down_th/node.volume
        A[row_thermal, cell_index_fast] = -current_xs.fast.s_out *node.volume #downscatter
        b[cell_index_fast] = (1 - self._beta) * current_xs.fast.nu_f * self.flux[cell_index_fast] * node.volume / k
        b[cell_index_fast] += (1 - self._beta) * current_xs.th.nu_f * self.flux[cell_index_thermal] * node.volume / k
        b[cell_index_fast] += self.precursorSum(node) / k
        F[cell_index_fast] = current_xs.fast.nu_f * node.volume * (1 - self._beta)
        F[cell_index_thermal] = current_xs.th.nu_f * node.volume * (1 - self._beta)
        return A, b, F

    def updateNeutronPopulation(self):
        """
        Updates the neutron population in the strategy.
        """
        self.solverStrategy.solve()
        absolute_power = 0
        for index, node in enumerate(self._coreNodes):
            xs = self.getXS(node)
            row_fast = index
            row_thermal = row_fast + self._ncell
            self._reactionRate[node.index] = xs.fast.nu_f * self.flux[row_fast] + xs.th.nu_f * self.flux[row_thermal]
            self._power[node.index] = xs.fast.kappa_f * self.flux[row_fast] + xs.th.kappa_f * self.flux[row_thermal]
            absolute_power += self._power[node.index] * node.volume

        if self._steadyState:
            self.normalizationFactor = self._P_0 / absolute_power
            print(f"Steady State Power Normalization: {self.normalizationFactor}")
        self.totalPower = absolute_power * self.normalizationFactor
        self._power *= self.normalizationFactor
        self._reactionRate *= self.normalizationFactor
        self.flux *= self.normalizationFactor

class DiffusionSolverStrategy(ABC):
    """
    A class representing a neutronics solver strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """
    @abstractmethod
    def solve(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """
        pass

class DiffusionSteadyStateStrategy(DiffusionSolverStrategy):
    """
    A class representing a steady-state neutronics strategy for fluid simulations.
    This will be Point Kinetics or 1D Diffusion.
    """

    def __init__(self, DiffusionStrategy):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._diffusion = DiffusionStrategy
        self._relax = 1.0


    def solve(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """
        A, b, F = self._diffusion.setUpMatrix()
        self._diffusion.flux = np.linalg.solve(A, b)
        F_old = F@self._diffusion.flux + self._diffusion.totalPrecursorSum()
        k_eff = 1.0
        for i in range (20):
            A, b, F = self._diffusion.setUpMatrix(k_eff)
            self._diffusion.flux = np.linalg.solve(A, b)
            F_new = F@self._diffusion.flux + self._diffusion.totalPrecursorSum()
            k_eff = k_eff * F_new / F_old
            F_old = F_new
            print(f"Iteration {i}: k-eff = {k_eff}")
            if abs(F_new - F_old) / F_new < 1e-6:
                break

class DiffusionTransientStrategy(DiffusionSolverStrategy):
    """
    A class representing a transient neutronics strategy for fluid simulations.
    This will be 1D Diffusion.
    """
    coefficent_matrix = np.array( [ [      1,        0,       0,        0,      0,       0],
                                    [    4/3,     -1/3,       0,        0,      0,       0],
                                    [  18/11,    -9/11,    2/11,        0,      0,       0],
                                    [  48/25,   -36/25,   16/25,    -3/25,      0,       0],
                                    [300/137, -300/137, 200/137,  -75/137, 12/137,       0],
                                    [360/147, -450/147, 400/147, -225/147, 72/147, -10/147] ] )
    gamma_vector = np.array([1, 3/2, 11/6, 25/12, 137/60, 147/60])

    def __init__(self, DiffusionStrategy, dt, scheme):
        """
        Initializes a given solver object

        Args:
            - coreNodes (List[int]): List of core node indices
            - mesh (Mesh): Mesh object containing the mesh information
            - fluidProperty (FluidProperty): FluidProperty object containing the fluid properties
        """
        self._diffusion = DiffusionStrategy
        self._relax = 0.75
        self._dt = dt
        self._order = order = int(scheme[-1])
        self._oldFlux = np.zeros((order, 2 * self._diffusion._ncell))  # 2 energy groups
        self._thermalVelocity = 2200 # m/s
        self._fastVelocity = 1e7 # m/s


    def setUpCurrentDataContainerInteractions(self, datacontainer):
        """
        Sets up interactions with required variables in data container like enthalpy used for material properties

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        for i in range(self._order):
            self._oldFlux[i] = self._diffusion.flux

    def updateOldVariableArray(self, datacontainer):
        """
        Updates the old variable array with the current neutron population

        Args:
            - datacontainer (datacontainer): Set up data container with all registered variables
        """
        self._oldFlux = np.roll(self._oldFlux, 1)
        self._oldFlux[0] = self._diffusion.flux
        # print("Updating Old Fluxes:")
        # print(f"Old Flux Updated: {self._oldFlux}")
        # core_temp = self._pk._fluidProperty.temperature(self._pk.getCoreAverage(self._pk._enthalpy))
        # T0 = self._pk._T0
        velocity = np.mean(datacontainer['velocity'])
        min_velocity = np.min(datacontainer['velocity'])
        # avg_temp = np.mean(self._pk._fluidProperty.temperature(datacontainer['enthalpy']))
        # n_coef = (self._pk._alpha() - self._pk._betaEffective) / self._pk._Lambda
        # core_inlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[0]])
        # core_outlet_temp = self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices[-1]])
        # max_core_temp = np.max(self._pk._fluidProperty.temperature(datacontainer['enthalpy'][self._pk._surface_indices]))
        power = self._diffusion.totalPower
        def fmt(x):
            return f"{x:.8e}" if (x is not None) else "None"
        # print(f"Neutron Population: {fmt(self._old_neutron_population)}, N Coef: {fmt(n_coef)}, Precursor Sum: {fmt(self._pk.getCorePrecursorSum())}, Reactivity Feedback Alpha: {fmt(alpha)} Core Power: {fmt(power)}\nCore Temperature: {fmt(core_temp)}, Initial Temperature: {fmt(T0)}, Avg Velocity: {fmt(velocity)}, Avg Temperature: {fmt(avg_temp)} Inlet Temp: {fmt(core_inlet_temp)}, Outlet Temp: {fmt(core_outlet_temp)} Max Core Temp: {fmt(max_core_temp)} ")
        print(f"Avg Velocity: {fmt(velocity)}, Min Velocity: {fmt(min_velocity)}, Core Power: {fmt(power)} ")
        # print(f"Max Old Fast Flux: {np.max(self._oldFlux[0][:self._diffusion._ncell])} Max Old Thermal Flux: {np.max(self._oldFlux[0][self._diffusion._ncell:])}")

    def solve(self):
        """
        Updates the neutron population in the strategy

        Args:
            - neutronPopulation (float): New neutron population
        """
        A, b, F = self._diffusion.setUpMatrix(k =1.0)
        gamma = self.gamma_vector[self._order - 1]/ self._dt
        coefficent_vector = self.coefficent_matrix[self._order - 1][:self._order] * gamma
        for index, n in enumerate(self._diffusion._coreNodes):
            # print(f"Updating flux at node {n.index}")
            # print(f"COEFFICENT VECTOR: {coefficent_vector}")
            # print(f"Old fast fluxes: {self._oldFlux.T[index]}   Old thermal fluxes: {self._oldFlux.T[index + self._diffusion._ncell]}")
            # print(f"Fast b addition; {(self._oldFlux.T[index]@ coefficent_vector)/ self._fastVelocity}")
            # print(f"A fast addition; {gamma/ self._fastVelocity}  A thermal addition; {gamma/ self._thermalVelocity}")
            # print(f"Thermal b addition; {(self._oldFlux.T[index + self._diffusion._ncell]@ coefficent_vector)/ self._thermalVelocity}")
            b[index] += n.volume * (self._oldFlux.T[index]@ coefficent_vector)/ self._fastVelocity
            A[index, index] += n.volume * gamma/ self._fastVelocity
            b[index + self._diffusion._ncell] += n.volume * (self._oldFlux.T[index + self._diffusion._ncell]@ coefficent_vector)/ self._thermalVelocity
            A[index + self._diffusion._ncell, index + self._diffusion._ncell] += n.volume * gamma/ self._thermalVelocity

        self._diffusion.flux = np.linalg.solve(A, b)