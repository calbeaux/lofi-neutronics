class CrossSections:

    def __init__(self, sigma_tr, sigma_a, nu_sigma_f, in_scatter, out_scatter, kappa_f, velocity):
        cm_to_m = 100  # cm^-1 to m^-1 conversion factor
        self.tr = sigma_tr * cm_to_m
        self.a = sigma_a * cm_to_m
        self.nu_f = nu_sigma_f * cm_to_m
        self.s_in = in_scatter * cm_to_m
        self.s_out = out_scatter * cm_to_m
        self.kappa_f = kappa_f * cm_to_m
        self.d_coef = 1 / (3 * self.tr)
        self.extrapolation_distance = 0.7104 / self.tr
        self.velocity = velocity * 1 / cm_to_m  # m/s, for thermal neutrons

    """
    Both of these class methods recreate cross sections from specifc mpact runs.
    this is a temporary solution until we have a more robust way to generate cross sections from mpact.
    UNITS in cm^-1, except for velocity which is in cm/s and kappa_f which is in J/cm.
    """
    @classmethod
    def fast(cls, temperature):
        dt = temperature
        removal = 3.362333e-03 + 2.000000e-08 * dt
        out_group_scatter = 1.081667e-02 - 1.333333e-07 * dt
        return cls(
            sigma_tr=3.205000e-01 + 0.000000e+00 * dt,
            sigma_a= removal - out_group_scatter,
            nu_sigma_f=4.864500e-04 - 2.166667e-09 * dt,
            in_scatter=1.345000e+00 + 0.000000e+00 * dt,
            out_scatter= out_group_scatter,
            kappa_f=6.453167e-15 - 2.833333e-20 * dt,
            velocity=3.091500e+07 + 2.500000e+02 * dt,
        )

    @classmethod
    def thermal(cls, temperature):
        dt = temperature
        removal = 4.929500e-03 - 1.583333e-07 * dt
        out_group_scatter = 0.0 # no upscatter in thermal group
        return cls(
            sigma_tr=4.055167e-01 + 5.000000e-07 * dt,
            sigma_a= removal - out_group_scatter,
            nu_sigma_f=9.281000e-03 - 3.100000e-07 * dt,
            in_scatter=1.607000e+00 + 3.333333e-06 * dt,
            out_scatter= out_group_scatter,
            kappa_f=1.235167e-13 - 4.166667e-18 * dt,
            velocity=5.571000e+05 + 1.533333e+01 * dt,
        )


class XSSet:
    def __init__(self, temperature):
        self.fast = CrossSections.fast(temperature)
        self.th = CrossSections.thermal(temperature)
