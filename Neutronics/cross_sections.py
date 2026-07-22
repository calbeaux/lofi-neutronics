class CrossSections:

    def __init__(self, sigma_tr, sigma_a, nu_sigma_f, in_scatter, out_scatter, kappa_f, velocity):
        cm_to_m = 100.0  # cm^-1 to m^-1 conversion factor
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
        return cls(
            sigma_tr=3.205000e-01 + 0.000000e+00 * dt,
            sigma_a= 3.362333e-03 + 2.000000e-08 * dt,
            nu_sigma_f=4.864500e-04 - 2.166667e-09 * dt,
            in_scatter=1.345000e+00 + 0.000000e+00 * dt,
            out_scatter= 1.081667e-02 - 1.333333e-07 * dt,
            kappa_f=6.453167e-15 - 2.833333e-20 * dt,
            velocity=3.091500e+07 + 2.500000e+02 * dt,
        )

    @classmethod
    def thermal(cls, temperature):
        dt = temperature
        return cls(
            sigma_tr=4.055167e-01 + 5.000000e-07 * dt,
            sigma_a= 4.929500e-03 - 1.583333e-07 * dt,
            nu_sigma_f=9.281000e-03 - 3.100000e-07 * dt,
            in_scatter=1.607000e+00 + 3.333333e-06 * dt,
            out_scatter=0.0,  # no upscatter in thermal group
            kappa_f=1.235167e-13 - 4.166667e-18 * dt,
            velocity=5.571000e+05 + 1.533333e+01 * dt,
        )


class XSSet:
    def __init__(self, temperature):
        self.fast = CrossSections.fast(temperature)
        self.th = CrossSections.thermal(temperature)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    temps = np.linspace(800, 1200, 50)
    rhos = []
    for temp in temps:
        xs = XSSet(temperature=temp)
        k_inf = (xs.fast.nu_f + (xs.fast.s_out / xs.th.a) * xs.th.nu_f) / (xs.fast.a + xs.fast.s_out)
        rho = (k_inf - 1) / k_inf
        rhos.append(rho)
        # print(f"Calculated k_inf: {k_inf}")
        # thermal_kappa = 2.43 *self.th.kappa_f / self.th.nu_f
        # fast_kappa = 2.43 * self.fast.kappa_f / self.fast.nu_f
        # print(f"Calculated thermal kappa: {thermal_kappa}")
        # print(f"Calculated fast kappa: {fast_kappa}")
        # fast_to_thermal_ratio = 2.601813e-02
        # weighted_kappa = fast_kappa * fast_to_thermal_ratio + thermal_kappa * (1 - fast_to_thermal_ratio)
        # print(f"Weighted kappa: {weighted_kappa}")

    temps = np.array(temps)
    rhos = np.array(rhos)

    # linear fit and slope
    coeffs = np.polyfit(temps, rhos, 1)
    slope, intercept = coeffs[0], coeffs[1]

    plt.figure()
    plt.plot(temps, rhos, 'o-', label='rho')
    plt.plot(temps, slope * temps + intercept, '--', label=f'fit slope={slope:.3e} per K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Reactivity rho')
    plt.title('Reactivity vs Temperature')
    plt.legend()
    plt.annotate(f'slope = {slope:.3e} /K', xy=(0.05, 0.05), xycoords='axes fraction')
    plt.grid(True)
    plt.show()
