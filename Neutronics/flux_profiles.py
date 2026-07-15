import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def _sine_harmonics(x, A1, A3, A5, L, D):
    return A1*np.sin(np.pi*x/L) + A3*np.sin(3*np.pi*x/L) + A5*np.sin(5*np.pi*x/L) + D


def _fit_profile(profile, n_nodes=None):
    """Normalize a flux/adjoint profile and fit sine harmonics.

    If n_nodes differs from len(profile), evaluates the fitted curve at n_nodes
    uniformly-spaced points to interpolate to the mesh. Returns (profile, fit_params).
    """
    x_orig = np.linspace(0, 1, len(profile))
    normalized = profile / (np.sum(profile) / len(profile))
    amp = normalized.max() - normalized.min()
    try:
        popt, _ = curve_fit(
            _sine_harmonics, x_orig, normalized,
            p0=(amp, amp, amp, 1.0, normalized.min()),
        )
    except Exception:
        popt = (amp, 0.0, 0.0, 1.0, normalized.min())

    if n_nodes is not None and n_nodes != len(profile):
        x_new = np.linspace(0, 1, n_nodes)
        return _sine_harmonics(x_new, *popt), popt

    return normalized, popt

# Raw flux and adjoint profiles from mpact run
_RAW_ADJOINT = np.array([
    2.48E+04, 4.15E+04, 5.27E+04, 6.10E+04, 6.75E+04, 7.26E+04,
    7.63E+04, 7.87E+04, 7.99E+04, 7.99E+04, 7.87E+04, 7.63E+04,
    7.26E+04, 6.75E+04, 6.10E+04, 5.27E+04, 4.15E+04, 2.48E+04,
])
_RAW_FLUX = np.array([
    1.63E+01, 4.02E+01, 6.49E+01, 8.74E+01, 1.07E+02, 1.24E+02,
    1.37E+02, 1.46E+02, 1.50E+02, 1.50E+02, 1.46E+02, 1.37E+02,
    1.24E+02, 1.07E+02, 8.74E+01, 6.49E+01, 4.02E+01, 1.63E+01,
])

if __name__ == "__main__":
    profiles = {
        "Adjoint": _RAW_ADJOINT/(np.sum(_RAW_ADJOINT) / len(_RAW_ADJOINT)),
        "Flux": _RAW_FLUX/(np.sum(_RAW_FLUX) / len(_RAW_FLUX)),
    }

    fig, axes = plt.subplots(len(profiles), 1, figsize=(8, 7), sharex=True)
    if len(profiles) == 1:
        axes = [axes]

    for ax, (label, profile) in zip(axes, profiles.items()):
        fitted_profile, fit_params = _fit_profile(profile)
        x_raw = np.linspace(0, 1, len(profile))
        x_fit = np.linspace(0, 1, 200)
        fit_curve = _sine_harmonics(x_fit, *fit_params)

        ax.plot(x_raw, profile, "o", label=f"{label} raw")
        ax.plot(x_fit, fit_curve, "-", label=f"{label} fit")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

        print(f"{label} fitted profile:", fitted_profile)
        print(f"{label} fit parameters:", fit_params)

    axes[-1].set_xlabel("Normalized position")
    fig.tight_layout()
    plt.show()