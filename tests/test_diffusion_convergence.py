"""
Convergence-order tests for the two-group diffusion solver (steady-state and
transient).

Unlike `test_diffusion.py` (which checks absolute accuracy against an
analytic bare-slab formula at a single, fixed resolution), these tests
perform a genuine order-of-accuracy study: solve at a sequence of
refinements, compare each to a much finer self-convergent reference, and fit
the observed order via log-log regression. This catches regressions that
change the *rate* of convergence even when a single-resolution check would
still pass.

Note: the analytic bare-slab formula (`analytic_keff_bare_slab`) is NOT used
as the reference here. It carries an intrinsic modeling bias (an "average
buckling" approximation shared across the two groups) that plateaus at a
fixed error regardless of mesh refinement, so it cannot anchor an order fit.
Self-convergence (comparing against a much finer numerical solution) is used
instead.

Run this file directly (`python tests/test_diffusion_convergence.py`) to
plot the convergence data on log-log axes alongside the fitted order.
"""

import sys
import os
import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from Coupling.state import State
from test_diffusion import (
    _XS, _make_xs_pair, _make_diffusion, _make_uniform_slab,
    _exact_critical_flux, CustomXSDiffusion,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def fit_order_of_accuracy(hs, errs):
    """
    Fit errs ~ C * hs**p via log-log least squares.

    hs, errs: sequences of equal length (>=3), hs decreasing, errs > 0.
    Returns (p, C, pairwise_orders) where pairwise_orders[i] is the local
    order estimated from consecutive points i, i+1 alone (diagnostic only).
    """
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    p, log_c = np.polyfit(np.log(hs), np.log(errs), 1)
    c = np.exp(log_c)
    pairwise_orders = [
        np.log(errs[i] / errs[i + 1]) / np.log(hs[i] / hs[i + 1])
        for i in range(len(hs) - 1)
    ]
    return float(p), float(c), pairwise_orders


# ---------------------------------------------------------------------------
# Spatial (steady-state) convergence
# ---------------------------------------------------------------------------

SPATIAL_SLAB_LENGTH = 200.0
SPATIAL_N_CELLS = [20, 40, 80, 160]
SPATIAL_N_REF = 1280


def _normalize_group(phi_group, dx):
    norm = np.sqrt(np.sum(phi_group ** 2) * dx)
    return phi_group / norm


def _solve_steady_state(n_cells, slab_length, fast_xs, th_xs):
    d, _ = _make_diffusion(n_cells=n_cells, slab_length=slab_length,
                           beta=0.0, fast_xs=fast_xs, th_xs=th_xs)
    state = State("steadystate")
    d.set_solver_method(state).solve()
    phi = d.flux.copy()
    mat, _, vf = d.set_up_matrix(k=1.0)
    k_eff = float((vf @ phi) / (mat @ phi).sum())
    return phi, k_eff


def compute_spatial_convergence_data(n_cells_list=SPATIAL_N_CELLS,
                                     n_ref=SPATIAL_N_REF,
                                     slab_length=SPATIAL_SLAB_LENGTH):
    """Return (dxs, keff_errs, flux_errs) for the given mesh-refinement sequence."""
    fast_xs, th_xs = _make_xs_pair()

    phi_ref, k_ref = _solve_steady_state(n_ref, slab_length, fast_xs, th_xs)
    dx_ref = slab_length / n_ref
    x_ref = (np.arange(n_ref) + 0.5) * dx_ref
    fast_ref = _normalize_group(phi_ref[:n_ref], dx_ref)
    th_ref = _normalize_group(phi_ref[n_ref:], dx_ref)

    dxs = []
    keff_errs = []
    flux_errs = []

    for n in n_cells_list:
        phi, k = _solve_steady_state(n, slab_length, fast_xs, th_xs)
        dx = slab_length / n
        x = (np.arange(n) + 0.5) * dx

        fast = _normalize_group(phi[:n], dx)
        th = _normalize_group(phi[n:], dx)

        fast_ref_interp = np.interp(x, x_ref, fast_ref)
        th_ref_interp = np.interp(x, x_ref, th_ref)

        # Fix arbitrary eigenvector sign against the reference.
        if np.dot(fast, fast_ref_interp) < 0:
            fast = -fast
            th = -th

        flux_err = np.sqrt(
            np.sum((fast - fast_ref_interp) ** 2) * dx
            + np.sum((th - th_ref_interp) ** 2) * dx
        )

        dxs.append(dx)
        keff_errs.append(abs(k - k_ref))
        flux_errs.append(flux_err)

    return dxs, keff_errs, flux_errs


class TestSpatialConvergence:
    """Mesh-refinement convergence study for the steady-state eigensolve."""

    def test_keff_and_flux_shape_converge_at_second_order(self):
        dxs, keff_errs, flux_errs = compute_spatial_convergence_data()

        assert all(keff_errs[i] > keff_errs[i + 1] for i in range(len(keff_errs) - 1)), (
            f"k_eff error did not decrease monotonically with refinement: {keff_errs}"
        )
        assert all(flux_errs[i] > flux_errs[i + 1] for i in range(len(flux_errs) - 1)), (
            f"Flux-shape error did not decrease monotonically with refinement: {flux_errs}"
        )

        p_keff, _, _ = fit_order_of_accuracy(dxs, keff_errs)
        p_flux, _, _ = fit_order_of_accuracy(dxs, flux_errs)

        assert 1.7 <= p_keff <= 2.3, (
            f"k_eff spatial order {p_keff:.2f} outside expected 2nd-order band [1.7, 2.3]; "
            f"errors={keff_errs}"
        )
        assert 1.5 <= p_flux <= 2.3, (
            f"Flux-shape spatial order {p_flux:.2f} outside expected 2nd-order band [1.5, 2.3]; "
            f"errors={flux_errs}"
        )


# ---------------------------------------------------------------------------
# Temporal (transient) convergence
# ---------------------------------------------------------------------------

TEMPORAL_N_CELLS = 10
TEMPORAL_SLAB_LENGTH = 200.0
TEMPORAL_DTS = [0.02, 0.01, 0.005, 0.0025]
TEMPORAL_T_FINAL = 0.04
TEMPORAL_REFERENCE_RATIO = 100


def _run_bdf_series(phi0, fast_xs, th_xs, n_cells, slab_length, scheme, dt, n_steps):
    """Run `n_steps` steps of `scheme` at step size `dt` starting from phi0."""
    d, _ = _make_diffusion(n_cells=n_cells, slab_length=slab_length,
                           beta=0.0, fast_xs=fast_xs, th_xs=th_xs)
    d.flux = phi0.copy()

    if scheme == "bdf1":
        state = State({"time_duration": dt, "time_step": dt, "scheme": "bdf1"})
        tr = d.set_solver_method(state)
        tr._old_flux[:] = phi0
        for _ in range(n_steps):
            tr.solve()
        return d.flux.copy()

    if scheme == "bdf2":
        # One BDF1 warmup step to get a real second history point.
        warmup_state = State({"time_duration": dt, "time_step": dt, "scheme": "bdf1"})
        warmup = d.set_solver_method(warmup_state)
        warmup._old_flux[:] = phi0
        warmup.solve()
        phi1 = d.flux.copy()

        state = State({"time_duration": dt, "time_step": dt, "scheme": "bdf2"})
        tr2 = d.set_solver_method(state)
        # update_old_variable_array() rolls history then overwrites slot 0
        # with the current flux (phi1). Setting slot 0 to phi0 here means
        # the first solve() sees post-roll history [phi1, phi0], i.e. the
        # correct two-point history (t, t-dt) = (phi1, phi0).
        tr2._old_flux[0] = phi0
        for _ in range(n_steps - 1):
            tr2.solve()
        return d.flux.copy()

    raise ValueError(f"Unknown scheme: {scheme}")


def _make_perturbed_xs(n_cells=TEMPORAL_N_CELLS, slab_length=TEMPORAL_SLAB_LENGTH):
    fast_xs, th_xs = _make_xs_pair()
    phi_crit, k_scipy = _exact_critical_flux(n_cells, slab_length, fast_xs, th_xs)
    fast_crit = _XS(fast_xs.tr, fast_xs.a, fast_xs.nu_f / k_scipy,
                    fast_xs.s_out, fast_xs.kappa_f, velocity=fast_xs.velocity)
    th_crit = _XS(th_xs.tr, th_xs.a, th_xs.nu_f / k_scipy,
                  th_xs.s_out, th_xs.kappa_f, velocity=th_xs.velocity)
    phi0, _ = _exact_critical_flux(n_cells, slab_length, fast_crit, th_crit)

    th_pert = _XS(th_crit.tr, th_crit.a * 1.20, th_crit.nu_f,
                 th_crit.s_out, th_crit.kappa_f, velocity=th_crit.velocity)
    return phi0, fast_crit, th_pert


def compute_temporal_convergence_data(scheme, dts=TEMPORAL_DTS, t_final=TEMPORAL_T_FINAL,
                                      reference_ratio=TEMPORAL_REFERENCE_RATIO,
                                      n_cells=TEMPORAL_N_CELLS,
                                      slab_length=TEMPORAL_SLAB_LENGTH):
    """Return (dts, errs) for the given scheme ("bdf1" or "bdf2") over the dt sequence."""
    phi0, fast_xs, th_xs = _make_perturbed_xs(n_cells, slab_length)

    dt_ref = dts[-1] / reference_ratio
    n_steps_ref = round(t_final / dt_ref)
    phi_ref = _run_bdf_series(phi0, fast_xs, th_xs, n_cells, slab_length,
                              scheme, dt_ref, n_steps_ref)

    errs = []
    for dt in dts:
        n_steps = round(t_final / dt)
        phi = _run_bdf_series(phi0, fast_xs, th_xs, n_cells, slab_length,
                              scheme, dt, n_steps)
        errs.append(np.linalg.norm(phi - phi_ref))

    return list(dts), errs


class TestTemporalConvergence:
    """Time-step-refinement convergence study for BDF1/BDF2 transient solves."""

    @pytest.mark.slow
    def test_bdf1_converges_at_first_order(self):
        dts, errs = compute_temporal_convergence_data("bdf1")
        p, _, _ = fit_order_of_accuracy(dts, errs)
        assert 0.7 <= p <= 1.3, (
            f"BDF1 temporal order {p:.2f} outside expected 1st-order band [0.7, 1.3]; "
            f"errors={errs}"
        )

    @pytest.mark.slow
    def test_bdf2_converges_at_second_order(self):
        dts, errs = compute_temporal_convergence_data("bdf2")
        p, _, _ = fit_order_of_accuracy(dts, errs)
        assert 1.6 <= p <= 2.4, (
            f"BDF2 temporal order {p:.2f} outside expected 2nd-order band [1.6, 2.4]; "
            f"errors={errs}"
        )


# ---------------------------------------------------------------------------
# Method of Manufactured Solutions (MMS) convergence
# ---------------------------------------------------------------------------
#
# Unlike the self-convergence studies above (which compare a solve against a
# much finer solve of the *same* method — proving consistency, not
# correctness), these tests compare against an independent closed-form exact
# solution, chosen so that it exactly satisfies the code's own discrete
# boundary condition.
#
# Important: the code's boundary term `D_g / (dx/2 + z_ext_g)` converges, as
# dx -> 0, to the *Robin* condition phi_g(0) = z_ext_g * phi_g'(0) — NOT to
# phi_g(-z_ext_g) = 0. A manufactured shape like sin(B*(x+z)) with
# B = pi/(L+2z) (the "linear extrapolation to zero" shape used elsewhere,
# e.g. analytic_keff_bare_slab) does NOT satisfy this and produces a
# dx-independent error plateau (measured fitted order: -0.02). Instead we
# solve for the group-specific B_g that satisfies the Robin condition
# exactly, and use phi_g(x) = A_g * cos(B_g * (x - L/2)).

def _find_robin_B(z_ext, slab_length):
    """
    Root of cos(B*L/2) - z_ext*B*sin(B*L/2) = 0 on (0, pi/L).

    This is the buckling for which A*cos(B*(x - L/2)) satisfies the Robin
    boundary condition phi(0) = z_ext*phi'(0) (and its mirror at x=L) exactly
    — the condition the code's boundary row actually converges to as dx -> 0.
    """
    def residual(b):
        return np.cos(b * slab_length / 2) - z_ext * b * np.sin(b * slab_length / 2)

    return brentq(residual, 1e-8, np.pi / slab_length - 1e-8)


def _mms_exact_flux(nodes, Bf, Bt, slab_length, t=0.0, alpha=0.0, a_fast0=1.0, a_th0=1.0):
    """Exact two-group manufactured flux phi_g(x,t) = A_g0*exp(alpha*t)*cos(B_g*(x-L/2))."""
    n = len(nodes)
    phi = np.zeros(2 * n)
    amp_fast = a_fast0 * np.exp(alpha * t)
    amp_th = a_th0 * np.exp(alpha * t)
    for node in nodes:
        x = (node.index + 0.5) * node.length
        phi[node.index] = amp_fast * np.cos(Bf * (x - slab_length / 2))
        phi[node.index + n] = amp_th * np.cos(Bt * (x - slab_length / 2))
    return phi


def _mms_source_vector(nodes, fast_xs, th_xs, Bf, Bt, slab_length,
                       t=0.0, alpha=0.0, a_fast0=1.0, a_th0=1.0):
    """
    Continuum source (reaction-rate density, scaled by node.length to match
    the code's convention) required for phi_g(x,t) = A_g0*exp(alpha*t)*
    cos(B_g*(x-L/2)) to exactly satisfy the two-group diffusion PDE with the
    given (nu_f = 0) cross sections:

        S_fast(x,t) = A_fast(t) * (alpha/v1 + D1*Bf^2 + Sa1 + Ss12) * cos(Bf(x-L/2))
        S_th(x,t)   = A_th(t) * (alpha/v2 + D2*Bt^2 + Sa2) * cos(Bt(x-L/2))
                      - Ss12 * A_fast(t) * cos(Bf(x-L/2))

    Setting alpha=0 (steady state) reduces this to the pure elliptic-operator
    source with no time-derivative contribution.
    """
    n = len(nodes)
    source = np.zeros(2 * n)
    d1, sa1, ss12, v1 = fast_xs.d_coef, fast_xs.a, fast_xs.s_out, fast_xs.velocity
    d2, sa2, v2 = th_xs.d_coef, th_xs.a, th_xs.velocity
    amp_fast = a_fast0 * np.exp(alpha * t)
    amp_th = a_th0 * np.exp(alpha * t)

    for node in nodes:
        x = (node.index + 0.5) * node.length
        cos_fast = np.cos(Bf * (x - slab_length / 2))
        cos_th = np.cos(Bt * (x - slab_length / 2))
        phi_fast = amp_fast * cos_fast

        s_fast = amp_fast * (alpha / v1 + d1 * Bf ** 2 + sa1 + ss12) * cos_fast
        s_th = amp_th * (alpha / v2 + d2 * Bt ** 2 + sa2) * cos_th - ss12 * phi_fast

        source[node.index] = s_fast * node.length
        source[node.index + n] = s_th * node.length

    return source


class MMSDiffusion(CustomXSDiffusion):
    """
    Diffusion subclass that adds an externally supplied manufactured source
    into vector_b every time set_up_matrix() is called. `source_fn(t)` must
    return a length-2*ncell vector (already scaled by node.length); `t` is
    read from `self._mms_time`, which the test driver advances manually.
    """

    def __init__(self, nodes, fast_xs, th_xs, beta=0.0):
        super().__init__(nodes, 1e6, 1, beta, fast_xs, th_xs)
        n = len(nodes)
        self._temperature = np.full(n, 300.0)
        self._precursor_source = np.zeros(n)
        self._power = np.zeros(n)
        self._mms_time = 0.0
        self.source_fn = lambda t: np.zeros(2 * n)

    def set_up_matrix(self, k=1.0):
        matrix_a, vector_b, vector_f = super().set_up_matrix(k=k)
        return matrix_a, vector_b + self.source_fn(self._mms_time), vector_f


# ---------------------------------------------------------------------------
# Spatial MMS (steady state)
# ---------------------------------------------------------------------------

SPATIAL_MMS_SLAB_LENGTH = 200.0
SPATIAL_MMS_N_CELLS = [20, 40, 80, 160, 320]


def compute_spatial_mms_data(n_cells_list=SPATIAL_MMS_N_CELLS, slab_length=SPATIAL_MMS_SLAB_LENGTH):
    """
    Return (dxs, errs): L2 error between the discrete steady-state solve and
    the exact manufactured solution, for each mesh in n_cells_list.

    No fine-mesh reference is needed here (unlike self-convergence) — the
    exact solution is closed-form, so error is measured directly against it.
    The eigenvalue search (DiffusionSteadyStateStrategy) is bypassed entirely:
    with nu_f=0 and a manufactured source, this is a single linear elliptic
    solve, not an eigenproblem.
    """
    fast_xs, th_xs = _make_xs_pair(nuf1=0.0, nuf2=0.0)
    Bf = _find_robin_B(fast_xs.extrapolation_distance, slab_length)
    Bt = _find_robin_B(th_xs.extrapolation_distance, slab_length)

    dxs = []
    errs = []
    for n in n_cells_list:
        _, phi_numerical, phi_exact = _solve_spatial_mms(n, slab_length, fast_xs, th_xs, Bf, Bt)
        dx = slab_length / n
        err = np.sqrt(np.sum((phi_numerical - phi_exact) ** 2) * dx)
        dxs.append(dx)
        errs.append(err)

    return dxs, errs


def _solve_spatial_mms(n_cells, slab_length, fast_xs, th_xs, Bf, Bt):
    """Solve the steady-state MMS problem on one mesh; return (x_centers, phi_numerical, phi_exact)."""
    nodes = _make_uniform_slab(n_cells, slab_length)
    d = MMSDiffusion(nodes, fast_xs, th_xs)
    source = _mms_source_vector(nodes, fast_xs, th_xs, Bf, Bt, slab_length, t=0.0, alpha=0.0)
    d.source_fn = lambda t, source=source: source
    d.flux = np.zeros(2 * n_cells)

    matrix_a, vector_b, _ = d.set_up_matrix(k=1.0)
    phi_numerical = np.linalg.solve(matrix_a, vector_b)
    phi_exact = _mms_exact_flux(nodes, Bf, Bt, slab_length, t=0.0, alpha=0.0)

    dx = slab_length / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    return x, phi_numerical, phi_exact


def compute_spatial_mms_profile(n_cells, slab_length=SPATIAL_MMS_SLAB_LENGTH):
    """
    Return (x, fast_numerical, th_numerical, fast_exact, th_exact) for a single
    mesh — for visualizing the discrete flux shape against the exact
    manufactured solution, as opposed to a single scalar error.
    """
    fast_xs, th_xs = _make_xs_pair(nuf1=0.0, nuf2=0.0)
    Bf = _find_robin_B(fast_xs.extrapolation_distance, slab_length)
    Bt = _find_robin_B(th_xs.extrapolation_distance, slab_length)
    x, phi_numerical, phi_exact = _solve_spatial_mms(n_cells, slab_length, fast_xs, th_xs, Bf, Bt)
    return (x, phi_numerical[:n_cells], phi_numerical[n_cells:],
           phi_exact[:n_cells], phi_exact[n_cells:])


class TestSpatialMMS:
    """Manufactured-solution convergence study for the steady-state elliptic operator."""

    def test_flux_converges_to_manufactured_solution_at_second_order(self):
        dxs, errs = compute_spatial_mms_data()

        assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), (
            f"MMS error did not decrease monotonically with refinement: {errs}"
        )

        p, _, _ = fit_order_of_accuracy(dxs, errs)
        assert 1.8 <= p <= 2.2, (
            f"Spatial MMS order {p:.2f} outside expected 2nd-order band [1.8, 2.2]; errors={errs}"
        )


# ---------------------------------------------------------------------------
# Temporal MMS (transient)
# ---------------------------------------------------------------------------

TEMPORAL_MMS_N_CELLS = 200
TEMPORAL_MMS_SLAB_LENGTH = 200.0
TEMPORAL_MMS_ALPHA = 10.0
TEMPORAL_MMS_A_FAST0 = 1e-3
TEMPORAL_MMS_A_TH0 = 1.0
TEMPORAL_MMS_DTS = [0.02, 0.01, 0.005, 0.0025]
TEMPORAL_MMS_T_FINAL = 0.08


def _run_mms_transient(scheme, dt, n_steps, nodes, fast_xs, th_xs, Bf, Bt, slab_length,
                       alpha, a_fast0, a_th0):
    """
    Run `n_steps` of `scheme` at step size `dt`, driving the manufactured
    source at the *new* time level each step (evaluating at t_n instead of
    t_{n+1} silently drops BDF2 to 1st order — confirmed empirically).

    _relax is set to 0 because, with nu_f=0, vector_b no longer depends on
    the flux being solved for, so the Picard loop's fixed point is reached
    in a single substitution (no damping needed) — measured bit-identical to
    the default relax=0.55, just far fewer iterations.
    """
    d = MMSDiffusion(nodes, fast_xs, th_xs)
    d.source_fn = lambda t: _mms_source_vector(
        nodes, fast_xs, th_xs, Bf, Bt, slab_length, t=t, alpha=alpha,
        a_fast0=a_fast0, a_th0=a_th0,
    )
    d.flux = _mms_exact_flux(nodes, Bf, Bt, slab_length, t=0.0, alpha=alpha,
                             a_fast0=a_fast0, a_th0=a_th0)

    state = State({"time_duration": dt, "time_step": dt, "scheme": scheme})
    tr = d.set_solver_method(state)
    tr._relax = 0.0

    if scheme == "bdf2":
        # update_old_variable_array() rolls history then overwrites slot 0
        # with the current flux, so pre-roll slot 0 becomes post-roll slot 1
        # (the "older", t-dt history point) on the very first solve() call.
        phi_before = _mms_exact_flux(nodes, Bf, Bt, slab_length, t=-dt, alpha=alpha,
                                     a_fast0=a_fast0, a_th0=a_th0)
        tr._old_flux[0] = phi_before

    for step in range(n_steps):
        d._mms_time = (step + 1) * dt
        tr.solve()

    return d.flux.copy()


def compute_temporal_mms_data(scheme, dts=TEMPORAL_MMS_DTS, t_final=TEMPORAL_MMS_T_FINAL,
                              n_cells=TEMPORAL_MMS_N_CELLS, slab_length=TEMPORAL_MMS_SLAB_LENGTH,
                              alpha=TEMPORAL_MMS_ALPHA, a_fast0=TEMPORAL_MMS_A_FAST0,
                              a_th0=TEMPORAL_MMS_A_TH0):
    """
    Return (dts, errs) for the given scheme ("bdf1"/"bdf2") against the exact
    manufactured solution at t=t_final.

    n_cells is fixed (not refined) so spatial truncation error stays a fixed,
    small floor and doesn't contaminate the fitted temporal order. a_fast0 is
    deliberately much smaller than a_th0 so the thermal group (whose larger
    alpha/velocity ratio makes its temporal truncation term dominate the
    error norm) sets the observed order, rather than the fast group's error
    being swamped by its own spatial floor.
    """
    fast_xs, th_xs = _make_xs_pair(nuf1=0.0, nuf2=0.0)
    Bf = _find_robin_B(fast_xs.extrapolation_distance, slab_length)
    Bt = _find_robin_B(th_xs.extrapolation_distance, slab_length)
    nodes = _make_uniform_slab(n_cells, slab_length)

    errs = []
    for dt in dts:
        n_steps = round(t_final / dt)
        phi_final = _run_mms_transient(scheme, dt, n_steps, nodes, fast_xs, th_xs, Bf, Bt,
                                       slab_length, alpha, a_fast0, a_th0)
        phi_exact = _mms_exact_flux(nodes, Bf, Bt, slab_length, t=t_final, alpha=alpha,
                                    a_fast0=a_fast0, a_th0=a_th0)
        dx = slab_length / n_cells
        err = np.sqrt(np.sum((phi_final - phi_exact) ** 2) * dx)
        errs.append(err)

    return list(dts), errs


def compute_temporal_mms_profile(scheme, dt, t_final=TEMPORAL_MMS_T_FINAL,
                                 n_cells=TEMPORAL_MMS_N_CELLS, slab_length=TEMPORAL_MMS_SLAB_LENGTH,
                                 alpha=TEMPORAL_MMS_ALPHA, a_fast0=TEMPORAL_MMS_A_FAST0,
                                 a_th0=TEMPORAL_MMS_A_TH0):
    """
    Return (x, fast_numerical, th_numerical, fast_exact, th_exact) at t=t_final
    for a single (scheme, dt) — for visualizing the discrete transient flux
    shape against the exact manufactured solution.
    """
    fast_xs, th_xs = _make_xs_pair(nuf1=0.0, nuf2=0.0)
    Bf = _find_robin_B(fast_xs.extrapolation_distance, slab_length)
    Bt = _find_robin_B(th_xs.extrapolation_distance, slab_length)
    nodes = _make_uniform_slab(n_cells, slab_length)

    n_steps = round(t_final / dt)
    phi_numerical = _run_mms_transient(scheme, dt, n_steps, nodes, fast_xs, th_xs, Bf, Bt,
                                       slab_length, alpha, a_fast0, a_th0)
    phi_exact = _mms_exact_flux(nodes, Bf, Bt, slab_length, t=t_final, alpha=alpha,
                                a_fast0=a_fast0, a_th0=a_th0)

    dx = slab_length / n_cells
    x = (np.arange(n_cells) + 0.5) * dx
    return (x, phi_numerical[:n_cells], phi_numerical[n_cells:],
           phi_exact[:n_cells], phi_exact[n_cells:])


class TestTemporalMMS:
    """Manufactured-solution convergence study for BDF1/BDF2 transient time-stepping."""

    @pytest.mark.slow
    def test_bdf1_converges_to_manufactured_solution_at_first_order(self):
        dts, errs = compute_temporal_mms_data("bdf1")
        p, _, _ = fit_order_of_accuracy(dts, errs)
        assert 0.85 <= p <= 1.15, (
            f"BDF1 MMS temporal order {p:.2f} outside expected 1st-order band [0.85, 1.15]; "
            f"errors={errs}"
        )

    @pytest.mark.slow
    def test_bdf2_converges_to_manufactured_solution_at_second_order(self):
        dts, errs = compute_temporal_mms_data("bdf2")
        p, _, _ = fit_order_of_accuracy(dts, errs)
        assert 1.75 <= p <= 2.25, (
            f"BDF2 MMS temporal order {p:.2f} outside expected 2nd-order band [1.75, 2.25]; "
            f"errors={errs}"
        )


# ---------------------------------------------------------------------------
# Standalone log-log convergence plots
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def _plot_convergence(ax, hs, errs, xlabel, title):
        hs = np.asarray(hs, dtype=float)
        errs = np.asarray(errs, dtype=float)
        p, c, _ = fit_order_of_accuracy(hs, errs)

        ax.loglog(hs, errs, "o", markersize=7, label="observed error")
        h_fit = np.array([hs.min(), hs.max()])
        ax.loglog(h_fit, c * h_fit ** p, "--", label=f"fit: O({p:.2f})")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("error (self-convergent reference)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        return p

    print("Computing spatial (steady-state) convergence data...")
    dxs, keff_errs, flux_errs = compute_spatial_convergence_data()

    print("Computing temporal (transient) convergence data (bdf1)...")
    dts_bdf1, errs_bdf1 = compute_temporal_convergence_data("bdf1")

    print("Computing temporal (transient) convergence data (bdf2)...")
    dts_bdf2, errs_bdf2 = compute_temporal_convergence_data("bdf2")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    p_keff = _plot_convergence(axes[0], dxs, keff_errs, "dx", "Steady-state k_eff")
    p_flux = _plot_convergence(axes[1], dxs, flux_errs, "dx", "Steady-state flux shape")

    ax_t = axes[2]
    p_bdf1, c_bdf1, _ = fit_order_of_accuracy(dts_bdf1, errs_bdf1)
    p_bdf2, c_bdf2, _ = fit_order_of_accuracy(dts_bdf2, errs_bdf2)
    dts_bdf1 = np.asarray(dts_bdf1, dtype=float)
    dts_bdf2 = np.asarray(dts_bdf2, dtype=float)
    ax_t.loglog(dts_bdf1, errs_bdf1, "o", markersize=7, color="C0", label="BDF1 observed")
    ax_t.loglog(dts_bdf1, c_bdf1 * dts_bdf1 ** p_bdf1, "--", color="C0",
               label=f"BDF1 fit: O({p_bdf1:.2f})")
    ax_t.loglog(dts_bdf2, errs_bdf2, "s", markersize=7, color="C1", label="BDF2 observed")
    ax_t.loglog(dts_bdf2, c_bdf2 * dts_bdf2 ** p_bdf2, "--", color="C1",
               label=f"BDF2 fit: O({p_bdf2:.2f})")
    ax_t.set_xlabel("dt")
    ax_t.set_ylabel("error (self-convergent reference)")
    ax_t.set_title("Transient BDF1 vs BDF2")
    ax_t.legend()
    ax_t.grid(True, which="both", alpha=0.3)

    fig.suptitle("Diffusion solver convergence study")
    fig.tight_layout()

    print(f"Spatial k_eff order:   {p_keff:.3f} (expected ~2)")
    print(f"Spatial flux order:    {p_flux:.3f} (expected ~2)")
    print(f"Temporal BDF1 order:   {p_bdf1:.3f} (expected ~1)")
    print(f"Temporal BDF2 order:   {p_bdf2:.3f} (expected ~2)")

    out_path = os.path.join(os.path.dirname(__file__), "diffusion_convergence.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    print("Computing spatial MMS convergence data...")
    dxs_mms, errs_mms = compute_spatial_mms_data()

    print("Computing temporal MMS convergence data (bdf1)...")
    dts_mms_bdf1, errs_mms_bdf1 = compute_temporal_mms_data("bdf1")

    print("Computing temporal MMS convergence data (bdf2)...")
    dts_mms_bdf2, errs_mms_bdf2 = compute_temporal_mms_data("bdf2")

    fig_mms, axes_mms = plt.subplots(1, 2, figsize=(11, 5))

    p_mms_spatial = _plot_convergence(axes_mms[0], dxs_mms, errs_mms, "dx",
                                      "Steady-state MMS flux error")
    axes_mms[0].set_ylabel("error (vs exact manufactured solution)")

    ax_mms_t = axes_mms[1]
    p_mms_bdf1, c_mms_bdf1, _ = fit_order_of_accuracy(dts_mms_bdf1, errs_mms_bdf1)
    p_mms_bdf2, c_mms_bdf2, _ = fit_order_of_accuracy(dts_mms_bdf2, errs_mms_bdf2)
    dts_mms_bdf1 = np.asarray(dts_mms_bdf1, dtype=float)
    dts_mms_bdf2 = np.asarray(dts_mms_bdf2, dtype=float)
    ax_mms_t.loglog(dts_mms_bdf1, errs_mms_bdf1, "o", markersize=7, color="C0",
                    label="BDF1 observed")
    ax_mms_t.loglog(dts_mms_bdf1, c_mms_bdf1 * dts_mms_bdf1 ** p_mms_bdf1, "--", color="C0",
                    label=f"BDF1 fit: O({p_mms_bdf1:.2f})")
    ax_mms_t.loglog(dts_mms_bdf2, errs_mms_bdf2, "s", markersize=7, color="C1",
                    label="BDF2 observed")
    ax_mms_t.loglog(dts_mms_bdf2, c_mms_bdf2 * dts_mms_bdf2 ** p_mms_bdf2, "--", color="C1",
                    label=f"BDF2 fit: O({p_mms_bdf2:.2f})")
    ax_mms_t.set_xlabel("dt")
    ax_mms_t.set_ylabel("error (vs exact manufactured solution)")
    ax_mms_t.set_title("Transient MMS: BDF1 vs BDF2")
    ax_mms_t.legend()
    ax_mms_t.grid(True, which="both", alpha=0.3)

    fig_mms.suptitle("Diffusion solver manufactured-solution (MMS) convergence study")
    fig_mms.tight_layout()

    print(f"Spatial MMS order:     {p_mms_spatial:.3f} (expected ~2)")
    print(f"Temporal MMS BDF1 order: {p_mms_bdf1:.3f} (expected ~1)")
    print(f"Temporal MMS BDF2 order: {p_mms_bdf2:.3f} (expected ~2)")

    out_path_mms = os.path.join(os.path.dirname(__file__), "diffusion_mms_convergence.png")
    fig_mms.savefig(out_path_mms, dpi=150)
    print(f"Saved plot to {out_path_mms}")

    print("Computing spatial MMS flux profile...")
    x_ss, fast_num_ss, th_num_ss, fast_exact_ss, th_exact_ss = compute_spatial_mms_profile(n_cells=40)

    print("Computing temporal MMS flux profile...")
    dt_profile = TEMPORAL_MMS_DTS[0]
    x_tr, fast_num_tr, th_num_tr, fast_exact_tr, th_exact_tr = compute_temporal_mms_profile(
        "bdf1", dt_profile
    )

    fig_profile, axes_profile = plt.subplots(1, 2, figsize=(11, 5))

    ax_ss = axes_profile[0]
    ax_ss.plot(x_ss, fast_exact_ss, "-", color="C0", label="fast exact (MMS)")
    ax_ss.plot(x_ss, fast_num_ss, "o", color="C0", markersize=4, label="fast numerical")
    ax_ss.plot(x_ss, th_exact_ss, "-", color="C1", label="thermal exact (MMS)")
    ax_ss.plot(x_ss, th_num_ss, "o", color="C1", markersize=4, label="thermal numerical")
    ax_ss.set_xlabel("x")
    ax_ss.set_ylabel("flux")
    ax_ss.set_title(f"Steady-state MMS flux (n_cells=40)")
    ax_ss.legend()
    ax_ss.grid(True, alpha=0.3)

    # a_fast0 << a_th0 by design (see TEMPORAL_MMS_A_FAST0/A_TH0 docstring), so the
    # fast group needs its own y-axis to be visible alongside the thermal group.
    ax_tr = axes_profile[1]
    ax_tr.plot(x_tr, th_exact_tr, "-", color="C1", label="thermal exact (MMS)")
    ax_tr.plot(x_tr, th_num_tr, "o", color="C1", markersize=3, label="thermal numerical")
    ax_tr.set_xlabel("x")
    ax_tr.set_ylabel("thermal flux", color="C1")
    ax_tr.tick_params(axis="y", labelcolor="C1")
    ax_tr.set_title(f"Transient MMS flux (BDF1, dt={dt_profile}, t={TEMPORAL_MMS_T_FINAL})")
    ax_tr.grid(True, alpha=0.3)

    ax_tr_fast = ax_tr.twinx()
    ax_tr_fast.plot(x_tr, fast_exact_tr, "-", color="C0", label="fast exact (MMS)")
    ax_tr_fast.plot(x_tr, fast_num_tr, "o", color="C0", markersize=3, label="fast numerical")
    ax_tr_fast.set_ylabel("fast flux", color="C0")
    ax_tr_fast.tick_params(axis="y", labelcolor="C0")

    lines_th, labels_th = ax_tr.get_legend_handles_labels()
    lines_f, labels_f = ax_tr_fast.get_legend_handles_labels()
    ax_tr.legend(lines_th + lines_f, labels_th + labels_f, loc="upper center")

    fig_profile.suptitle("Manufactured solution vs. discrete diffusion flux")
    fig_profile.tight_layout()

    out_path_profile = os.path.join(os.path.dirname(__file__), "diffusion_mms_flux_profile.png")
    fig_profile.savefig(out_path_profile, dpi=150)
    print(f"Saved plot to {out_path_profile}")

    plt.show()
