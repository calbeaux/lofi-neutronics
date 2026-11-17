"""In-memory temperature library for MPACT two-group HDF5 edits.

All stored dimensional data use SI. Timing estimates use the collapsed nodal
forward/adjoint fields, rather than claiming to reproduce a transport estimator.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.optimize import minimize

from Neutronics.cross_sections import CrossSections


_HARMONICS = np.array([1, 3, 5]) * np.pi


def _basis(x):
    x = np.asarray(x)
    return np.column_stack((np.sin(x[:, None] * _HARMONICS), np.ones(len(x))))


def _cell_basis(edges):
    edges = np.asarray(edges)
    waves = (np.cos(edges[:-1, None] * _HARMONICS)
             - np.cos(edges[1:, None] * _HARMONICS))
    waves /= np.diff(edges)[:, None] * _HARMONICS
    return np.column_stack((waves, np.ones(len(edges) - 1)))


@dataclass(frozen=True)
class ProfileFit:
    coefficients: np.ndarray  # A1, A3, A5, offset; normalized axial coordinate
    relative_residual: float

    def cell_averages(self, edges):
        edges = np.asarray(edges, dtype=float)
        if (edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges))
                or edges[0] < 0 or edges[-1] > 1 or np.any(np.diff(edges) <= 0)):
            raise ValueError("Profile edges must increase within [0, 1].")
        return np.maximum(_cell_basis(edges) @ self.coefficients, 0.0)


@dataclass(frozen=True)
class PKProfiles:
    power: ProfileFit
    adjoint: ProfileFit
    fueled_extent: tuple  # meters in the MPACT mesh

    def map_nodes(self, nodes):
        lengths = np.array([n.length for n in nodes])
        volumes = np.array([n.volume for n in nodes])
        if (not len(nodes) or not np.all(np.isfinite(lengths + volumes))
                or np.any(lengths <= 0) or np.any(volumes <= 0)):
            raise ValueError("Core nodes require positive finite lengths and volumes.")
        edges = np.concatenate(([0.0], np.cumsum(lengths) / lengths.sum()))
        edges[-1] = 1.0
        power = self.power.cell_averages(edges)
        adjoint = self.adjoint.cell_averages(edges)
        power_norm = power @ volumes
        if power_norm <= 0 or adjoint @ (power * volumes) <= 0:
            raise ValueError("Fitted PK profiles have zero normalization.")
        power = power / power_norm
        # Source is node-integrated neutrons/s. Normalize importance to unit
        # power-weighted mean; use the same importance with volume for temperature.
        adjoint = adjoint / (adjoint @ (power * volumes))
        return power, adjoint


def _fit_profile(edges, values, weights):
    values = values / np.average(values, weights=weights)
    design = _cell_basis(edges)
    w = weights / weights.sum()
    constraint = _basis(np.linspace(0, 1, 257))
    result = minimize(
        lambda c: np.sum(w * (design @ c - values) ** 2),
        np.array([0., 0., 0., 1.]),
        jac=lambda c: 2 * design.T @ (w * (design @ c - values)),
        constraints=[{"type": "ineq", "fun": lambda c: constraint @ c,
                      "jac": lambda c: constraint}],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"PK profile fit failed: {result.message}")
    c = result.x
    # Check every stationary point analytically, including minima between the
    # constraint grid points. d/dtheta is a fifth-degree polynomial in cos(theta).
    a, b, d, _ = c
    roots = np.polynomial.polynomial.polyroots([0, a - 9*b + 25*d, 0,
                                               12*b - 100*d, 0, 80*d])
    real = roots.real[(abs(roots.imag) < 1e-8) & (abs(roots.real) <= 1)]
    candidates = np.concatenate(([0., 1.], np.arccos(real) / np.pi))
    c[3] += max(0., -np.min(_basis(candidates) @ c))
    residual = np.sqrt(np.sum(w * (design @ c - values)**2)
                       / np.sum(w * values**2))
    return ProfileFit(c, float(residual))


def _read(group, key, shape=None, positive=False, nonnegative=False):
    if key not in group:
        raise ValueError(f"Missing MPACT dataset {group.name}/{key}")
    value = np.asarray(group[key][()], dtype=float)
    if shape is not None and value.shape != shape:
        raise ValueError(f"{group.name}/{key}: expected shape {shape}, got {value.shape}")
    if (not np.all(np.isfinite(value)) or (positive and np.any(value <= 0))
            or (nonnegative and np.any(value < 0))):
        raise ValueError(f"Invalid values in {group.name}/{key}")
    return value


def _spatial_data(state):
    if "NODAL_XS" not in state:
        raise ValueError("PK estimates and profiles require NODAL_XS.")
    g = state["NODAL_XS"]
    volume = _read(g, "VOL", nonnegative=True) * 1e-6
    if volume.ndim != 3 or not np.any(volume > 0):
        raise ValueError("NODAL_XS/VOL must have (radial, axial, assembly) dimensions.")
    shape = (2,) + volume.shape
    # Zero-volume padding may contain zeros or NaNs. Discard it before any
    # arithmetic or positivity checks on group-dependent nodal data.
    valid = volume > 0
    data = {}
    for key in ("FLUX", "ADJOINT FLUX", "VELOCITY", "NXSF", "XSF", "KXSF", "CHI", "CHID"):
        if key not in g or g[key].shape != shape:
            raise ValueError(f"NODAL_XS/{key} requires shape {shape}.")
        a = np.asarray(g[key][()], dtype=float)
        if not np.all(np.isfinite(a[:, valid])) or np.any(a[:, valid] < 0):
            raise ValueError(f"Invalid populated NODAL_XS/{key} values.")
        a[:, ~valid] = 0
        data[key] = a
    if np.any(data["VELOCITY"][:, valid] <= 0):
        raise ValueError("Populated nodal velocities must be positive.")
    for key in ("NXSF", "XSF", "KXSF"):
        data[key] *= 100
    data["VELOCITY"] /= 100
    mesh = _read(g, "AXIALMESH") / 100
    if mesh.shape != (volume.shape[1] + 1,) or np.any(np.diff(mesh) <= 0):
        raise ValueError("Invalid NODAL_XS/AXIALMESH.")
    phi, adj = data["FLUX"], data["ADJOINT FLUX"]
    inventory = np.sum(volume[valid] * np.sum(
        adj[:, valid] * phi[:, valid] / data["VELOCITY"][:, valid], axis=0))
    production = np.sum(data["NXSF"] * phi, axis=0)
    prompt_importance = np.sum(adj * data["CHI"], axis=0)
    delayed_importance = np.sum(adj * data["CHID"], axis=0)
    denominator = np.sum(volume * prompt_importance * production)
    power = np.sum(data["KXSF"] * phi, axis=0)
    weighted_power = np.sum(volume * delayed_importance * power)
    if min(inventory, denominator, weighted_power) <= 0:
        raise ValueError("Nodal flux/adjoint fields cannot support PK normalization.")
    timing = inventory / denominator
    # Effective production per joule uses the same adjoint normalization as n0.
    neutrons_per_joule = denominator / weighted_power
    axial_power = np.sum(volume * power, axis=(0, 2))
    fueled = np.flatnonzero(axial_power > 0)
    lo, hi = fueled[0], fueled[-1] + 1
    axial_volume = np.sum(volume, axis=(0, 2))[lo:hi]
    if np.any(axial_volume <= 0):
        raise ValueError("Fueled axial extent contains empty planes.")
    axial_adj = np.sum(volume * delayed_importance, axis=(0, 2))[lo:hi] / axial_volume
    edges = (mesh[lo:hi+1] - mesh[lo]) / (mesh[hi] - mesh[lo])
    profiles = PKProfiles(
        _fit_profile(edges, axial_power[lo:hi] / axial_volume, axial_volume),
        _fit_profile(edges, axial_adj, axial_volume), (float(mesh[lo]), float(mesh[hi])),
    )
    return timing, neutrons_per_joule, profiles, mesh, volume


class MPACTLibrary:
    """Temperature samples for one unchanged core; all queries are in kelvin."""

    @classmethod
    def from_files(cls, paths, state="STATE_0001", *, temperature_source="dataset"):
        """Read samples using FUELTEMP or an explicit ``_T0700`` filename token.

        Filename temperatures are kelvin; no temperature is inferred from the
        containing directory, and a malformed filename never falls back to HDF5.
        """
        if temperature_source not in {"dataset", "filename"}:
            raise ValueError("temperature_source must be 'dataset' or 'filename'.")
        samples = []
        signatures = []
        for path in paths:
            path = Path(path).expanduser().resolve()
            with h5py.File(path, "r") as f:
                if state not in f or "CORE_XS" not in f[state]:
                    raise ValueError(f"{path}: missing {state}/CORE_XS")
                s, core = f[state], f[state]["CORE_XS"]
                recorded_temperature = None
                if "FUELTEMP" in core:
                    recorded = np.asarray(core["FUELTEMP"][()], dtype=float)
                    if recorded.shape == () and np.isfinite(recorded):
                        recorded_temperature = float(recorded)
                if temperature_source == "dataset":
                    temperature = float(_read(core, "FUELTEMP", (), positive=True))
                else:
                    matches = re.findall(r"(?:^|_)T(\d+(?:\.\d+)?)(?=_|$)", path.stem)
                    if len(matches) != 1:
                        raise ValueError(f"{path.name}: expected one temperature token such as _T0700 in filename.")
                    temperature = float(matches[0])
                    if not np.isfinite(temperature) or temperature <= 0:
                        raise ValueError(f"{path.name}: filename temperature must be positive finite kelvin.")
                xs = {key: _read(core, key, (2,), positive=key in {"XSTR", "VELOCITY"},
                                 nonnegative=True)
                      for key in ("XSTR", "XSRM", "NXSF", "XSF", "KXSF", "VELOCITY", "CHI", "CHID")}
                xs["XSS"] = _read(core, "XSS", (2, 2), nonnegative=True)
                if not np.allclose(xs["CHI"], [1, 0], atol=1e-7, rtol=0):
                    raise ValueError("Two-group diffusion requires CHI=[1, 0].")
                if not np.allclose(xs["CHID"], [1, 0], atol=1e-7, rtol=0):
                    raise ValueError("Two-group diffusion requires CHID=[1, 0].")
                if abs(xs["XSS"][1, 0]) > 1e-12:
                    raise ValueError("Two-group diffusion does not support upscatter.")
                xs["XSS"][1, 0] = 0
                out = xs["XSS"].sum(axis=1) - np.diag(xs["XSS"])
                if np.any(xs["XSRM"] - out < 0):
                    raise ValueError("Negative absorption inferred from removal minus outscatter.")
                for key in ("XSTR", "XSRM", "XSS", "NXSF", "XSF", "KXSF"):
                    xs[key] *= 100
                xs["VELOCITY"] /= 100
                beta = _read(core, "Beta", nonnegative=True)
                decay = _read(core, "Delayed Neutron Precursor Decay Constants", positive=True)
                if beta.ndim != 1 or not beta.size or beta.shape != decay.shape or beta.sum() >= 1:
                    raise ValueError("Invalid delayed-neutron group data.")
                flux = _read(core, "FLUX", (2,), nonnegative=True)
                energy_rate = xs["KXSF"] @ flux
                fission_rate = xs["XSF"] @ flux
                if min(energy_rate, fission_rate) <= 0:
                    raise ValueError("Core data must support positive fission and energy production.")
                k = float(_read(s, "keff", (), positive=True))
                timing, production, profiles, mesh, volumes = _spatial_data(s)
                grid = _read(core, "energy_grid", (2,), positive=True)
                boundaries = []
                for boundary in ("bot", "top", "rad"):
                    value = np.asarray(f[f"CORE/bc_{boundary}"][()]).item()
                    boundaries.append(value.decode().strip() if isinstance(value, bytes) else str(value).strip())
                boundaries = tuple(boundaries)
                signatures.append((grid, mesh, volumes, boundaries, beta.shape, profiles.fueled_extent))
                samples.append(dict(temperature=temperature, recorded_temperature=recorded_temperature,
                                    xs=xs, beta_groups=beta,
                                    decay_constants=decay, keff=k, rho=1 - 1/k,
                                    generation_time=timing, neutron_lifetime=k*timing,
                                    neutrons_per_joule=production,
                                    fissions_per_joule=fission_rate/energy_rate,
                                    profiles=profiles, source=str(path)))
        if len(samples) < 2:
            raise ValueError("An MPACT library requires at least two temperature samples.")
        first = signatures[0]
        for signature in signatures[1:]:
            for a, b in zip(first[:3], signature[:3]):
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-8, atol=1e-12):
                    raise ValueError("MPACT energy grids, core meshes, and volumes must agree.")
            if first[3:] != signature[3:]:
                raise ValueError("MPACT boundaries, delayed groups, and fueled extents must agree.")
        obj = cls()
        obj._samples = sorted(samples, key=lambda s: s["temperature"])
        obj.temperatures = np.array([s["temperature"] for s in obj._samples])
        if np.any(np.diff(obj.temperatures) <= 1e-8):
            raise ValueError("Duplicate MPACT sample temperatures.")
        obj.metadata = dict(
            sources=[s["source"] for s in obj._samples], state=state,
            temperatures=obj.temperatures.tolist(), units="SI; temperature K",
            temperature_source=temperature_source,
            recorded_temperatures=[s["recorded_temperature"] for s in obj._samples],
            timing="Spatially collapsed NODAL_XS adjoint-weighted estimate; lifetime=keff*generation_time",
            reactivity="MPACT keff; rho=1-1/keff",
            beta="CORE_XS/Beta, adjoint weighted",
            profiles="NODAL_XS axial volume collapse; nonnegative harmonics 1,3,5 plus offset",
            boundaries=dict(zip(("bot", "top", "rad"), first[3])),
        )
        return obj

    def _interval(self, temperature):
        t = float(temperature)
        if not np.isfinite(t) or t < self.temperatures[0] or t > self.temperatures[-1]:
            raise ValueError(f"Temperature {t} K outside MPACT range "
                             f"[{self.temperatures[0]}, {self.temperatures[-1]}] K")
        i = min(np.searchsorted(self.temperatures, t, side="right") - 1, len(self._samples)-2)
        fraction = (t - self.temperatures[i]) / (self.temperatures[i+1] - self.temperatures[i])
        return i, fraction

    def _mix(self, key, i, fraction):
        return (1-fraction)*self._samples[i][key] + fraction*self._samples[i+1][key]

    def cross_sections(self, temperature, *, radial_buckling_squared=0.0):
        i, f = self._interval(temperature)
        xs = {key: (1-f)*self._samples[i]["xs"][key] + f*self._samples[i+1]["xs"][key]
              for key in self._samples[i]["xs"]}
        out = xs["XSS"].sum(axis=1) - np.diag(xs["XSS"])
        groups = [CrossSections(xs["XSTR"][g], xs["XSRM"][g]-out[g], xs["NXSF"][g],
                                xs["XSS"][g, g], out[g], xs["KXSF"][g], xs["VELOCITY"][g],
                                sigma_f=xs["XSF"][g], si=True, radial_buckling_squared=radial_buckling_squared) for g in range(2)]
        return SimpleNamespace(fast=groups[0], th=groups[1], chi=xs["CHI"], chid=xs["CHID"])

    def kinetics(self, temperature):
        i, f = self._interval(temperature)
        keys = ("beta_groups", "decay_constants", "rho", "generation_time",
                "neutrons_per_joule", "fissions_per_joule")
        values = {key: self._mix(key, i, f) for key in keys}
        values["beta_effective"] = float(values["beta_groups"].sum())
        values["keff"] = 1 / (1-values["rho"])
        values["neutron_lifetime"] = values["keff"] * values["generation_time"]
        values["alpha_T"] = ((self._samples[i+1]["rho"] - self._samples[i]["rho"])
                             / (self.temperatures[i+1]-self.temperatures[i]))
        return SimpleNamespace(**values)

    def profiles(self, temperature):
        i, f = self._interval(temperature)
        a, b = self._samples[i]["profiles"], self._samples[i+1]["profiles"]
        def mix(name):
            x, y = getattr(a, name), getattr(b, name)
            return ProfileFit((1-f)*x.coefficients + f*y.coefficients,
                              (1-f)*x.relative_residual + f*y.relative_residual)
        return PKProfiles(mix("power"), mix("adjoint"), a.fueled_extent)
