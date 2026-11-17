"""Geometry shared by the representative axial neutronics models."""

import numpy as np


class CoreGeometry:
    """Aligned vertical channels, with original SyTH node indices preserved.

    Bounds contain two opposite corners, not necessarily ordered minima/maxima.
    Without bounds, the supplied node order describes one legacy axial channel.
    """

    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        if not self.nodes:
            raise ValueError("Core geometry requires at least one node.")
        if len({n.index for n in self.nodes}) != len(self.nodes):
            raise ValueError("Core node indices must be unique.")
        if any(not np.isfinite(n.length) or n.length <= 0
               or not np.isfinite(n.volume) or n.volume <= 0 for n in self.nodes):
            raise ValueError("Core nodes require positive finite lengths and volumes.")
        bounded = [n.bounds is not None for n in self.nodes]
        if any(bounded) and not all(bounded):
            raise ValueError("Core bounds must be provided for every node or none.")
        if not any(bounded):
            self.channels = (self.nodes,)
            self.center_channel = self.nodes
        else:
            groups = []
            for n in sorted(self.nodes, key=lambda n: n.index):
                bounds = np.asarray(n.bounds, dtype=float)
                if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
                    raise ValueError("Core bounds require six finite coordinates.")
                lo, hi = np.minimum(bounds[:3], bounds[3:]), np.maximum(bounds[:3], bounds[3:])
                if not np.isclose(hi[2] - lo[2], n.length, rtol=1e-8, atol=1e-10):
                    raise ValueError("Core nodes must be vertical with axial bounds matching length.")
                xy = (lo[:2] + hi[:2]) / 2
                group = next((g for g in groups if np.allclose(g[0], xy, rtol=0, atol=1e-8)), None)
                if group is None:
                    group = (xy, [])
                    groups.append(group)
                group[1].append((lo[2], hi[2], n))
            groups.sort(key=lambda g: tuple(g[0]))
            for _, cells in groups:
                cells.sort(key=lambda c: c[0])
                if not np.allclose([c[1] for c in cells[:-1]], [c[0] for c in cells[1:]], rtol=1e-8, atol=1e-10):
                    raise ValueError("Core channels must have contiguous axial cells.")
            reference = np.array([(c[0], c[1]) for c in groups[0][1]])
            for _, cells in groups[1:]:
                edges = np.array([(c[0], c[1]) for c in cells])
                if edges.shape != reference.shape or not np.allclose(edges, reference, rtol=1e-8, atol=1e-10):
                    raise ValueError("Core channels must have matching axial boundaries.")
            self.channels = tuple(tuple(c[2] for c in cells) for _, cells in groups)
            centers = np.array([g[0] for g in groups])
            midpoint = (centers.min(axis=0) + centers.max(axis=0)) / 2
            distances = np.sum((centers - midpoint)**2, axis=1)
            candidates = np.flatnonzero(np.isclose(distances, distances.min(), rtol=1e-10, atol=1e-16))
            selected = min(candidates, key=lambda i: min(n.index for n in self.channels[i]))
            self.center_channel = self.channels[selected]
        self.axial_index = {n.index: i for channel in self.channels for i, n in enumerate(channel)}
        self.axial_volumes = np.zeros(len(self.center_channel))
        for n in self.nodes:
            self.axial_volumes[self.axial_index[n.index]] += n.volume

    @classmethod
    def from_nodes(cls, nodes):
        return nodes if isinstance(nodes, cls) else cls(nodes)

    @property
    def volume(self):
        return sum(n.volume for n in self.nodes)

    def validate_diffusion(self):
        lengths = np.array([n.length for n in self.center_channel])
        if len(lengths) < 3:
            raise ValueError("The diffusion matrix builder requires at least three core nodes.")
        if not np.allclose(lengths, lengths[0], rtol=1e-8, atol=1e-10):
            raise ValueError("The diffusion solver requires a uniform axial mesh.")

    def broadcast(self, axial_values, target):
        """Write axial power densities into the existing node-indexed array."""
        for n in self.nodes:
            target[n.index] = axial_values[self.axial_index[n.index]]
