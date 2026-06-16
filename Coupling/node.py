from typing import Optional
import h5py


class Node:
    """A simple node class that owns its length, volume, index, and optional bounds."""

    def __init__(
        self,
        length: float,
        volume: float,
        index: int,
        bounds: Optional[tuple[float, float, float, float, float, float]] = None,
    ):
        """
        Initialize a Node.

        Args:
            length: The length of the node.
            volume: The volume of the node.
            index: The index of the node.
            bounds: Optional bounds as (x_min, x_max, y_min, y_max, z_min, z_max).
        """
        self.length = length
        self.volume = volume
        self.index = index
        self.bounds = bounds
