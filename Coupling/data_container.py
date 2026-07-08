import functools
import numpy as np


class DataContainer:
    """
    Simple container class for data
    """

    def __init__(self, nmesh: int):
        """
        Initializes the data container

        Args:
            - nmesh: number of unknowns each solution vector
        """
        self._nmesh = nmesh
        self._data = {}

    def register(self, name: str, default_value: float = 1.0) -> np.ndarray:
        """
        Registers a new variable into the container

        Args:
            - name (str): The name of the solution variable
            - data (np.ndarray): the data array associated with the name
        """
        assert not name in self._data
        data = default_value * np.ones(self._nmesh)
        self._data[name] = data
        return data

    def __getitem__(self, key):
        """
        Returns solution variable vector

        Args:
            - key (str or (str, int)): The name of a solution variable or tuple with name and index

        Returns:
            - the full solution vector (np.ndarray)

        Raises:
            - KeyError: if the key does not exist in the data container
            - IndexError: if the index is out of range for the key
        """
        output = self.get(key, default_value=None)  # returns None if key does not exist
        if output is None:
            raise KeyError(f"The key {key} does not exist in the data container.")
        return output

    def __contains__(self, item):
        """
        Method to return boolean if variable is in the data container

        Args:
            - name (str): The name of a solution variable

        Returns:
            - boolean: if name is in the data container
        """
        return item in self._data

    def get(self, key, default_value=None):
        """
        Returns solution variable vector without raising key errors

        Args:
            - key (str or (str, int)): The name of a solution variable or tuple with name and index
            - default_value: The value to return if the key does not exist

        Returns:
            - the full solution vector (np.ndarray) if the key exists
            - `default_value` if the key does not exist
        """
        if isinstance(key, tuple):
            key, idx = key
            if idx >= self._nmesh:
                raise IndexError(
                    f"The index {idx} is out of range for the data container (n = {self._nmesh})."
                )
            data = self._data.get(key, default_value)
            return data[idx]

        return self._data.get(key, default_value)

    def get_value(self, key, idx: int) -> float:
        """
        Returns solution variable at a location

        Args:
            - name (str): The name of a solution variable
            - idx (int): The position in the solution vector to get the value

        Returns:
            - the solution value (float)
        """
        assert idx < self._nmesh
        return self[key, idx]


def check_memory_address(*attributes):
    """
    Returns a decorator that checks if the memory address of the given attributes has changed.

    The decorator will raise a RuntimeError if the memory address of any of the given attributes has changed.

    The reason for this check is so that solvers (and other classes) are always dealing with the
    same memory address for data vectors, which allows the data to be updated in place for the
    data container and other classes.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the initial memory addresses
            initial_ids = {attr: id(getattr(self, attr)) for attr in attributes}

            # Call the function
            result = func(self, *args, **kwargs)

            # Check if the memory addresses are the same
            for attr in attributes:
                if id(getattr(self, attr)) != initial_ids[attr]:
                    raise RuntimeError(f"The memory address of {attr} has changed.")

            return result

        return wrapper

    return decorator
