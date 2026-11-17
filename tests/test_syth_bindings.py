"""Portable checks for the SyTH ctypes boundary; no native library required."""
import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import h5py
import pytest


@pytest.fixture
def bindings(monkeypatch):
    calls = []

    class Library:
        def __init__(self):
            self.functions = {}
            self.syTH_getCoupledNodeCount = ctypes.CFUNCTYPE(
                None, ctypes.POINTER(ctypes.c_int)
            )(lambda output: output.__setitem__(0, 3))
            self.syTH_getFirstCoupledNode = ctypes.CFUNCTYPE(
                None, ctypes.POINTER(ctypes.c_int)
            )(lambda output: output.__setitem__(0, 2))
            self.syTH_stagePreviousConcentration = ctypes.CFUNCTYPE(
                None, ctypes.c_int, ctypes.c_int, ctypes.c_double
            )(lambda group, node, concentration: calls.append((group, node, concentration)))

        def __getattr__(self, name):
            if name == 'syTH_getNodeCount':
                raise AttributeError('The updated C API has no full-mesh node-count getter')
            if name not in self.functions:
                def function(*args):
                    calls.append((name, args))
                self.functions[name] = function
            return self.functions[name]

    library = Library()
    monkeypatch.setattr(ctypes, 'CDLL', lambda *args, **kwargs: library)
    # The production loader targets Linux; portable tests don't load a library.
    monkeypatch.setattr(os, 'RTLD_DEEPBIND', getattr(os, 'RTLD_DEEPBIND', 0), raising=False)
    package_name = '_syth_binding_test'
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / 'SyTH')]
    monkeypatch.setitem(sys.modules, package_name, package)

    def load(name):
        path = Path(package.__path__[0]) / f'{name}.py'
        spec = importlib.util.spec_from_file_location(f'{package_name}.{name}', path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module

    raw = load('syth_raw')
    wrapper = load('syth')
    return raw, wrapper.Syth, calls


def test_new_c_api_signatures(bindings):
    raw, _, _ = bindings
    assert raw.syTH_stagePreviousConcentration.argtypes == [
        ctypes.c_int, ctypes.c_int, ctypes.c_double]
    assert raw.syTH_getCoupledNodeCount.argtypes == [ctypes.POINTER(raw.SIK)]
    assert raw.syTH_getFirstCoupledNode.argtypes == [ctypes.POINTER(raw.SIK)]
    for name, function in vars(raw).items():
        if name.startswith('syTH_'):
            assert function.restype is None
            assert function.argtypes is not None
    assert not hasattr(raw, 'syTH_getNodeCount')


def test_full_mesh_count_is_distinct_from_coupled_count(bindings, tmp_path):
    _, Syth, _ = bindings
    path = tmp_path / 'input.h5'
    with h5py.File(path, 'w') as f:
        group = f.create_group('fluid_solve/fluid_mesh/nodes')
        group['number_of_nodes'] = 12
        # Coupled indices need not be contiguous or start at zero.
        for index in (2, 8, 11):
            group.create_group(f'node_{index:04d}')['is_coupled'] = True
    syth = Syth(str(path))
    assert syth.get_node_count() == 12
    assert syth.get_coupled_node_count() == 3
    assert syth.get_first_coupled_node() == 2


def test_stage_concentration_preserves_group_and_node_indices(bindings, tmp_path):
    _, Syth, calls = bindings
    syth = Syth(str(tmp_path / 'input.h5'))
    syth.stage_previous_concentration(species_id=2, node_index=8, concentration=1.25e12)
    assert calls[-1] == (2, 8, 1.25e12)
