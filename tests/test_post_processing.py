"""Check core selection and state-local time against native HDF5 layout."""
import h5py
import numpy as np
import pytest
import post_processing as pp


def test_native_coupled_nodes_and_continuing_output_indices(tmp_path):
    path = tmp_path / 'native.h5'
    with h5py.File(path, 'w') as f:
        f['fluid_solve/state_2/transient_simulation/dt'] = .05
        for i, coupled in enumerate([False, True, True]):
            base = f'fluid_solve/fluid_mesh/nodes/node_{i:04d}'
            f[base+'/is_coupled'] = coupled
            f[base+'/volume'] = [100, 2, 3][i]
        for key in ['0100', '0101', '0102']:
            base = f'solution/fluid_data_0002/{key}'
            f[base+'/power'] = [999, 10, 20]
            f[base+'/average_temperature'] = [999, 800, 900]
    assert pp._get_core_nodes(path) == [(1, 2), (2, 3)]
    t, power = pp.compute_total_core_power(path, stride=1, state=2)
    np.testing.assert_allclose(t, [.05, .10, .15])
    np.testing.assert_allclose(power, 80)
    t, means = pp.compute_core_averages(path, ['average_temperature'], stride=2, state=2)
    np.testing.assert_allclose(t, [.05, .15])
    np.testing.assert_allclose(means['average_temperature'], 850)
    assert pp.read_timestep(path, 101, state=2)['time'] == pytest.approx(.10)
    data = pp.read_solution(path, start=101, state=2)
    np.testing.assert_allclose(data['time'], [.10, .15])
