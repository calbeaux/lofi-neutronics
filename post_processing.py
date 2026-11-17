import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


PRECURSOR_LAMBDAS = np.array([0.01244, 0.03051, 0.11144, 0.30137, 1.13631, 3.01368])

SCALAR_DATASETS = [
    "average_density",
    "average_enthalpy",
    "average_mass_flow_rate",
    "average_pressure",
    "average_temperature",
    "average_velocity",
    "power",
]
PRECURSOR_GROUP = "average_neutron_precursor_mass_conc"


def _sorted_timestep_keys(group):
    return sorted(group.keys(), key=lambda k: int(k))


def _fluid_data_path(state):
    """Return the 'solution/fluid_data_NNNN' path for a given state number."""
    return f"solution/fluid_data_{state:04d}"


def _dt(filepath, state=1):
    with h5py.File(filepath, "r") as f:
        try:
            return float(f[f"fluid_solve/state_{state}/transient_simulation/dt"][()])
        except KeyError:
            return None


def read_steady_state(filepath):
    """Return the initial steady-state solution as a dict of numpy arrays."""
    data = {}
    with h5py.File(filepath, "r") as f:
        fd0 = f["solution/fluid_data_0000"]
        for name in SCALAR_DATASETS:
            if name in fd0:
                data[name] = fd0[name][()]
        prec_group = fd0.get(PRECURSOR_GROUP)
        if prec_group is not None:
            data[PRECURSOR_GROUP] = {
                k: prec_group[k][()] for k in sorted(prec_group.keys())
            }
    return data


def read_timestep(filepath, timestep_key, state=1):
    """
    Return data for a single transient timestep.

    Args:
        filepath: path to the H5 file
        timestep_key: integer or zero-padded string key (e.g. 118166 or '118166')
        state: state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        dict with numpy arrays for each dataset plus 'time' (float, seconds)
    """
    key = str(int(timestep_key))
    dt = _dt(filepath, state=state)
    data = {}
    with h5py.File(filepath, "r") as f:
        group = f[_fluid_data_path(state)]
        # Accept zero-padded native keys as well as integer caller input.
        key = next(k for k in group if int(k) == int(key))
        first_index = min(map(int, group.keys()))
        ts = group[key]
        for name in SCALAR_DATASETS:
            if name in ts:
                data[name] = ts[name][()]
        prec_group = ts.get(PRECURSOR_GROUP)
        if prec_group is not None:
            data[PRECURSOR_GROUP] = {
                k: prec_group[k][()] for k in sorted(prec_group.keys())
            }
        data["time"] = float(int(key) - first_index + 1) * dt if dt is not None else None
    return data


def read_solution(filepath, stride=1, start=None, stop=None, state=1):
    """
    Load transient solution data from an H5 file into memory.

    With 250k timesteps and 376 nodes this can be very large; use stride to
    downsample.  All arrays have shape (n_timesteps, n_nodes).

    Args:
        filepath: path to the H5 file
        stride:   load every Nth timestep (default 1 = all)
        start:    first timestep integer index to include (inclusive, default first)
        stop:     last timestep integer index to include (inclusive, default last)
        state:    state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        dict with keys:
            'timestep_indices'  int array (n_timesteps,)
            'time'              float array (n_timesteps,)  seconds; None if dt unknown
            'average_density'                   (n_timesteps, 376)
            'average_enthalpy'                  (n_timesteps, 376)
            'average_mass_flow_rate'            (n_timesteps, 376)
            'average_pressure'                  (n_timesteps, 376)
            'average_temperature'               (n_timesteps, 376)
            'average_velocity'                  (n_timesteps, 376)
            'power'                             (n_timesteps, 375)
            'neutron_precursor_concentrations'  (n_precursors, n_timesteps, 376)
    """
    dt = _dt(filepath, state=state)

    with h5py.File(filepath, "r") as f:
        fd1 = f[_fluid_data_path(state)]
        all_keys = _sorted_timestep_keys(fd1)

        start_idx = int(all_keys[0]) if start is None else start
        stop_idx = int(all_keys[-1]) if stop is None else stop
        selected = [
            k for k in all_keys
            if start_idx <= int(k) <= stop_idx
        ][::stride]

        if not selected:
            raise ValueError("No timesteps match the requested range/stride.")

        n = len(selected)
        indices = np.array([int(k) for k in selected], dtype=np.int64)
        times = (indices - int(all_keys[0]) + 1) * dt if dt is not None else None

        # Pre-allocate from first timestep to get shapes
        first = fd1[selected[0]]
        shapes = {name: first[name].shape for name in SCALAR_DATASETS if name in first}
        prec_keys = sorted(first[PRECURSOR_GROUP].keys()) if PRECURSOR_GROUP in first else []

        arrays = {name: np.empty((n, *shapes[name]), dtype=np.float64) for name in shapes}
        prec_arrays = (
            np.empty((len(prec_keys), n, *first[PRECURSOR_GROUP][prec_keys[0]].shape),
                     dtype=np.float64)
            if prec_keys else None
        )

        for i, key in enumerate(selected):
            ts = fd1[key]
            for name in arrays:
                arrays[name][i] = ts[name][()]
            if prec_arrays is not None:
                pg = ts[PRECURSOR_GROUP]
                for j, pk in enumerate(prec_keys):
                    prec_arrays[j, i] = pg[pk][()]

    result = {"timestep_indices": indices, "time": times}
    result.update(arrays)
    if prec_arrays is not None:
        result["neutron_precursor_concentrations"] = prec_arrays
    return result


def summarize(filepath, state=1):
    """Print a concise summary of the H5 file contents."""
    dt = _dt(filepath, state=state)
    with h5py.File(filepath, "r") as f:
        meta = f.get("metadata")
        if meta:
            version = meta["version"][()].decode() if "version" in meta else "unknown"
            exported = meta["exported_at_utc"][()].decode() if "exported_at_utc" in meta else "unknown"
            print(f"Version : {version}")
            print(f"Exported: {exported}")

        fd1 = f[_fluid_data_path(state)]
        keys = _sorted_timestep_keys(fd1)
        n_ts = len(keys)
        first_key, last_key = int(keys[0]), int(keys[-1])

        print(f"Timesteps       : {n_ts:,}")
        print(f"Timestep range  : {first_key} – {last_key}")
        if dt is not None:
            t_start = dt
            t_end = (last_key - first_key + 1) * dt
            print(f"dt              : {dt} s")
            print(f"Time range      : {t_start:.3f} – {t_end:.3f} s")

        first = fd1[keys[0]]
        for name in SCALAR_DATASETS:
            if name in first:
                print(f"  {name}: shape {first[name].shape}")
        if PRECURSOR_GROUP in first:
            pg = first[PRECURSOR_GROUP]
            n_prec = len(pg)
            shape = pg[sorted(pg.keys())[0]].shape
            print(f"  {PRECURSOR_GROUP}: {n_prec} groups, each shape {shape}")


def _get_core_nodes(filepath):
    """Return sorted (index, volume) for tagged or explicitly coupled core nodes."""

    def _decode(val):
        if isinstance(val, bytes):
            return val.decode()
        if hasattr(val, "item"):
            val = val.item()
            if isinstance(val, bytes):
                return val.decode()
        return val

    core = []
    with h5py.File(filepath, "r") as f:
        for name, grp in f["fluid_solve/fluid_mesh/nodes"].items():
            if not isinstance(grp, h5py.Group):
                continue
            tag_ds = grp.get("tag")
            coupled = grp.get("is_coupled")
            if not ((tag_ds is not None and _decode(tag_ds[()]) == "core")
                    or (coupled is not None and bool(coupled[()]))):
                continue
            vol_ds = grp.get("volume")
            if not isinstance(vol_ds, h5py.Dataset):
                continue
            core.append((int(name.split("_")[-1]), float(vol_ds[()])))
    return sorted(core, key=lambda x: x[0])


def compute_total_core_power(filepath, stride=10, state=1):
    """
    Compute total reactor power (W) at every ``stride`` transient timesteps.

    Power stored in the H5 file is volumetric power density (W/m³).  Total
    power is the dot product of the core-node density values and their volumes.

    Args:
        filepath: path to the H5 file
        stride:   sample every Nth timestep (default 10)
        state:    state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        times  (n,)  float array  – time in seconds
        powers (n,)  float array  – total core power in watts
    """
    core = _get_core_nodes(filepath)
    core_indices = np.array([idx for idx, _ in core], dtype=int)
    volumes = np.array([vol for _, vol in core], dtype=np.float64)

    dt = _dt(filepath, state=state)

    with h5py.File(filepath, "r") as f:
        fd1 = f[_fluid_data_path(state)]
        all_keys = _sorted_timestep_keys(fd1)
        selected = all_keys[::stride]

        times = np.empty(len(selected), dtype=np.float64)
        powers = np.empty(len(selected), dtype=np.float64)

        for i, key in enumerate(selected):
            ts = fd1[key]
            power_density = ts["power"][core_indices]
            powers[i] = float(np.dot(power_density, volumes))
            # Native output numbering continues across states; time resets.
            times[i] = (int(key) - int(all_keys[0]) + 1) * dt if dt is not None else float(i * stride)

    return times, powers


QUANTITY_LABELS = {
    "average_density": "Avg Core Density (kg/m³)",
    "average_enthalpy": "Avg Core Enthalpy (J/kg)",
    "average_mass_flow_rate": "Avg Core Mass Flow Rate (kg/s)",
    "average_pressure": "Avg Core Pressure (Pa)",
    "average_temperature": "Avg Core Temperature (K)",
    "average_velocity": "Avg Core Velocity (m/s)",
    "power": "Total Core Power (MW)",
}


def compute_quantity(filepath, quantity, stride=10, state=1):
    """
    Compute a single quantity vs time, sampled every ``stride`` timesteps.

    ``power`` is special-cased to the volume-weighted total core power (W);
    every other name in SCALAR_DATASETS is the plain core-node spatial mean.

    Args:
        filepath: path to the H5 file
        quantity: dataset name, e.g. 'average_temperature' or 'power'
        stride:   sample every Nth timestep (default 10)
        state:    state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        times  (n,) float array – seconds
        values (n,) float array
    """
    if quantity == "power":
        return compute_total_core_power(filepath, stride=stride, state=state)
    times, results = compute_core_averages(filepath, [quantity], stride=stride, state=state)
    return times, results[quantity]


def plot_quantity(filepath, quantity, stride=10, save_path=None, state=1):
    """
    Plot a single quantity versus time sampled at every ``stride`` timesteps.

    Args:
        filepath:  path to the H5 file
        quantity:  dataset name, e.g. 'average_temperature' or 'power'
        stride:    sample every Nth timestep (default 10)
        save_path: if given, save the figure to this path instead of showing it
        state:     state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)
    """
    times, values = compute_quantity(filepath, quantity, stride=stride, state=state)
    if quantity == "power":
        values = values / 1e6

    ylabel = QUANTITY_LABELS.get(quantity, quantity)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, values, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Time")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_quantity_comparison(filepath1, filepath2, quantity, labels=None, stride=10, save_path=None, state=1):
    """
    Plot a single quantity vs time for two H5 files on the same axes.

    Args:
        filepath1:  path to first H5 file
        filepath2:  path to second H5 file
        quantity:   dataset name, e.g. 'average_temperature' or 'power'
        labels:     (label1, label2) tuple; defaults to the file basenames
        stride:     sample every Nth timestep (default 10)
        save_path:  save figure here instead of displaying
        state:      state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)
    """
    import os

    if labels is None:
        labels = (
            os.path.splitext(os.path.basename(filepath1))[0],
            os.path.splitext(os.path.basename(filepath2))[0],
        )

    t1, v1 = compute_quantity(filepath1, quantity, stride=stride, state=state)
    t2, v2 = compute_quantity(filepath2, quantity, stride=stride, state=state)
    if quantity == "power":
        v1, v2 = v1 / 1e6, v2 / 1e6

    ylabel = QUANTITY_LABELS.get(quantity, quantity)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t1, v1, linewidth=1.2, label=labels[0])
    ax.plot(t2, v2, linewidth=1.2, label=labels[1], linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def compute_core_averages(filepath, quantities, stride=10, state=1):
    """
    Compute spatial mean over core nodes for one or more datasets at each timestep.

    A single pass through the H5 file reads all requested quantities together.

    Args:
        filepath:   path to the H5 file
        quantities: list of dataset names, e.g. ['average_temperature', 'average_velocity']
        stride:     sample every Nth timestep (default 10)
        state:      state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        times   (n,) float array – seconds
        results dict  quantity name -> (n,) float array of core-node spatial means
    """
    core = _get_core_nodes(filepath)
    core_indices = np.array([idx for idx, _ in core], dtype=int)
    dt = _dt(filepath, state=state)

    with h5py.File(filepath, "r") as f:
        fd1 = f[_fluid_data_path(state)]
        all_keys = _sorted_timestep_keys(fd1)
        selected = all_keys[::stride]

        n = len(selected)
        times = np.empty(n, dtype=np.float64)
        results = {q: np.empty(n, dtype=np.float64) for q in quantities}

        for i, key in enumerate(selected):
            ts = fd1[key]
            # Native output numbering continues across states; time resets.
            times[i] = (int(key) - int(all_keys[0]) + 1) * dt if dt is not None else float(i * stride)
            for q in quantities:
                results[q][i] = float(ts[q][core_indices].mean())

    return times, results


def compute_total_delayed_source(filepath, stride=10, state=1):
    """
    Compute total core-average delayed neutron source (sum_i lambda_i * C_i_bar)
    at each sampled timestep, where C_i_bar is the spatial mean of precursor
    group i over core nodes.

    Args:
        filepath: path to the H5 file
        stride:   sample every Nth timestep (default 10)
        state:    state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)

    Returns:
        times   (n,) float array – seconds
        source  (n,) float array – total delayed source (same units as precursor
                                   concentrations * s^-1)
    """
    core = _get_core_nodes(filepath)
    core_indices = np.array([idx for idx, _ in core], dtype=int)
    dt = _dt(filepath, state=state)
    n_groups = len(PRECURSOR_LAMBDAS)

    prec_names = [
        f"neutron_precursor_mass_concentration_{i}" for i in range(n_groups)
    ]

    with h5py.File(filepath, "r") as f:
        fd1 = f[_fluid_data_path(state)]
        all_keys = _sorted_timestep_keys(fd1)
        selected = all_keys[::stride]

        n = len(selected)
        times = np.empty(n, dtype=np.float64)
        source = np.empty(n, dtype=np.float64)

        for i, key in enumerate(selected):
            ts = fd1[key]
            pg = ts[PRECURSOR_GROUP]
            # Native output numbering continues across states; time resets.
            times[i] = (int(key) - int(all_keys[0]) + 1) * dt if dt is not None else float(i * stride)
            total = 0.0
            for j, name in enumerate(prec_names):
                core_mean = float(pg[name][core_indices].mean())
                total += PRECURSOR_LAMBDAS[j] * core_mean
            source[i] = total

    return times, source


def plot_delayed_source(filepath, stride=10, save_path=None, state=1):
    """
    Plot the core-average delayed neutron source (sum_i lambda_i * C_i_bar)
    vs. time for a single H5 file.

    Args:
        filepath:  path to the H5 file
        stride:    sample every Nth timestep (default 10)
        save_path: if given, save the figure to this path instead of showing it
        state:     state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)
    """
    times, source = compute_total_delayed_source(filepath, stride=stride, state=state)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, source, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\sum_i \lambda_i \bar{C}_i$ (kg m$^{-3}$ s$^{-1}$)")
    ax.set_title("Core-Avg Delayed Neutron Source vs. Time")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    return fig, ax


def plot_full_comparison(filepath1, filepath2, labels=None, stride=10, save_path=None, state=1):
    """
    Four-panel comparison plot: total core power, avg core temperature, avg core
    velocity, and core-average delayed neutron source (sum_i lambda_i * C_i_bar).

    Args:
        filepath1:  path to first H5 file
        filepath2:  path to second H5 file
        labels:     (label1, label2) tuple; defaults to file basenames
        stride:     sample every Nth timestep (default 10)
        save_path:  save figure here instead of displaying
        state:      state number to read from (i.e. 'solution/fluid_data_000<state>'; default 1)
    """
    import os

    if labels is None:
        labels = (
            os.path.splitext(os.path.basename(filepath1))[0],
            os.path.splitext(os.path.basename(filepath2))[0],
        )

    quantities = ["average_temperature", "average_velocity"]

    t1_pwr, p1 = compute_total_core_power(filepath1, stride=stride, state=state)
    t2_pwr, p2 = compute_total_core_power(filepath2, stride=stride, state=state)
    t1, avg1 = compute_core_averages(filepath1, quantities, stride=stride, state=state)
    t2, avg2 = compute_core_averages(filepath2, quantities, stride=stride, state=state)
    t1_dn, dn1 = compute_total_delayed_source(filepath1, stride=stride, state=state)
    t2_dn, dn2 = compute_total_delayed_source(filepath2, stride=stride, state=state)

    def _make_plot(t1, y1, t2, y2, ylabel, title, path):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t1, y1, linewidth=1.2, label=labels[0])
        ax.plot(t2, y2, linewidth=1.2, label=labels[1], linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
            print(f"Figure saved to {path}")
        else:
            plt.show()
        return fig, ax

    import os

    def _insert_suffix(path, suffix):
        if path is None:
            return None
        base, ext = os.path.splitext(path)
        return f"{base}_{suffix}{ext}"

    fig_pwr, ax_pwr = _make_plot(
        t1_pwr, p1 / 1e6, t2_pwr, p2 / 1e6,
        "Total Core Power (MW)", "Total Core Power vs. Time",
        _insert_suffix(save_path, "power"),
    )
    fig_tmp, ax_tmp = _make_plot(
        t1, avg1["average_temperature"], t2, avg2["average_temperature"],
        "Avg Core Temperature (K)", "Avg Core Temperature vs. Time",
        _insert_suffix(save_path, "temperature"),
    )
    fig_vel, ax_vel = _make_plot(
        t1, avg1["average_velocity"], t2, avg2["average_velocity"],
        "Avg Core Velocity (m/s)", "Avg Core Velocity vs. Time",
        _insert_suffix(save_path, "velocity"),
    )
    fig_dn, ax_dn = _make_plot(
        t1_dn, dn1, t2_dn, dn2,
        r"$\sum_i \lambda_i \bar{C}_i$ (kg m$^{-3}$ s$^{-1}$)",
        "Core-Avg Delayed Neutron Source vs. Time",
        _insert_suffix(save_path, "delayed_source"),
    )

    return (fig_pwr, ax_pwr), (fig_tmp, ax_tmp), (fig_vel, ax_vel), (fig_dn, ax_dn)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process a lofi-neutronics H5 output file."
    )
    parser.add_argument("filepath", help="Path to the .h5 file")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Load every Nth timestep when reading full solution (default: 1)"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="First timestep index to include"
    )
    parser.add_argument(
        "--stop", type=int, default=None,
        help="Last timestep index to include"
    )
    parser.add_argument(
        "--timestep", type=int, default=None,
        help="Read and print a single timestep's data"
    )
    parser.add_argument(
        "--plot-quantity", default=None, metavar="QUANTITY", choices=SCALAR_DATASETS,
        help="Plot a quantity vs time (sampled every --stride timesteps); "
             f"choices: {', '.join(SCALAR_DATASETS)}"
    )
    parser.add_argument(
        "--plot-delayed-source", action="store_true",
        help="Plot core-avg delayed neutron source vs time (sampled every --stride timesteps)"
    )
    parser.add_argument(
        "--compare", default=None, metavar="FILEPATH2",
        help="Second H5 file to overlay on the --plot-quantity plot for comparison"
    )
    parser.add_argument(
        "--labels", nargs=2, default=None, metavar=("LABEL1", "LABEL2"),
        help="Legend labels for --compare (default: file basenames)"
    )
    parser.add_argument(
        "--full-comparison", action="store_true",
        help="Plot power, temperature, and velocity comparison (requires --compare)"
    )
    parser.add_argument(
        "--save", default=None,
        help="Save the plot to this file path instead of displaying it"
    )
    parser.add_argument(
        "--state", type=int, default=1,
        help="State number to read from, i.e. 'solution/fluid_data_000<state>' (default: 1)"
    )
    args = parser.parse_args()

    if args.full_comparison:
        if not args.compare:
            parser.error("--full-comparison requires --compare FILEPATH2")
        plot_full_comparison(
            args.filepath, args.compare,
            labels=args.labels,
            stride=args.stride,
            save_path=args.save,
            state=args.state,
        )
        return

    if args.plot_quantity:
        if args.compare:
            plot_quantity_comparison(
                args.filepath, args.compare, args.plot_quantity,
                labels=args.labels,
                stride=args.stride,
                save_path=args.save,
                state=args.state,
            )
        else:
            plot_quantity(args.filepath, args.plot_quantity, stride=args.stride,
                           save_path=args.save, state=args.state)
        return

    if args.plot_delayed_source:
        plot_delayed_source(args.filepath, stride=args.stride,
                             save_path=args.save, state=args.state)
        return

    if args.timestep is not None:
        data = read_timestep(args.filepath, args.timestep, state=args.state)
        t = data.get("time")
        print(f"Timestep {args.timestep}" + (f" (t = {t:.4f} s)" if t else ""))
        for name in SCALAR_DATASETS:
            if name in data:
                arr = data[name]
                print(f"  {name}: min={arr.min():.6g}  max={arr.max():.6g}  mean={arr.mean():.6g}")
        if PRECURSOR_GROUP in data:
            for pk, arr in data[PRECURSOR_GROUP].items():
                print(f"  {pk}: min={arr.min():.6g}  max={arr.max():.6g}")
        return

    summarize(args.filepath, state=args.state)

    if args.stride != 1 or args.start is not None or args.stop is not None:
        print(f"\nLoading solution (stride={args.stride}) ...")
        sol = read_solution(args.filepath, stride=args.stride,
                            start=args.start, stop=args.stop, state=args.state)
        print(f"Loaded {len(sol['timestep_indices'])} timesteps.")
        for name in SCALAR_DATASETS:
            if name in sol:
                arr = sol[name]
                print(f"  {name}: shape {arr.shape}  min={arr.min():.6g}  max={arr.max():.6g}")
        if "neutron_precursor_concentrations" in sol:
            arr = sol["neutron_precursor_concentrations"]
            print(f"  neutron_precursor_concentrations: shape {arr.shape}")


if __name__ == "__main__":
    main()
