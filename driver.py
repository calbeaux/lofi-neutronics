import argparse
import multiprocessing as mp
from pathlib import Path
import time
from Coupling.simulation import Simulation
from Neutronics.mpact_library import MPACTLibrary


def _run_simulation(input_file, verbose=True, library=None, mpact_dir=None,
                    temperature_source="dataset", **kwargs):
    start_time = time.time()

    if mpact_dir is not None:
        if library is not None:
            raise ValueError("Supply either library or mpact_dir, not both.")
        directory = Path(mpact_dir).expanduser()
        paths = sorted(directory.glob("*.h5"))
        if not paths:
            raise ValueError(f"No MPACT .h5 files found in {directory}")
        library = MPACTLibrary.from_files(paths, temperature_source=temperature_source)
    simulation = Simulation(input_file, verbose=verbose, library=library)
    try:
        simulation.set_up_data_container()
        for index, state in enumerate(simulation.states):
            if verbose:
                print(f"Running state {index}")
            simulation.set_state(index, state)
            simulation.run()

        if verbose:
            print(f"Program executed in {time.time() - start_time:.3f} seconds.\n")
    finally:
        simulation._syth.close()


def main(input_file, verbose=True, **kwargs):
    context = mp.get_context("spawn")
    process = context.Process(
        target=_run_simulation, args=(input_file, verbose), kwargs=kwargs
    )
    process.start()

    try:
        process.join()
    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupt received, stopping SyTH...")
        process.terminate()
        process.join()
        if process.is_alive():
            process.kill()
            process.join()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem.")
    parser.add_argument("input", help="Syth Input .h5 file")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mpact-dir", help="Directory of MPACT .h5 temperature samples; overrides embedded MPACT config")
    parser.add_argument("--mpact-temperature-source", choices=("dataset", "filename"),
                        default="dataset", help="Temperature source for --mpact-dir (default: dataset)")
    args = parser.parse_args()

    main(input_file=args.input, verbose=not args.quiet,
         mpact_dir=args.mpact_dir, temperature_source=args.mpact_temperature_source)
