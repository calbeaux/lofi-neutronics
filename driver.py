import argparse
import multiprocessing as mp
import time
from Coupling.simulation import Simulation


def _run_simulation(input_file, verbose=True, **kwargs):
    start_time = time.time()

    simulation = Simulation(input_file, verbose=verbose)
    try:
        simulation.setUpDataContainer()
        for index, state in enumerate(simulation.states):
            if verbose:
                print(f"Running state {index}")
            simulation.setState(index, state)
            simulation.run()

        if verbose:
            print(f"Program executed in {time.time() - start_time:.3f} seconds.\n")
    finally:
        simulation._syth.close()


def main(input_file, verbose=True, **kwargs):
    context = mp.get_context("spawn")
    process = context.Process(target=_run_simulation, args=(input_file, verbose), kwargs=kwargs)
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
    args = parser.parse_args()

    main(input_file=args.input, verbose=not args.quiet)
