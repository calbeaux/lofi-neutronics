import argparse
import time
from Coupling.simulation import Simulation


def main(input_file, verbose = True, **kwargs):
    start_time = time.time()

    simulation = Simulation(input_file, verbose=verbose)
    simulation.setUpDataContainer()

    if verbose:
        print(f"Program executed in {time.time() - start_time:.3f} seconds.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a SyTH problem or create an h5 SyTH input.")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    main(input_file=args.input, verbose=not args.quiet)
