"""
Run every experiment, in order, each through its own main().

    python src/run.py

Options: --only characterization dynamic_tracking (only these, still in the
order below), --quick (a short smoke run of each that proves the code works;
its numbers mean nothing), --output DIR (write under DIR instead).

The experiments, in order:

  estimators         the window-size check behind the 11 x 11 window
  characterization   error of each estimator against SNR, and its bias
  realtime           time per frame
  dynamic_tracking   a moving spot and a periodic disturbance
  real_world         fading, daylight, spot shape, a second source, temperature

Where the results go:

  default        results/<name>/, replacing the committed results
  --quick        quick_output/<name>/, which git ignores, so a smoke run
                 never replaces a committed result
  --output DIR   DIR/<name>/, quick or not

The simulator and the estimators themselves are code that the experiments
call; they have nothing of their own to regenerate.

realtime measures time, so its numbers depend on the machine and on what
else it is doing. Every other number is seeded and reproduces exactly.
"""

import os

# One thread for the linear-algebra library, set before numpy loads (see the
# same lines in characterize.py). Set here because this file is the first to
# run, and the modules below import numpy.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import time
from pathlib import Path

import characterize
import dynamic
import real_world
import realtime
import window_size

ROOT = Path(__file__).parent.parent
QUICK_DIR = ROOT / "quick_output"

# Name (also the folder under results/): entry point. In the order they run.
EXPERIMENTS = {
    "estimators": window_size.main,
    "characterization": characterize.main,
    "realtime": realtime.main,
    "dynamic_tracking": dynamic.main,
    "real_world": real_world.main,
}


def experiment_arguments(name: str, quick: bool, output: Path | None) -> list[str]:
    """The command line handed to one experiment's main()."""
    arguments = ["--quick"] if quick else []
    if output is None and quick:
        output = QUICK_DIR
    if output is not None:
        arguments += ["--output", str(output / name)]
    return arguments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS),
                        metavar="NAME", help="experiments to run, from: " + " ".join(EXPERIMENTS))
    parser.add_argument("--quick", action="store_true",
                        help="short smoke run of each experiment; written to quick_output/")
    parser.add_argument("--output", type=Path, default=None, metavar="DIR",
                        help="write to DIR/<name>/ instead of results/<name>/")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    for name, experiment_main in EXPERIMENTS.items():
        if name not in args.only:
            continue
        arguments = experiment_arguments(name, args.quick, args.output)
        print(f"\n=== {name} {' '.join(arguments)}".rstrip(), flush=True)
        experiment_started = time.perf_counter()
        experiment_main(arguments)
        print(f"=== {name} done in {time.perf_counter() - experiment_started:.0f} s", flush=True)
    print(f"\nall done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
