# Sub-pixel spot localization

Find the centre of a laser spot on a small camera frame to a fraction of a pixel, and measure how well that can be done. The repository holds a camera simulator, four position estimators, and the experiments on them: window size, accuracy against signal-to-noise ratio, time per frame, tracking a moving spot, and outdoor conditions.

## Reproduce every result

Set up once, with [conda](https://docs.conda.io/) (Anaconda or Miniconda):

```
conda env create -f environment.yml
conda activate spot-localization
```

Then one command:

```
python src/run.py
```

It runs every experiment in order and writes each one's CSV files, tables and charts to `results/<name>/`, replacing the committed ones. On an 8-core Apple M3 the full run takes about 42 minutes: under 1 for `estimators`, 6 for `characterization`, under 1 for `realtime`, 17 each for `dynamic_tracking` and `real_world`.

| To | Run |
|---|---|
| check that everything runs, in about 3 minutes | `python src/run.py --quick` |
| run some experiments only | `python src/run.py --only characterization dynamic_tracking` |
| run one experiment on its own | `python src/characterize.py` (or `window_size.py`, `realtime.py`, `dynamic.py`, `real_world.py`) |
| write somewhere else | add `--output DIR` |

`--quick` is a smoke run: few frames and short records. Its numbers mean nothing, and it writes to `quick_output/`, which git ignores, so it never replaces a committed result.

## What reproduces, and how exactly

Every experiment except `realtime` is seeded: every random stream is derived from a fixed seed, and the result does not depend on the number of worker processes or the order they finish in. In an environment created from `environment.yml` on the machine that produced the committed results (Apple M3, macOS 14), a full rerun of `characterization`, `dynamic_tracking` and `real_world` gives back all 90,793 CSV cells and the three `tables.md` files unchanged, byte for byte, and a rerun of `estimators` gives back its window-size table unchanged. The charts were restyled afterwards and redrawn from those same results by the same chart functions.

On another machine expect the last digits to move. The cause is the library NumPy and SciPy use for linear algebra, not the code. `environment.yml` asks for OpenBLAS. A rerun with the same package versions on Apple's Accelerate changed 27,147 of the 90,793 cells, the largest by 3 parts in 10,000, and every number in the three `tables.md` files still came out the same as printed.

`realtime` measures time per frame. Its numbers depend on the machine and on what else the machine is doing, so they will not match from one run to the next. The ranking of the methods should.

## Where things are

| Folder | Holds |
|---|---|
| `src/` | all the code, as one flat set of modules |
| `configs/` | the two camera presets, `silicon.toml` and `ingaas.toml` |
| `results/` | one folder per experiment, with its CSV files, tables and charts |

| Part | Question | Code | Results |
|---|---|---|---|
| Simulator | What does the camera record? | `src/simulator.py` | `results/simulator/` |
| Estimators | Where is the spot? | `src/estimators.py`; `src/window_size.py` (why the window is 11 x 11) | `results/estimators/` |
| Characterization | How large is the error at each signal level, against the theoretical floor? | `src/characterize.py`, `src/precision_floor.py` | `results/characterization/` |
| Real time | Which methods fit a 1 kHz loop? | `src/realtime.py` | `results/realtime/` |
| Dynamic tracking | Can a moving spot be followed and a periodic disturbance identified? | `src/dynamic.py`, with `trajectory.py`, `spectral.py`, `tracking_experiment.py`, `tracking_plots.py` | `results/dynamic_tracking/` |
| Real world | What do fading, daylight, spot shape, a second source and temperature do? | `src/real_world.py` | `results/real_world/` |

`src/run.py` runs the experiments through each script's own `main()`. The simulator and the estimators are code the experiments call; `results/simulator/summary.md` shows one frame per preset and the snippet that produces it.
