# Simulator: one frame per preset

Seed 42, spot at (15.3, 14.7), 32 x 32 frame, signal taken from each preset.

| Preset | Nominal SNR | Peak spot pixel (e) | Saturated pixels | DN range |
|---|---|---|---|---|
| silicon | 88.5 | 494 | 0 | 98 to 303 |
| ingaas | 591.8 | 49,569 | 0 | 87 to 1754 |

The InGaAs peak is about 40% of the 120,000 e full well. The silicon peak is about 5% of its 10,000 e full well, so the silicon default sits at the dim end of the usable range.

`simulator.py` is a module with no command line of its own. To regenerate the table, run this from `src/` with `python`:

```python
from simulator import load_config, make_rngs, make_sensor_maps, simulate_frame

for preset in ("silicon", "ingaas"):
    camera, spot, background = load_config(preset)
    sensor_rng, frame_rng = make_rngs(seed=42)
    maps = make_sensor_maps(camera, sensor_rng)
    frame = simulate_frame(spot, camera, background, maps, frame_rng)
    print(preset, f"SNR {frame.nominal_snr:.1f}", f"peak {frame.expected_spot_e.max():,.0f} e",
          f"saturated {frame.n_saturated_pixels}",
          f"DN {frame.image_dn.min()} to {frame.image_dn.max()}")
```
