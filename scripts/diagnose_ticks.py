"""Diagnose a missing heading tick. Run from the repo root:  python scripts/diagnose_tick.py

Prints which jax_render.py is loaded (and its hash), the JAX setup, and the tick pixels rendered
on CPU vs the default device (GPU if present), so file differences and device differences can be told apart.
"""
import hashlib, inspect, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax, jax.numpy as jnp, numpy as np
import env.jax_image_renderer as jr

EXPECTED_MD5 = "7ea45734e6c663eed631fc9ef230c1e3"   # env/jax_render.py as delivered (2 px tick)

path = inspect.getfile(jr)
md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
print(f"jax_render.py loaded from: {path}")
print(f"  md5 = {md5}  ({'matches delivered file' if md5 == EXPECTED_MD5 else 'DIFFERS from delivered file'})")
print(f"  RenderConfig defaults: {jr.RenderConfig()}")
print(f"jax {jax.__version__} | default backend: {jax.default_backend()} | devices: {jax.devices()} "
      f"| x64: {jax.config.jax_enable_x64}")


def tick_pixels(device):
    with jax.default_device(device):
        img = jr._render_one(jr.RenderConfig(), 0.3, jnp.zeros(2, jnp.float32), jnp.zeros(2, jnp.float32),
                             jnp.float32(0.3), jnp.array([3.0, 0.0], jnp.float32), jnp.zeros((0, 2), jnp.float32),
                             jnp.zeros((0,), jnp.float32), jnp.zeros((0,), jnp.float32))
        rc = jr.RenderConfig()
        res = jr.meters_per_pixel(rc)
        grid = jr._pixel_grid(rc)
        a = jnp.zeros(2, jnp.float32)
        seg = jr._soft_segment(grid, a, a + 1.6 * 0.3 * jnp.array([0.0, 1.0], jnp.float32),
                               rc.tick_width_px * res, res)
    return np.asarray(img)[36:41, 40:44, 0], np.asarray(seg)[36:41, 40:44]


for dev in [jax.devices("cpu")[0]] + ([jax.devices()[0]] if jax.default_backend() != "cpu" else []):
    img, seg = tick_pixels(dev)
    print(f"\n[{dev}] robot channel, rows 36-40, cols 40-43 (expected: row 38 -> 0 179 179 0, row 39 -> 0 255 255 0):")
    print(img)
    print(f"[{dev}] raw tick coverage from _soft_segment (expected ~0.7 at row 38, 1.0 at row 39):")
    print(np.round(seg, 2))