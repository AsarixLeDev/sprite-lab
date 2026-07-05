"""Small OKLab conversion helpers.

The functions here are dependency-free array conversions used by palette
quantization. They are deterministic and operate on RGB arrays with a trailing
channel dimension of size 3.
"""

from __future__ import annotations

import numpy as np


def rgb_u8_array_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 values with shape ``(..., 3)`` to OKLab floats."""

    rgb_array = np.asarray(rgb)
    if rgb_array.shape[-1:] != (3,):
        raise ValueError("rgb array must have shape (..., 3).")

    srgb = rgb_array.astype(np.float64) / 255.0
    linear = _srgb_to_linear(srgb)
    red = linear[..., 0]
    green = linear[..., 1]
    blue = linear[..., 2]

    l_val = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m_val = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s_val = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    l_root = np.cbrt(l_val)
    m_root = np.cbrt(m_val)
    s_root = np.cbrt(s_val)

    lab_l = 0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root
    lab_a = 1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root
    lab_b = 0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root

    return np.stack([lab_l, lab_a, lab_b], axis=-1)


def oklab_array_to_rgb_u8(oklab: np.ndarray) -> np.ndarray:
    """Convert OKLab floats with shape ``(..., 3)`` to clipped RGB uint8."""

    lab = np.asarray(oklab, dtype=np.float64)
    if lab.shape[-1:] != (3,):
        raise ValueError("oklab array must have shape (..., 3).")

    lab_l = lab[..., 0]
    lab_a = lab[..., 1]
    lab_b = lab[..., 2]

    l_root = lab_l + 0.3963377774 * lab_a + 0.2158037573 * lab_b
    m_root = lab_l - 0.1055613458 * lab_a - 0.0638541728 * lab_b
    s_root = lab_l - 0.0894841775 * lab_a - 1.2914855480 * lab_b

    l_val = l_root * l_root * l_root
    m_val = m_root * m_root * m_root
    s_val = s_root * s_root * s_root

    red = 4.0767416621 * l_val - 3.3077115913 * m_val + 0.2309699292 * s_val
    green = -1.2684380046 * l_val + 2.6097574011 * m_val - 0.3413193965 * s_val
    blue = -0.0041960863 * l_val - 0.7034186147 * m_val + 1.7076147010 * s_val

    linear = np.stack([red, green, blue], axis=-1)
    srgb = _linear_to_srgb(linear)
    clipped = np.clip(np.rint(srgb * 255.0), 0, 255)
    return clipped.astype(np.uint8)


def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((srgb + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
