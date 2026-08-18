"""Minimal NumPy compatibility runtime for the TorchSig calls used by Cliff.

This module is a smoke-test runtime only.  It preserves the public call shapes
used by ``run_pilot.py`` and common-random-number pairing, but it is not a
replacement for the frozen TorchSig 2.1.1 environment used for formal runs.
"""

from __future__ import annotations

import sys
import types

import numpy as np


def _constellation(name: str) -> np.ndarray:
    name = name.lower()
    if name == "bpsk":
        points = np.asarray([-1.0, 1.0], dtype=np.complex128)
    elif name == "qpsk":
        points = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
    elif name == "8psk":
        points = np.exp(1j * (np.pi / 8.0 + np.arange(8) * np.pi / 4.0))
    elif name == "16qam":
        levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
        points = np.asarray([i + 1j * q for i in levels for q in levels])
    else:
        raise ValueError(f"unsupported compatibility constellation: {name}")
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def _srrc_taps(samples_per_symbol: int, alpha: float, span_symbols: int = 8) -> np.ndarray:
    half = span_symbols * samples_per_symbol // 2
    time = np.arange(-half, half + 1, dtype=float) / samples_per_symbol
    taps = np.empty_like(time)
    for index, value in enumerate(time):
        if abs(value) < 1e-12:
            taps[index] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif alpha > 0 and abs(abs(value) - 1.0 / (4.0 * alpha)) < 1e-10:
            taps[index] = (
                alpha
                / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - alpha))
                + 4.0 * alpha * value * np.cos(np.pi * value * (1.0 + alpha))
            )
            denominator = np.pi * value * (1.0 - (4.0 * alpha * value) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def constellation_modulator(
    *,
    constellation_name: str,
    pulse_shape_name: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    alpha_rolloff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if pulse_shape_name.lower() != "srrc":
        raise ValueError("compatibility runtime implements only SRRC pulse shaping")
    samples_per_symbol = max(2, int(round(sample_rate / bandwidth)))
    guard = 12
    symbol_count = int(np.ceil(num_samples / samples_per_symbol)) + 2 * guard
    points = _constellation(constellation_name)
    symbols = points[rng.integers(0, len(points), size=symbol_count)]
    upsampled = np.zeros(symbol_count * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols
    shaped = np.convolve(
        upsampled,
        _srrc_taps(samples_per_symbol, float(alpha_rolloff)),
        mode="same",
    )
    start = guard * samples_per_symbol
    result = shaped[start : start + int(num_samples)]
    if result.size != int(num_samples):
        raise RuntimeError("compatibility modulator produced a short signal")
    return result.astype(np.complex64)


def nonlinear_amplifier(
    x: np.ndarray,
    *,
    gain: float,
    psat_backoff: float,
    phi_max: float,
    phi_slope: float,
    auto_scale: bool,
) -> np.ndarray:
    value = np.asarray(x, dtype=np.complex128) * float(gain)
    if auto_scale:
        value = value / np.sqrt(np.mean(np.abs(value) ** 2) + 1e-12)
    magnitude = np.abs(value)
    saturation = np.sqrt(10.0 ** (float(psat_backoff) / 10.0))
    compressed = magnitude / np.power(1.0 + (magnitude / saturation) ** 4, 0.25)
    phase_rotation = float(phi_max) * (1.0 - np.exp(-float(phi_slope) * magnitude**2))
    return (compressed * np.exp(1j * (np.angle(value) + phase_rotation))).astype(np.complex64)


def carrier_phase_noise(
    x: np.ndarray, *, phase_noise_degrees: float, rng: np.random.Generator
) -> np.ndarray:
    standard_deviation = np.deg2rad(float(phase_noise_degrees))
    phase = rng.normal(0.0, standard_deviation, size=len(x))
    return (np.asarray(x) * np.exp(1j * phase)).astype(np.complex64)


def awgn(x: np.ndarray, *, noise_power_db: float, rng: np.random.Generator) -> np.ndarray:
    noise_power = 10.0 ** (float(noise_power_db) / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))
    )
    return (np.asarray(x) + noise).astype(np.complex64)


def install() -> None:
    """Install import-compatible smoke modules only when real packages are absent."""
    if "torchsig" in sys.modules:
        return
    torch = types.ModuleType("torch")
    torch.__version__ = "compat-no-tensor-runtime"
    # SciPy's array-API detector performs this attribute lookup even when no
    # tensor operation is requested.
    torch.Tensor = type("Tensor", (), {})
    torchsig = types.ModuleType("torchsig")
    torchsig.__version__ = "2.1.1-numpy-smoke-compat"
    signals = types.ModuleType("torchsig.signals")
    builders = types.ModuleType("torchsig.signals.builders")
    constellation = types.ModuleType("torchsig.signals.builders.constellation")
    transforms = types.ModuleType("torchsig.transforms")
    functional = types.ModuleType("torchsig.transforms.functional")
    constellation.constellation_modulator = constellation_modulator
    functional.nonlinear_amplifier = nonlinear_amplifier
    functional.carrier_phase_noise = carrier_phase_noise
    functional.awgn = awgn
    transforms.functional = functional
    sys.modules.update(
        {
            "torch": torch,
            "torchsig": torchsig,
            "torchsig.signals": signals,
            "torchsig.signals.builders": builders,
            "torchsig.signals.builders.constellation": constellation,
            "torchsig.transforms": transforms,
            "torchsig.transforms.functional": functional,
        }
    )
