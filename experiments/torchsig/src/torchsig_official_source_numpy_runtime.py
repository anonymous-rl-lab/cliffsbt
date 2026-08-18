"""Source-faithful NumPy execution of the TorchSig 2.1.1 calls used by Cliff.

This is deliberately narrower than an installed TorchSig distribution.  It
ports the exact tagged algorithms exercised by ``run_pilot.generate_iq``:
constellation modulation at the experiment's unit resampling rate, SRRC pulse
shaping, nonlinear amplification, carrier phase noise, and AWGN.  The purpose
is to remove the approximate signal-kernel confound while the standard
PyTorch/TorchSig wheel remains unavailable in this execution environment.

Upstream provenance
-------------------
TorchSig tag: v2.1.1
Source commit: d9abfe1af2b0216d2bacc31c677407ed31878086
Wheel SHA-256: 2e6ea54df639028b4914fee4daea0ed7e87ef53d15c4dddd939d0d867e24d2e1

This module must never be reported as a standard TorchSig package runtime.
"""

from __future__ import annotations

from copy import copy
import sys
import types

import numpy as np
from scipy import signal as sp


TORCHSIG_TAG = "v2.1.1"
TORCHSIG_SOURCE_COMMIT = "d9abfe1af2b0216d2bacc31c677407ed31878086"
TORCHSIG_WHEEL_SHA256 = (
    "2e6ea54df639028b4914fee4daea0ed7e87ef53d15c4dddd939d0d867e24d2e1"
)
RUNTIME_KIND = "torchsig_2.1.1_official_source_numpy_execution"
TorchSigComplexDataType = np.complex64


def _symbol_map(name: str) -> np.ndarray:
    """Return the v2.1.1 constellation map for the four experiment classes."""
    name = name.lower()
    if name == "bpsk":
        value = np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j)))
    elif name == "qpsk":
        value = np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2)),
            )
        )
    elif name == "8psk":
        value = np.exp(2j * np.pi * np.linspace(0, 7, 8) / 8.0)
    elif name == "16qam":
        value = np.add(
            *map(
                np.ravel,
                np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4)),
            )
        )
    else:
        raise ValueError(f"unsupported experiment constellation: {name}")
    return np.asarray(value)


def _estimate_filter_length(
    transition_bandwidth: float, attenuation_db: float, sample_rate: float
) -> int:
    length = int(np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22)))
    if np.equal(np.mod(length, 2), 0):
        length += 1
    return length


def _srrc_taps(samples_per_symbol: int, span_symbols: int, alpha: float) -> np.ndarray:
    m = span_symbols
    n_s = float(samples_per_symbol)
    n = np.arange(-m * n_s, m * n_s + 1)
    taps = np.zeros(int(2 * m * n_s + 1))
    for index in range(len(taps)):
        if n[index] * 4 * alpha == n_s or n[index] * 4 * alpha == -n_s:
            taps[index] = 0.5 * (
                (1 + alpha) * np.sin((1 + alpha) * np.pi / (4.0 * alpha))
                - (1 - alpha) * np.cos((1 - alpha) * np.pi / (4.0 * alpha))
                + (4 * alpha)
                / np.pi
                * np.sin((1 - alpha) * np.pi / (4.0 * alpha))
            )
        else:
            taps[index] = 4 * alpha / (
                np.pi * (1 - 16 * alpha**2 * (n[index] / n_s) ** 2)
            )
            taps[index] *= (
                np.cos((1 + alpha) * np.pi * n[index] / n_s)
                + np.sinc((1 - alpha) * n[index] / n_s)
                * (1 - alpha)
                * np.pi
                / (4.0 * alpha)
            )
    return taps


def _pad_head_tail_to_length(value: np.ndarray, length: int) -> np.ndarray:
    result = copy(value)
    if len(result) < length:
        zeros = length - len(result)
        result = np.concatenate(
            (np.zeros(int(np.ceil(zeros / 2))), result, np.zeros(int(np.floor(zeros / 2))))
        )
    elif len(result) > length:
        raise ValueError("signal is too long to be zero padded")
    return result


def _slice_tail_to_length(value: np.ndarray, length: int) -> np.ndarray:
    result = copy(value)
    if len(result) > length:
        result = result[:length]
    elif len(result) < length:
        raise ValueError("signal too short to be sliced")
    return result


def _slice_head_tail_to_length(value: np.ndarray, length: int) -> np.ndarray:
    result = copy(value)
    if len(result) > length:
        extra = len(result) - length
        result = result[int(np.ceil(extra / 2)) :]
        tail = int(np.floor(extra / 2))
        if tail > 0:
            result = result[:-tail]
    elif len(result) < length:
        raise ValueError("signal too short to be sliced")
    return result


def _constellation_modulator_baseband(
    constellation_name: str,
    pulse_shape_name: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    alpha_rolloff: float | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if max_num_samples <= 0 or oversampling_rate_nominal <= 0:
        raise ValueError("sample counts and oversampling rate must be positive")
    rng = np.random.default_rng() if rng is None else rng
    symbol_map = _symbol_map(constellation_name)
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))
    samples_per_symbol = oversampling_rate_nominal
    if pulse_shape_name == "rectangular":
        pulse_shape = np.ones(samples_per_symbol)
        filter_span = 0
    elif pulse_shape_name == "srrc":
        if alpha_rolloff is None or not 0 < alpha_rolloff < 1:
            raise ValueError("alpha_rolloff must be between 0 and 1")
        filter_length = _estimate_filter_length(alpha_rolloff, 120, 1)
        filter_span = int(np.ceil((filter_length - 1) / (2 * samples_per_symbol)))
        pulse_shape = _srrc_taps(samples_per_symbol, filter_span, alpha_rolloff)
    else:
        raise ValueError(f"pulse shape {pulse_shape_name} not supported")
    num_symbols = int(np.floor(max_num_samples / samples_per_symbol)) - 2 * filter_span
    num_symbols = max(num_symbols, 1)
    symbols = np.zeros(1)
    while np.equal(np.sum(np.abs(symbols)), 0):
        indices = rng.integers(low=0, high=len(symbol_map), size=num_symbols)
        symbols = symbol_map[indices]
    result = sp.upfirdn(pulse_shape, symbols, up=samples_per_symbol, down=1)
    if len(result) < max_num_samples:
        result = _pad_head_tail_to_length(result, max_num_samples)
    elif len(result) > max_num_samples:
        result = _slice_tail_to_length(result, max_num_samples)
    return result.astype(TorchSigComplexDataType)


def constellation_modulator(
    *,
    constellation_name: str,
    pulse_shape_name: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    alpha_rolloff: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """TorchSig v2.1.1 algorithm for the experiment's unit-resampling case."""
    if bandwidth <= 0 or sample_rate <= 0 or num_samples <= 0:
        raise ValueError("bandwidth, sample_rate, and num_samples must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    rng = np.random.default_rng() if rng is None else rng
    oversampling_rate = sample_rate / bandwidth
    oversampling_rate_baseband = 4
    resample_rate = oversampling_rate / oversampling_rate_baseband
    if not np.isclose(resample_rate, 1.0, rtol=0.0, atol=1e-15):
        raise NotImplementedError(
            "narrow official-source runtime is validated only for resample_rate_ideal == 1"
        )
    baseband_count = int(np.floor(num_samples / resample_rate))
    baseband_count = (
        oversampling_rate_baseband if baseband_count <= 0 else baseband_count
    )
    baseband = _constellation_modulator_baseband(
        constellation_name,
        pulse_shape_name,
        baseband_count,
        oversampling_rate_baseband,
        alpha_rolloff,
        rng,
    )
    result = baseband
    result = (
        _slice_head_tail_to_length(result, num_samples)
        if len(result) > num_samples
        else _pad_head_tail_to_length(result, num_samples)
    )
    if len(result) != num_samples:
        raise ValueError("constellation modulator produced an incorrect sample count")
    return result.astype(TorchSigComplexDataType)


def nonlinear_amplifier(
    data: np.ndarray,
    *,
    gain: float = 1.0,
    psat_backoff: float = 10.0,
    phi_max: float = 0.1,
    phi_slope: float = 0.01,
    auto_scale: bool = True,
) -> np.ndarray:
    n = len(data)
    magnitude = np.abs(data)
    phase = np.angle(data)
    in_power = magnitude**2
    mean_power = np.mean(in_power)
    psat = mean_power * psat_backoff
    scale_factor = psat / gain
    out_power = psat * np.tanh(in_power / scale_factor)
    out_magnitude = out_power**0.5
    phase_shift: float | np.ndarray = 0.0
    if not np.equal(phi_max, 0.0) and not np.equal(phi_slope, 0.0):
        slope = np.abs(phi_slope)
        phase_shift = (phi_max / 2) * (
            np.tanh((in_power - scale_factor) / slope) + 1
        )
    result = out_magnitude * np.exp(1j * (phase + phase_shift))
    if auto_scale:
        window = sp.windows.blackmanharris(n)
        input_power = np.max(np.abs(np.fft.fft(data * window)))
        output_power = np.max(np.abs(np.fft.fft(result * window)))
        result *= input_power / output_power
    return result.astype(TorchSigComplexDataType)


def carrier_phase_noise(
    data: np.ndarray,
    *,
    phase_noise_degrees: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    phase_degrees = rng.normal(0, phase_noise_degrees, data.size)
    return (data * np.exp(1j * phase_degrees * np.pi / 180)).astype(
        TorchSigComplexDataType
    )


def awgn(
    data: np.ndarray,
    *,
    noise_power_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    real_noise = rng.standard_normal(*data.shape)
    imag_noise = rng.standard_normal(*data.shape)
    result = data + 10.0 ** (noise_power_db / 20.0) * (
        real_noise + 1j * imag_noise
    ) / np.sqrt(2)
    return result.astype(TorchSigComplexDataType)


def install() -> None:
    """Expose only the import surface needed by the frozen Cliff generators."""
    if "torchsig" in sys.modules:
        raise RuntimeError("install official-source runtime before importing torchsig")
    torch = types.ModuleType("torch")
    torch.__version__ = "compat-no-tensor-runtime"
    torch.Tensor = type("Tensor", (), {})
    torchsig = types.ModuleType("torchsig")
    torchsig.__version__ = "2.1.1-official-source-numpy-execution"
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
