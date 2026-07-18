"""
tests.test_acoustics
=======================
Pytest suite for ci_engine.acoustics.AcousticWave.
"""

from __future__ import annotations

import numpy as np
import pytest

from ci_engine.acoustics import AcousticWave


def test_empty_wave_is_silence() -> None:
    """No components added -> signal is all zeros, correct length."""
    wave = AcousticWave(sample_rate=8000, duration=0.25)
    signal = wave.get_signal()

    assert signal.shape == (2000,)
    assert np.allclose(signal, 0.0)


def test_signal_length_matches_sample_rate_and_duration() -> None:
    wave = AcousticWave(sample_rate=16000, duration=0.5)
    assert wave.num_samples == 8000
    assert wave.get_signal().shape == (8000,)


def test_single_sine_wave_matches_analytical_formula() -> None:
    """S(t) for one component should exactly match A*sin(2*pi*f*t)
    evaluated on the same time axis, per the spec's formula."""
    sample_rate = 8000
    duration = 0.1
    freq = 100.0
    amp = 0.75

    wave = AcousticWave(sample_rate=sample_rate, duration=duration)
    wave.add_sine_wave(frequency=freq, amplitude=amp)
    signal = wave.get_signal()

    t = wave.time_axis
    expected = amp * np.sin(2.0 * np.pi * freq * t)

    assert np.allclose(signal, expected)


def test_superposition_is_additive() -> None:
    """Two components together should equal the elementwise sum of each
    component computed alone (linearity of S(t) = sum A_n sin(2 pi f_n t))."""
    sample_rate = 8000
    duration = 0.1

    combined = AcousticWave(sample_rate=sample_rate, duration=duration)
    combined.add_sine_wave(frequency=100.0, amplitude=1.0)
    combined.add_sine_wave(frequency=300.0, amplitude=0.4)
    combined_signal = combined.get_signal()

    only_a = AcousticWave(sample_rate=sample_rate, duration=duration)
    only_a.add_sine_wave(frequency=100.0, amplitude=1.0)

    only_b = AcousticWave(sample_rate=sample_rate, duration=duration)
    only_b.add_sine_wave(frequency=300.0, amplitude=0.4)

    assert np.allclose(combined_signal, only_a.get_signal() + only_b.get_signal())


def test_add_sine_wave_returns_self_for_chaining() -> None:
    wave = AcousticWave(sample_rate=8000, duration=0.1)
    result = wave.add_sine_wave(frequency=200.0, amplitude=1.0)
    assert result is wave


def test_clear_resets_to_silence() -> None:
    wave = AcousticWave(sample_rate=8000, duration=0.1)
    wave.add_sine_wave(frequency=200.0, amplitude=1.0)
    assert not np.allclose(wave.get_signal(), 0.0)

    wave.clear()
    assert np.allclose(wave.get_signal(), 0.0)
    assert wave.components == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0, "duration": 0.5},
        {"sample_rate": -8000, "duration": 0.5},
        {"sample_rate": 8000, "duration": 0.0},
        {"sample_rate": 8000, "duration": -0.5},
    ],
)
def test_invalid_constructor_args_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AcousticWave(**kwargs)


def test_add_sine_wave_above_nyquist_raises() -> None:
    wave = AcousticWave(sample_rate=8000, duration=0.5)  # Nyquist = 4000 Hz
    with pytest.raises(ValueError):
        wave.add_sine_wave(frequency=4000.0, amplitude=1.0)  # >= Nyquist
    with pytest.raises(ValueError):
        wave.add_sine_wave(frequency=5000.0, amplitude=1.0)


def test_add_sine_wave_negative_amplitude_raises() -> None:
    wave = AcousticWave(sample_rate=8000, duration=0.5)
    with pytest.raises(ValueError):
        wave.add_sine_wave(frequency=200.0, amplitude=-1.0)


def test_add_sine_wave_nonpositive_frequency_raises() -> None:
    wave = AcousticWave(sample_rate=8000, duration=0.5)
    with pytest.raises(ValueError):
        wave.add_sine_wave(frequency=0.0, amplitude=1.0)
    with pytest.raises(ValueError):
        wave.add_sine_wave(frequency=-100.0, amplitude=1.0)
