"""
ci_engine.acoustics
====================
Time-domain acoustic physics: generates complex sound waves as a linear
superposition of individual sine wave components.

Mathematical model
-------------------
    S(t) = sum_{n=1}^{k} A_n * sin(2 * pi * f_n * t)

Each call to `add_sine_wave` appends one (frequency, amplitude) component
to the superposition. `get_signal` evaluates S(t) over the configured
sample grid and returns the aggregated 1D waveform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _SineComponent:
    """A single sinusoidal term A_n * sin(2*pi*f_n*t) in the superposition."""

    frequency: float
    amplitude: float

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")
        if self.amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {self.amplitude}")


class AcousticWave:
    """
    Builds a composite acoustic waveform S(t) from superposed sine waves.

    Parameters
    ----------
    sample_rate : int
        Samples per second (Hz) used to discretize time. Must satisfy the
        Nyquist criterion for any frequency added later (sample_rate >
        2 * frequency), or `add_sine_wave` will raise.
    duration : float
        Length of the signal in seconds. Must be positive.

    Examples
    --------
    >>> wave = AcousticWave(sample_rate=16000, duration=0.5)
    >>> wave.add_sine_wave(frequency=440.0, amplitude=1.0)
    >>> wave.add_sine_wave(frequency=880.0, amplitude=0.5)
    >>> signal = wave.get_signal()
    >>> signal.shape
    (8000,)
    """

    def __init__(self, sample_rate: int, duration: float) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        self.sample_rate = int(sample_rate)
        self.duration = float(duration)

        # Precompute the time axis once; every component is evaluated on it.
        n_samples = int(round(self.sample_rate * self.duration))
        self._t = np.linspace(
            0.0, self.duration, num=n_samples, endpoint=False, dtype=np.float64
        )

        self._components: list[_SineComponent] = []

    @property
    def num_samples(self) -> int:
        """Number of discrete time samples in the signal."""
        return self._t.shape[0]

    @property
    def time_axis(self) -> np.ndarray:
        """The discretized time axis t used to evaluate S(t)."""
        return self._t

    @property
    def components(self) -> list[tuple[float, float]]:
        """List of (frequency, amplitude) pairs currently in the superposition."""
        return [(c.frequency, c.amplitude) for c in self._components]

    def add_sine_wave(self, frequency: float, amplitude: float) -> "AcousticWave":
        """
        Append one sine wave term A_n * sin(2*pi*f_n*t) to the superposition.

        Parameters
        ----------
        frequency : float
            Frequency f_n in Hz. Must be positive and below the Nyquist
            frequency (sample_rate / 2) to avoid aliasing.
        amplitude : float
            Amplitude A_n. Must be non-negative.

        Returns
        -------
        AcousticWave
            self, to allow chaining: wave.add_sine_wave(...).add_sine_wave(...)
        """
        nyquist = self.sample_rate / 2.0
        if frequency >= nyquist:
            raise ValueError(
                f"frequency {frequency} Hz exceeds Nyquist limit "
                f"({nyquist} Hz) for sample_rate={self.sample_rate}"
            )

        self._components.append(_SineComponent(frequency=frequency, amplitude=amplitude))
        return self

    def clear(self) -> None:
        """Remove all sine wave components, resetting the superposition to silence."""
        self._components.clear()

    def get_signal(self) -> np.ndarray:
        """
        Evaluate and return the aggregated waveform S(t).

        Returns
        -------
        np.ndarray
            1D array of shape (num_samples,) representing S(t). If no
            components have been added, returns an array of zeros
            (silence) of the correct length.
        """
        signal = np.zeros_like(self._t)
        for comp in self._components:
            signal += comp.amplitude * np.sin(2.0 * np.pi * comp.frequency * self._t)
        return signal

    def __repr__(self) -> str:
        return (
            f"AcousticWave(sample_rate={self.sample_rate}, duration={self.duration}, "
            f"components={len(self._components)})"
        )
