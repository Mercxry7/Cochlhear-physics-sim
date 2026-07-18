"""
ci_engine.processor
=====================
Digital signal processing layer: simulates the internal computer of a
cochlear implant. Takes a raw acoustic waveform S(t) and produces a set
of discrete electrical charges q_i, one per electrode, by running a
spectrogram and binning spectral energy into logarithmically spaced
frequency channels (mirroring how real CI filterbanks allocate apex-to
-base frequency bands across the electrode array).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


class NeuroprostheticFilterbank:
    """
    Maps an acoustic waveform to per-electrode charge values via a
    logarithmically spaced filterbank, mimicking a cochlear implant's
    channel allocation from apex (low frequency) to base (high frequency).

    Parameters
    ----------
    num_electrodes : int
        Number of discrete electrode channels N. Must be >= 1.
    low_freq : float
        Lower bound of the frequency range (Hz), corresponding to the
        cochlear apex. Must be positive.
    high_freq : float
        Upper bound of the frequency range (Hz), corresponding to the
        cochlear base. Must be greater than low_freq.

    Examples
    --------
    >>> from ci_engine.acoustics import AcousticWave
    >>> wave = AcousticWave(sample_rate=16000, duration=0.5)
    >>> wave.add_sine_wave(frequency=1000.0, amplitude=1.0)
    >>> signal = wave.get_signal()
    >>> bank = NeuroprostheticFilterbank(num_electrodes=12, low_freq=200, high_freq=7000)
    >>> charges = bank.extract_electrode_charges(signal, sample_rate=16000)
    >>> charges.shape
    (12,)
    """

    def __init__(self, num_electrodes: int, low_freq: float, high_freq: float) -> None:
        if num_electrodes < 1:
            raise ValueError(f"num_electrodes must be >= 1, got {num_electrodes}")
        if low_freq <= 0:
            raise ValueError(f"low_freq must be positive, got {low_freq}")
        if high_freq <= low_freq:
            raise ValueError(
                f"high_freq ({high_freq}) must be greater than low_freq ({low_freq})"
            )

        self.num_electrodes = int(num_electrodes)
        self.low_freq = float(low_freq)
        self.high_freq = float(high_freq)

        # Boundaries between adjacent electrode bands (N+1 edges -> N bands).
        self._band_edges = self._generate_frequency_map()

    def _generate_frequency_map(self) -> np.ndarray:
        """
        Create logarithmic frequency band boundaries from low_freq to
        high_freq. Log spacing mirrors the tonotopic (frequency-to-place)
        organization of the biological cochlea, which real CI filterbanks
        replicate.

        Returns
        -------
        np.ndarray
            Array of shape (num_electrodes + 1,) giving the edges of each
            of the num_electrodes logarithmically spaced bands.
        """
        return np.logspace(
            np.log10(self.low_freq),
            np.log10(self.high_freq),
            num=self.num_electrodes + 1,
            base=10.0,
        )

    @property
    def band_edges(self) -> np.ndarray:
        """The (num_electrodes + 1) frequency boundaries defining each electrode band."""
        return self._band_edges

    @property
    def band_centers(self) -> np.ndarray:
        """Geometric-mean center frequency of each electrode band (length num_electrodes)."""
        return np.sqrt(self._band_edges[:-1] * self._band_edges[1:])

    def extract_electrode_charges(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Run a spectrogram on the acoustic waveform, extract the spectral
        envelope within each logarithmic electrode band, and return the
        resulting per-electrode charge array q.

        Parameters
        ----------
        waveform : np.ndarray
            1D array representing S(t), as produced by
            `AcousticWave.get_signal()`.
        sample_rate : int
            Sampling rate (Hz) at which `waveform` was generated. Must
            match the rate used to build the waveform, or frequency
            binning will be misaligned.

        Returns
        -------
        np.ndarray
            1D array of shape (num_electrodes,), normalized to [0, 1],
            giving the charge q_i assigned to each electrode. An entirely
            silent waveform returns an all-zero array.
        """
        waveform = np.asarray(waveform, dtype=np.float64)
        if waveform.ndim != 1:
            raise ValueError(f"waveform must be 1D, got shape {waveform.shape}")
        if waveform.size < 2:
            raise ValueError("waveform must contain at least 2 samples")

        nyquist = sample_rate / 2.0
        if self.high_freq > nyquist:
            raise ValueError(
                f"filterbank high_freq ({self.high_freq} Hz) exceeds Nyquist "
                f"limit ({nyquist} Hz) for sample_rate={sample_rate}"
            )

        # nperseg must not exceed the signal length; scipy warns/errors otherwise.
        nperseg = min(256, waveform.size)
        freqs, _, Sxx = spectrogram(waveform, fs=sample_rate, nperseg=nperseg)

        # Spectral envelope: peak magnitude per frequency bin across all time frames.
        # Using max (not mean) matches the spec's "maximum amplitude within each band".
        envelope = Sxx.max(axis=1)

        charges = np.zeros(self.num_electrodes, dtype=np.float64)
        for i in range(self.num_electrodes):
            lo, hi = self._band_edges[i], self._band_edges[i + 1]
            # Last band includes its upper edge so high_freq itself is captured.
            if i == self.num_electrodes - 1:
                mask = (freqs >= lo) & (freqs <= hi)
            else:
                mask = (freqs >= lo) & (freqs < hi)

            if np.any(mask):
                charges[i] = envelope[mask].max()
            # else: no FFT bins fall in this band (band narrower than freq
            # resolution) -> charge stays 0.0, which is the correct silent
            # default rather than raising.

        # Normalize to [0, 1] so charges are comparable regardless of input
        # amplitude scale; guards against divide-by-zero on silent input.
        peak = charges.max()
        if peak > 0:
            charges = charges / peak

        return charges

    def __repr__(self) -> str:
        return (
            f"NeuroprostheticFilterbank(num_electrodes={self.num_electrodes}, "
            f"low_freq={self.low_freq}, high_freq={self.high_freq})"
        )
