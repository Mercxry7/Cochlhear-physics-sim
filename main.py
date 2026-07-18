"""
app.main
==========
Streamlit entry point. Connects sidebar controls (sound wave frequencies/
amplitudes, electrode count, fluid depth) to the backend physics engine
in `ci_engine` and renders the resulting scalar potential and vector
electric field plots via `app.visualizations.FieldPlotter`.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Streamlit renders figures itself; avoid GUI backend issues.

import numpy as np
import streamlit as st

from ci_engine.acoustics import AcousticWave
from ci_engine.biophysics import CochlearFluidModel
from ci_engine.processor import NeuroprostheticFilterbank
from app.visualizations import FieldPlotter


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cochlear Electrophysics & Channel Interaction Simulator",
    layout="wide",
)

SAMPLE_RATE = 16000  # Hz. Fixed: high enough for all UI-permitted tone frequencies.
DURATION = 0.5  # seconds. Short signal is enough for a stable spectrogram envelope.
MAX_TONES = 3  # Number of independently controllable sine wave components.


def _build_sidebar() -> dict:
    """
    Render all sidebar controls and return their current values as a dict.
    Kept separate from the render loop so the parameter set is easy to see
    and to unit-test/extend independently of the plotting code.
    """
    st.sidebar.header("Sound Source")
    st.sidebar.caption("Add up to three sine-wave tones that make up the input sound.")

    tones = []
    for i in range(MAX_TONES):
        with st.sidebar.expander(f"Tone {i + 1}", expanded=(i == 0)):
            enabled = st.checkbox("Enabled", value=(i == 0), key=f"tone_enabled_{i}")
            freq = st.slider(
                "Frequency (Hz)",
                min_value=100,
                max_value=7000,
                value=440 * (i + 1),
                step=10,
                key=f"tone_freq_{i}",
            )
            amp = st.slider(
                "Amplitude",
                min_value=0.0,
                max_value=1.0,
                value=1.0 if i == 0 else 0.5,
                step=0.05,
                key=f"tone_amp_{i}",
            )
            tones.append({"enabled": enabled, "frequency": float(freq), "amplitude": float(amp)})

    st.sidebar.header("Cochlear Implant")
    num_electrodes = st.sidebar.slider(
        "Electrode count (N)", min_value=4, max_value=22, value=12, step=1
    )
    low_freq = st.sidebar.number_input(
        "Filterbank low frequency (Hz)", min_value=50, max_value=1000, value=200, step=50
    )
    high_freq = st.sidebar.number_input(
        "Filterbank high frequency (Hz)", min_value=2000, max_value=7900, value=7000, step=100
    )

    st.sidebar.header("Cochlear Fluid")
    fluid_depth = st.sidebar.slider(
        "Fluid/tissue depth constant (d)",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1,
        help="Larger d spreads current more (models a more conductive or "
        "deeper fluid path); smaller d makes electrode stimulation more focal.",
    )
    resolution = st.sidebar.select_slider(
        "Field grid resolution",
        options=[40, 60, 80, 100, 140, 180],
        value=100,
        help="Higher resolution is a smoother-looking field but slower to compute.",
    )

    return {
        "tones": tones,
        "num_electrodes": int(num_electrodes),
        "low_freq": float(low_freq),
        "high_freq": float(high_freq),
        "fluid_depth": float(fluid_depth),
        "resolution": int(resolution),
    }


def _run_pipeline(params: dict) -> dict:
    """
    Execute the full backend pipeline: acoustic wave -> electrode charges
    -> scalar potential -> electric field. Returns every intermediate
    result needed for both plotting and diagnostic display.
    """
    wave = AcousticWave(sample_rate=SAMPLE_RATE, duration=DURATION)
    for tone in params["tones"]:
        if tone["enabled"] and tone["amplitude"] > 0:
            wave.add_sine_wave(frequency=tone["frequency"], amplitude=tone["amplitude"])
    signal = wave.get_signal()

    bank = NeuroprostheticFilterbank(
        num_electrodes=params["num_electrodes"],
        low_freq=params["low_freq"],
        high_freq=params["high_freq"],
    )
    charges = bank.extract_electrode_charges(signal, sample_rate=SAMPLE_RATE)

    model = CochlearFluidModel(
        num_electrodes=params["num_electrodes"],
        fluid_depth=params["fluid_depth"],
        resolution=params["resolution"],
    )
    V = model.calculate_scalar_potential(charges)
    Ex, Ey = model.calculate_electric_field()
    X, Y = model.mesh

    return {
        "signal": signal,
        "bank": bank,
        "charges": charges,
        "model": model,
        "V": V,
        "Ex": Ex,
        "Ey": Ey,
        "X": X,
        "Y": Y,
    }


def _render_results(results: dict) -> None:
    """Render the main-panel plots and diagnostic panels from a pipeline run."""
    plotter = FieldPlotter()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scalar Potential Field")
        fig_contour = plotter.plot_scalar_contour(
            results["X"],
            results["Y"],
            results["V"],
            electrode_positions=results["model"].electrode_positions,
        )
        st.pyplot(fig_contour, clear_figure=False)

    with col2:
        st.subheader("Electric Field (Current Flow)")
        fig_combined = plotter.plot_scalar_contour(
            results["X"],
            results["Y"],
            results["V"],
            electrode_positions=results["model"].electrode_positions,
            title="Potential + Vector Field Overlay",
        )
        fig_combined = plotter.plot_vector_quiver(
            results["X"], results["Y"], results["Ex"], results["Ey"], base_figure=fig_combined
        )
        st.pyplot(fig_combined, clear_figure=False)

    st.subheader("Electrode Charge Allocation")
    st.caption(
        "How much of the input sound's energy landed on each electrode. "
        "Adjacent non-zero electrodes indicate channel interaction / current spread."
    )
    charge_data = {
        f"E{i + 1} ({freq:.0f} Hz)": float(q)
        for i, (freq, q) in enumerate(
            zip(results["bank"].band_centers, results["charges"])
        )
    }
    st.bar_chart(charge_data)


def main() -> None:
    st.title("🔊 Cochlear Electrophysics & Channel Interaction Simulator")
    st.markdown(
        "Simulates how a cochlear implant decomposes sound into electrode "
        "charges, and visualizes the resulting electrical potential and "
        "current spread (channel interaction) in the cochlear fluid."
    )

    params = _build_sidebar()

    if not any(t["enabled"] and t["amplitude"] > 0 for t in params["tones"]):
        st.warning("Enable at least one tone in the sidebar with amplitude > 0 to see a result.")
        return

    if params["high_freq"] >= SAMPLE_RATE / 2:
        st.error(
            f"Filterbank high frequency ({params['high_freq']:.0f} Hz) must be below "
            f"the Nyquist limit ({SAMPLE_RATE / 2:.0f} Hz). Lower it in the sidebar."
        )
        return

    with st.spinner("Running acoustic-to-biophysics pipeline..."):
        results = _run_pipeline(params)

    _render_results(results)


if __name__ == "__main__":
    main()
