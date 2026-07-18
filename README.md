# Cochlear Electrophysics & Channel Interaction Simulator

A computational physics engine and interactive visualization platform that
models how modern cochlear implants (CIs) convert sound into electrical
stimulation, and how the resulting electric fields overlap in the
cochlear fluid — a phenomenon called **channel interaction** or
**current spread**, which is a major limiting factor in how well CI
users can perceive pitch and music.

The simulator combines three domains in a single pipeline: acoustic wave
superposition, digital signal processing (a CI-style filterbank), and
multivariable calculus (scalar potential fields and their gradients).

## Why this matters

Cochlear implants bypass damaged hair cells and stimulate the auditory
nerve directly with an array of electrodes. Because the cochlea is
filled with highly conductive perilymph fluid, current from one
electrode doesn't stay contained to its intended location — it spreads
and overlaps with current from neighboring electrodes. This spatial
overlap blurs the spectral resolution a CI can deliver, which is a
central reason CI users often struggle with music and tonal languages
even when speech understanding is good.

This project makes that phenomenon visible and interactive: adjust the
input sound, the electrode count, or the fluid conductivity, and watch
how the electric field spreads and overlaps in real time.

## The Math

**1. Acoustic wave superposition** — the input sound is built from `k`
sine wave components:

```
S(t) = Σ (n=1 to k) A_n · sin(2π f_n t)
```

**2. Scalar potential field** — each of the `N` active electrodes
contributes an inverse-square potential to every point `(x, y)` in the
2D cochlear fluid plane:

```
V(x, y) = Σ (i=1 to N)  q_i / √((x - x_i)² + (y - y_i)² + d²)
```

`d` is a small tissue-depth constant that keeps the field finite
directly at an electrode's location (no singularity).

**3. Vector electric field** — the actual force felt by the auditory
nerve is the negative gradient of the potential:

```
E = -∇V = ( -∂V/∂x, -∂V/∂y )
```

Computed numerically via finite differences (`np.gradient`) over the
potential field.

## Architecture

The backend physics/DSP engine (`ci_engine/`) is completely decoupled
from the UI (`app/`) — every class in `ci_engine` can be imported and
used from a plain Python script or a Jupyter notebook with no Streamlit
dependency at all.

```
cochlear_physics_sim/
├── ci_engine/                 # BACKEND: math & signal processing
│   ├── acoustics.py           # AcousticWave — wave superposition, S(t)
│   ├── processor.py           # NeuroprostheticFilterbank — DSP, charge extraction
│   └── biophysics.py          # CochlearFluidModel — scalar potential & gradient
│
├── app/                       # FRONTEND: Streamlit UI
│   ├── main.py                # Entry point — sidebar controls, pipeline wiring
│   └── visualizations.py      # FieldPlotter — contour & quiver rendering
│
├── tests/                     # Pytest suite
│   ├── test_acoustics.py
│   └── test_biophysics.py
│
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <this-repo>
cd cochlear_physics_sim
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app/main.py
```

This opens the simulator in your browser. Use the sidebar to:

- Enable up to three tones (frequency + amplitude) to build the input sound
- Set the number of implant electrodes and the filterbank's frequency range
- Adjust the fluid depth constant `d`, which controls how focal vs.
  spread-out the current appears — a direct, tunable knob on channel
  interaction

The main panel shows the scalar potential heatmap, a combined
potential + vector field view, and a bar chart of how much charge
landed on each electrode.

## Running the tests

```bash
pytest tests/ -v
```

The suite includes the two required correctness proofs for the
biophysics engine:

- `test_zero_charge_zero_gradient` — silence produces an exactly zero
  potential and exactly zero gradient everywhere
- `test_gradient_direction` — a single point charge produces a field
  whose vectors point strictly away from that charge, verifying both the
  sign convention of `E = -∇V` and the correct axis handling of
  `np.gradient` on a `meshgrid`-based grid

along with additional tests for superposition linearity, the
depth-regularized potential value at an electrode (`V = q/d`), and input
validation across both `ci_engine.acoustics` and `ci_engine.biophysics`.

## Using the backend without the UI

Every `ci_engine` class works standalone:

```python
from ci_engine.acoustics import AcousticWave
from ci_engine.processor import NeuroprostheticFilterbank
from ci_engine.biophysics import CochlearFluidModel

wave = AcousticWave(sample_rate=16000, duration=0.5)
wave.add_sine_wave(frequency=440.0, amplitude=1.0)
signal = wave.get_signal()

bank = NeuroprostheticFilterbank(num_electrodes=12, low_freq=200, high_freq=7000)
charges = bank.extract_electrode_charges(signal, sample_rate=16000)

model = CochlearFluidModel(num_electrodes=12, fluid_depth=0.5, resolution=100)
V = model.calculate_scalar_potential(charges)
Ex, Ey = model.calculate_electric_field()
```

## Tech stack

| Purpose         | Library      |
|------------------|--------------|
| Numerical core   | `numpy`      |
| Signal processing| `scipy`      |
| Visualization    | `matplotlib` |
| UI               | `streamlit`  |
| Testing          | `pytest`     |
