"""
tests.test_biophysics
========================
Pytest suite for ci_engine.biophysics.CochlearFluidModel.

Verifies the two core physical correctness properties of the
multivariable calculus engine:
  1. Silence (zero charge everywhere) produces zero potential and zero
     gradient everywhere on the grid.
  2. A single point charge produces a field whose vectors point strictly
     away from that charge's location (repulsive, as expected for the
     inverse-square law with a positive charge).

Also covers input-validation edge cases so regressions in error handling
are caught, not just regressions in the physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from ci_engine.biophysics import CochlearFluidModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model() -> CochlearFluidModel:
    """A small, fast model for general-purpose tests."""
    return CochlearFluidModel(num_electrodes=5, fluid_depth=0.5, resolution=50)


@pytest.fixture
def single_electrode_model() -> CochlearFluidModel:
    """
    A model with exactly one electrode, used for the gradient-direction
    test where we need an unambiguous single source point.
    """
    return CochlearFluidModel(num_electrodes=1, fluid_depth=0.5, resolution=101)


# ---------------------------------------------------------------------------
# Required tests (per project spec)
# ---------------------------------------------------------------------------


def test_zero_charge_zero_gradient(small_model: CochlearFluidModel) -> None:
    """If the input wave is silent (all charges zero), grad(V) must be
    precisely zero everywhere on the grid."""
    charges = np.zeros(small_model.num_electrodes)

    V = small_model.calculate_scalar_potential(charges)
    assert np.allclose(V, 0.0), "Potential should be exactly zero with no active charges"

    Ex, Ey = small_model.calculate_electric_field()
    assert np.allclose(Ex, 0.0), "E_x should be exactly zero with no active charges"
    assert np.allclose(Ey, 0.0), "E_y should be exactly zero with no active charges"


def test_gradient_direction(single_electrode_model: CochlearFluidModel) -> None:
    """Placing a single positive charge must create a field whose vectors
    point strictly away from that charge's coordinate (repulsive field),
    at every sampled point on the grid (excluding the singular point
    itself, which fluid_depth regularizes but where the direction is
    numerically undefined for a point exactly on the electrode)."""
    charges = np.array([1.0])
    single_electrode_model.calculate_scalar_potential(charges)
    Ex, Ey = single_electrode_model.calculate_electric_field()

    X, Y = single_electrode_model.mesh
    elec_x, elec_y = single_electrode_model.electrode_positions[0]

    radial_x = X - elec_x
    radial_y = Y - elec_y

    # Dot product of the field with the radial (outward) direction at
    # every grid point. Strictly positive everywhere except exactly at
    # the source (where the radial vector itself is zero-length and the
    # dot product is trivially zero, not a violation of "points away").
    dot = radial_x * Ex + radial_y * Ey
    on_source = (np.abs(radial_x) < 1e-9) & (np.abs(radial_y) < 1e-9)

    assert np.all(dot[~on_source] > 0), (
        "Electric field must point strictly away from a single positive "
        "charge at every grid point other than the source itself"
    )


# ---------------------------------------------------------------------------
# Additional correctness tests
# ---------------------------------------------------------------------------


def test_potential_decays_with_distance(single_electrode_model: CochlearFluidModel) -> None:
    """Potential should be highest near the electrode and monotonically
    smaller far away from it."""
    charges = np.array([1.0])
    V = single_electrode_model.calculate_scalar_potential(charges)

    X, Y = single_electrode_model.mesh
    elec_x, elec_y = single_electrode_model.electrode_positions[0]
    ix = np.argmin(np.abs(X[0, :] - elec_x))
    iy = np.argmin(np.abs(Y[:, 0] - elec_y))

    v_near = V[iy, ix]
    v_far_corner = V[0, 0]

    assert v_near > v_far_corner


def test_potential_at_electrode_equals_q_over_d() -> None:
    """With the fluid_depth regularization, V directly at an electrode's
    (x, y) should equal q / d exactly (no singularity, correct value)."""
    d = 0.5
    q = 2.0
    model = CochlearFluidModel(num_electrodes=1, fluid_depth=d, resolution=101)
    V = model.calculate_scalar_potential(np.array([q]))

    X, Y = model.mesh
    elec_x, elec_y = model.electrode_positions[0]
    ix = np.argmin(np.abs(X[0, :] - elec_x))
    iy = np.argmin(np.abs(Y[:, 0] - elec_y))

    assert np.isclose(V[iy, ix], q / d)


def test_superposition_is_additive() -> None:
    """V from two simultaneously active charges should equal the sum of
    the V fields each charge would produce alone (linearity of the
    inverse-square superposition)."""
    model_a = CochlearFluidModel(num_electrodes=3, fluid_depth=0.5, resolution=60)
    model_b = CochlearFluidModel(num_electrodes=3, fluid_depth=0.5, resolution=60)
    model_both = CochlearFluidModel(num_electrodes=3, fluid_depth=0.5, resolution=60)

    V_a = model_a.calculate_scalar_potential(np.array([1.0, 0.0, 0.0]))
    V_b = model_b.calculate_scalar_potential(np.array([0.0, 0.7, 0.0]))
    V_both = model_both.calculate_scalar_potential(np.array([1.0, 0.7, 0.0]))

    assert np.allclose(V_a + V_b, V_both)


def test_potential_field_shape_matches_resolution() -> None:
    """Output field shape should always be (resolution, resolution)."""
    model = CochlearFluidModel(num_electrodes=4, fluid_depth=0.3, resolution=73)
    V = model.calculate_scalar_potential(np.zeros(4))
    Ex, Ey = model.calculate_electric_field()

    assert V.shape == (73, 73)
    assert Ex.shape == (73, 73)
    assert Ey.shape == (73, 73)


def test_calculate_electric_field_before_potential_raises() -> None:
    """Calling calculate_electric_field before calculate_scalar_potential
    must raise, not silently return garbage or crash with an unrelated
    AttributeError."""
    model = CochlearFluidModel(num_electrodes=3, fluid_depth=0.5, resolution=40)
    with pytest.raises(RuntimeError):
        model.calculate_electric_field()


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_electrodes": 0, "fluid_depth": 0.5, "resolution": 50},
        {"num_electrodes": -1, "fluid_depth": 0.5, "resolution": 50},
        {"num_electrodes": 5, "fluid_depth": 0.0, "resolution": 50},
        {"num_electrodes": 5, "fluid_depth": -0.5, "resolution": 50},
        {"num_electrodes": 5, "fluid_depth": 0.5, "resolution": 1},
        {"num_electrodes": 5, "fluid_depth": 0.5, "resolution": 0},
    ],
)
def test_invalid_constructor_args_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CochlearFluidModel(**kwargs)


def test_charges_wrong_shape_raises(small_model: CochlearFluidModel) -> None:
    with pytest.raises(ValueError):
        small_model.calculate_scalar_potential(np.zeros(small_model.num_electrodes + 1))
