"""
ci_engine.biophysics
=======================
Multivariable calculus core: models the cochlear perilymph as a
continuous 2D scalar field and derives the vector electric field acting
on the auditory nerve.

Mathematical model
-------------------
Scalar potential field (inverse-square law, summed over N electrodes):

    V(x,y) = sum_{i=1}^{N} q_i / sqrt((x - x_i)^2 + (y - y_i)^2 + d^2)

Vector electric field (negative gradient of the scalar potential):

    E = -grad(V) = (-dV/dx, -dV/dy)

`d` is a tissue depth constant that keeps the denominator away from zero
directly beneath an electrode, avoiding a singularity at the electrode
surface.
"""

from __future__ import annotations

import numpy as np


class CochlearFluidModel:
    """
    Models the 2D cochlear fluid plane and computes the scalar potential
    field and its gradient (the electric field) given a set of charged
    electrodes.

    Electrodes are placed along a single line (the y=0 axis) spanning the
    simulated cochlear length, evenly spaced by electrode index — this
    mirrors a linear CI electrode array threaded along the cochlear duct.

    Parameters
    ----------
    num_electrodes : int
        Number of electrodes N placed along the array. Must be >= 1.
    fluid_depth : float
        Tissue/fluid depth constant `d` (> 0) that regularizes the
        inverse-square law and prevents division by zero directly at an
        electrode's location.
    resolution : int
        Number of grid points per axis for the spatial mesh. Must be
        >= 2. Total grid size is resolution x resolution.

    Examples
    --------
    >>> model = CochlearFluidModel(num_electrodes=12, fluid_depth=0.5, resolution=100)
    >>> charges = np.zeros(12); charges[6] = 1.0
    >>> V = model.calculate_scalar_potential(charges)
    >>> Ex, Ey = model.calculate_electric_field()
    >>> V.shape, Ex.shape, Ey.shape
    ((100, 100), (100, 100), (100, 100))
    """

    def __init__(self, num_electrodes: int, fluid_depth: float, resolution: int) -> None:
        if num_electrodes < 1:
            raise ValueError(f"num_electrodes must be >= 1, got {num_electrodes}")
        if fluid_depth <= 0:
            raise ValueError(f"fluid_depth must be positive, got {fluid_depth}")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")

        self.num_electrodes = int(num_electrodes)
        self.fluid_depth = float(fluid_depth)
        self.resolution = int(resolution)

        # Electrode array spans a normalized cochlear length of [0, N-1]
        # along y=0, one unit apart, matching electrode index spacing.
        self.electrode_positions = np.array(
            [(float(i), 0.0) for i in range(self.num_electrodes)]
        )

        self._X: np.ndarray
        self._Y: np.ndarray
        self._generate_mesh()

        # Populated by calculate_scalar_potential(); required before
        # calculate_electric_field() can be called.
        self._V: np.ndarray | None = None

    def _generate_mesh(self) -> None:
        """
        Build the 2D spatial grid (X, Y) covering the electrode array with
        a margin, using np.meshgrid. Margin ensures the field is visible
        beyond the outermost electrodes, not just between them.
        """
        margin = max(2.0, self.fluid_depth * 4.0)
        x_min = -margin
        x_max = (self.num_electrodes - 1) + margin
        y_min = -margin
        y_max = margin

        x = np.linspace(x_min, x_max, self.resolution)
        y = np.linspace(y_min, y_max, self.resolution)
        self._X, self._Y = np.meshgrid(x, y)

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """The (X, Y) spatial coordinate matrices used for field evaluation."""
        return self._X, self._Y

    @property
    def potential(self) -> np.ndarray | None:
        """
        The most recently computed scalar potential field V(x,y), or None
        if `calculate_scalar_potential` has not yet been called.
        """
        return self._V

    def calculate_scalar_potential(self, charges: np.ndarray) -> np.ndarray:
        """
        Compute the scalar potential field V(x,y) by summing the
        inverse-square contribution of every active electrode charge.

        Parameters
        ----------
        charges : np.ndarray
            1D array of length num_electrodes giving charge q_i for each
            electrode, e.g. as produced by
            `NeuroprostheticFilterbank.extract_electrode_charges`.

        Returns
        -------
        np.ndarray
            2D array of shape (resolution, resolution) giving V at every
            grid point. Also cached internally for use by
            `calculate_electric_field`.
        """
        charges = np.asarray(charges, dtype=np.float64)
        if charges.shape != (self.num_electrodes,):
            raise ValueError(
                f"charges must have shape ({self.num_electrodes},), "
                f"got {charges.shape}"
            )

        V = np.zeros_like(self._X, dtype=np.float64)
        d_sq = self.fluid_depth ** 2

        for (x_i, y_i), q_i in zip(self.electrode_positions, charges):
            if q_i == 0.0:
                continue  # Inactive electrode contributes nothing; skip for speed.
            r_sq = (self._X - x_i) ** 2 + (self._Y - y_i) ** 2 + d_sq
            V += q_i / np.sqrt(r_sq)

        self._V = V
        return V

    def calculate_electric_field(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the vector electric field E = -grad(V) from the most
        recently calculated scalar potential, via finite differences
        (`np.gradient`).

        Returns
        -------
        tuple of np.ndarray
            (E_x, E_y), each of shape (resolution, resolution), giving the
            x- and y-components of the electric field at every grid
            point.

        Raises
        ------
        RuntimeError
            If called before `calculate_scalar_potential`.
        """
        if self._V is None:
            raise RuntimeError(
                "calculate_electric_field() requires a potential field; "
                "call calculate_scalar_potential(charges) first."
            )

        # np.gradient returns derivatives in the order of the array's axes.
        # For a (row=y, col=x) grid from meshgrid, axis 0 is d/dy, axis 1 is d/dx.
        y_axis = self._Y[:, 0]
        x_axis = self._X[0, :]
        dV_dy, dV_dx = np.gradient(self._V, y_axis, x_axis)

        Ex = -dV_dx
        Ey = -dV_dy
        return Ex, Ey

    def __repr__(self) -> str:
        return (
            f"CochlearFluidModel(num_electrodes={self.num_electrodes}, "
            f"fluid_depth={self.fluid_depth}, resolution={self.resolution})"
        )
