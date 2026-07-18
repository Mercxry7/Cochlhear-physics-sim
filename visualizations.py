"""
app.visualizations
=====================
Isolates all Matplotlib rendering logic from the Streamlit UI layer.
`FieldPlotter` turns raw field arrays (scalar potential V, vector field
Ex/Ey) produced by `ci_engine.biophysics.CochlearFluidModel` into
publication-style contour and quiver figures.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure


class FieldPlotter:
    """
    Renders the cochlear fluid's electrical fields as Matplotlib figures.

    Two plot types are supported:
      - A filled contour ("heatmap") of the scalar potential V(x,y).
      - A quiver overlay of the vector electric field (Ex, Ey), which can
        be layered on top of the contour plot for a combined view.

    Parameters
    ----------
    figsize : tuple of float, optional
        (width, height) in inches for generated figures. Defaults to
        (8, 6).
    cmap : str, optional
        Matplotlib colormap name used for the potential heatmap. Defaults
        to "inferno", which reads well for a scalar intensity field.
    """

    def __init__(self, figsize: tuple[float, float] = (8.0, 6.0), cmap: str = "inferno") -> None:
        self.figsize = figsize
        self.cmap = cmap

    def plot_scalar_contour(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        V: np.ndarray,
        electrode_positions: np.ndarray | None = None,
        levels: int = 40,
        title: str = "Scalar Potential Field V(x, y)",
    ) -> Figure:
        """
        Generate a filled contour heatmap of the scalar potential field.

        Parameters
        ----------
        X, Y : np.ndarray
            2D spatial coordinate matrices, as returned by
            `CochlearFluidModel.mesh`.
        V : np.ndarray
            2D scalar potential field, as returned by
            `CochlearFluidModel.calculate_scalar_potential`. Must match
            the shape of X and Y.
        electrode_positions : np.ndarray, optional
            Array of shape (N, 2) of (x, y) electrode coordinates. If
            given, electrodes are marked on the plot.
        levels : int, optional
            Number of contour levels/color bands. Defaults to 40.
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered figure, ready for `st.pyplot(fig)` or saving.
        """
        self._validate_shapes(X, Y, V=V)

        fig = Figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        contour = ax.contourf(X, Y, V, levels=levels, cmap=self.cmap)
        fig.colorbar(contour, ax=ax, label="Potential V (a.u.)")

        if electrode_positions is not None:
            electrode_positions = np.asarray(electrode_positions)
            ax.scatter(
                electrode_positions[:, 0],
                electrode_positions[:, 1],
                c="cyan",
                edgecolors="black",
                s=60,
                marker="o",
                zorder=5,
                label="Electrodes",
            )
            ax.legend(loc="upper right")

        ax.set_xlabel("x (cochlear length, a.u.)")
        ax.set_ylabel("y (radial depth, a.u.)")
        ax.set_title(title)
        ax.set_aspect("equal")

        return fig

    def plot_vector_quiver(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Ex: np.ndarray,
        Ey: np.ndarray,
        base_figure: Figure | None = None,
        stride: int = 4,
        title: str = "Electric Field E = -\u2207V",
    ) -> Figure:
        """
        Overlay (or plot standalone) a quiver plot of the vector electric
        field showing directional current flow.

        Parameters
        ----------
        X, Y : np.ndarray
            2D spatial coordinate matrices matching Ex, Ey.
        Ex, Ey : np.ndarray
            2D vector field components, as returned by
            `CochlearFluidModel.calculate_electric_field`.
        base_figure : matplotlib.figure.Figure, optional
            If provided (e.g. the output of `plot_scalar_contour`), the
            quiver arrows are drawn on top of that figure's existing axes
            instead of creating a new figure.
        stride : int, optional
            Subsample the grid every `stride` points per axis before
            plotting arrows, since plotting one arrow per grid cell is
            usually too dense to read. Defaults to 4.
        title : str, optional
            Plot title. Ignored if `base_figure` is provided (the base
            figure's title/axes are preserved).

        Returns
        -------
        matplotlib.figure.Figure
            The figure with quiver arrows drawn (either `base_figure`,
            mutated in place, or a newly created standalone figure).
        """
        self._validate_shapes(X, Y, Ex=Ex, Ey=Ey)

        if base_figure is not None:
            fig = base_figure
            ax = fig.axes[0]
        else:
            fig = Figure(figsize=self.figsize)
            ax = fig.add_subplot(111)
            ax.set_xlabel("x (cochlear length, a.u.)")
            ax.set_ylabel("y (radial depth, a.u.)")
            ax.set_title(title)
            ax.set_aspect("equal")

        s = max(1, int(stride))
        ax.quiver(
            X[::s, ::s],
            Y[::s, ::s],
            Ex[::s, ::s],
            Ey[::s, ::s],
            color="white" if base_figure is not None else "black",
            alpha=0.85,
            scale_units="xy",
            angles="xy",
        )

        return fig

    @staticmethod
    def _validate_shapes(X: np.ndarray, Y: np.ndarray, **fields: np.ndarray) -> None:
        """Ensure X, Y and every provided field array share identical shapes."""
        if X.shape != Y.shape:
            raise ValueError(f"X shape {X.shape} does not match Y shape {Y.shape}")
        for name, arr in fields.items():
            if arr.shape != X.shape:
                raise ValueError(
                    f"{name} shape {arr.shape} does not match grid shape {X.shape}"
                )

    def __repr__(self) -> str:
        return f"FieldPlotter(figsize={self.figsize}, cmap={self.cmap!r})"
