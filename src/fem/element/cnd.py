"""
Finite element continuum (Cnd) definitions for plane stress/strain elements.

This module defines constitutive and geometric element classes for
2D finite element analysis (constant strain triangles and quads).
Each element combines geometry, shape functions, and
constitutive update behavior (via Material.eval). This file does
*not* change logic — it only adds docstrings for readability.

Classes
-------
CnD
    Base class for continuum elements (directional + shear components).
CPX3
    Base constant strain triangle with 3 nodes.
CPS3, CPE3
    Plane stress and plane strain 3‑node triangles.
CPX4
    Base constant strain quadrilateral with 4 nodes.
CPS4, CPE4
    Plane stress and plane strain 4‑node quadrilaterals.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..constants import Ux
from ..constants import Uy
from ..material import Material
from .geom import P3
from .geom import P4
from .isop import IsoparametricElement


class CnD:
    """
    Continuum element behavior mixin.

    Provides directional (`ndir`) and shear (`nshr`) dimension counts and
    a common state update method using the provided Material object.

    Attributes
    ----------
    ndir : int
        Number of normal strain directions (e.g., 2 for plane stress).
    nshr : int
        Number of shear components.
    """

    ndir: int
    nshr: int

    @property
    def ntens(self) -> int:
        """
        Total number of tensor components.

        Returns
        -------
        int
            Sum of directional and shear components.
        """
        return self.ndir + self.nshr

    def update_state(
        self,
        material: Material,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        p: NDArray,
        u: NDArray,
        e: NDArray,
        de: NDArray,
        hsv: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Update constitutive state variables using material model.

        This calls material.eval and stores strain and stress in
        history variables.

        Parameters
        ----------
        material
            Constitutive material object with `eval` method.
        step
            Load step index.
        increment
            Increment index within the step.
        time
            Simulation time history.
        dt
            Time increment.
        eleno
            Element index.
        p
            Nodal coordinate array.
        u
            Current nodal displacement
        e
            Current strain state.
        de
            Strain increment.
        hsv
            History variables array to be updated.

        Returns
        -------
        tuple of (D, s)
            Material stiffness matrix D and stress vector s.
        """
        D, s = material.eval(e, ndir=self.ndir, nshr=self.nshr)
        hsv[: self.ntens] = e
        hsv[self.ntens : 2 * self.ntens] = s
        return D, s


class CPX3(P3, CnD, IsoparametricElement):
    """
    Base constant strain triangle element (3 nodes).

    Combines geometric shape functions (P3) with continuum behavior (CnD)
    and isoparametric quadrature definitions.
    """

    gauss_pts = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=float) / 6.0
    gauss_wts = np.array([1.0, 1.0, 1.0], dtype=float) / 6.0
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """
        Degree‑of‑freedom layout for a 3 node element.

        Returns
        -------
        List of tuples mapping node DOFs (u,v, other reserved slots).
        """
        return [(Ux, Uy), (Ux, Uy), (Ux, Uy)]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """
        Constructs displacement‑to‑nodal matrix P.

        Parameters
        ----------
        xi
            Local parametric coordinates.

        Returns
        -------
        P matrix for shape function interpolation.
        """
        N = self.shape(xi)
        P = np.zeros((6, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P


class CPS3(CPX3):
    """Plane stress constant strain triangle element."""

    ndir = 2
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Compute strain–displacement matrix B at a given point.

        Parameters
        ----------
        p
            Nodal coordinates.
        xi
            Local parametric location.

        Returns
        -------
        B matrix relating nodal displacements to strains.
        """
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        """
        List of history variable names (strains and stresses).

        Returns
        -------
        List of string labels.
        """
        return ["strain_xx", "strain_yy", "strain_xy", "stress_xx", "stress_yy", "stress_xy"]


class CPE3(CPX3):
    """Plane strain constant strain triangle element."""

    ndir = 3
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return [
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
        ]


class CPX4(P4, CnD, IsoparametricElement):
    """
    Base constant strain quadrilateral element (4 nodes).

    Geometric shape (P4) with continuum material update behavior.
    """

    gauss_pts = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]) / np.sqrt(3.0)
    gauss_wts = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)], dtype=float)
    edge_gauss_wts = np.array([1.0, 1.0], dtype=float)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [(Ux, Uy), (Ux, Uy), (Ux, Uy), (Ux, Uy)]

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((8, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P


class CPS4(CPX4):
    """Plane stress constant strain quadrilateral element."""

    ndir = 2
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return ["strain_xx", "strain_yy", "strain_xy", "stress_xx", "stress_yy", "stress_xy"]


class CPE4(CPX4):
    """Plane strain constant strain quadrilateral element."""

    ndir = 3
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 8))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B

    def history_variables(self) -> list[str]:
        return [
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
        ]
