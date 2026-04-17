"""
Finite element conductive (heat transfer) elements (DCPn).

Analogous to cnd.py, but for thermal conduction.
Temperature DOF at index 3 of 10 in the node freedom table.
History variables track element-level quantities (e.g., flux),
not nodal temperature.
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..dof_manager import DOF
from ..material import Material
from . import gauss
from .isop import IsoparametricElement
from .reference import Quad4
from .reference import Tri3


class DiffusiveContinueElement:
    """Conductive element mixin."""

    ndir: int
    nshr: int = 0

    def history_variables(self) -> list[str]:
        if self.ndir == 2:
            return ["DTx", "DTy", "Qx", "Qy"]
        raise NotImplementedError

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
        Compute element conductivity and internal flux variables.

        Parameters
        ----------
        material : Material
            Material with thermal conductivity evaluation.
        step, increment : int
            Step and increment counters.
        time : sequence of float
            Current time values.
        dt : float
            Time increment.
        eleno : int
            Element number.
        p : NDArray
            Nodal coordinates.
        u
            Current nodal temperature
        e : NDArray
            Temperature gradient
        de : NDArray
            Temperature increment gradient
        hsv : NDArray
            Element history variable array to store flux or energy.

        Returns
        -------
        K : NDArray
            Element conductivity matrix.
        q : NDArray
            Element internal flux vector.
        """
        ndim = p.shape[1]
        n = len(self.history_variables())
        temp = dtemp = 0.0  # place holder
        D, q = material.eval(hsv[n:], e, de, time, dt, temp, dtemp, ndim, 0, eleno, step, increment)
        hsv[:2] = e
        hsv[2:] = q  # store element flux or internal quantities
        return D, q


class DCP3(Tri3, DiffusiveContinueElement, IsoparametricElement):
    """3-node constant conductivity triangle element."""

    ndir = 2
    gauss_pts, gauss_wts = gauss.gauss_tri3()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(2)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """Node freedom table Temperature DOF at 4th entry."""
        return [(DOF.T,), (DOF.T,), (DOF.T,)]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """Interpolation matrix for temperature DOF."""
        N = self.shape(xi)
        P = np.zeros((3, 1))
        P[:, 0] = N
        return P

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((2, 3))
        B[0] = dNdx[0]
        B[1] = dNdx[1]
        return B


class DCP4(Quad4, DiffusiveContinueElement, IsoparametricElement):
    """4-node constant conductivity quadrilateral element."""

    ndir = 2
    gauss_pts, gauss_wts = gauss.gauss2x2()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(2)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """Node freedom table.  Temperature DOF at 4th entry."""
        return [(DOF.T,), (DOF.T,), (DOF.T,), (DOF.T,)]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """Interpolation matrix for temperature DOF."""
        N = self.shape(xi)
        P = np.zeros((4, 1))
        P[:, 0] = N
        return P

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((2, 4))
        B[0] = dNdx[0]
        B[1] = dNdx[1]
        return B
