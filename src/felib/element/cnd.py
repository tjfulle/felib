"""
Finite element continuum (Cnd) definitions for plane stress/strain elements.

This module defines constitutive and geometric element classes for
2D finite element analysis (constant strain triangles and quads).
Each element combines geometry, shape functions, and
constitutive update behavior (via Material.eval). This file does
*not* change logic — it only adds docstrings for readability.

Classes
-------
ContinuumElement
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

from typing import TYPE_CHECKING
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..dof_manager import DOF
from ..material import Material
from . import gauss
from .isop import IsoparametricElement
from .reference import Quad4
from .reference import Quad8
from .reference import Tri3

if TYPE_CHECKING:
    from ..collections import DistributedLoad
    from ..collections import DistributedSurfaceLoad
    from ..collections import RobinLoad


class ContinuumElement:
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

    def history_variables(self) -> list[str]:
        """
        List of history variable names (strains and stresses).

        Returns
        -------
        List of string labels.
        """
        if self.ndir == 2 and self.nshr == 1:
            return ["exx", "eyy", "exy", "sxx", "syy", "sxy"]
        if self.ndir == 3 and self.nshr == 1:
            return ["exx", "eyy", "ezz", "exy", "sxx", "syy", "szz", "sxy"]
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
        temp = dtemp = 0.0  # place holder
        n = len(self.history_variables())
        D, s = material.eval(
            hsv[n:], e, de, time, dt, temp, dtemp, self.ndir, self.nshr, eleno, step, increment
        )
        hsv[: self.ntens] = e
        hsv[self.ntens : 2 * self.ntens] = s
        return D, s


class CPX3(Tri3, ContinuumElement, IsoparametricElement):
    """
    Base constant strain triangle element (3 nodes).

    Combines geometric shape functions (Tri3) with continuum behavior (ContinuumElement)
    and isoparametric quadrature definitions.
    """

    gauss_pts, gauss_wts = gauss.gauss_tri3()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(2)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """
        Degree‑of‑freedom layout for a 3 node element.

        Returns
        -------
        List of tuples mapping node DOFs (u,v, other reserved slots).
        """
        return [(DOF.ux, DOF.uy), (DOF.ux, DOF.uy), (DOF.ux, DOF.uy)]

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


class CPX4(Quad4, ContinuumElement, IsoparametricElement):
    """
    Base constant strain quadrilateral element (4 nodes).

    Geometric shape (Quad4) with continuum material update behavior.
    """

    gauss_pts, gauss_wts = gauss.gauss2x2()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(2)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
        ]

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


class CPE4R(CPS4):
    """Plane strain constant strain quadrilateral element, reduced integration"""

    gauss_pts, gauss_wts = gauss.gauss1x1()
    hg_alpha: float = 0.1

    def eval(
        self,
        material: Material,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        p: NDArray,
        u: NDArray,
        du: NDArray,
        pdata: NDArray,
        dloads: list["DistributedLoad"] | None = None,
        dsloads: list[tuple[int, "DistributedSurfaceLoad"]] | None = None,
        rloads: list["RobinLoad"] | None = None,
    ) -> tuple[NDArray, NDArray]:
        ke, re = super().eval(
            material,
            step,
            increment,
            time,
            dt,
            eleno,
            p,
            u,
            du,
            pdata,
            dloads=dloads,
            dsloads=dsloads,
            rloads=rloads,
        )
        khg, rhg = self.hourglass_terms(p, u)
        ke += khg
        re += rhg
        return ke, re

    def hourglass_terms(self, p, u):
        ndof = self.nnode * self.dof_per_node
        Khg = np.zeros((ndof, ndof))
        Rhg = np.zeros(ndof)
        xi = np.zeros(2)
        J = self.jacobian(p, xi)
        for h in self.hourglass_vectors(p):
            q = np.dot(h, u)
            Khg += np.outer(h, h)
            Rhg += q * h
        scale = self.hg_alpha * J
        Khg *= scale
        Rhg *= scale
        return Khg, Rhg

    def hourglass_vectors(self, p: NDArray) -> list[NDArray]:

        # shape gradients at center
        xi = np.zeros(2)
        dNdx = self.shape_gradient(p, xi)  # (2, 4)

        # canonical scalar patterns (node space)
        patterns = [
            np.array([ 1, -1,  1, -1], dtype=float),
            np.array([ 1,  1, -1, -1], dtype=float),
        ]

        H = []

        for g in patterns:
            g = g.copy()
            # ---- projection: remove strain-producing part ----
            for i in range(dNdx.shape[0]):  # x,y directions
                xi_i = np.dot(g, p[:, i])  # same as old code
                for a in range(len(g)):
                    g[a] -= xi_i * dNdx[i, a]
            # normalize
            nrm = np.linalg.norm(g)
            if nrm > 1e-12:
                g /= nrm
            # expand to DOFs (u,v)
            h = np.zeros(2 * len(g))
            h[0::2] = g
            h[1::2] = g
            H.append(h)
        return H


class CPX8(Quad8, ContinuumElement, IsoparametricElement):
    """
    Base 8 node quadrilateral element

    Geometric shape (Quad8) with continuum material update behavior.
    """

    gauss_pts, gauss_wts = gauss.gauss3x3()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(3)

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
            (DOF.ux, DOF.uy),
        ]

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((16, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P


class CPS8(CPX8):
    """Plane stress constant strain quadrilateral element."""

    ndir = 2
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 16))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B


class CPE8(CPX8):
    """Plane strain constant strain quadrilateral element."""

    ndir = 3
    nshr = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 16))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B


class CPS4I(CPS4):
    """
    Quad4 with incompatible modes (shear improved).

    - Full integration (2x2)
    - Uses 2 internal incompatible DOFs
    - Static condensation applied as a correction to Kdd
    """

    # ------------------------------------------------------------
    # Main eval with static condensation correction
    # ------------------------------------------------------------
    def eval(
        self,
        material: Material,
        step: int,
        increment: int,
        time,
        dt: float,
        eleno: int,
        p: NDArray,
        u: NDArray,
        du: NDArray,
        pdata: NDArray,
        dloads=None,
        dsloads=None,
        rloads=None,
    ):
        # --- get standard stiffness + all load contributions ---
        hsv = pdata.copy()
        ke, re = super().eval(
            material, step, increment, time, dt,
            eleno, p, u, du, pdata,
            dloads=dloads, dsloads=dsloads, rloads=rloads
        )

        ndof = self.nnode * self.dof_per_node
        na = 2  # incompatible DOFs

        Kda = np.zeros((ndof, na), dtype=float)
        Kaa = np.zeros((na, na), dtype=float)

        # --------------------------------------------------------
        # Second loop: build coupling terms
        # --------------------------------------------------------
        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            G = self.gmatrix(p, xi)

            # consistent strain state
            e = np.dot(B, u)
            de = np.dot(B, du)

            D, _ = self.update_state(
                material,
                step,
                increment,
                time,
                dt,
                eleno,
                p,
                u,
                e,
                de,
                hsv[ipt],
            )

            # assemble coupling blocks
            Bt = B.T
            Gt = G.T

            Kda += w * J * np.dot(np.dot(Bt, D), G)
            Kaa += w * J * np.dot(np.dot(Gt, D), G)

        # --------------------------------------------------------
        # Static condensation: K = Kdd - Kda Kaa^{-1} Kda^T
        # --------------------------------------------------------
        Kcorr = np.dot(np.dot(Kda, np.linalg.inv(Kaa)), Kda.T)
        ke -= Kcorr

        return ke, re

    def gmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Construct incompatible strain-displacement matrix G.

        Returns shape: (3, 2)
        """
        r, s = xi

        G = np.zeros((3, 2), dtype=float)

        # Normal components
        G[0, 0] = r
        G[1, 1] = s

        # Shear component (key for shear locking)
        G[2, 0] = s
        G[2, 1] = r

        return G
