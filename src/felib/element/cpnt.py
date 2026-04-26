"""
Coupled thermo-mechanical continuum elements (CPnT).

Each node carries three DOFs: [Ux, Uy, T].

The element is solved sequentially (staggered):
  - Stage 1 (thermal):    assembles K_TT and thermal residual -> solves for T
  - Stage 2 (mechanical): assembles K_uu including thermal strain as a known
                          load -> solves for [Ux, Uy]

Thermal strain at a Gauss point:
    epsilon_thermal = alpha * (T_gp - T_ref) * [1, 1, 0]   (plane stress)
    epsilon_thermal = alpha * (T_gp - T_ref) * [1, 1, 0, 0] (plane strain)

where T_gp is interpolated from nodal temperatures using standard shape
functions, alpha is the coefficient of thermal expansion, and T_ref is the
stress-free reference temperature.

Classes
-------
CPnT4S  - 4-node quad, plane stress
CPnT4E  - 4-node quad, plane strain
"""

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..constants import T
from ..constants import Ux
from ..constants import Uy
from ..material import Material
from . import gauss
from .cnd import ContinuumElement
from .isop import IsoparametricElement
from .reference import Quad4


# ---------------------------------------------------------------------------
# Helper: split combined [Ux, Uy, T, ...] DOF vector
# ---------------------------------------------------------------------------

def _split_dofs(u: NDArray, nnode: int) -> tuple[NDArray, NDArray]:
    """
    Split the combined [Ux, Uy, T, Ux, Uy, T, ...] DOF vector into
    mechanical and thermal parts.

    Parameters
    ----------
    u     : element DOF vector of length 3*nnode
    nnode : number of nodes

    Returns
    -------
    u_mech  : NDArray shape (2*nnode,) - displacement DOFs [Ux, Uy, ...]
    u_therm : NDArray shape (nnode,)   - temperature DOFs  [T, T, ...]
    """
    u_mech = np.empty(2 * nnode, dtype=float)
    u_therm = np.empty(nnode, dtype=float)
    for n in range(nnode):
        u_mech[2 * n]     = u[3 * n]       # Ux
        u_mech[2 * n + 1] = u[3 * n + 1]   # Uy
        u_therm[n]        = u[3 * n + 2]   # T
    return u_mech, u_therm


# ---------------------------------------------------------------------------
# Base CPnT element
# ---------------------------------------------------------------------------

class CPnTElement(Quad4, ContinuumElement, IsoparametricElement):
    """
    Base class for 4-node coupled thermo-mechanical quad elements.

    DOF layout per node: [Ux, Uy, T]

    The stiffness matrix has the block structure:

        K = [ K_uu   0   ]
            [  0    K_TT ]

    K_uu  - mechanical stiffness (thermal strain enters residual only)
    K_TT  - thermal conductivity stiffness

    Subclasses must set:
        ndir, nshr  - mechanical tensor dimensions
    """

    ndir: int
    nshr: int

    # Thermal material properties
    alpha: float = 1.2e-5   # coefficient of thermal expansion [1/K]
    T_ref: float = 0.0      # stress-free reference temperature [K or C]

    gauss_pts, gauss_wts = gauss.gauss2x2()
    edge_gauss_pts, edge_gauss_wts = gauss.gauss1d(2)

    # ------------------------------------------------------------------
    # DOF layout
    # ------------------------------------------------------------------

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """Each node carries Ux, Uy, and T DOFs."""
        return [(Ux, Uy, T)] * self.nnode

    # ------------------------------------------------------------------
    # History variables
    # ------------------------------------------------------------------

    def history_variables(self) -> list[str]:
        """Strains, stresses (from ContinuumElement) plus thermal strain."""
        return ContinuumElement.history_variables(self) + ["e_thermal"]

    # ------------------------------------------------------------------
    # P-matrix
    # ------------------------------------------------------------------

    def pmatrix(self, xi: NDArray) -> NDArray:
        """
        Interpolation matrix for body forces.

        Maps [fx, fy] into the full 3*nnode DOF space.
        Only Ux/Uy rows are populated; T rows are zero.
        """
        N = self.shape(xi)
        nnode = self.nnode
        P = np.zeros((3 * nnode, 2))
        for n in range(nnode):
            P[3 * n,     0] = N[n]   # Ux
            P[3 * n + 1, 1] = N[n]   # Uy
        return P

    # ------------------------------------------------------------------
    # B-matrices
    # ------------------------------------------------------------------

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Full strain-displacement B matrix in 3-DOF-per-node layout.

        Maps the full [Ux, Uy, T, ...] DOF vector to mechanical strains.
        T columns are zero.
        """
        dNdx = self.shape_gradient(p, xi)
        nnode = self.nnode
        B = np.zeros((self.ntens, 3 * nnode))
        for n in range(nnode):
            col_ux = 3 * n
            col_uy = 3 * n + 1
            B[0, col_ux]  = dNdx[0, n]   # exx = dUx/dx
            B[1, col_uy]  = dNdx[1, n]   # eyy = dUy/dy
            B[-1, col_ux] = dNdx[1, n]   # gxy = dUx/dy
            B[-1, col_uy] = dNdx[0, n]   #     + dUy/dx
        return B

    def _bmatrix_thermal(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Thermal gradient B matrix (2 x nnode).

        Maps nodal temperatures to temperature gradient [dT/dx, dT/dy].
        """
        dNdx = self.shape_gradient(p, xi)
        nnode = self.nnode
        B_th = np.zeros((2, nnode))
        B_th[0] = dNdx[0]
        B_th[1] = dNdx[1]
        return B_th

    # ------------------------------------------------------------------
    # Thermal strain
    # ------------------------------------------------------------------

    def _thermal_strain(self, T_gp: float) -> NDArray:
        """
        Isotropic thermal strain vector at a Gauss point.

        epsilon_thermal = alpha * (T_gp - T_ref) * [1, 1, 0]    (plane stress)
        epsilon_thermal = alpha * (T_gp - T_ref) * [1, 1, 0, 0] (plane strain)
        """
        dT = T_gp - self.T_ref
        e_th = np.zeros(self.ntens)
        e_th[0] = self.alpha * dT   # exx
        e_th[1] = self.alpha * dT   # eyy
        return e_th

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

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
        Update mechanical material state subtracting thermal strain.

        The mechanical strain passed to the material is:
            e_mech = e_total - e_thermal

        T_gp is read from the last slot of hsv, which is written
        by eval() before calling update_state().
        """
        T_gp = float(hsv[-1])
        e_thermal = self._thermal_strain(T_gp)
        e_mech = e - e_thermal
        de_mech = de

        temp = dtemp = 0.0
        nhsv_mech = 2 * self.ntens   # strains + stresses
        D, s = material.eval(
            hsv[nhsv_mech:-1],
            e_mech,
            de_mech,
            time, dt, temp, dtemp,
            self.ndir, self.nshr,
            eleno, step, increment,
        )
        hsv[: self.ntens] = e_mech
        hsv[self.ntens : 2 * self.ntens] = s
        return D, s

    # ------------------------------------------------------------------
    # Main eval
    # ------------------------------------------------------------------

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
        dloads=None,
        dsloads=None,
        rloads=None,
    ) -> tuple[NDArray, NDArray]:
        """
        Assemble element stiffness and residual for [Ux, Uy, T] DOFs.

        Parameters
        ----------
        u     : full DOF vector [Ux0,Uy0,T0, Ux1,Uy1,T1, ...]
        du    : DOF increment vector (same layout)
        pdata : integration point state array (npts x n_hsv)

        Returns
        -------
        ke : (3*nnode, 3*nnode) element stiffness
        re : (3*nnode,)         element residual
        """
        dloads = dloads or []
        dsloads = dsloads or []
        rloads = rloads or []

        nnode = self.nnode
        ndof = 3 * nnode
        ke = np.zeros((ndof, ndof))
        re = np.zeros(ndof)

        # Split DOFs
        u_mech, u_therm = _split_dofs(u, nnode)

        # Thermal conductivity matrix (isotropic)
        try:
            k_cond = float(material.conductivity)
        except AttributeError:
            k_cond = 1.0
        K_mat = k_cond * np.eye(2)

        # T DOF indices in the element DOF vector
        T_dofs = [3 * n + 2 for n in range(nnode)]

        # ----------------------------------------------------------------
        # Volume integration
        # ----------------------------------------------------------------
        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            N = self.shape(xi)
            x = self.interpolate(p, xi)

            # Interpolate temperature to Gauss point
            T_gp = float(np.dot(N, u_therm))

            # Store T_gp in last hsv slot for update_state
            pdata[ipt, -1] = T_gp

            # ---- Thermal block ----------------------------------------
            B_th = self._bmatrix_thermal(p, xi)        # (2, nnode)
            grad_T = np.dot(B_th, u_therm)              # (2,)
            q_flux = np.dot(K_mat, grad_T)              # (2,)

            K_TT = w * J * np.dot(B_th.T, np.dot(K_mat, B_th))
            ke[np.ix_(T_dofs, T_dofs)] += K_TT
            re[T_dofs] += w * J * np.dot(B_th.T, q_flux)

            # ---- Mechanical block -------------------------------------
            B = self.bmatrix(p, xi)                    # (ntens, 3*nnode)
            P = self.pmatrix(xi)                       # (3*nnode, 2)

            e_total = np.dot(B, u)
            de_total = np.dot(B, du) / dt

            D, s = self.update_state(
                material, step, increment, time, dt,
                eleno, p, u, e_total, de_total, pdata[ipt],
            )

            # Store scalar thermal strain magnitude in history
            e_th = self._thermal_strain(T_gp)
            pdata[ipt, -1] = float(np.linalg.norm(e_th))

            ke += w * J * np.dot(np.dot(B.T, D), B)
            re += w * J * np.dot(B.T, s)

            # Body forces
            for dload in dloads:
                value = dload(step, increment, time, dt, eleno, ipt, x.tolist())
                re -= w * J * np.dot(P, value)

        # ----------------------------------------------------------------
        # Surface loads (mechanical only)
        # ----------------------------------------------------------------
        for edge_no, dsload in dsloads:
            nodes = self.edges[edge_no]
            nft = [3 * n + d for n in nodes for d in range(2)]
            for ipt, (w, xi) in enumerate(self.edge_integration_points()):
                x = self.interpolate_edge(edge_no, p, xi)
                n_vec = self.edge_normal(edge_no, p, xi)
                traction = dsload(
                    step, increment, time, dt, eleno, edge_no, ipt, x.tolist(), n_vec
                )
                st = self.ref_edge_coords(edge_no, xi)
                P_edge = self.pmatrix(st)[nft]
                Jedge = self.edge_jacobian(edge_no, p, xi)
                re[nft] -= w * Jedge * np.dot(P_edge, traction)

        # ----------------------------------------------------------------
        # Robin boundary conditions (thermal convection)
        # ----------------------------------------------------------------
        for rload in rloads:
            nodes = self.edges[rload.edge]
            T_dofs_edge = [3 * n + 2 for n in nodes]
            H = np.asarray(rload.H)
            u0_vec = np.asarray(rload.u0)
            u_T_edge = u[T_dofs_edge]
            for ipt, (w, xi) in enumerate(self.edge_integration_points()):
                st = self.ref_edge_coords(rload.edge, xi)
                N_edge = self.shape(st)[list(nodes)]
                P_T = N_edge.reshape(-1, 1)
                Jedge = self.edge_jacobian(rload.edge, p, xi)
                kr = w * Jedge * np.dot(np.dot(P_T, H), P_T.T)
                fr = w * Jedge * np.dot(P_T, np.dot(H, u0_vec))
                ke[np.ix_(T_dofs_edge, T_dofs_edge)] += kr
                re[T_dofs_edge] += np.dot(kr, u_T_edge) - fr.ravel()

        return ke, re


# ---------------------------------------------------------------------------
# Concrete elements
# ---------------------------------------------------------------------------

class CPnT4S(CPnTElement):
    """
    4-node plane stress thermo-mechanical element.

    DOFs per node : [Ux, Uy, T]
    ndir = 2, nshr = 1  ->  ntens = 3  (sxx, syy, sxy)
    """

    ndir = 2
    nshr = 1


class CPnT4E(CPnTElement):
    """
    4-node plane strain thermo-mechanical element.

    DOFs per node : [Ux, Uy, T]
    ndir = 3, nshr = 1  ->  ntens = 4  (sxx, syy, szz, sxy)
    """

    ndir = 3
    nshr = 1
