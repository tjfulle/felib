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
    Plane stress and plane strain 3-node triangles.
CPX4
    Base constant strain quadrilateral with 4 nodes.
CPS4, CPE4
    Plane stress and plane strain 4-node quadrilaterals.
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
        Degree-of-freedom layout for a 3 node element.

        Returns
        -------
        List of tuples mapping node DOFs (u,v, other reserved slots).
        """
        return [(DOF.ux, DOF.uy), (DOF.ux, DOF.uy), (DOF.ux, DOF.uy)]

    def pmatrix(self, xi: NDArray) -> NDArray:
        """
        Constructs displacement-to-nodal matrix P.

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
        Compute strain-displacement matrix B at a given point.

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
            np.array([1, -1, 1, -1], dtype=float),
            np.array([1, 1, -1, -1], dtype=float),
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

        ndof = self.nnode * self.dof_per_node
        na = 4  # incompatible DOFs

        Kda = np.zeros((ndof, na), dtype=float)
        Kaa = np.zeros((na, na), dtype=float)
        ra = np.zeros(na, dtype=float)

        # --------------------------------------------------------
        # Second loop: build coupling terms
        # --------------------------------------------------------
        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            G = self.gmatrix(p, xi, J)

            # consistent strain state
            e = np.dot(B, u)
            de = np.dot(B, du)

            D, s = self.update_state(
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
            Kda += w * J * np.dot(np.dot(B.T, D), G)
            Kaa += w * J * np.dot(np.dot(G.T, D), G)
            ra += w * J * np.dot(G.T, s)

        # --------------------------------------------------------
        # Static condensation: K = Kdd - Kda Kaa^{-1} Kda^T
        # --------------------------------------------------------
        ke -= np.dot(np.dot(Kda, np.linalg.inv(Kaa)), Kda.T)
        re -= np.dot(np.dot(Kda, np.linalg.inv(Kaa)), ra)
        return ke, re

    def gmatrix(self, p: NDArray, xi: NDArray, J: float) -> NDArray:
        """
        Construct incompatible strain-displacement matrix G.

        Algorithm in

        The Finite Element Method:  Its Basis and Fundamentals
        By Olek C Zienkiewicz, Robert L Taylor, J.Z. Zhu

        """
        dNdxi = self.shape_derivative(np.zeros(2))
        dxdxi = np.dot(dNdxi, p)
        dxidx = np.linalg.inv(dxdxi)
        J0 = np.linalg.det(dxidx)

        s, t = xi
        dNdxi = np.array([[-2.0 * s, 0.0], [0.0, -2.0 * t]], dtype=float)
        dNdx = J0 / J * np.dot(dxidx, dNdxi)

        G1 = np.array([[dNdx[0, 0], 0], [0, dNdx[0, 1]], [dNdx[0, 1], dNdx[0, 0]]])
        G2 = np.array([[dNdx[1, 0], 0], [0, dNdx[1, 1]], [dNdx[1, 1], dNdx[1, 0]]])
        G = np.concatenate((G1, G2), axis=1)
        return G


class NLGeom(ContinuumElement, IsoparametricElement):
    def green_lagrange_2d(self, p: NDArray, u: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)  # shape: (2, nnode)

        ux = u[0::2]
        uy = u[1::2]

        gradu = np.array(
            [
                [dNdx[0] @ ux, dNdx[1] @ ux],
                [dNdx[0] @ uy, dNdx[1] @ uy],
            ],
            dtype=float,
        )

        F = np.eye(2) + gradu
        E = 0.5 * (F.T @ F - np.eye(2))
        return E

    def pack_green_lagrange(self, E: NDArray) -> NDArray:
        if self.ndir == 2 and self.nshr == 1:  # plane stress
            return np.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]], dtype=float)
        if self.ndir == 3 and self.nshr == 1:  # plane strain
            return np.array([E[0, 0], E[1, 1], 0.0, 2.0 * E[0, 1]], dtype=float)
        raise NotImplementedError

    def pk2_voigt_to_tensor(self, s: NDArray) -> NDArray:
        if self.ndir == 2 and self.nshr == 1:  # plane stress
            return np.array([[s[0], s[2]], [s[2], s[1]]], dtype=float)
        if self.ndir == 3 and self.nshr == 1:  # plane strain
            return np.array([[s[0], s[3]], [s[3], s[1]]], dtype=float)
        raise NotImplementedError

    def deformation_gradient_2d(
        self, p: NDArray, u: NDArray, xi: NDArray
    ) -> tuple[NDArray, NDArray]:
        dNdx = self.shape_gradient(p, xi)

        ux = u[0::2]
        uy = u[1::2]

        grad_u = np.array(
            [
                [dNdx[0] @ ux, dNdx[1] @ ux],
                [dNdx[0] @ uy, dNdx[1] @ uy],
            ],
            dtype=float,
        )

        F = np.eye(2) + grad_u
        E = 0.5 * (F.T @ F - np.eye(2))
        return F, E

    def piola_tangent_2d(
        self,
        p: NDArray,
        u: NDArray,
        xi: NDArray,
        w: float,
        J: float,
        s_voigt: NDArray,
        D_voigt: NDArray,
    ) -> NDArray:
        """
        Consistent tangent for 2D nonlinear element residual using Piola stress.

        Returns the element stiffness contribution for one integration point.
        """
        dNdx = self.shape_gradient(p, xi)
        F, _ = self.deformation_gradient_2d(p, u, xi)
        S = self.pk2_voigt_to_tensor(s_voigt)

        ndof = self.nnode * self.dof_per_node
        K = np.zeros((ndof, ndof), dtype=float)

        for b in range(self.nnode):
            Gb = dNdx[:, b]  # [dN_b/dx, dN_b/dy]

            for beta in range(2):
                # dF from one nodal dof perturbation
                dF = np.zeros((2, 2), dtype=float)
                dF[beta, :] = Gb

                # dE = 0.5*(F^T dF + dF^T F)
                dE = 0.5 * (F.T @ dF + dF.T @ F)
                de = self.pack_green_lagrange(dE)

                # dS from material tangent
                dS_voigt = D_voigt @ de
                dS = self.pk2_voigt_to_tensor(dS_voigt)

                # dP = dF S + F dS
                dP = dF @ S + F @ dS

                # contribution to all test-node forces
                for a in range(self.nnode):
                    Ga = dNdx[:, a]
                    df = dP @ Ga

                    ia = self.dof_per_node * a
                    ib = self.dof_per_node * b + beta
                    K[ia : ia + 2, ib] += w * J * df

        return K

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
        """
        Perform element integration and compute stiffness and residual.

        This routine handles:
        - Volume integrals
        - Body forces
        - Surface loads
        - Robin boundary conditions

        Args:
            material: Constitutive model
            step: Step number
            increment: Increment count
            time: Time history sequence
            dt: Time increment
            eleno: Element index
            p: Nodal coordinates
            u: Element DOF vector
            du: Incremental DOFs
            pdata: Integration point storages
            dloads: Volume distributed loads
            dsloads: Surface distributed loads
            rloads: Robin loads

        Returns:
            Tuple of (element stiffness ke, element residual re)
        """
        dloads = dloads or []
        dsloads = dsloads or []
        rloads = rloads or []

        ndof = self.nnode * self.dof_per_node
        re = np.zeros(ndof)
        ke = np.zeros((ndof, ndof))

        # ————————————————— Volume integration —————————————————
        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            Pmat = self.pmatrix(xi)
            x = self.interpolate(p, xi)

            # Large strain kinematics
            E = self.green_lagrange_2d(p, u, xi)
            e = self.pack_green_lagrange(E)

            E_prev = self.green_lagrange_2d(p, u - du, xi)
            e_prev = self.pack_green_lagrange(E_prev)
            de = (e - e_prev) / dt

            D, s = self.update_state(
                material, step, increment, time, dt, eleno, p, u, e, de, pdata[ipt]
            )

            # Convert PK2 stress to tensor and build Piola stress
            F, _ = self.deformation_gradient_2d(p, u, xi)
            S = self.pk2_voigt_to_tensor(s)
            P = F @ S

            # Residual from Piola stress
            dNdx = self.shape_gradient(p, xi)
            for a in range(self.nnode):
                ia = self.dof_per_node * a
                re[ia : ia + 2] += w * J * (P @ dNdx[:, a])

            # Tangent from linearization of the Piola residual
            ke += self.piola_tangent_2d(p, u, xi, w, J, s, D)

            # — Body forces
            for dload in dloads:
                value = dload(step, increment, time, dt, eleno, ipt, x.tolist())
                re -= w * J * np.dot(Pmat, value)

        # ————————————————— Surface loads —————————————————
        for edge_no, dsload in dsloads:
            nodes = self.edges[edge_no]
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            for ipt, (w, xi) in enumerate(self.edge_integration_points()):
                x = self.interpolate_edge(edge_no, p, xi)
                n = self.edge_normal(edge_no, p, xi)
                traction = dsload(step, increment, time, dt, eleno, edge_no, ipt, x.tolist(), n)
                st = self.ref_edge_coords(edge_no, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(edge_no, p, xi)
                re[nft] -= w * J * np.dot(P, traction)

        # ————————————————— Robin boundary conditions —————————————————
        for rload in rloads:
            nodes = self.edges[rload.edge]
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            H = np.asarray(rload.H)
            u0 = np.asarray(rload.u0)
            for ipt, (w, xi) in enumerate(self.edge_integration_points()):
                st = self.ref_edge_coords(rload.edge, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(rload.edge, p, xi)
                kr = w * J * np.dot(np.dot(P, H), P.T)
                fr = w * J * np.dot(P, np.dot(H, u0))
                ke[np.ix_(nft, nft)] += kr
                re[nft] += np.dot(kr, u[nft]) - fr

        return ke, re


class CPS3NL(CPS3, NLGeom):
    pass


class CPE3NL(CPE3, NLGeom):
    pass


class CPS4NL(CPS4, NLGeom):
    pass


class CPE4NL(CPE4, NLGeom):
    pass


class CPS8NL(CPS8, NLGeom):
    pass


class CPE8NL(CPE8, NLGeom):
    pass
