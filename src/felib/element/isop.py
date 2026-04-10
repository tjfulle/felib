from abc import abstractmethod
from typing import Any
from typing import Generator
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..collections import DistributedLoad
from ..collections import DistributedSurfaceLoad
from ..collections import RobinLoad
from ..material import Material
from .base import Element


class IsoparametricElement(Element):
    """
    Abstract base class for isoparametric finite elements.

    Implements shared geometric routines (shape interpolation,
    jacobian, shape gradients, etc.) and defines the interface
    for specific element families.

    Concrete subclasses must implement shape functions, their
    derivatives, and edge interpolation/tangent/normal routines.
    """

    edges: NDArray
    ref_coords: NDArray
    gauss_wts: NDArray
    gauss_pts: NDArray
    edge_gauss_wts: NDArray
    edge_gauss_pts: NDArray
    uses_local_pressure: bool = False
    npressure: int = 0

    # —————————————————————————————————————————————————————————————
    # Integration point accessors
    # —————————————————————————————————————————————————————————————
    def integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        """
        Yield integration weights and local coordinates.

        Returns:
            A generator yielding tuples (weight, xi) for each Gauss point.
        """
        yield from zip(self.gauss_wts, self.gauss_pts)

    def edge_integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        """
        Yield edge integration weights and local 1D coordinates.

        Returns:
            Generator over (weight, xi) for each edge Gauss point.
        """
        yield from zip(self.edge_gauss_wts, self.edge_gauss_pts)

    # —————————————————————————————————————————————————————————————
    # DOF and dimension routines
    # —————————————————————————————————————————————————————————————

    @property
    def dof_per_node(self) -> int:
        """
        Compute degrees of freedom per node from node freedom table.

        Returns:
            Integer number of DOFs per node.
        """
        return max([len(_) for _ in self.node_freedom_table])

    @property
    def dimensions(self) -> int:
        """
        Spatial dimension based on node freedom table.

        Returns:
            Integer spatial dimension (e.g., 2 or 3).
        """
        return sum(self.node_freedom_table[0][:3])

    @property
    def nnode(self) -> int:
        """
        Number of nodes in this element.

        Returns:
            Integer node count.
        """
        return len(self.node_freedom_table)

    @property
    def npts(self) -> int:
        """
        Number of volume integration points.

        Returns:
            Count of Gauss points.
        """
        return self.gauss_wts.size

    # —————————————————————————————————————————————————————————————
    # Abstract shape and kinematic methods
    # —————————————————————————————————————————————————————————————

    @abstractmethod
    def shape(self, xi: NDArray) -> NDArray:
        """
        Evaluate shape functions at reference coordinate xi.

        Args:
            xi: Local element coordinate

        Returns:
            Nodal shape function values
        """
        ...

    @abstractmethod
    def shape_derivative(self, xi: NDArray) -> NDArray:
        """
        Derivative of shape functions with respect to reference coordinates.

        Args:
            xi: Local element coordinate

        Returns:
            Derivative matrix of shape functions
        """
        ...

    @abstractmethod
    def pmatrix(self, xi: NDArray) -> NDArray:
        """
        Construct the P matrix for body or traction loads.

        Args:
            xi: Reference coordinate

        Returns:
            Matrix mapping loads into DOF space
        """
        ...

    @abstractmethod
    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Construct the strain-displacement (B) matrix.

        Args:
            p: Physical nodal coordinates
            xi: Reference coordinate

        Returns:
            B matrix for this element
        """
        ...

    @abstractmethod
    def ref_edge_coords(self, edge_no: int, xi: float) -> NDArray:
        """
        Reference-space coordinates along an edge.

        Args:
            edge_no: Edge index
            xi: 1D local coordinate

        Returns:
            Reference coordinate on the edge
        """
        ...

    @abstractmethod
    def edge_tangent(self, edge_no: int, p: NDArray, xi: float) -> NDArray:
        """
        Tangent vector along a physical edge.

        Args:
            edge_no: Edge index
            p: Physical coordinates
            xi: Edge Gauss coordinate

        Returns:
            Tangent vector at that location
        """
        ...

    @abstractmethod
    def edge_normal(self, edge_no: int, p: NDArray, xi: float) -> NDArray:
        """
        Surface normal vector on an edge.

        Args:
            edge_no: Edge index
            p: Physical coordinates
            xi: Edge Gauss coordinate

        Returns:
            Normal vector
        """
        ...

    @abstractmethod
    def interpolate_edge(self, edge_no: int, p: NDArray, xi: float) -> NDArray:
        """
        Interpolate physical coordinates along an edge.

        Args:
            edge_no: Edge index
            p: Physical node coordinates
            xi: 1D local coordinate

        Returns:
            Interpolated point on edge
        """
        ...

    # —————————————————————————————————————————————————————————————
    # Geometric utilities
    # —————————————————————————————————————————————————————————————

    def centroid(self, p: NDArray) -> NDArray:
        """
        Compute element centroid.

        Args:
            p: Nodal coordinate array

        Returns:
            Mean of nodal coordinates
        """
        return np.asarray(p).mean(axis=0)

    def area(self, p: NDArray) -> float:
        A: float = 0.0
        for w, xi in self.integration_points():
            dN = self.shape_derivative(xi)
            J = np.dot(dN, p)
            A += w * np.linalg.det(J)
        return A

    def jacobian(self, p: NDArray, xi: NDArray) -> float:
        """
        Compute the volume Jacobian determinant at xi.

        Args:
            p: Physical nodal coordinates
            xi: Local coordinate

        Returns:
            Determinant of the Jacobian
        """
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        return np.linalg.det(dxdxi)

    def shape_gradient(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Compute the global derivative of shape functions.

        Args:
            p: Physical nodal coordinates
            xi: Local coordinate

        Returns:
            Gradient of shape functions wrt physical coordinates
        """
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        # dNdx = dNdxi * (dxdxi)^(-1)
        dNdx = np.dot(np.linalg.inv(dxdxi), dNdxi)
        return dNdx

    def interpolate(self, p: NDArray, xi: NDArray) -> NDArray:
        """
        Interpolate physical coordinates from nodal coords.

        Args:
            p: Physical nodal coordinates
            xi: Local coordinate

        Returns:
            Interpolated point in physical space
        """
        N = self.shape(xi)
        return np.array([np.dot(N, p[:, i]) for i in range(p.shape[1])], dtype=float)

    def edge_jacobian(self, edge_no: int, p: NDArray, xi: float) -> float:
        """
        Compute 1D edge Jacobian (|dx/dξ|) for edge integration.

        The 1D jacobian is the norm of the tangent vector.

        Args:
            edge_no: Edge index
            p: Nodal coordinates
            xi: Local edge coordinate

        Returns:
            The scalar edge jacobian
        """
        return float(np.linalg.norm(self.edge_tangent(edge_no, p, xi)))
    def axisymmetric_weight(self, x: NDArray) -> float:
        if not getattr(self, "axisymmetric", False):
            return 1.0
        r = float(x[0]) if x.size > 0 else 0.0
        if r <= 0.0:
            r = 1e-12
        return 2.0 * np.pi * r

    def bmatrix_vol(self, p: NDArray, xi: NDArray) -> NDArray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define a volumetric B matrix"
        )

    def pressure_shape(self, xi: NDArray) -> NDArray:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define local pressure interpolation"
        )

    def pressure_stress_vector(self, pressure: float, size: int) -> NDArray:
        sp = np.zeros(size, dtype=float)
        sp[: min(3, size)] = pressure
        return sp
    # —————————————————————————————————————————————————————————————
    # Abstract update_state required for eval
    # —————————————————————————————————————————————————————————————

    @abstractmethod
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
        Update integration-point material state.

        Args:
            material: Constitutive model
            step: Step index
            increment: Increment index
            time: Time history sequence
            dt: Time delta
            eleno: Element number
            p: Physical coordinates
            e: Displacement
            e: Strain
            de: Strain rate
            hsv: State memory array

        Returns:
            Elastic stiffness tangent and stress
        """
        ...
    def _apply_surface_loads(
        self,
        re: NDArray,
        p: NDArray,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        dsloads: list[tuple[int, DistributedSurfaceLoad]],
    ) -> None:
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
                axfact = self.axisymmetric_weight(x)
                re[nft] -= w * J * axfact * np.dot(P, traction)

    def _apply_robin_loads(
        self,
        ke: NDArray,
        re: NDArray,
        u: NDArray,
        p: NDArray,
        rloads: list[RobinLoad],
    ) -> None:
        for rload in rloads:
            nodes = self.edges[rload.edge]
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            H = np.asarray(rload.H)
            u0 = np.asarray(rload.u0)
            for w, xi in self.edge_integration_points():
                st = self.ref_edge_coords(rload.edge, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(rload.edge, p, xi)
                x = self.interpolate_edge(rload.edge, p, xi)
                axfact = self.axisymmetric_weight(x)
                kr = w * J * axfact * np.dot(np.dot(P, H), P.T)
                fr = w * J * axfact * np.dot(P, np.dot(H, u0))
                ke[np.ix_(nft, nft)] += kr
                re[nft] += np.dot(kr, u[nft]) - fr

    def eval_mixed_condensed(
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
        dloads: list[DistributedLoad],
        dsloads: list[tuple[int, DistributedSurfaceLoad]],
        rloads: list[RobinLoad],
    ) -> tuple[NDArray, NDArray]:
        ndof_u = self.nnode * self.dof_per_node
        npressure = self.npressure
        if npressure <= 0:
            raise RuntimeError(
                f"{self.__class__.__name__} requests local pressure condensation without "
                "declaring any pressure DOFs"
            )

        Kuu = np.zeros((ndof_u, ndof_u), dtype=float)
        Kup = np.zeros((ndof_u, npressure), dtype=float)
        Kpu = np.zeros((npressure, ndof_u), dtype=float)
        Kpp = np.zeros((npressure, npressure), dtype=float)
        Ru = np.zeros(ndof_u, dtype=float)
        Rp = np.zeros(npressure, dtype=float)

        strains: list[NDArray] = []
        dev_stresses: list[NDArray] = []
        pressure_shapes: list[NDArray] = []
        volumetric_strains: list[float] = []
#Code lines 359-438 generated with CODEX from ChatGPT, adapted for ME7540

        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            Bv = self.bmatrix_vol(p, xi)
            P = self.pmatrix(xi)
            Np = np.atleast_2d(self.pressure_shape(xi))
            x = self.interpolate(p, xi)
            axfact = self.axisymmetric_weight(x)

            e = np.dot(B, u)
            de = np.dot(B, du) / dt
            ev = float(np.dot(Bv, u)[0])
            temp = dtemp = 0.0
            Ddev, sdev, Kbulk = material.eval_hybrid(
                pdata[ipt][len(self.history_variables()) :],
                e,
                de,
                time,
                dt,
                temp,
                dtemp,
                getattr(self, "ndir"),
                getattr(self, "nshr"),
                eleno,
                step,
                increment,
            )

            Kuu += w * J * axfact * np.dot(np.dot(B.T, Ddev), B)
            coupling = w * J * axfact * np.dot(Bv.T, Np)
            Kup -= coupling
            Kpu -= coupling.T
            Kpp -= w * J * axfact * (1.0 / Kbulk) * np.dot(Np.T, Np)
            Ru += w * J * axfact * np.dot(B.T, sdev)
            Rp -= w * J * axfact * Np.ravel() * ev

            strains.append(np.asarray(e, dtype=float))
            dev_stresses.append(np.asarray(sdev, dtype=float))
            pressure_shapes.append(Np.ravel())
            volumetric_strains.append(ev)

            for dload in dloads:
                value = dload(step, increment, time, dt, eleno, ipt, x.tolist())
                Ru -= w * J * axfact * np.dot(P, value)

        try:
            pressure_dofs = -np.linalg.solve(Kpp, Rp)
            ke = Kuu - Kup @ np.linalg.solve(Kpp, Kpu)
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                f"Local pressure condensation failed for {self.__class__.__name__}"
            ) from err
        re = Ru + Kup @ pressure_dofs

        ntens = len(strains[0]) if strains else 0
        for ipt, (e, sdev, Np, ev) in enumerate(
            zip(strains, dev_stresses, pressure_shapes, volumetric_strains)
        ):
            pressure = float(np.dot(Np, pressure_dofs))
            sigma = sdev - self.pressure_stress_vector(pressure, ntens)
            hsv = pdata[ipt]
            if ntens:
                hsv[:ntens] = e
                hsv[ntens : 2 * ntens] = sigma
            offset = 2 * ntens
            nstore = min(npressure, max(0, hsv.size - offset))
            if nstore:
                hsv[offset : offset + nstore] = pressure_dofs[:nstore]
                offset += nstore
            if offset < hsv.size:
                hsv[offset] = ev

        self._apply_surface_loads(re, p, step, increment, time, dt, eleno, dsloads)
        self._apply_robin_loads(ke, re, u, p, rloads)
        return ke, re

    # —————————————————————————————————————————————————————————————
    # Main evaluation routine
    # —————————————————————————————————————————————————————————————

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
        dloads: list[DistributedLoad] | None = None,
        dsloads: list[tuple[int, DistributedSurfaceLoad]] | None = None,
        rloads: list[RobinLoad] | None = None,
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
        if getattr(self, "uses_local_pressure", False):
              return self.eval_mixed_condensed(
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
                dloads,
                dsloads,
                rloads,
            )

        # ————————————————— Volume integration —————————————————
        for ipt, (w, xi) in enumerate(self.integration_points()):
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            P = self.pmatrix(xi)
            x = self.interpolate(p, xi)
            axfact = self.axisymmetric_weight(x)
            # — Internal contribution
            e = np.dot(B, u)
            de = np.dot(B, du) / dt
            D, s = self.update_state(
                material, step, increment, time, dt, eleno, p, u, e, de, pdata[ipt]
            )
            ke += w * J * axfact * np.dot(np.dot(B.T, D), B)
            re += w * J * axfact * np.dot(B.T, s)

            # — Body forces
            for dload in dloads:
                value = dload(step, increment, time, dt, eleno, ipt, x.tolist())
                re -= w * J * axfact * np.dot(P, value)

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
                axfact = self.axisymmetric_weight(x)
                re[nft] -= w * J * axfact * np.dot(P, traction)

        # ————————————————— Robin boundary conditions —————————————————
        self._apply_robin_loads(ke, re, u, p, rloads)
        

        return ke, re
