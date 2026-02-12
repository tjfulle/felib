from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generator

import numpy as np
from numpy.typing import NDArray

from .collections import DistributedLoad
from .collections import RobinLoad
from .collections import SurfaceLoad
from .constants import GRAVITY
from .constants import PRESSURE
from .constants import TRACTION
from .material import Material


class Element(ABC):
    @property
    @abstractmethod
    def nnode(self) -> int: ...

    @property
    @abstractmethod
    def dof_per_node(self) -> int: ...

    @property
    @abstractmethod
    def dimensions(self) -> int: ...

    @property
    @abstractmethod
    def node_freedom_table(self) -> list[tuple[int, ...]]: ...

    @abstractmethod
    def area(self, p: NDArray) -> float:
        """Find the area of the triangle subtended by points p

        Args:
            p: p[:,0] are the x points, p[:,1] are the y points

        """
        ...

    @abstractmethod
    def side_centroid(self, side: int, p: NDArray) -> NDArray: ...

    @abstractmethod
    def eval(
        self,
        material: Material,
        p: NDArray,
        u: NDArray,
        dloads: list[DistributedLoad] | None = None,
        sloads: list[SurfaceLoad] | None = None,
        rloads: list[RobinLoad] | None = None,
    ) -> tuple[NDArray, NDArray]: ...


class IsoparametricElement(Element):
    ndir: int
    nshr: int
    sides: NDArray
    ref_coords: NDArray
    gauss_pts: NDArray
    gauss_wts: NDArray
    edge_gauss_pts: NDArray
    edge_gauss_wts: NDArray

    @property
    def integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        yield from zip(self.gauss_wts, self.gauss_pts)

    @property
    def edge_integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        yield from zip(self.edge_gauss_wts, self.edge_gauss_pts)

    @property
    def dof_per_node(self) -> int:
        return max([sum(_) for _ in self.node_freedom_table])

    @property
    def dimensions(self) -> int:
        return sum(self.node_freedom_table[0][:3])

    @property
    def nnode(self) -> int:
        return len(self.node_freedom_table)

    @abstractmethod
    def shape(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def shape_derivative(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def pmatrix(self, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray: ...

    @abstractmethod
    def ref_edge_coords(self, edge: int, xi: float) -> NDArray: ...

    @abstractmethod
    def edge_tangent(self, edge: int, p: NDArray, xi: float) -> NDArray: ...

    @abstractmethod
    def edge_normal(self, edge: int, p: NDArray, xi: float) -> NDArray: ...

    @abstractmethod
    def interpolate_edge(self, p: NDArray, xi: float) -> tuple[float, ...]: ...

    def jacobian(self, p: NDArray, xi: NDArray) -> float:
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        return np.linalg.det(dxdxi)

    def shape_gradient(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdxi = self.shape_derivative(xi)
        dxdxi = np.dot(dNdxi, p)
        dNdx = np.dot(np.linalg.inv(dxdxi), dNdxi)
        return dNdx

    def interpolate(self, p: NDArray, xi: NDArray) -> tuple[float, ...]:
        N = self.shape(xi)
        return tuple(np.dot(N, p[:, i]) for i in range(p.shape[1]))

    def edge_jacobian(self, edge: int, p: NDArray, xi: float) -> float:
        """
        Compute Jacobian |dx/dxi| for each 1D Gauss point along a physical edge.

        Args:
            p_edge: array of shape (n_edge_nodes, 2) with coordinates of edge nodes
        Returns:
            J: array of length ngauss with Jacobian at each Gauss point
        """
        return float(np.linalg.norm(self.edge_tangent(edge, p, xi)))

    def eval(
        self,
        material: Material,
        p: NDArray,
        u: NDArray,
        dloads: list[DistributedLoad] | None = None,
        sloads: list[SurfaceLoad] | None = None,
        rloads: list[RobinLoad] | None = None,
    ) -> tuple[NDArray, NDArray]:
        dloads = dloads or []
        sloads = sloads or []
        rloads = rloads or []

        ndof = self.nnode * self.dof_per_node
        re = np.zeros(ndof)
        ke = np.zeros((ndof, ndof))

        for w, xi in self.integration_points:
            # ------------------
            # Volume intetration
            # ------------------
            J = self.jacobian(p, xi)
            B = self.bmatrix(p, xi)
            P = self.pmatrix(xi)

            # --- Internal terms
            e = np.dot(B, u)
            D, s = material.eval(e, ndir=self.ndir, nshr=self.nshr)
            ke += w * J * np.dot(np.dot(B.T, D), B)
            re += w * J * np.dot(B.T, s)

            for dload in dloads:
                value = np.asarray(dload.value)
                if dload.load_type == GRAVITY:
                    value = value * material.density
                re -= w * J * np.dot(P, value)

        for sload in sloads:
            nodes = self.sides[sload.edge]  # local node freedom table
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            for w, xi in self.edge_integration_points:
                if sload.load_type == TRACTION:
                    traction = np.asarray(sload.value)
                elif sload.load_type == PRESSURE:
                    n = self.edge_normal(sload.edge, p, xi)
                    traction = float(sload.value) * n
                else:
                    traction = np.asarray(sload.value)
                st = self.ref_edge_coords(sload.edge, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(sload.edge, p, xi)
                re[nft] -= w * J * np.dot(P, traction)

        for rload in rloads:
            nodes = self.sides[rload.edge]  # local node freedom table
            nft = [self.dof_per_node * n + d for n in nodes for d in range(self.dof_per_node)]
            H = np.asarray(rload.H)
            u0 = np.asarray(rload.u0)
            for w, xi in self.edge_integration_points:
                st = self.ref_edge_coords(rload.edge, xi)
                P = self.pmatrix(st)[nft]
                J = self.edge_jacobian(rload.edge, p, xi)
                kr = w * J * np.dot(np.dot(P, H), P.T)
                fr = w * J * np.dot(P, np.dot(H, u0))
                ke[np.ix_(nft, nft)] += kr
                re[nft] += np.dot(kr, u[nft]) - fr

        return ke, re


class CPX3(IsoparametricElement):
    sides = np.array([[0, 1], [1, 2], [2, 0]])
    ref_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    gauss_pts = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
    gauss_wts = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    edge_gauss_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    edge_gauss_wts = np.array([1.0, 1.0])

    @property
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        return [
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        ]

    def area(self, p: NDArray) -> float:
        xp, yp = p[:, 0], p[:, 1]
        d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
        assert d > 0
        return d / 2.0

    def side_centroid(self, side: int, p: NDArray) -> NDArray:
        ix = self.sides[side]
        x = p[ix]
        return 0.5 * (x[0] + x[1])

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        N = np.array([1.0 - s - t, s, t])
        return N

    def shape_derivative(self, xi: NDArray) -> NDArray:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def pmatrix(self, xi: NDArray) -> NDArray:
        N = self.shape(xi)
        P = np.zeros((6, 2))
        P[0::2, 0] = N
        P[1::2, 1] = N
        return P

    def ref_edge_coords(self, edge: int, xi: float) -> NDArray:
        """
        Map 1D Gauss points on the edge to reference element (s,t).

        Returns:
            s, t: array of shape (2,) with reference coordinates for xi
        """
        a, b = self.sides[edge]
        sa, ta = self.ref_coords[a]
        sb, tb = self.ref_coords[b]
        s = 0.5 * (1 - xi) * sa + 0.5 * (1 + xi) * sb
        t = 0.5 * (1 - xi) * ta + 0.5 * (1 + xi) * tb
        return np.array((s, t))

    def edge_tangent(self, edge: int, p: NDArray, xi: float) -> NDArray:
        """
        Compute Jacobian dx/dxi for each 1D Gauss point along a physical edge.

        Args:
            p_edge: array of shape (n_edge_nodes, 2) with coordinates of edge nodes
        Returns:
            J: array of length ngauss with Jacobian at each Gauss point
        """
        # Map xi -> s, t
        st = self.ref_edge_coords(edge, xi)
        dNdst = self.shape_derivative(st)
        dxdst = np.dot(dNdst, p)
        a, b = self.sides[edge]
        sa, ta = self.ref_coords[a]
        sb, tb = self.ref_coords[b]
        dstdxi = 0.5 * np.array([sb - sa, tb - ta])
        dxdxi = np.dot(dxdst.T, dstdxi)
        return dxdxi

    def edge_normal(self, edge: int, p: NDArray, xi: float) -> NDArray:
        t = self.edge_tangent(edge, p, xi)
        n = np.array([t[1], -t[0]])  # rotate tangent
        return n / np.linalg.norm(n)

    def interpolate_edge(self, p: NDArray, xi: float) -> tuple[float, float]:
        xp, yp = p[:, 0], p[:, 1]
        x = 0.5 * (1.0 - xi) * xp[0] + 0.5 * (1 + xi) * xp[1]
        y = 0.5 * (1.0 - xi) * yp[0] + 0.5 * (1 + xi) * yp[1]
        return x, y


class CPS3(CPX3):
    """Plane stress CST element"""

    ndir: int = 2
    nshr: int = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((3, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[2, 0::2] = dNdx[1]
        B[2, 1::2] = dNdx[0]
        return B


class CPE3(CPX3):
    """Plane strain CST element"""

    ndir: int = 3
    nshr: int = 1

    def bmatrix(self, p: NDArray, xi: NDArray) -> NDArray:
        dNdx = self.shape_gradient(p, xi)
        B = np.zeros((4, 6))
        B[0, 0::2] = dNdx[0]
        B[1, 1::2] = dNdx[1]
        B[3, 0::2] = dNdx[1]
        B[3, 1::2] = dNdx[0]
        return B
