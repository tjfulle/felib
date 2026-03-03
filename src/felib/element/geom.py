import numpy as np
from numpy.typing import NDArray


class Pn:
    edges: NDArray
    ref_coords: NDArray

    # Geometry-only methods
    def shape(self, xi: NDArray) -> NDArray:
        raise NotImplementedError

    def shape_derivative(self, xi: NDArray) -> NDArray:
        raise NotImplementedError

    def area(self, p: NDArray) -> float:
        raise NotImplementedError

    def edge_shape(self, xi: float, n: int) -> NDArray:
        if n == 2:
            return np.array([0.5 * (1.0 - xi), 0.5 * (1.0 + xi)])
        if n == 3:
            return np.array([0.5 * xi * (xi - 1.0), 1 - xi**2, 0.5 * xi * (xi + 1)])
        raise NotImplementedError

    def edge_shape_derivative(self, xi: float, n: int) -> NDArray:
        if n == 2:
            return np.array([-0.5, 0.5])
        if n == 3:
            return np.array([(xi - 0.5), -2.0 * xi, xi + 0.5])
        raise NotImplementedError

    def edge_coords(self, edge_no: int, p: NDArray) -> NDArray:
        return p[self.edges[edge_no]]

    def ref_edge_coords(self, edge_no: int, xi: float) -> NDArray:
        ix = self.edges[edge_no]
        p = self.ref_coords[ix]
        n = len(ix)
        N = self.edge_shape(xi, n)
        st = np.dot(N, p)
        return st

    def interpolate_edge(self, edge_no: int, p: NDArray, xi: float) -> NDArray:
        st = self.ref_edge_coords(edge_no, xi)
        ix = self.edges[edge_no]
        N = self.shape(st)
        return np.dot(N[ix], p[ix])

    def edge_tangent(self, edge_no: int, p: NDArray, xi: float) -> NDArray:
        ix = self.edges[edge_no]
        dN = self.edge_shape_derivative(xi, len(ix))
        return np.dot(dN, p[ix])

    def edge_normal(self, edge_no: int, p: NDArray, xi: float = 0.0) -> NDArray:
        t = self.edge_tangent(edge_no, p, xi)
        n = np.array([t[1], -t[0]])
        n = n / np.linalg.norm(n)
        return n

    def edge_centroid(self, edge_no: int, p: NDArray) -> NDArray:
        ix = self.edges[edge_no]
        n = len(ix)
        pd = p[ix]
        if n == 2:
            return 0.5 * (pd[0] + pd[1])
        if n == 3:
            return np.mean(pd, axis=0)
        raise NotImplementedError


class P3(Pn):
    """Linear 3-node triangle"""

    family = "TRI3"
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    ref_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        return np.array([1 - s - t, s, t])

    def shape_derivative(self, xi: NDArray) -> NDArray:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def area(self, p: NDArray) -> float:
        xp, yp = p[:, 0], p[:, 1]
        d = xp[0] * (yp[1] - yp[2]) + xp[1] * (yp[2] - yp[0]) + xp[2] * (yp[0] - yp[1])
        assert d > 0
        return d / 2.0


class P4(Pn):
    """Linear 4-node quad

    Notes
    -----
    Node and element face numbering

               [2]
            3-------2
            |       |
       [3]  |       | [1]
            |       |
            0-------1
               [0]

    """

    family = "QUAD4"
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        a = np.array(
            [
                (1.0 - s) * (1.0 - t),
                (1.0 + s) * (1.0 - t),
                (1.0 + s) * (1.0 + t),
                (1.0 - s) * (1.0 + t),
            ]
        )
        return a / 4.0

    def shape_derivative(self, xi):
        s, t = xi
        a = np.array(
            [[-1.0 + t, 1.0 - t, 1.0 + t, -1.0 - t], [-1.0 + s, -1.0 - s, 1.0 + s, 1.0 - s]]
        )
        return a / 4.0

    def area(self, p: NDArray) -> float:
        xp, yp = p[:, 0], p[:, 1]
        d = (xp[0] * yp[1] - xp[1] * yp[0]) + (xp[1] * yp[2] - xp[2] * yp[1])
        d += (xp[2] * yp[3] - xp[3] * yp[2]) + (xp[3] * yp[0] - xp[0] * yp[3])
        assert d > 0
        return d / 2.0
