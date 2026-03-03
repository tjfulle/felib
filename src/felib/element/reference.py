import numpy as np
from numpy.typing import NDArray


class ReferenceElement:
    family: str
    edges: NDArray
    faces: NDArray
    ref_coords: NDArray

    @property
    def ndim(self) -> int:
        return self.ref_coords.shape[1]

    @property
    def nnode(self) -> int:
        return self.ref_coords.shape[0]

    @property
    def nedge(self) -> int:
        return self.edges.shape[0]

    @property
    def nface(self) -> int:
        return self.faces.shape[0]

    # Geometry-only methods
    def shape(self, xi: NDArray) -> NDArray:
        raise NotImplementedError

    def shape_derivative(self, xi: NDArray) -> NDArray:
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

    def face_nodes(self, face_no) -> NDArray:
        return self.faces[face_no]

    def edge_nodes(self, edge_no) -> NDArray:
        return self.edges[edge_no]

    def edge_coords(self, edge_no: int, p: NDArray) -> NDArray:
        return p[self.edge_nodes(edge_no)]

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
        return n / np.linalg.norm(n)

    def edge_centroid(self, edge_no: int, p: NDArray) -> NDArray:
        ix = self.edges[edge_no]
        n = len(ix)
        pd = p[ix]
        if n == 2:
            return 0.5 * (pd[0] + pd[1])
        if n == 3:
            return np.mean(pd, axis=0)
        raise NotImplementedError


class Tri3(ReferenceElement):
    """Linear 3-node triangle"""

    family = "TRI3"
    faces = np.array([[0, 1, 2, 3]], dtype=int)
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    ref_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        return np.array([1 - s - t, s, t])

    def shape_derivative(self, xi: NDArray) -> NDArray:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])


class Quad4(ReferenceElement):
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
    faces = np.array([[0, 1, 2, 3, 4]], dtype=int)
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


class Quad8(ReferenceElement):
    """Quadratic 8-node quad

    Notes
    -----
    Node and element face numbering

               [2]
            3---6---2
            |       |
       [3]  7       5 [1]
            |       |
            0---4---1
               [0]

    """

    family = "QUAD8"
    edges = np.array([[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]], dtype=int)
    faces = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int)
    ref_coords = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )

    def shape(self, xi: NDArray) -> NDArray:
        s, t = xi
        N = np.zeros(8, dtype=float)
        N[0] = 0.25 * (1.0 - s) * (1.0 - t) * (-s - t - 1)
        N[1] = 0.25 * (1.0 + s) * (1.0 - t) * (s - t - 1)
        N[2] = 0.25 * (1.0 + s) * (1.0 + t) * (s + t - 1)
        N[3] = 0.25 * (1.0 - s) * (1.0 + t) * (-s + t - 1)
        N[4] = 0.5 * (1.0 - s**2) * (1.0 - t)
        N[5] = 0.5 * (1.0 + s) * (1.0 - t**2)
        N[6] = 0.5 * (1.0 - s**2) * (1.0 + t)
        N[7] = 0.5 * (1.0 - s) * (1.0 - t**2)
        return N

    def shape_derivative(self, xi):
        s, t = xi
        dN = np.zeros((2, 8), dtype=float)
        dN[0, 0] = 0.25 * (1.0 - t) * (2.0 * s + t)
        dN[1, 0] = 0.25 * (1.0 - s) * (2.0 * t + s)

        dN[0, 1] = 0.25 * (1.0 - t) * (2.0 * s - t)
        dN[1, 1] = 0.25 * (1.0 + s) * (2.0 * t + s)

        dN[0, 2] = 0.25 * (1.0 + t) * (2.0 * s + t)
        dN[1, 2] = 0.25 * (1.0 + s) * (2.0 * t + s)

        dN[0, 3] = 0.25 * (1.0 + t) * (2.0 * s - t)
        dN[1, 3] = 0.25 * (1.0 - s) * (2.0 * t - s)

        dN[0, 4] = -s * (1.0 - t)
        dN[1, 4] = -0.5 * (1.0 - s**2)

        dN[0, 5] = 0.5 * (1.0 - t**2)
        dN[1, 5] = -(1.0 + s) * t

        dN[0, 6] = -s * (1.0 + t)
        dN[1, 6] = 0.5 * (1.0 - s**2)

        dN[0, 7] = -0.5 * (1.0 - t**2)
        dN[1, 7] = -(1.0 - s) * t

        return dN
