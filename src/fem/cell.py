from abc import ABC
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray


class Cell(ABC):
    dim: int
    nnode: int
    nedge: int
    nface: int

    @staticmethod
    @abstractmethod
    def edge_nodes(edge_no: int) -> list[int]: ...

    @staticmethod
    @abstractmethod
    def face_nodes(face_no: int) -> list[int]: ...

    @staticmethod
    @abstractmethod
    def edge_normal(edge_no: int, p: NDArray) -> NDArray: ...

    @staticmethod
    @abstractmethod
    def edge_centroid(edge_no: int, p: NDArray) -> NDArray: ...


class Tri3(Cell):
    dim = 2
    nnode = 3
    nedge = 3
    nface = 1

    _edges = [[0, 1], [1, 2], [2, 0]]
    _faces = [[0, 1, 2]]

    @staticmethod
    def edge_nodes(edge_no: int) -> list[int]:
        try:
            return Tri3._edges[edge_no]
        except IndexError:
            raise ValueError(f"Illegal edge number {edge_no}") from None

    @staticmethod
    def face_nodes(face_no: int) -> list[int]:
        try:
            return Tri3._faces[face_no]
        except IndexError:
            raise ValueError(f"Illegal face number {face_no}") from None

    @staticmethod
    def edge_normal(edge_no: int, p: NDArray) -> NDArray:
        na, nb = Tri3.edge_nodes(edge_no)
        tangent = p[nb] - p[na]
        normal = np.array([-tangent[1], tangent[0]])
        centroid = np.mean(p, axis=0)
        midpoint = 0.5 * (p[na] + p[nb])
        if np.dot(normal, centroid - midpoint) > 0:
            normal = -normal
        return normal / np.linalg.norm(normal)

    @staticmethod
    def edge_centroid(edge_no: int, p: NDArray) -> NDArray:
        ix = Tri3.edge_nodes(edge_no)
        x = p[ix]
        return 0.5 * (x[0] + x[1])
