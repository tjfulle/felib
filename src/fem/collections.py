from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import Type

from numpy.typing import NDArray

if TYPE_CHECKING:
    from .cell import Cell


class Map:
    def __init__(self, gids: list[int]) -> None:
        self.lid_to_gid = list(gids)
        self.gid_to_lid = {gid: lid for lid, gid in enumerate(gids)}

    def __getitem__(self, lid: int) -> int:
        return self.lid_to_gid[lid]

    def __contains__(self, gid: int) -> bool:
        return gid in self.lid_to_gid

    def local(self, gid: int) -> int:
        return self.gid_to_lid[gid]


class Value(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> Any: ...


class Constant(Value):
    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> Any:
        return self.value


class Vector(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[float]: ...


class ConstantVector(Vector):
    def __init__(self, value: list[float]) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[float]:
        return self.value


class Matrix(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[list[float]]: ...


class ConstantMatrix(Matrix):
    def __init__(self, value: list[list[float]]) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[list[float]]:
        return self.value


class BoundaryCondition(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]: ...


class PinnedBoundary(BoundaryCondition):
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim)]


class RollerBoundary(BoundaryCondition):
    def __init__(self, free_dof: int, value: float = 0.0) -> None:
        self.free_dof = free_dof
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim) if dof != self.free_dof]


class PointLoad(BoundaryCondition):
    def __init__(self, free_dof: int, value: float = 0.0) -> None:
        self.free_dof = free_dof
        self.value = value

    def __call__(self, x: Sequence[float], t: float = 0.0) -> list[tuple[int, float]]:
        ndim = len(x)
        return [(dof, self.value) for dof in range(ndim) if dof != self.free_dof]


class RegionSelector(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], on_boundary: bool) -> bool: ...


@dataclass
class Node:
    lid: int
    gid: int
    x: list[float]
    normal: list[float] = field(default_factory=list)
    on_boundary: bool = field(default=False)


@dataclass
class Edge:
    element: int
    edge: int
    x: list[float]
    normal: list[float]


@dataclass
class BlockSpec:
    name: str
    cell_type: Type["Cell"]
    region: RegionSelector


@dataclass
class NodeSetSpec:
    name: str
    region: RegionSelector


@dataclass
class SideSetSpec:
    name: str
    region: RegionSelector


@dataclass
class ElemSetSpec:
    name: str
    region: RegionSelector


@dataclass
class DistributedLoad:
    load_type: int
    value: NDArray


@dataclass
class SurfaceLoad:
    load_type: int
    edge: int
    value: NDArray


@dataclass
class RobinLoad:
    edge: int
    H: NDArray
    u0: NDArray
