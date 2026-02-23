from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import Type
from .typing import DSLoadT, DLoadT, RLoadT

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .cell import Cell


class Map:
    def __init__(self, gids: list[int]) -> None:
        self.lid_to_gid = list(gids)
        self.gid_to_lid = {gid: lid for lid, gid in enumerate(gids)}

    def __getitem__(self, lid: int) -> int:
        try:
            return self.lid_to_gid[lid]
        except IndexError:
            raise ValueError(f"Invalid index {lid}") from None

    def __contains__(self, gid: int) -> bool:
        return gid in self.lid_to_gid

    def __len__(self) -> int:
        return len(self.lid_to_gid)

    def local(self, gid: int) -> int:
        return self.gid_to_lid[gid]


class Field(ABC):
    @abstractmethod
    def __call__(self, x: NDArray, time: Sequence[float]) -> Any: ...


class ScalarField(Field):
    @abstractmethod
    def __call__(self, x: NDArray, time: Sequence[float]) -> float: ...


class ConstantScalarField(ScalarField):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, x: NDArray, time: Sequence[float]) -> float:
        return self.value


class VectorField(Field):
    @abstractmethod
    def __call__(self, x: NDArray, time: Sequence[float]) -> NDArray: ...


class ConstantVectorField(VectorField):
    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        vec = np.asarray(direction, dtype=float)
        vec /= np.linalg.norm(vec)
        self.value = magnitude * vec

    def __call__(self, x: NDArray, time: Sequence[float]) -> NDArray:
        return self.value


class BoundaryCondition(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        step: int,
        increment: int,
        node: int,
        dof: int,
        time: list[float],
        dt: float,
        x: NDArray,
    ) -> float: ...


class Load(ABC):
    @property
    @abstractmethod
    def field(self) -> Field: ...

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...


class DistributedLoad(Load):
    """
    Load integrated over element domain (volume or length).
    """

    def __init__(self, field: Field) -> None:
        self._field = field

    @property
    def field(self) -> Field:
        return self._field

    def __call__(
        self,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        ipt: int,
        x: NDArray,
    ) -> NDArray:
        """
        Evaluate the load at point x and time t.

        Args:

          step: Current analysis step number
          increment: Current step increment
          time:
            time[0]: Current value of step time
            time[1]: Current value of total time
          eleno: Element number
          ipt: Integration point number
          x: Coordinates of the load integration point
        """
        return self._field(x, time)


class DistributedSurfaceLoad(Load):
    """
    Load integrated over element boundary (codim 1).
    """

    def __init__(self, field: Field) -> None:
        self._field = field

    @property
    def field(self) -> Field:
        return self._field

    def __call__(
        self,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        sideno: int,
        ipt: int,
        x: NDArray,
        n: NDArray,
    ) -> NDArray:
        """
        Evaluate the load at point x and time t.

        Args:

          step: Current analysis step number
          increment: Current step increment
          time:
            time[0]: Current value of step time
            time[1]: Current value of total time
          eleno: Element number
          sideno: Edge number
          ipt: Integration point number
          x: Coordinates of the load integration point
        """
        return self._field(x, time)


class GravityLoad(DistributedLoad):
    """
    Mass-proportional body force.
    """

    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        field = ConstantVectorField(magnitude, direction)
        super().__init__(field=field)


class TractionLoad(DistributedSurfaceLoad):
    """
    Mechanical traction applied on element surfaces.
    """

    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        field = ConstantVectorField(magnitude, direction)
        super().__init__(field=field)


class PressureLoad(DistributedSurfaceLoad):
    """
    Mechanical traction applied on element surfaces.
    """

    def __init__(self, magnitude: float) -> None:
        field = ConstantScalarField(magnitude)
        super().__init__(field=field)

    def __call__(
        self,
        step: int,
        increment: int,
        time: Sequence[float],
        dt: float,
        eleno: int,
        sideno: int,
        ipt: int,
        x: NDArray,
        n: NDArray,
    ) -> NDArray:
        return -self.field(x, time) * n


@dataclass
class Solution:
    stiff: NDArray
    force: NDArray
    dofs: NDArray
    react: NDArray
    iterations: int = field(default=1)
    lagrange_multipliers: NDArray = field(default_factory=lambda: np.empty((0,)))


class RegionSelector(ABC):
    @abstractmethod
    def __call__(self, x: NDArray, on_boundary: bool) -> bool: ...


@dataclass
class Node:
    lid: int
    gid: int
    x: Sequence[float]
    normal: list[float] = field(default_factory=list)
    on_boundary: bool = field(default=False)


@dataclass
class Edge:
    element: int
    edge: int
    x: Sequence[float]
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
class SurfaceLoad:
    load_type: int
    edge: int
    value: NDArray


@dataclass
class RobinLoad:
    edge: int
    H: NDArray
    u0: NDArray


@dataclass
class SolverState:
    u0: NDArray
    R0: NDArray
    ddofs: NDArray
    dvals: NDArray
    fdofs: NDArray
    time: list[float]
    dt: float
    step: int
    dsloads: DSLoadT
    dloads: DLoadT
    rloads: RLoadT
    equations: list[list[int | float]]
