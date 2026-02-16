from abc import ABC
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
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


class Field(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> Any: ...


class ScalarField(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> float: ...


class ConstantScalarField(ScalarField):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
        return self.value


class VectorField(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> NDArray: ...


class ConstantVectorField(VectorField):
    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        self.value = magnitude * direction

    def __call__(self, x: Sequence[float], time: Sequence[float]) -> NDArray:
        return self.value


class Load(ABC):
    @property
    @abstractmethod
    def field(self) -> Field: ...

    @abstractmethod
    def __call__(self, *args) -> Any: ...


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


class TractionLoad(DistributedSurfaceLoad):
    """
    Mechanical traction applied on element surfaces.
    """

    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        field = ConstantVectorField(magnitude, direction)
        super().__init__(field=field)



class Value(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], t: float = 0.0) -> Any: ...


class Constant(Value):
    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, *args: Any) -> Any:
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


class NodalFieldEvaluator(ABC):
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
        x: Sequence[float],
    ) -> float: ...


class ConstantNodalField(NodalFieldEvaluator):
    def __init__(self, value: float) -> None:
        self.value = value
    def __call__(
        self,
        *,
        step: int,
        increment: int,
        node: int,
        dof: int,
        time: list[float],
        dt: float,
        x: Sequence[float],
    ) -> float:
        return self.value


class ElementFieldEvaluator(ABC):
    @abstractmethod
    def __call__(
        self,
        step: int,
        increment: int,
        element: int,
        time: list[float],
        dt: float,
        x: Sequence[float],
    ) -> Sequence[float]: ...


class ConstantElementField(ElementFieldEvaluator):
    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        self.magnitude = magnitude
        self.direction = direction
    def __call__(
        self,
        step: int,
        increment: int,
        element: int,
        time: list[float],
        dt: float,
        x: Sequence[float],
    ) -> Sequence[float]:
        return [self.magnitude * dir for dir in self.direction]


class GravityField(ElementFieldEvaluator):
    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        self.magnitude = magnitude
        self.direction = direction
    def __call__(
        self,
        step: int,
        increment: int,
        element: int,
        time: list[float],
        dt: float,
        x: Sequence[float],
    ) -> Sequence[float]:
        return [self.magnitude * dir for dir in self.direction]


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
class SurfaceLoad:
    load_type: int
    edge: int
    value: NDArray


@dataclass
class RobinLoad:
    edge: int
    H: NDArray
    u0: NDArray


# --------------------------------------------------------------------------
# Surface/Edge Loads
# --------------------------------------------------------------------------
