from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import Sequence
from typing import Type

import numpy as np
from numpy.typing import NDArray

from .typing import DLoadT
from .typing import DSLoadT
from .typing import RLoadT

if TYPE_CHECKING:
    from .constants import NodeVariable
    from .dof_manager import DOFManager
    from .element import ReferenceElement


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


class ElementBlockData:
    def __init__(self, block: str, nelem: int, ngauss: int, elem_var_names: list[str]) -> None:
        self.block = block
        shape = (2, nelem, ngauss, len(elem_var_names))
        self.data: NDArray = np.zeros(shape, dtype=float)
        self.var_names = list(elem_var_names)

    def advance_state(self) -> None:
        self.data[0, :] = self.data[1, :]

    @property
    def scratch(self) -> NDArray:
        return self.data[1]

    def sync(self) -> None:
        self.data[1] = self.data[0]

    def items(self, projection: str = "centroid") -> Generator[tuple[str, NDArray], None, None]:
        if projection == "centroid":
            centroid_data = self.data[0].mean(axis=1)
            for i, name in enumerate(self.var_names):
                yield name, centroid_data[:, i]
        else:
            raise NotImplementedError(projection)


class NodeData:
    """
    Container for all node-centered data in the simulation.

    Attributes
    ----------
    data : np.ndarray
        Node variable storage, shape (2, nnode, nvars)
        [0] = converged, [1] = scratch space.
    var_names : list[str]
        Names of all variables (DOFs first, then extra node variables)
    """

    def __init__(
        self, dof_manager: "DOFManager", node_vars: list["NodeVariable"] | None = None
    ) -> None:
        """
        Parameters
        ----------
        dof_manager : DOFManager
            Provides node_freedom_table, dof_map, and node_dof_cols.
        node_vars : list[str], optional
            Names of additional node variables not associated with DOFs.
        """
        self.nnode: int = dof_manager.nnode
        self.dof_manager: "DOFManager" = dof_manager

        # Build variable names: DOFs first, then extras
        node_vars = node_vars or []
        self.var_names = [dof_type.label for dof_type in dof_manager.node_dof_types]
        for node_var in node_vars:
            names = node_var.labels(dof_manager.ndim)
            for name in names:
                if name not in self.var_names:
                    self.var_names.append(name)
        self.nvars = len(self.var_names)

        self.vectors: dict[str, list[int]] = {}
        self.exodus_labels: list[str] = []
        for i, name in enumerate(self.var_names):
            base, sep, comp = name.partition(".")
            if sep:
                self.exodus_labels.append(f"displ{comp}" if base == "u" else f"{base}{comp}")
                self.vectors.setdefault(base, []).append(i)
            else:
                self.exodus_labels.append(name)

        # Allocate storage: [converged, scratch, nnode, nvars]
        nnode = dof_manager.nnode
        self.data = np.zeros((2, nnode, self.nvars), dtype=float)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.gather(name)

    def __setitem__(self, name: str, values: np.ndarray) -> None:
        return self.scatter(name, values)

    def gather_dofs(self) -> np.ndarray:
        """
        Gather all active DOFs into a flat vector suitable for assembly.

        Returns
        -------
        dofs : np.ndarray
            Global DOF vector of length dof_manager.ndof
        """
        ndof = self.dof_manager.ndof
        dofs = np.zeros(ndof, dtype=float)

        for n in range(self.nnode):
            for local_idx, gdof in enumerate(self.dof_manager.dof_map[n]):
                if gdof >= 0:
                    dof_type = self.dof_manager.node_freedom_type(local_idx)
                    col = self.dof_manager.node_dof_cols[dof_type]
                    dofs[gdof] = self.data[0, n, col]

        return dofs

    def scatter_dofs(self, dofs: np.ndarray) -> None:
        """
        Scatter global DOF vector back into NodeData.

        Parameters
        ----------
        dofs : np.ndarray
            Global DOF vector (length = dof_manager.ndof)
        """
        for n in range(self.nnode):
            for local_idx, gdof in enumerate(self.dof_manager.dof_map[n]):
                if gdof >= 0:
                    dof_type = self.dof_manager.node_freedom_type(local_idx)
                    col = self.dof_manager.node_dof_cols[dof_type]
                    self.data[1, n, col] = dofs[gdof]

    def gather(self, name: str) -> np.ndarray:
        """
        Gather a scalar node variable (DOF or extra) across all nodes.

        Returns a vector of length nnode.
        """
        d = self.data[0]
        if name in self.vectors:
            return d[:, self.vectors[name]]
        else:
            col = self.var_names.index(name)
            return d[:, col]

    def scatter(self, name: str, values: np.ndarray) -> None:
        """
        Scatter a scalar node variable back into NodeData.
        """
        d = self.data[1]
        if name in self.vectors:
            d[:, self.vectors[name]] = values
        else:
            col = self.var_names.index(name)
            self.data[1, :, col] = values

    @property
    def scratch(self) -> NDArray:
        return self.data[1]

    def advance_state(self) -> None:
        self.data[0, :] = self.data[1, :]

    def sync(self) -> None:
        self.data[1] = self.data[0]

    def items(self, exodus_labels: bool = False) -> Generator[tuple[str, NDArray], None, None]:
        for i, name in enumerate(self.var_names):
            if exodus_labels:
                name = self.exodus_labels[i]
            yield name, self.data[0, :, i]


class Field(ABC):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> Any: ...


class ScalarField(Field):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> float: ...


class ConstantScalarField(ScalarField):
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
        return self.value


class VectorField(Field):
    @abstractmethod
    def __call__(self, x: Sequence[float], time: Sequence[float]) -> NDArray: ...


class ConstantVectorField(VectorField):
    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        vec = np.asarray(direction, dtype=float)
        vec /= np.linalg.norm(vec)
        self.value = magnitude * vec

    def __call__(self, x: Sequence[float], time: Sequence[float]) -> NDArray:
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
    _scale: float

    @property
    @abstractmethod
    def field(self) -> Field: ...

    @abstractmethod
    def __call__(self, *args: Any) -> Any: ...

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, arg: float) -> None:
        self._scale = arg


class DistributedLoad(Load):
    """
    Load integrated over element domain (volume or length).
    """

    def __init__(self, field: Field) -> None:
        self._field: Field = field
        self._scale = 1.0

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
        x: Sequence[float],
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
        return self.scale * self._field(x, time)


class DistributedSurfaceLoad(Load):
    """
    Load integrated over element boundary (codim 1).
    """

    def __init__(self, field: Field) -> None:
        self._field = field
        self._scale = 1.0

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
        x: Sequence[float],
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
        return self.scale * self._field(x, time)


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
        x: Sequence[float],
        n: NDArray,
    ) -> NDArray:
        return -self.scale * self.field(x, time) * n


class HeatSource(DistributedLoad):
    """
    Mass-proportional body force.
    """

    def __init__(self, field: Field) -> None:
        super().__init__(field=field)

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
        x: Sequence[float],
    ) -> NDArray:
        return self.scale * np.array([self.field(x, time)])


class HeatFlux(DistributedSurfaceLoad):
    """
    Heat flux applied on element surfaces.
    """

    def __init__(self, magnitude: float, direction: Sequence[float]) -> None:
        field = ConstantVectorField(magnitude, direction)
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
        x: Sequence[float],
        n: NDArray,
    ) -> NDArray:
        # return -np.array([np.dot(self.field(x, time), n)])
        return -self.scale * np.array([np.dot(self.field(x, time), n)])


@dataclass
class Solution:
    stiff: NDArray
    force: NDArray
    dofs: NDArray
    react: NDArray
    iterations: int = field(default=1)
    lagrange_multipliers: NDArray = field(default_factory=lambda: np.empty((0,)))


class RegionSelector(ABC): ...


@dataclass
class Node:
    lid: int
    gid: int
    x: Sequence[float]
    normal: list[float] = field(default_factory=list)
    on_boundary: bool = field(default=False)


@dataclass
class Element:
    lid: int
    gid: int
    x: Sequence[float]  # element centroid
    block: str | None = None


@dataclass
class Edge:
    element: int
    edge: int
    x: Sequence[float]
    normal: list[float]


@dataclass
class Side:
    element: Element
    side: int  # global side number -> lid + 1
    x: Sequence[float]
    normal: list[float]
    on_boundary: bool


class NodeSelector(RegionSelector):
    @abstractmethod
    def __call__(self, node: Node) -> bool: ...


class NodeXSelector(NodeSelector):
    def __init__(self, nodes: list[int]) -> None:
        self.nodes = nodes

    def __call__(self, node: Node) -> bool:
        return node.gid in self.nodes


class ElementSelector(RegionSelector):
    @abstractmethod
    def __call__(self, element: Element) -> bool: ...


class ElementXSelector(ElementSelector):
    def __init__(self, elements: list[int]) -> None:
        self.elements = elements

    def __call__(self, element: Element) -> bool:
        return element.gid in self.elements


class SideSelector(RegionSelector):
    @abstractmethod
    def __call__(self, node: Side) -> bool: ...


class SideXSelector(SideSelector):
    def __init__(self, sides: list[Sequence[int]]) -> None:
        self.sides: list[tuple[int, ...]] = [tuple(side[:2]) for side in sides]

    def __call__(self, side: Side) -> bool:
        return (side.element.gid, side.side) in self.sides


@dataclass
class BlockSpec:
    name: str
    cell_type: Type["ReferenceElement"]
    region: Callable[[Element], bool] | ElementSelector


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
