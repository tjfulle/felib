from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..collections import NodeData
from ..collections import Solution
from ..constants import NodeVariable
from ..typing import DLoadT
from ..typing import DSLoadT
from ..typing import RLoadT

if TYPE_CHECKING:
    from ..dof_manager import DOFManager
    from ..model import Model


class Step(ABC):
    def __init__(self, name: str, ndim: int, period: float = 1.0) -> None:
        self.name = name
        self.ndim = ndim
        self.period = period
        self.metadata: dict[str, dict] = defaultdict(dict)

    @abstractmethod
    def node_variables(self) -> list[NodeVariable]: ...

    @abstractmethod
    def compile(
        self, model: "Model", dof_manager: "DOFManager", parent: "CompiledStep | None"
    ) -> "CompiledStep": ...


@dataclass
class CompiledStep(ABC):
    name: str = ""
    parent: "CompiledStep | None" = None
    period: float = 1.0
    dbcs: list[tuple[int, float]] = field(default_factory=list)
    nbcs: list[tuple[int, float]] = field(default_factory=list)
    dloads: DLoadT = field(default_factory=dict)
    dsloads: DSLoadT = field(default_factory=dict)
    rloads: RLoadT = field(default_factory=dict)
    equations: list[list[int | float]] = field(default_factory=list)
    node_variables: list[str] = field(default_factory=list)
    start: float = field(init=False, default=0.0)
    number: int = field(init=False, repr=False)
    fdofs: NDArray = field(init=False, repr=False)
    ddofs: NDArray = field(init=False, repr=False)
    dvals: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.number = 1 if self.parent is None else self.parent.number + 1
        self.name = self.name or f"Step-{self.number}"
        if self.parent is not None:
            self.start = self.parent.start + self.parent.period

        my_ddofs = np.array([dof for dof, _ in self.dbcs], dtype=int)
        my_dvals = np.array([val for _, val in self.dbcs], dtype=float)

        inherited_ddofs: list[int] = []
        inherited_dvals: list[float] = []
        if self.parent:
            inherited_ddofs.extend(self.parent.ddofs)
            inherited_dvals.extend(self.parent.dvals[1, :])

        ddofs = np.array(sorted(set(inherited_ddofs) | set(my_ddofs)), dtype=int)
        dvals = np.zeros((2, len(ddofs)), dtype=float)

        if inherited_ddofs:
            mask = np.isin(ddofs, inherited_ddofs)
            dvals[0, mask] = inherited_dvals[np.searchsorted(inherited_ddofs, ddofs[mask])]
            dvals[1, mask] = dvals[0, mask]

        mask = np.isin(ddofs, my_ddofs)
        dvals[1, mask] = my_dvals[np.searchsorted(my_ddofs, ddofs[mask])]

        self.ddofs = ddofs
        self.dvals = dvals

        self._solution: Solution | None = None

    @abstractmethod
    def solve(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        u0: NDArray,
        ndata: NodeData,
        args: tuple[Any, ...] = (),
    ) -> NDArray: ...

    @property
    def solution(self) -> Solution:
        if self._solution is None:
            raise RuntimeError("Solution has not been set")
        return self._solution

    @solution.setter
    def solution(self, arg: Solution) -> None:
        self._solution = arg
