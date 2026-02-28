from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generator
from typing import Sequence

from numpy.typing import NDArray

from ..collections import DistributedLoad
from ..collections import DistributedSurfaceLoad
from ..collections import RobinLoad
from ..material import Material


class Element(ABC):
    """
    Abstract base class for all finite elements.

    Defines the required interface for element-level computations,
    including stiffness assembly, state updates, and integration rules.

    All concrete element types must implement required abstract methods.
    """

    @property
    @abstractmethod
    def node_freedom_table(self) -> list[tuple[int, ...]]:
        """
        Node freedom table defining active DOFs per node.

        Returns:
            A list where each entry corresponds to a node and contains
            a tuple of flags indicating active degrees of freedom.
        """
        ...

    @property
    @abstractmethod
    def dof_per_node(self) -> int:
        """
        Number of degrees of freedom per node.

        Returns:
            Integer number of DOFs associated with each node.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Spatial dimensionality of the element.

        Returns:
            Integer spatial dimension (e.g., 2 or 3).
        """
        ...

    @property
    @abstractmethod
    def nnode(self) -> int:
        """
        Number of nodes in the element.

        Returns:
            Integer node count.
        """
        ...

    @property
    @abstractmethod
    def npts(self) -> int:
        """
        Number of integration (Gauss) points.

        Returns:
            Integer number of quadrature points.
        """
        ...

    @abstractmethod
    def history_variables(self) -> list[str]:
        """
        Names of history/state variables stored per integration point.

        Returns:
            List of variable names corresponding to entries stored in pdata.
        """
        ...

    @abstractmethod
    def integration_points(self) -> Generator[tuple[Any, Any], None, None]:
        """
        Generator over integration points.

        Yields:
            Tuples of (weight, local_coordinate).
        """
        ...

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
        Update material state at an integration point.

        Args:
            material: Material model instance.
            step: Step number.
            increment: Increment number within step.
            time: Time history sequence.
            dt: Time increment.
            eleno: Element number.
            p: Element nodal coordinates.
            e: Nodal dof values.
            e: Current strain.
            de: Strain rate.
            hsv: History variable array for this integration point.

        Returns:
            Tuple of (tangent matrix D, stress vector s).
        """
        ...

    @abstractmethod
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
        Evaluate element stiffness matrix and residual vector.

        Performs numerical integration over the element volume
        and includes contributions from distributed loads,
        surface loads, and Robin boundary terms.

        Args:
            material: Material model.
            step: Step number.
            increment: Increment index.
            time: Time history.
            dt: Time increment.
            eleno: Element number.
            p: Nodal coordinates.
            u: Element displacement vector.
            du: Displacement increment vector.
            pdata: Integration point state data.
            dloads: Volume distributed loads.
            dsloads: Surface distributed loads.
            rloads: Robin boundary loads.

        Returns:
            Tuple of (element stiffness matrix ke, element residual vector re).
        """
        ...
