from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..collections import Solution
from .base import CompiledStep
from .base import Step
from .heat_transfer import CompiledHeatTransferStep
from .heat_transfer import HeatTransferStep
from .static import CompiledStaticStep
from .static import StaticStep

if TYPE_CHECKING:
    from ..model import Model


class StaggeredStep(Step):
    """
    Staggered (sequential) thermo-mechanical step.

    Solves in two stages per increment:
      1. Thermal stage  - HeatTransferStep solves for nodal temperatures T.
      2. Mechanical stage - StaticStep solves for displacements [u, v],
         with thermal strains computed inside the element from the
         temperatures found in stage 1.

    Usage
    -----
    Create via ``Simulation.staggered_step()``, then configure the
    two sub-steps through the ``.thermal`` and ``.mechanical`` attributes::

        step = sim.staggered_step(name="thermo-mech")
        step.thermal.temperature(nodes="left", value=100.0)
        step.mechanical.boundary(nodes="bottom", dofs=[Ux, Uy], value=0.0)
    """

    def __init__(self, name: str, period: float = 1.0, **options: Any) -> None:
        super().__init__(name=name, period=period)
        self.thermal = HeatTransferStep(name=f"{name}-thermal", period=period)
        self.mechanical = StaticStep(name=f"{name}-mechanical", period=period, **options)

    def compile(self, model: "Model", parent: CompiledStep | None) -> "CompiledStaggeredStep":
        # Each sub-step inherits from the same parent (not chained to each other)
        # This prevents thermal Dirichlet BCs from polluting the mechanical DOFs
        compiled_thermal = self.thermal.compile(model, parent=parent)
        compiled_mechanical = self.mechanical.compile(model, parent=parent)

        return CompiledStaggeredStep(
            name=self.name,
            parent=parent,
            period=self.period,
            compiled_thermal=compiled_thermal,
            compiled_mechanical=compiled_mechanical,
        )


@dataclass
class CompiledStaggeredStep(CompiledStep):
    """
    Compiled form of StaggeredStep.

    Runs thermal solve first, then passes updated temperatures to the
    mechanical solve so the element can compute thermal strains.
    """

    compiled_thermal: CompiledHeatTransferStep = field(default_factory=CompiledHeatTransferStep)
    compiled_mechanical: CompiledStaticStep = field(default_factory=CompiledStaticStep)

    def solve(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        u0: NDArray,
        args: tuple[Any, ...] = (),
    ) -> tuple[NDArray, NDArray]:
        """
        Sequential thermal -> mechanical solve.

        Stage 1: Solve for temperatures T.
        Stage 2: Solve for displacements [u, v] using updated T.
        """
        # Stage 1: thermal solve - updates T DOFs
        u_thermal, flux_thermal = self.compiled_thermal.solve(fun, u0, args=args)

        # Stage 2: mechanical solve - reads T from u_thermal to compute thermal strains
        u_mechanical, flux_mechanical = self.compiled_mechanical.solve(
            fun, u_thermal, args=args
        )

        return u_mechanical, flux_mechanical
