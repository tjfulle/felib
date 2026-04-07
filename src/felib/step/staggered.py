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
      1. Thermal stage  – HeatTransferStep solves for nodal temperatures T.
      2. Mechanical stage – StaticStep solves for displacements [u, v],
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
        # Sub-steps: user configures BCs / loads on these directly
        self.thermal = HeatTransferStep(name=f"{name}-thermal", period=period)
        self.mechanical = StaticStep(name=f"{name}-mechanical", period=period, **options)

    def compile(self, model: "Model", parent: CompiledStep | None) -> "CompiledStaggeredStep":
        # Compile thermal sub-step first (inherits parent BCs / start time)
        compiled_thermal = self.thermal.compile(model, parent=parent)

        # Mechanical sub-step uses the compiled thermal step as its parent so
        # that Dirichlet inheritance and step numbering stay consistent.
        compiled_mechanical = self.mechanical.compile(model, parent=compiled_thermal)

        return CompiledStaggeredStep(
            name=self.name,
            parent=parent,
            period=self.period,
            compiled_thermal=compiled_thermal,       # type: ignore[arg-type]
            compiled_mechanical=compiled_mechanical,  # type: ignore[arg-type]
        )


@dataclass
class CompiledStaggeredStep(CompiledStep):
    """
    Compiled form of StaggeredStep.

    Holds two compiled sub-steps and orchestrates the sequential solve:
      1. Run the thermal solve  → get full DOF vector with temperatures updated.
      2. Pass that DOF vector to the mechanical solve so the element can read
         nodal temperatures and compute thermal strains.

    The ``solve`` signature matches every other CompiledStep so that
    ``Simulation.run()`` can call it without modification.
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
        Sequential thermal → mechanical solve.

        Parameters
        ----------
        fun:
            Global assembly function – ``model.assemble``.
        u0:
            Full DOF vector at the start of this step (previous converged state).
        args:
            Extra arguments forwarded to ``fun`` (typically ``(ebdata,)``).

        Returns
        -------
        u_final : NDArray
            Full DOF vector after both solves (temperatures + displacements).
        flux_final : NDArray
            Flux / reaction vector from the mechanical solve.
        """

        # ------------------------------------------------------------------
        # Stage 1: Thermal solve
        #   Solves K_TT * T = F_T for nodal temperatures.
        #   Returns the full DOF vector with T DOFs updated; [u,v] DOFs are
        #   whatever they were in u0 (unchanged by the thermal solve).
        # ------------------------------------------------------------------
        u_thermal, flux_thermal = self.compiled_thermal.solve(fun, u0, args=args)

        # ------------------------------------------------------------------
        # Stage 2: Mechanical solve
        #   The element reads T from the DOF vector to compute thermal strains
        #   ε_thermal = α (T - T_ref) I, treated as a known load.
        #   We pass u_thermal as the starting point so the element sees the
        #   updated nodal temperatures in the [u, v, T] DOF layout.
        # ------------------------------------------------------------------
        u_mechanical, flux_mechanical = self.compiled_mechanical.solve(
            fun, u_thermal, args=args
        )

        # The final DOF vector has:
        #   - T DOFs  from u_thermal  (set by Stage 1)
        #   - u,v DOFs from u_mechanical (set by Stage 2)
        # Because u_mechanical starts from u_thermal, T DOFs are preserved.
        return u_mechanical, flux_mechanical
