from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..collections import NodeData
from ..collections import Solution
from ..solver import DirectSolver
from .assemble import AssemblyKernel
from .base import CompiledStep
from .static import StaticStep

if TYPE_CHECKING:
    from ..dof_manager import DOFManager
    from ..model import Model


class DirectStep(StaticStep):
    def __init__(self, name: str, ndim: int, period: float = 1.0) -> None:
        super().__init__(name, ndim=ndim, period=period)

    def compile(
        self, model: "Model", dof_manager: "DOFManager", parent: CompiledStep | None
    ) -> CompiledStep:
        return CompiledDirectStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.compile_dbcs(model, dof_manager),
            nbcs=self.compile_nbcs(model, dof_manager),
            dloads=self.compile_dloads(model),
            dsloads=self.compile_dsloads(model),
            rloads=self.compile_rloads(model),
            equations=self.compile_constraints(model, dof_manager),
        )


@dataclass
class CompiledDirectStep(CompiledStep):
    """
    Single linear static step.

    Performs one global assembly and a single linear solve.
    No Newton iteration is performed.
    """

    def solve(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        u0: NDArray,
        ndata: NodeData,
        args: tuple[Any, ...] = (),
    ) -> NDArray:
        ddofs = self.ddofs
        ndof = len(u0)
        fdofs = np.array(sorted(set(range(ndof)) - set(ddofs)))
        nf = len(fdofs)
        neq = len(self.equations) if self.equations else 0

        x0 = u0[fdofs]
        if neq > 0:
            x0 = np.hstack([x0, np.zeros(neq)])
        increment = 1
        time = (0.0, self.start)
        dt = self.period
        kernel = AssemblyKernel(
            fun,
            u0,
            step=self.number,
            increment=increment,
            time=time,
            dt=dt,
            ddofs=ddofs,
            dvals=self.dvals[1, :],
            nbcs=self.nbcs,
            dloads=self.dloads,
            dsloads=self.dsloads,
            rloads=self.rloads,
            equations=self.equations,
            args=args,
        )
        solver = DirectSolver()
        state = solver(kernel, x0)

        # -------------------------------------------------
        # Construct final displacement
        # -------------------------------------------------
        u = u0.copy()
        u[fdofs] = state.x[:nf]
        u[ddofs] = self.dvals[1, :]

        R = kernel.resid
        K = kernel.stiff
        react = np.zeros_like(R)
        react[ddofs] = R[ddofs]

        self.solution = Solution(
            stiff=K[:ndof, :ndof],
            force=R[:ndof],
            dofs=u[:ndof],
            react=react,
            lagrange_multipliers=state.x[nf:],
            iterations=1,
        )

        flux = np.dot(K[:ndof, :ndof], u[:ndof])
        #        ndata[NodeVariable.Fx] = flux[0::2]
        #        ndata[NodeVariable.Fy] = flux[1::2]
        return u[:ndof]
