from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..collections import Solution
from ..solver import DirectSolver
from .base import CompiledStep
from .constraint import build_linear_constraint
from .static import StaticStep

if TYPE_CHECKING:
    from ..model import Model
    from ..solver import SolverState


class DirectStep(StaticStep):
    def __init__(self, name: str, period: float = 1.0) -> None:
        super().__init__(name, period=period)

    def compile(self, model: "Model", parent: CompiledStep | None) -> CompiledStep:
        return CompiledDirectStep(
            name=self.name,
            parent=parent,
            period=self.period,
            dbcs=self.compile_dbcs(model),
            nbcs=self.compile_nbcs(model),
            dloads=self.compile_dloads(model),
            dsloads=self.compile_dsloads(model),
            rloads=self.compile_rloads(model),
            equations=self.compile_constraints(model),
        )


@dataclass
class CompiledDirectStep(CompiledStep):
    """
    Single linear static step.

    Performs one global assembly and a single linear solve.
    No Newton iteration is performed.
    """

    def solve(self, model: "Model") -> Solution:
        n = model.num_dof

        ddofs = np.asarray(self.ddofs, dtype=int)
        dvals = self.dvals[1, :]
        fdofs = np.array(sorted(set(range(n)) - set(ddofs)))
        nf = len(fdofs)

        neq = len(self.equations) if self.equations else 0

        # -------------------------------------------------
        # Assemble linear system
        # -------------------------------------------------

        # For linear step, incremental displacement is irrelevant.
        # We solve directly for total displacement.
        model.u[1, :] = model.u[0, :]
        model.u[1, ddofs] = dvals
        K, R = model.assemble(self, 1, [0.0, self.start], self.period, model.u[1], np.zeros(n))
        for dof, value in self.nbcs:
            R[dof] -= value

        # Reduced free system
        K_ff = K[np.ix_(fdofs, fdofs)]
        R_f = R[fdofs]

        state: "SolverState"
        solver = DirectSolver()
        if neq == 0:
            state = solver(K_ff, R_f)
        else:
            # ------------------------------------------
            # Linear constraint system
            # ------------------------------------------
            C, r = build_linear_constraint(model.num_dof, self.equations)
            C_f = C[:, fdofs]
            g = np.dot(C, model.u[1]) - r
            Ka = np.block([[K_ff, C_f.T], [C_f, np.zeros((neq, neq))]])
            Ra = np.hstack([R_f, g])
            state = solver(Ka, Ra)

        # -------------------------------------------------
        # Construct final displacement
        # -------------------------------------------------
        model.u[1, fdofs] += state.x[:nf]

        # Reassemble to compute reactions
        K, R = model.assemble(self, 1, [0.0, self.start], self.period, model.u[1], np.zeros(n))
        for dof, value in self.nbcs:
            R[dof] -= value
        react = np.zeros_like(R)
        react[ddofs] = R[ddofs]

        self.solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=model.u[1, :n].reshape((model.nnode, -1)),
            react=react.reshape((model.nnode, -1)),
            lagrange_multipliers=state.x[nf:],
            iterations=1,
        )
        return self.solution
