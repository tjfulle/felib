from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import ElementBlockData
from .collections import NodeData
from .constants import NodeVariable
from .dof_manager import DOFManager
from .io import ExodusFileWriter
from .step import CompiledStep
from .step import DirectStep
from .step import HeatTransferStep
from .step import StaticStep
from .step import ExplicitStep
from .step import Step
from .typing import DLoadT
from .typing import DSLoadT
from .typing import RLoadT

if TYPE_CHECKING:
    from .model import Model


class Simulation:
    def __init__(self, model: "Model") -> None:
        self.model: "Model" = model
        self.model.freeze()
        self.dof_manager: DOFManager = DOFManager(model)
        self.steps: list[Step] = []
        self.csteps: list[CompiledStep] = []

        self.ndata: NodeData
        self.ebdata: list[ElementBlockData] = []
        b: ElementBlock
        for b in self.model.blocks:
            ebd = ElementBlockData(b.name, b.nelem, b.element.npts, b.element_variable_names())
            self.ebdata.append(ebd)

    def advance_state(self) -> None:
        """
        Advance the state from current solution to previous solution.

        Copies the contents of self.u[1] -> self.u[0]
        and self.R[1] -> self.R[0], preparing for next step.
        """
        self.ndata.advance_state()
        for d in self.ebdata:
            d.advance_state()

    def static_step(
        self, name: str | None = None, period: float = 1.0, **options: Any
    ) -> StaticStep:
        name = name or f"step-{len(self.steps)}"
        step = StaticStep(name=name, ndim=self.model.ndim, period=period, **options)
        self.steps.append(step)
        return step
    
    def explicit_step(
        self, name: str | None = None, period: float = 1.0, **options: Any
    ) -> ExplicitStep:
        name = name or f"step-{len(self.steps)}"
        step = ExplicitStep(name=name, ndim=self.model.ndim, period=period, **options)
        self.steps.append(step)
        return step

    def direct_step(self, name: str | None = None, period: float = 1.0) -> DirectStep:
        name = name or f"step-{len(self.steps)}"
        step = DirectStep(name=name, ndim=self.model.ndim, period=period)
        self.steps.append(step)
        return step

    def heat_transfer_step(self, name: str | None = None, period: float = 1.0) -> HeatTransferStep:
        name = name or f"step-{len(self.steps)}"
        step = HeatTransferStep(name=name, ndim=self.model.ndim, period=period)
        self.steps.append(step)
        return step

    def allocate_node_storage(self) -> None:
        node_vars: list[NodeVariable] = []
        for step in self.steps:
            for var in step.node_variables():
                if var not in node_vars:
                    node_vars.append(var)
        self.ndata = NodeData(self.dof_manager, node_vars=node_vars)

    def run(self) -> None:
        """
        Run through all analysis steps and solve.

        For each step, triggers CompiledStep.solve(), advances state, and writes results to
        the Exodus output file.
        """
        self.allocate_node_storage()
        file = ExodusFileWriter(self)
        parent: CompiledStep | None = None
        for i, step in enumerate(self.steps):
            cstep = step.compile(self.model, self.dof_manager, parent=parent)
            u0 = self.ndata.gather_dofs()
            u = cstep.solve(
                assemble, u0, self.ndata, args=(self.dof_manager, self.model.blocks, self.ebdata)
            )
            if not isinstance(step, ExplicitStep):
                self.ndata.scatter_dofs(u)

            self.advance_state()
            file.update(i + 1, cstep.start, cstep.period, self.ndata, self.ebdata)
            parent = cstep
            self.csteps.append(cstep)


def assemble(
    step: int,
    increment: int,
    time: Sequence[float],
    dt: float,
    u: NDArray,
    du: NDArray,
    dloads: DLoadT,
    dsloads: DSLoadT,
    rloads: RLoadT,
    dofs: DOFManager,
    blocks: list[ElementBlock],
    ebdata: list[ElementBlockData],
) -> tuple[NDArray, NDArray]:
    """
    Global matrix and residual assembly.

    Calls ElementBlock.assemble() for each block, inserting into global stiffness
    matrix and force vector.

    Args:
        step: Current analysis step.
        increment: Sub-increment index.
        time: Current time history.
        dt: Time step size.
        u: Current displacement vector.
        du: Displacement increment vector.

    Returns:
        Tuple of (K_global, R_global)
    """
    K = np.zeros((dofs.size, dofs.size), dtype=float)
    R = np.zeros(dofs.size, dtype=float)
    for b, block in enumerate(blocks):
        bft = dofs.block_freedom_table(b)
        kb, rb = block.assemble(
            step,
            increment,
            time,
            dt,
            u[bft],
            du[bft],
            ebdata[b],
            dloads=dloads.get(b),
            dsloads=dsloads.get(b),
            rloads=rloads.get(b),
        )
        K[np.ix_(bft, bft)] += kb
        R[bft] += rb
    return K, R
