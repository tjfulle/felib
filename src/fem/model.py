from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import Map
from .mesh import Mesh
from .step import Step

ArrayLike = NDArray | Sequence


@dataclass
class Model:
    mesh: Mesh
    blocks: list[ElementBlock]

    num_dof: int
    dof_map: NDArray
    block_dof_map: NDArray
    node_signatures: NDArray

    steps: list[Step] = field(default_factory=list)
    u: NDArray = field(init=False)
    R: NDArray = field(init=False)
    ndata: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.u = np.zeros((2, self.num_dof), dtype=float)
        self.R = np.zeros((2, self.num_dof), dtype=float)
        self.ndata = np.zeros((2, self.coords.shape[0], self.coords.shape[1]), dtype=float)

    @property
    def nnode(self) -> int:
        return self.mesh.coords.shape[0]

    @property
    def nelem(self) -> int:
        return self.mesh.coords.shape[0]

    @property
    def node_map(self) -> Map:
        return self.mesh.node_map

    @property
    def elem_map(self) -> Map:
        return self.mesh.elem_map

    @property
    def coords(self) -> NDArray:
        return self.mesh.coords

    @property
    def connect(self) -> NDArray:
        return self.mesh.connect

    @property
    def elemsets(self) -> dict[str, list[int]]:
        return self.mesh.elemsets

    @property
    def nodesets(self) -> dict[str, list[int]]:
        return self.mesh.nodesets

    @property
    def sidesets(self) -> dict[str, list[tuple[int, int]]]:
        return self.mesh.sidesets

    @property
    def block_elem_map(self) -> dict[int, int]:
        return self.mesh.block_elem_map

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def solve(self) -> None:
        for step in self.steps:
            step.solution = step.solve(self)

    def assemble(
        self, step: Step, increment: int, time: Sequence[float], dt: float, u: NDArray, du: NDArray
    ) -> tuple[NDArray, NDArray]:
        K = np.zeros((self.num_dof, self.num_dof), dtype=float)
        R = np.zeros(self.num_dof, dtype=float)
        for b, block in enumerate(self.blocks):
            ix = self.block_dof_map[b]
            bft = ix[ix != -1]
            kb, rb = block.assemble(
                step.number,
                increment,
                time,
                dt,
                u[bft],
                du[bft],
                dloads=step.dloads.get(b),
                dsloads=step.dsloads.get(b),
                rloads=step.rloads.get(b),
            )
            K[np.ix_(bft, bft)] += kb
            R[bft] += rb
        return K, R

    def build_linear_constraint(
        self, step: Step, u: NDArray, du: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Enforce homogeneous linear constraints using Lagrange multiplier method.

        Procuedure
        ----------

        The standard augmented Lagrange system for a set of linear constraints

            C.u = r

        is written as

            ⎡ K   C.T⎤ ⎧ u ⎫   ⎧ F ⎫
            ⎢        ⎥ ⎨   ⎬ = ⎨   ⎬
            ⎣ C    0 ⎦ ⎩ 𝜆 ⎭   ⎩ r ⎭

        where:
        - K is the global stiffness
        - C is the constraint matrix
        - 𝜆 are the Lagrange multipliers enforcing the constraings
        - F is the external force vector
        - r is the rhs of the constraint.

        If any DOFs participating in the constraint equations are prescribed (known), they must be
        eliminated before assembling the augmented system.

        For example, if

            u_1 - u_5 = 0

        and u_1 is prescribed (∆), this becomes:

            -u_5 = -∆

        The corresponding row of C has the column for DOF 1 zeroed and r modified by -C[i, 1] * ∆

        The augmented system becomes

            ⎡ K_ff   C_f.T⎤ ⎧ u_f ⎫   ⎧ F_f ⎫
            ⎢             ⎥ ⎨     ⎬ = ⎨     ⎬
            ⎣ C_f     0   ⎦ ⎩  𝜆  ⎭   ⎩  r  ⎭

        """
        m = len(step.equations)
        if not m:
            return np.empty(0), np.empty(0)

        # Build the linear constrain matrix
        m = len(step.equations)
        n = self.num_dof

        C: NDArray = np.zeros(shape=(m, n), dtype=float)
        r: NDArray = np.zeros(shape=m, dtype=float)

        for i, equation in enumerate(step.equations):
            rhs = equation[-1]
            for j in range(0, len(equation[:-1]), 2):
                dof = int(equation[j])
                coeff = float(equation[j + 1])
                C[i, dof] = coeff
            r[i] = rhs

        # Gather prescribed DOFs and values
        for i, row in enumerate(C):
            for dof, value in step.dbcs:
                coeff = row[dof]
                if abs(coeff) > 0.0:
                    r[i] -= coeff * (value - u[dof])
                    row[dof] = 0.0

        return C, r
