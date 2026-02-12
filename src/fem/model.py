from dataclasses import dataclass
from dataclasses import field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .block import ElementBlock
from .collections import DistributedLoad
from .collections import Map
from .collections import RobinLoad
from .collections import SurfaceLoad

ArrayLike = NDArray | Sequence


@dataclass
class Model:
    coords: NDArray
    connect: NDArray
    blocks: list[ElementBlock]

    num_dof: int
    dof_map: NDArray
    block_dof_map: NDArray
    node_signatures: NDArray

    node_map: Map
    elem_map: Map

    dirichlet_dofs: NDArray
    dirichlet_vals: NDArray
    neumann_dofs: NDArray
    neumann_vals: NDArray

    robin_loads: dict[int, dict[int, list[RobinLoad]]]
    surface_loads: dict[int, dict[int, list[SurfaceLoad]]]
    distributed_loads: dict[int, dict[int, list[DistributedLoad]]]

    equations: list[list[int | float]] = field(default_factory=list)

    nnode: int = field(init=False, default=-1)
    nelem: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        self.nnode = self.coords.shape[0]
        self.nelem = self.connect.shape[0]


def assemble(model: Model, u: NDArray, du: NDArray) -> tuple[NDArray, NDArray]:
    K = np.zeros((model.num_dof, model.num_dof), dtype=float)
    R = np.zeros(model.num_dof, dtype=float)
    for b, block in enumerate(model.blocks):
        ix = model.block_dof_map[b]
        bft = ix[ix != -1]
        kb, rb = block.assemble(
            u[bft],
            du[bft],
            dloads=model.distributed_loads.get(b),
            sloads=model.surface_loads.get(b),
            rloads=model.robin_loads.get(b),
        )
        K[np.ix_(bft, bft)] += kb
        R[bft] += rb
    return K, R


def build_linear_constraint(model: Model, u: NDArray, du: NDArray) -> tuple[NDArray, NDArray]:
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
    m = len(model.equations)
    if not m:
        return np.empty(0), np.empty(0)

    # Build the linear constrain matrix
    m = len(model.equations)
    n = model.num_dof

    C: NDArray = np.zeros(shape=(m, n), dtype=float)
    r: NDArray = np.zeros(shape=m, dtype=float)

    for i, equation in enumerate(model.equations):
        rhs = equation[-1]
        for j in range(0, len(equation[:-1]), 3):
            node = int(equation[j])
            dof = int(equation[j + 1])
            coeff = float(equation[j + 2])
            I = model.dof_map[node, dof]
            C[i, I] = coeff
        r[i] = rhs

    # Gather prescribed DOFs and values
    for i, row in enumerate(C):
        for dof, val in zip(model.dirichlet_dofs, model.dirichlet_vals):
            coeff = row[dof]
            if abs(coeff) > 0.0:
                r[i] -= coeff * (val - u[dof])
                row[dof] = 0.0

    return C, r
