import numpy as np
from numpy.typing import NDArray


def build_linear_constraint(n: int, equations: list[list]) -> tuple[NDArray, NDArray]:
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
    m = len(equations)
    if not m:
        return np.empty(0), np.empty(0)

    # Build the linear constrain matrix
    C: NDArray = np.zeros(shape=(m, n), dtype=float)
    r: NDArray = np.zeros(shape=m, dtype=float)

    for i, equation in enumerate(equations):
        rhs = equation[-1]
        for j in range(0, len(equation[:-1]), 2):
            dof = int(equation[j])
            coeff = float(equation[j + 1])
            C[i, dof] = coeff
        r[i] = rhs

    return C, r
