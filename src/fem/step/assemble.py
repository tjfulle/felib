from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..typing import DLoadT
from ..typing import DSLoadT
from ..typing import RLoadT
from .constraint import build_linear_constraint


class AssemblyKernel:
    def __init__(
        self,
        assemble_fun: Callable[..., tuple[NDArray, NDArray]],
        u0: NDArray,
        *,
        step: int,
        increment: int,
        time: tuple[float, float],
        dt: float,
        ddofs: NDArray,
        dvals: NDArray,
        nbcs: list[tuple[int, float]],
        dloads: DLoadT,
        dsloads: DSLoadT,
        rloads: RLoadT,
        equations: list[list],
    ) -> None:
        self.assemble_fun = assemble_fun
        self.u0 = u0
        self.step = step
        self.increment = increment
        self.time = time
        self.dt = dt
        self.ddofs = ddofs
        self.dvals = dvals  # Target Dirichlet values at end of step
        self.nbcs = nbcs
        self.dloads = dloads
        self.dsloads = dsloads
        self.rloads = rloads
        self.equations = equations

    def __call__(self, x: NDArray):
        """
        Assemble the nonlinear equilibrium system for the current Newton iterate.

        Parameters
        ----------
        x : NDArray
            Current Newton unknown vector.

            If no linear constraint equations are present:
                x = u_f
                where u_f are the free (non-Dirichlet) displacement DOFs.

            If linear constraint equations are present:
                x = [u_f, λ]
                where:
                    u_f : free displacement DOFs
                    λ   : Lagrange multipliers associated with linear constraints.

        Assembly Procedure
        ------------------
        1. Construct the full trial displacement vector:
                - Free DOFs are taken from x.
                - Dirichlet DOFs are enforced strongly using prescribed values.
                - model.u[1, :] stores the full trial displacement.

        2. Compute incremental displacement:
                du = model.u[1] - model.u[0]

            This incremental field is passed to model.assemble(...) so that
            material models receive strain increments relative to the last
            converged configuration.

        3. Assemble the full global stiffness matrix K and residual vector R:
                K, R = model.assemble(...)

            These are the full ndof-sized system including Dirichlet DOFs.

        4. Eliminate Dirichlet DOFs by extracting the reduced system:
                K_ff = K[fdofs, fdofs]
                R_f  = R[fdofs]

            where fdofs are the free DOF indices.

        Constraint Handling
        -------------------
        If linear constraint equations of the form

                C u = r

        are present, a saddle-point (augmented) system is formed:

            [ K_ff   C_f^T ] [ Δu_f ] = -[ R_f + C_f^T λ ]
            [  C_f    0    ] [ Δλ   ]   [ C u - r        ]

        where:
            C_f : constraint matrix restricted to free DOFs
            λ   : Lagrange multipliers
            g   : constraint residual (C u - r)

        The augmented Jacobian and residual are returned as:

            K_aug = [[K_ff, C_f^T],
                        [C_f,      0]]

            R_aug = [R_f + C_f^T λ, g]

        Returns
        -------
        (K_ff, R_f) : tuple[NDArray, NDArray]
            If no constraint equations are present.

        (K_aug, R_aug) : tuple[NDArray, NDArray]
            If constraint equations are present, representing the
            saddle-point system for the unknown vector [u_f, λ].

        Notes
        -----
        - Dirichlet DOFs are enforced strongly and do not appear in the
            Newton unknown vector.
        - The system with constraints is symmetric but indefinite.
        - Lagrange multipliers represent constraint reaction forces.
        """
        ndof = len(self.u0)
        fdofs = np.array(sorted(set(range(ndof)) - set(self.ddofs)))
        nf = len(fdofs)
        neq = len(self.equations) if self.equations else 0

        u = self.u0.copy()
        u[fdofs] = x[:nf]
        u[self.ddofs] = self.dvals
        du = u - self.u0

        K, R = self.assemble_fun(
            self.step,
            self.increment,
            self.time,
            self.dt,
            u,
            du,
            self.dloads,
            self.dsloads,
            self.rloads,
        )
        for dof, value in self.nbcs:
            R[dof] -= value
        R_f = R[fdofs]
        K_ff = K[np.ix_(fdofs, fdofs)]

        if neq == 0:
            return K_ff, R_f

        C, r = build_linear_constraint(ndof, self.equations)
        C_f = C[:, fdofs]
        g = np.dot(C, u) - r
        Ka = np.block([[K_ff, C_f.T], [C_f, np.zeros((neq, neq))]])
        Ra = np.hstack([R_f + np.dot(C_f.T, x[nf:]), g])
        return Ka, Ra
