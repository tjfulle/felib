from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .collections import Solution
from .solver import DirectSolver
from .solver import NonlinearNewtonSolver
from .typing import DLoadT
from .typing import DSLoadT
from .typing import RLoadT

if TYPE_CHECKING:
    from .model import Model
    from .solver import SolverState


@dataclass
class Step(ABC):
    name: str = ""
    parent: "Step | None" = None
    period: float = 1.0
    dbcs: list[tuple[int, float]] = field(default_factory=list)
    nbcs: list[tuple[int, float]] = field(default_factory=list)
    dsloads: DSLoadT = field(default_factory=dict)
    dloads: DLoadT = field(default_factory=dict)
    rloads: RLoadT = field(default_factory=dict)
    equations: list[list[int | float]] = field(default_factory=list)
    start: float = field(init=False, default=0.0)
    number: int = field(init=False, repr=False)
    fdofs: NDArray = field(init=False, repr=False)
    ddofs: NDArray = field(init=False, repr=False)
    dvals: NDArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.number = 1 if self.parent is None else self.parent.number + 1
        self.name = self.name or f"Step-{self.number}"
        if self.parent is not None:
            self.start = self.parent.start + self.parent.period

        my_ddofs = np.array([dof for dof, _ in self.dbcs], dtype=int)
        my_dvals = np.array([val for _, val in self.dbcs], dtype=float)

        inherited_ddofs: list[int] = []
        inherited_dvals: list[float] = []
        if self.parent:
            inherited_ddofs.extend(self.parent.ddofs)
            inherited_dvals.extend(self.parent.dvals[1, :])

        ddofs = np.array(sorted(set(inherited_ddofs) | set(my_ddofs)), dtype=int)
        dvals = np.zeros((2, len(ddofs)), dtype=float)

        if inherited_ddofs:
            mask = np.isin(ddofs, inherited_ddofs)
            dvals[0, mask] = inherited_dvals[np.searchsorted(inherited_ddofs, ddofs[mask])]
            dvals[1, mask] = dvals[0, mask]

        mask = np.isin(ddofs, my_ddofs)
        dvals[1, mask] = my_dvals[np.searchsorted(my_ddofs, ddofs[mask])]

        self.ddofs = ddofs
        self.dvals = dvals

        self._solution: Solution | None = None

    @abstractmethod
    def solve(self, model: "Model") -> Solution: ...

    @property
    def solution(self) -> Solution:
        if self._solution is None:
            raise RuntimeError("Solution has not been set")
        return self._solution

    @solution.setter
    def solution(self, arg: Solution) -> None:
        self._solution = arg


@dataclass
class DirectStep(Step):
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
        K, R = model.assemble(self, 1, [0.0, self.start], self.period, model.u[1], np.zeros(n))

        # Enforce Dirichlet DOFs strongly
        u = model.u[1].copy()
        u[ddofs] = dvals

        # Modify RHS for prescribed values:
        # R_f = R_f + K_fd * u_d
        R_mod = R.copy()
        R_mod -= np.dot(K, u)

        # Reduced free system
        K_ff = K[np.ix_(fdofs, fdofs)]
        R_f = R_mod[fdofs]

        state: "SolverState"
        solver = DirectSolver()
        if neq == 0:
            state = solver(K_ff, R_f)
        else:
            # ------------------------------------------
            # Linear constraint system
            # ------------------------------------------
            C, r = model.build_linear_constraint(self, u, du=np.zeros_like(u))
            C_f = C[:, fdofs]
            g = np.dot(C, u) - r
            Ka = np.block([[K_ff, C_f.T], [C_f, np.zeros((neq, neq))]])
            Ra = np.hstack([R_f, g])
            state = solver(Ka, Ra)

        # -------------------------------------------------
        # Construct final displacement
        # -------------------------------------------------

        model.u[1, :] = model.u[0, :]
        model.u[1, fdofs] = state.x[:nf]
        model.u[1, ddofs] = dvals

        # Reassemble to compute reactions
        K, R = model.assemble(self, 1, [0.0, self.start], self.period, model.u[1], np.zeros(n))
        react = np.zeros_like(R)
        react[ddofs] = R[ddofs]

        return Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=model.u[1, :n].reshape((model.nnode, -1)),
            react=react.reshape((model.nnode, -1)),
            lagrange_multipliers=state.x[nf:],
            iterations=1,
        )


@dataclass
class StaticStep(Step):
    solver_options: dict[str, Any] = field(default_factory=dict)

    def solve(self, model: "Model") -> Solution:
        ddofs = self.ddofs
        dvals = self.dvals[1, :]  # Target Dirichlet values at end of step
        fdofs = np.array(sorted(set(range(model.num_dof)) - set(ddofs)))
        nf = len(fdofs)
        neq = len(self.equations) if self.equations else 0

        def fun(x: NDArray, model: "Model", step: "StaticStep"):
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

            model : Model
                Finite element model containing the previously converged state
                in model.u[0, :] and internal material variables.

            step : StaticStep
                Current load step definition.

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
            model.u[1, fdofs] = x[:nf]
            model.u[1, ddofs] = dvals
            du = model.u[1] - model.u[0]

            time = [0, step.start]
            dt = step.period
            K, R = model.assemble(step, 1, time, dt, model.u[1], du)
            R_f = R[fdofs]
            K_ff = K[np.ix_(fdofs, fdofs)]

            if neq == 0:
                return K_ff, R_f

            C, r = model.build_linear_constraint(step, model.u[1], du)
            C_f = C[:, fdofs]
            g = np.dot(C, model.u[1]) - r
            Ka = np.block([[K_ff, C_f.T], [C_f, np.zeros((neq, neq))]])
            Ra = np.hstack([R_f + np.dot(C_f.T, x[nf:]), g])
            return Ka, Ra

        x0 = model.u[0, fdofs]
        if neq > 0:
            x0 = np.hstack([x0, np.zeros(neq)])

        n = model.num_dof
        solver = NonlinearNewtonSolver()
        state = solver(
            fun,
            model.u[0, fdofs],
            args=(model, self),
            atol=self.solver_options.get("atol"),
            rtol=self.solver_options.get("rtol"),
            maxiter=self.solver_options.get("maxiter"),
        )
        model.u[1, fdofs] = state.x[:nf]
        model.u[1, ddofs] = dvals
        K, R = model.assemble(
            self, 1, [0, self.start], self.period, model.u[1], model.u[1] - model.u[0]
        )
        react = np.zeros_like(R)
        react[ddofs] = R[ddofs]
        solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=model.u[1, :n].reshape((model.nnode, -1)),
            react=react.reshape((model.nnode, -1)),
            lagrange_multipliers=state.x[nf:],
            iterations=state.iterations,
        )

        return solution
