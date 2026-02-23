from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .collections import Solution

if TYPE_CHECKING:
    from .model import Model
    from .step import Step


class Solver(ABC):
    @abstractmethod
    def __call__(self, model: "Model", step: "Step") -> Solution: ...


class DirectSolver(Solver):
    def __call__(self, model: "Model", step: "Step") -> Solution:
        """Solve the 2D plane stress/strain problem"""
        n = model.num_dof
        m = len(step.equations)
        u = np.zeros(n)
        du = np.zeros(n)
        lam = np.zeros(m)

        K: NDArray = np.zeros((n + m, n + m), dtype=float)
        R: NDArray = np.zeros(n + m, dtype=float)

        time = [step.start, 0.0]
        dt = step.period

        ddofs, dvals = model.evaluate_dirichlet_bcs(step)
        fdofs = np.array(sorted(set(range(n)).difference(ddofs)), dtype=int)

        K[:n, :n], R[:n] = model.assemble(step, 1, time, dt, u, du)

        if step.equations:
            C, r = model.build_linear_constraint(step, u, du)
            K[:n, n:] = C.T
            K[n:, :n] = C
            R[n:] = r

        # Condense out Dirichlet DOFs and solve condensed system
        delta = np.zeros(n + m)
        delta[fdofs] = np.linalg.solve(
            K[np.ix_(fdofs, fdofs)], -R[fdofs] - np.dot(K[np.ix_(fdofs, ddofs)], dvals)
        )

        u[ddofs] = dvals
        u[fdofs] += delta[fdofs]
        if m:
            lam = delta[n:]

        react = np.zeros_like(R[:n])
        K[:n, :n], R[:n] = model.assemble(step, 1, time, dt, u, du)
        react[ddofs] = np.dot(K[np.ix_(ddofs, ddofs)], u[ddofs]) - R[:n][ddofs]

        solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=u[:n].reshape((model.nnode, -1)),
            react=react[:n].reshape((model.nnode, -1)),
            lagrange_multipliers=lam,
            iterations=1,
        )
        return solution


class NonlinearNewtonSolver(Solver):
    def __init__(self, **options: Any) -> None:
        self.atol: float = options.get("absolute tolerance", -1.0)
        self.rtol: float = options.get("relative tolerance", 1e-8)
        self.maxiter: int = options.get("max iterations", 25)

    def __call__(self, model: "Model", step: "Step") -> Solution:
        n = model.num_dof
        m = len(step.equations)

        u = model.u[0, :].copy()
        du = np.zeros_like(u, dtype=float)
        lam = np.zeros(m, dtype=float)

        K: NDArray = np.zeros((n + m, n + m), dtype=float)
        R: NDArray = np.zeros(n + m, dtype=float)
        R0 = 1.0

        it = 0
        time = [0.0, step.start]
        dt = step.period

        fac: float = 1.0
        ddofs = step.ddofs
        fdofs = np.array(sorted(set(range(model.num_dof)) - set(ddofs)))

        while it < self.maxiter:

            it += 1

            K.fill(0.0)
            R.fill(0.0)

            K[:n, :n], R[:n] = model.assemble(step, 1, time, dt, u, du)

            if step.equations:
                C, r = model.build_linear_constraint(step, u, du)
                K[:n, n:] = C.T
                K[n:, :n] = C
                R[n:] = r

            if it == 1:
                R0 = float(np.linalg.norm(R[fdofs]))
                if self.atol < 0.0:
                    self.atol = 1e-8 * R0

            # Condense out Dirichlet DOFs and solve condensed system
            dvals = step.dvals[0, :] + fac * (step.dvals[1, :] - step.dvals[0, :])
            delta = np.zeros(n + m)
            delta[fdofs] = np.linalg.solve(
                K[np.ix_(fdofs, fdofs)],
                -R[fdofs] - np.dot(K[np.ix_(fdofs, ddofs)], u[ddofs] - dvals),
            )
            du.fill(0.0)
            du[fdofs] = delta[fdofs]
            u += du
            if m:
                lam += delta[n:]

            res_norm = np.linalg.norm(R[fdofs])
            if res_norm < self.atol or res_norm / R0 < self.rtol:
                break

        else:
            raise RuntimeError(f"Newton iterations failed to converge after {it} iterations")

        du = np.zeros_like(u)
        K[:n, :n], R[:n] = model.assemble(step, 1, time, dt, u, du)

        react = np.zeros(n)
        react[ddofs] = np.dot(K[np.ix_(ddofs, ddofs)], u[ddofs]) - R[:n][ddofs]

        solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=u[:n].reshape((model.nnode, -1)),
            react=react.reshape((model.nnode, -1)),
            lagrange_multipliers=lam,
            iterations=it - 1,  # return only nonlinear iterations
        )
        return solution
