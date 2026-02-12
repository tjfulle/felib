from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import model


@dataclass
class Solution:
    stiff: NDArray
    force: NDArray
    dofs: NDArray
    react: NDArray
    iterations: int = field(default=1)
    lagrange_multipliers: NDArray = field(default_factory=lambda: np.empty((0,)))


class Solver(ABC):
    @abstractmethod
    def __call__(self, mod: model.Model) -> Solution: ...


class DirectSolver(Solver):
    def __call__(self, mod: model.Model) -> Solution:
        """Solve the 2D plane stress/strain problem

        Args:
            model: Model

        Returns:
            d: nodal displacements returned with shape (nnode, 2)
            r: nodal reactions (including Neumann terms) returned with shape (nnode, 2)

        """
        n = mod.num_dof
        m = len(mod.equations)
        u = np.zeros(n)
        du = np.zeros(n)
        K: NDArray = np.zeros((n + m, n + m), dtype=float)
        R: NDArray = np.zeros(n + m, dtype=float)

        K[:n, :n], R[:n] = model.assemble(mod, u, du)

        if mod.equations:
            C, r = model.build_linear_constraint(mod, u, du)
            K[:n, n:] = C.T
            K[n:, :n] = C
            R[n:] = r

        # Condense out Dirichlet DOFs and solve condensed system
        ddofs, dvals = mod.dirichlet_dofs, mod.dirichlet_vals
        fdofs = np.array(sorted(set(range(n)).difference(ddofs)), dtype=int)
        d = np.zeros(n + m)
        d[ddofs] = dvals
        d[fdofs] = np.linalg.solve(
            K[np.ix_(fdofs, fdofs)], R[fdofs] - np.dot(K[np.ix_(fdofs, ddofs)], d[ddofs])
        )

        react = np.zeros_like(R[:n])
        du = np.zeros_like(u)
        K[:n, :n], R[:n] = model.assemble(mod, d, du)
        react[ddofs] = np.dot(K[np.ix_(ddofs, ddofs)], u[ddofs]) - R[:n][ddofs]

        solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=d[:n].reshape((mod.nnode, -1)),
            react=react[:n].reshape((mod.nnode, -1)),
            lagrange_multipliers=react[n:],
        )
        return solution


class NonlinearNewtonSolver(Solver):
    def __init__(self, **options: Any) -> None:
        self.atol: float = options.get("absolute tolerance", -1.0)
        self.rtol: float = options.get("relative tolerance", 1e-8)
        self.maxiter: int = options.get("max iterations", 25)

    def __call__(self, mod: model.Model) -> Solution:
        n = mod.num_dof
        m = len(mod.equations)

        u = np.zeros(n, dtype=float)
        du = np.zeros(n, dtype=float)
        lam = np.zeros(m, dtype=float)

        ddofs, dvals = mod.dirichlet_dofs, mod.dirichlet_vals
        fdofs = np.array(sorted(set(range(n)).difference(ddofs)), dtype=int)

        K: NDArray = np.zeros((n + m, n + m), dtype=float)
        R: NDArray = np.zeros(n + m, dtype=float)
        R0 = 1.0

        it = 0
        while it < self.maxiter:
            it += 1

            K.fill(0.0)
            R.fill(0.0)

            K[:n, :n], R[:n] = model.assemble(mod, u, du)

            if mod.equations:
                C, r = model.build_linear_constraint(mod, u, du)
                K[:n, n:] = C.T
                K[n:, :n] = C
                R[n:] = r

            if it == 1:
                R0 = float(np.linalg.norm(R[fdofs]))
                if self.atol < 0.0:
                    self.atol = 1e-8 * R0

            # Condense out Dirichlet DOFs and solve condensed system
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

        react = np.zeros_like(R[:n])
        du = np.zeros_like(u)
        K[:n, :n], R[:n] = model.assemble(mod, u, du)
        react = np.zeros(n)
        react[ddofs] = np.dot(K[np.ix_(ddofs, ddofs)], u[ddofs]) - R[:n][ddofs]

        solution = Solution(
            stiff=K[:n, :n],
            force=R[:n],
            dofs=u[:n].reshape((mod.nnode, -1)),
            react=react.reshape((mod.nnode, -1)),
            lagrange_multipliers=lam,
            iterations=it - 1,  # return only nonlinear iterations
        )
        return solution
