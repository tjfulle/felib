from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class SolverState:
    x: NDArray
    residual_norm: float
    iterations: int
    converged: bool


class Solver(ABC):
    @abstractmethod
    def __call__(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        x0: NDArray,
        args: tuple = (),
        atol: float | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
    ) -> SolverState: ...


class DirectSolver(Solver):
    """
    Direct linear solver for a single equilibrium solve.

    Solves:

        K x = -R

    where K and R are already reduced or augmented
    by the calling Step.
    """

    def __call__(
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        x0: NDArray,
        args: tuple = (),
        atol: float | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
    ) -> SolverState:
        try:
            K, R = fun(x0, *args)
            x = np.linalg.solve(K, -R)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Linear solve failed.") from e

        return SolverState(
            x=x, residual_norm=float(np.linalg.norm(R)), iterations=1, converged=True
        )


# TODO: Solver awareness of reduced systems
# Solvers currently accept already-reduced or augmented matrices. When using
# homogeneous-MPCs we will provide a reduced system K_red/f_red to the solver.
# The solver should remain agnostic, but the calling Step must:
#  - pass K_red, f_red when using `T`-based reduction, or
#  - pass augmented K/R when using Lagrange multipliers.
# After the solver returns `x_red` the Step must expand to full `u_full`
# via `u_full = T @ x_red` before updating `model.u` and post-processing.


class NonlinearNewtonSolver(Solver):
    def __call__(  # type: ignore
        self,
        fun: Callable[..., tuple[NDArray, NDArray]],
        x0: NDArray,
        args: tuple = (),
        atol: float | None = None,
        rtol: float | None = None,
        maxiter: int | None = None,
    ) -> SolverState:
        x = x0.copy()
        R0_norm: float | None = None
        it: int = 0
        atol = atol or -1.0
        rtol = rtol or 1e-8
        maxiter = maxiter or 25
        res_norm: float = 1.0
        while it < maxiter:
            it += 1
            K, R = fun(x, *args)
            res_norm = float(np.linalg.norm(R))
            if R0_norm is None:
                R0_norm = res_norm if res_norm != 0.0 else 1.0
                if atol <= 0.0:
                    atol = 1e-8 * R0_norm
            if res_norm < atol or res_norm / R0_norm < rtol:
                break
            try:
                dx = np.linalg.solve(K, -R)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f"Linear solve failed at iteration {it}") from e
            x += dx
        else:
            raise RuntimeError(
                f"Newton iterations failed to converge in {maxiter} iterations. "
                f"Final residuls = {res_norm:.3e}"
            )
        return SolverState(x=x, residual_norm=res_norm, iterations=it - 1, converged=True)
