from typing import Sequence

from numpy.typing import NDArray


class Material:
    _density: float | None

    def __init__(self, *, density: float | None = None, **properties: float) -> None:
        self._density = density

    def eval(
        self,
        hsv: NDArray,
        e: NDArray,
        de: NDArray,
        time: Sequence[float],
        dtime: float,
        temp: float,
        dtemp: float,
        ndir: int,
        nshr: int,
        eleno: int,
        step: int,
        increment: int,
    ) -> tuple[NDArray, NDArray]:
        raise NotImplementedError

    def eval_hybrid(
        self,
        hsv: NDArray,
        e: NDArray,
        de: NDArray,
        time: Sequence[float],
        dtime: float,
        temp: float,
        dtemp: float,
        ndir: int,
        nshr: int,
        eleno: int,
        step: int,
        increment: int,
    ) -> tuple[NDArray, NDArray, float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement hybrid u-p support"
        )

    def has_density(self) -> bool:
        return self._density is not None

    def history_variables(self) -> list[str]:
        return []

    @property
    def density(self) -> float:
        if self._density is None:
            raise RuntimeError("Density has not been defined")
        return self._density
