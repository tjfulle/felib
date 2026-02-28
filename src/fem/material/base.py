from numpy.typing import NDArray


class Material:
    _density: float | None

    def __init__(self, *, density: float | None = None, **properties: float) -> None:
        self._density = density

    def eval(self, e: NDArray, ndir: int, nshr: int) -> tuple[NDArray, NDArray]:
        raise NotImplementedError

    def has_density(self) -> bool:
        return self._density is not None

    @property
    def density(self) -> float:
        if self._density is None:
            raise RuntimeError("Density has not been defined")
        return self._density
