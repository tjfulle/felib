from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import Material


class HeatConduction(Material):
    def __init__(
        self,
        *,
        conductivity: float | NDArray,
        specific_heat: float,
        density: float | None = None,
    ) -> None:
        super().__init__(density=density)
        self._conductivity = conductivity
        self.specific_heat = specific_heat
        assert self._conductivity > 0
        assert self.specific_heat > 0

    def conductivity(self, dim: int) -> NDArray:
        if isinstance(self._conductivity, float):
            return self._conductivity * np.eye(dim)
        assert isinstance(self._conductivity, np.ndarray)
        assert self._conductivity.shape[0] == dim
        return self._conductivity

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
        D = self.conductivity(ndir)
        return D, np.dot(D, e)
