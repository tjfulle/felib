import numpy as np
from numpy.typing import NDArray
from .collections import ScalarField


class Material:
    _density: ScalarField | ScalarField

    def __init__(self, *, density: float | ScalarField | None = None, **properties: float) -> None:
        from .collections import ConstantScalarField
        if isinstance(density, (float, int)):
            self._density = ConstantScalarField(density)
        else:
            self._density = density

    def eval(self, e: NDArray, ndir: int, nshr: int) -> tuple[NDArray, NDArray]:
        raise NotImplementedError

    def has_density(self) -> bool:
        return self._density is not None

    def density(self, x: NDArray, t: float) -> float:
        if self._density is None:
            raise RuntimeError("Density has not been defined")
        return self._density(x, t)


class LinearElastic(Material):
    def __init__(
        self,
        *,
        density: float | ScalarField | None = None,
        youngs_modulus: float,
        poissons_ratio: float,
    ) -> None:
        super().__init__(density=density)
        self.youngs_modulus = youngs_modulus
        assert self.youngs_modulus > 0
        self.poissons_ratio = poissons_ratio
        assert -1 <= self.poissons_ratio < 0.5

    def eval(self, e: NDArray, ndir: int, nshr: int) -> tuple[NDArray, NDArray]:
        E = self.youngs_modulus
        nu = self.poissons_ratio
        if ndir == 2 and nshr == 1:
            # Plane stress: 2 direct components of stress and 1 shear component
            D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
            s = np.dot(D, e)
            return D, s
        elif ndir == 3 and nshr == 1:
            # Plane strain: 3 direct components of stress and 1 shear component
            factor = E / (1 - nu) / (1 - 2 * nu)
            D = factor * np.array(
                [
                    [1 - nu, nu, nu, 0],
                    [nu, 1 - nu, nu, 0],
                    [nu, nu, 1 - nu, 0],
                    [0, 0, 0, (1 - 2 * nu) / 2],
                ]
            )
            s = np.dot(D, e)
            return D, s
        raise NotImplementedError(f"{ndir=}, {nshr=}")
