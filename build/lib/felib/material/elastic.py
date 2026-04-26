from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import Material


class LinearElastic(Material):
    def __init__(
        self,
        *,
        youngs_modulus: float,
        poissons_ratio: float,
        density: float | None = None,
    ) -> None:
        super().__init__(density=density)
        self.youngs_modulus = youngs_modulus
        assert self.youngs_modulus > 0
        self.poissons_ratio = poissons_ratio
        assert -1 <= self.poissons_ratio < 0.5

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
        if ndir != 3 or nshr != 1:
            raise NotImplementedError(f"{ndir=}, {nshr=}")

        E = self.youngs_modulus
        nu = self.poissons_ratio

        # 3D isotropic bulk and shear moduli
        Kbulk = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))

        # Engineering strain vector ordering: [exx, eyy, ezz, gxy_like_exy]
        eps = np.asarray(e, dtype=float)
        tr_eps = eps[0] + eps[1] + eps[2]

        # Deviatoric projection in Voigt-like form for [xx, yy, zz, xy]
        Pdev = np.array(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0],
                [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0],
                [-1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        eps_dev = Pdev @ eps
        sdev = 2.0 * G * eps_dev
        Ddev = 2.0 * G * Pdev

        return Ddev, sdev, Kbulk
