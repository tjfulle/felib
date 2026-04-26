import numpy as np
from numpy.typing import NDArray


def gauss1d(n: int) -> tuple[NDArray, NDArray]:
    wts: NDArray
    pts: NDArray
    if n == 1:
        pts = np.array([0.0], dtype=float)
        wts = np.array([2.0], dtype=float)
    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        pts = np.array([-a, a], dtype=float)
        wts = np.array([1.0, 1.0], dtype=float)
    elif n == 3:
        a = np.sqrt(3.0 / 5.0)
        pts = np.array([-a, 0.0, a], dtype=float)
        wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    elif n == 4:
        pa, pb = 0.8611363115940526, 0.3399810435848563
        pts = np.array([-pa, -pb, pb, pa], dtype=float)
        wa, wb = 0.3478548451374539, 0.6521451548625461
        wts = np.array([wa, wb, wb, wa], dtype=float)
    else:
        raise NotImplementedError
    pts.flags.writeable = False
    wts.flags.writeable = False
    return pts, wts


def gauss2d(n: int) -> tuple[NDArray, NDArray]:
    w, p = [], []
    xi, wi = gauss1d(n)
    for i in range(n):
        for j in range(n):
            p.append([xi[i], xi[j]])
            w.append(wi[i] * wi[j])
    pts, wts = np.asarray(p), np.asanyarray(w)
    pts.flags.writeable = False
    wts.flags.writeable = False
    return pts, wts


def gauss2x2() -> tuple[NDArray, NDArray]:
    return gauss2d(2)


def gauss3x3() -> tuple[NDArray, NDArray]:
    return gauss2d(3)


def gauss4x4() -> tuple[NDArray, NDArray]:
    return gauss2d(4)


# Triangle quadrature rules
# Reference triangle:
#  (r, s): r >= 0, s >= 0, r + s <= 1
# Area = 1/2


def gauss_tri1() -> tuple[NDArray, NDArray]:
    pts = np.array([[1.0, 1.0]], dtype=float) / 3.0
    wts = np.array([0.5], dtype=float)
    pts.flags.writeable = False
    wts.flags.writeable = False
    return pts, wts


def gauss_tri3() -> tuple[NDArray, NDArray]:
    wts = np.array([1.0, 1.0, 1.0], dtype=float) / 6.0
    pts = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=float) / 6.0
    wts.flags.writeable = False
    pts.flags.writeable = False
    return pts, wts


def gauss_tri7() -> tuple[NDArray, NDArray]:
    """7-point Gaussian quadrature rule on a triangle
    (exact for polynomials up to degree 5; Dunavant 7-point rule)
    """
    # Weights for integrating over the reference triangle (area = 1/2)
    wts = np.array(
        [
            0.112500000000000,
            0.066197076394253,
            0.066197076394253,
            0.066197076394253,
            0.062969590272414,
            0.062969590272414,
            0.062969590272414,
        ],
        dtype=float,
    )
    pts = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.059715871789770],
            [0.470142064105115, 0.470142064105115],
            [0.101286507323456, 0.797426985353087],
            [0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456],
        ],
        dtype=float,
    )
    return pts, wts
