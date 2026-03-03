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
    else:
        raise NotImplementedError
    wts.flags.writeable = False
    pts.flags.writeable = False
    return wts, pts


def gauss2d(n: int) -> tuple[NDArray, NDArray]:
    xi, wi = gauss1d(n)
    w = []
    p = []
    for i in range(n):
        for j in range(n):
            p.append([xi[i], xi[j]])
            w.append([wi[i], wi[j]])
    wts, pts = np.asarray(w), np.asanyarray(p)
    wts.flags.writeable = False
    pts.flags.writeable = False
    return wts, pts


def gauss2x2() -> tuple[NDArray, NDArray]:
    return gauss2d(2)


def gauss3x3() -> tuple[NDArray, NDArray]:
    return gauss2d(3)


# Triangle quadrature rules
# Reference triangle:
#  (r, s): r >= 0, s >= 0, r + s <= 1
# Area = 1/2


def gauss_tri1() -> tuple[NDArray, NDArray]:
    wts = np.array([0.5], dtype=float)
    pts = np.array([[1.0, 1.0]], dtype=float) / 3.0
    wts.flags.writeable = False
    pts.flags.writeable = False
    return wts, pts


def gauss_tri3() -> tuple[NDArray, NDArray]:
    wts = np.array([1.0, 1.0, 1.0], dtype=float) / 6.0
    pts = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]], dtype=float) / 6.0
    wts.flags.writeable = False
    pts.flags.writeable = False
    return wts, pts
