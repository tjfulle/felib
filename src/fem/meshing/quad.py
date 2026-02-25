from typing import Callable

import numpy as np
from numpy.typing import NDArray

MapFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def gridmesh2d(
    nx: int, ny: int, mapfn: MapFunction
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Structured quad mesh engine

    Returns
    -------
    coords : (N, 2) ndarray
    conn   : (M, 4) ndarray
    """
    s = np.linspace(0.0, 1.0, nx + 1)
    t = np.linspace(0.0, 1.0, ny + 1)

    s, t = np.meshgrid(s, t, indexing="xy")
    x, y = mapfn(s, t)

    coords = np.column_stack((x.ravel(), y.ravel()))

    nnx = nx + 1
    i = np.arange(nx)
    j = np.arange(ny)
    i, j = np.meshgrid(i, j, indexing="xy")

    n1 = j * nnx + i
    n2 = n1 + 1
    n3 = n2 + nnx
    n4 = n1 + nnx

    conn = np.column_stack((n1.ravel(), n2.ravel(), n3.ravel(), n4.ravel())).astype(np.int64)

    return coords, conn


def rectmesh(
    bbox: tuple[float, float, float, float],
    h: float,
    biasx: float = 1.0,
    biasy: float = 1.0,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Structured rectangular quad mesh with optional bias.

    Returns
    -------
    coords : [[nid, x, y], ...]   (1-based node ids)
    conn   : [[eid, n1, n2, n3, n4], ...] (1-based ids)
    """
    xmin, xmax, ymin, ymax = bbox
    lx = xmax - xmin
    ly = ymax - ymin

    nx = int(np.ceil(lx / h))
    ny = int(np.ceil(ly / h))

    def mapfn(
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sb = s**biasx
        tb = t**biasy
        x = xmin + lx * sb
        y = ymin + ly * tb
        return x, y

    coords0, conn0 = gridmesh2d(nx, ny, mapfn)

    coords = [[nid + 1, float(x), float(y)] for nid, (x, y) in enumerate(coords0)]

    conn = [[eid + 1, n1 + 1, n2 + 1, n3 + 1, n4 + 1] for eid, (n1, n2, n3, n4) in enumerate(conn0)]

    return coords, conn


def wedgemesh(
    rinner: float,
    router: float,
    theta0: float,
    theta1: float,
    h: float,
    biasr: float = 1.0,
    biastheta: float = 1.0,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Structured quad mesh of a cylindrical wedge with optional bias.

    Returns
    -------
    coords : [[nid, x, y], ...]   (1-based node ids)
    conn   : [[eid, n1, n2, n3, n4], ...] (1-based ids)
    """
    dr = router - rinner
    dtheta = theta1 - theta0
    rmean = 0.5 * (rinner + router)

    nr = int(np.ceil(dr / h))
    nt = int(np.ceil((rmean * dtheta) / h))

    def mapfn(
        s: NDArray[np.float64],
        t: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sb = s**biasr
        tb = t**biastheta
        r = rinner + dr * sb
        theta = theta0 + dtheta * tb
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    coords0, conn0 = gridmesh2d(nr, nt, mapfn)

    coords = [[nid + 1, float(x), float(y)] for nid, (x, y) in enumerate(coords0)]

    conn = [[eid + 1, n1 + 1, n2 + 1, n3 + 1, n4 + 1] for eid, (n1, n2, n3, n4) in enumerate(conn0)]

    return coords, conn
