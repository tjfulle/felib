from typing import Callable

import numpy as np
from numpy.typing import NDArray

MapFunction = Callable[
    [NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
]


def plate_with_hole_Q4(
    bbox: tuple[float, float, float, float],
    nx: int = 1,
    ny: int = 1,
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
    ix, jx = np.meshgrid(i, j, indexing="xy")

    n1 = jx * nnx + ix
    n2 = n1 + 1
    n3 = n2 + nnx
    n4 = n1 + nnx

    conn = np.column_stack((n1.ravel(), n2.ravel(), n3.ravel(), n4.ravel())).astype(np.int64)

    return coords, conn


def rectmesh(
    bbox: tuple[float, float, float, float],
    nx: int = 1,
    ny: int = 1,
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
    nr: int = 1,
    nt: int = 1,
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


def gridmesh2d_quad8(
    nx: int,
    ny: int,
    mapfn: MapFunction,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Structured Quad8 mesh engine (serendipity).

    Returns
    -------
    coords : (N, 2) ndarray
    conn   : (M, 8) ndarray

    Node ordering per element:
        3--6--2
        |     |
        7     5
        |     |
        0--4--1
    """

    # Refined logical grid
    nnx = 2 * nx + 1
    nny = 2 * ny + 1

    s = np.linspace(0.0, 1.0, nnx)
    t = np.linspace(0.0, 1.0, nny)

    s, t = np.meshgrid(s, t, indexing="xy")
    x, y = mapfn(s, t)

    node_map = -np.ones((nny, nnx), dtype=int)
    nid = 0
    xc = []
    for j in range(nny):
        for i in range(nnx):
            if i % 2 == 1 and j % 2 == 1:
                continue  # unused interior node
            node_map[j, i] = nid
            xc.append([x[j, i], y[j, i]])
            nid += 1
    coords = np.array(xc, dtype=float)

    ec: list[list[int]] = []
    for ey in range(ny):
        for ex in range(nx):
            ii = 2 * ex
            jj = 2 * ey
            n0 = node_map[jj, ii]
            n1 = node_map[jj, ii + 2]
            n2 = node_map[jj + 2, ii + 2]
            n3 = node_map[jj + 2, ii]
            n4 = node_map[jj, ii + 1]
            n5 = node_map[jj + 1, ii + 2]
            n6 = node_map[jj + 2, ii + 1]
            n7 = node_map[jj + 1, ii]
            ec.append([n0, n1, n2, n3, n4, n5, n6, n7])
    conn = np.array(ec, dtype=int)

    return coords, conn


def rectmesh_quad8(
    bbox: tuple[float, float, float, float],
    nx: int = 1,
    ny: int = 1,
    biasx: float = 1.0,
    biasy: float = 1.0,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Structured rectangular Quad8 mesh with optional bias.

    Returns
    -------
    coords : [[nid, x, y], ...]   (1-based node ids)
    conn   : [[eid, n0..n7], ...] (1-based ids)
    """

    xmin, xmax, ymin, ymax = bbox
    lx = xmax - xmin
    ly = ymax - ymin

    def mapfn(
        s: np.ndarray,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sb = s**biasx
        tb = t**biasy
        x = xmin + lx * sb
        y = ymin + ly * tb
        return x, y

    coords0, conn0 = gridmesh2d_quad8(nx, ny, mapfn)

    coords = [[nid + 1, float(x), float(y)] for nid, (x, y) in enumerate(coords0)]

    conn = [[eid + 1] + [nid + 1 for nid in row] for eid, row in enumerate(conn0)]

    return coords, conn
