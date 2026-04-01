import numpy as np
from distmeshpy import distmesh2d
from distmeshpy.utils import dcircle
from distmeshpy.utils import ddiff
from distmeshpy.utils import drectangle
from distmeshpy.utils import huniform


def plate_with_hole(esize: float) -> tuple[list, list]:
    fd = lambda p, **kwargs: ddiff(drectangle(p, -1, 1, -1, 1), dcircle(p, 0, 0, 0.5))
    fh = lambda p, **kwargs: 0.05 + 0.3 * dcircle(p, 0, 0, 0.5)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed, seed=7)
    nodes = [[i + 1, *x] for i, x in enumerate(p)]
    elements = [[i + 1, *(e + 1)] for i, e in enumerate(t)]
    return nodes, elements


def uniform_plate(esize: float) -> tuple[list, list]:
    fd = lambda p, **kwargs: drectangle(p, -1, 1, -1, 1)
    fh = lambda p, **kwargs: huniform(p)
    bbox = ((-1.0, 1.0), (-1.0, 1.0))
    fixed = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    p, t = distmesh2d(fd, fh, esize, bbox, fixed, seed=7)
    nodes = [[i + 1, *x] for i, x in enumerate(p)]
    elements = [[i + 1, *(e + 1)] for i, e in enumerate(t)]
    return nodes, elements
