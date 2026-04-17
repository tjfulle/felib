import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.figure import SubFigure
from numpy.typing import NDArray

from .element import ReferenceElement


def rplot1(p: NDArray, r: NDArray) -> None:
    """Make plots of reactions on left and right edges of a uniform square"""
    _, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left reaction
    ilo = [n for n, x in enumerate(p) if isclose(x[0], -1.0)]
    ylo = p[ilo, 1]
    rlo = r[ilo]
    ix = np.argsort(ylo)
    ylo = ylo[ix]
    rlo = rlo[ix]

    axs[0].plot(ylo, rlo, "o", label="LHS")
    axs[0].set_title("LHS reaction")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("Heat flux/reaction")
    axs[0].grid(True)

    # Right reaction
    ihi = [n for n, x in enumerate(p) if isclose(x[0], 1.0)]
    yhi = p[ihi, 1]
    rhi = r[ihi]
    ix = np.argsort(yhi)
    yhi = yhi[ix]
    rhi = rhi[ix]

    axs[1].plot(yhi, rhi, "o", label="RHS")
    axs[1].set_title("RHS reaction")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("Heat flux/reaction")
    axs[1].grid(True)

    print(np.sum(rlo))
    print(np.sum(rhi))

    plt.tight_layout()
    plt.show()


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def mesh_plot_quad4(
    p: NDArray,
    connect: NDArray,
    n_edge: int = 10,
    ax: Axes | None | None = None,
    label: str | None = None,
    color: str = "k",
    lw: float = 0.6,
    ls: str = "-",
) -> tuple[Figure | SubFigure, Axes]:
    from .element import Quad4

    return mesh_plot(
        Quad4(), p, connect, label=label, n_edge=n_edge, ax=ax, color=color, lw=lw, ls=ls
    )


def mesh_plot_quad8(
    p: NDArray,
    connect: NDArray,
    n_edge: int = 10,
    ax: Axes | None | None = None,
    label: str | None = None,
    color: str = "k",
    lw: float = 0.6,
    ls: str = "-",
) -> tuple[Figure | SubFigure, Axes]:
    from .element import Quad8

    return mesh_plot(
        Quad8(), p, connect, label=label, n_edge=n_edge, ax=ax, color=color, lw=lw, ls=ls
    )


def mesh_plot(
    element: ReferenceElement,
    p: NDArray,
    connect: NDArray,
    n_edge: int = 10,
    ax: Axes | None | None = None,
    label: str | None = None,
    color: str = "k",
    lw: float = 0.6,
    ls: str = "-",
) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot FE mesh connectivity with correct element edges.

    Args:
        element : IsoparametricElement instance
        p       : global nodal coordinates (nnode_total, ndim)
        connect : element connectivity (nel, element.nnode)
        title   : plot title
        n_edge  : number of points per edge to draw curved edges
    """
    fig: Figure | SubFigure | None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    assert fig is not None
    seen: set[tuple[int, ...]] = set()
    linspace = np.linspace(-1.0, 1.0, n_edge)
    for elem in connect:
        pe = p[elem]
        for edge_no in range(len(element.edges)):
            ix = tuple(sorted(elem[element.edges[edge_no]]))
            if ix in seen:
                continue
            edge_pts = np.array([element.interpolate_edge(edge_no, pe, x) for x in linspace])
            ax.plot(edge_pts[:, 0], edge_pts[:, 1], color=color, linewidth=lw, label=label, ls=ls)
            label = None
            seen.add(ix)
    return fig, ax


def tplot(p: NDArray, t: NDArray, z: NDArray, title: str = "FEA Solution") -> None:
    """Make a 2D contour plot

    Args:
      p: mesh point coordinates (n, 2)
      t: mesh connectivity (triangulation) (N, 3)
      z: array of points to plot (n)

    """
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    plt.figure(figsize=(7, 5))
    countour = plt.tricontourf(triang, z, levels=50, cmap="turbo")
    plt.triplot(triang, color="k", linewidth=0.3)
    plt.colorbar(countour, label=None)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")


def _triangles_from_connect(connect: NDArray) -> NDArray:
    connect = np.asarray(connect, dtype=int)
    if connect.ndim != 2:
        raise ValueError("Expected a 2D connectivity array")

    valid = connect[connect >= 0]
    if valid.size == 0:
        raise ValueError("Connectivity array does not contain any valid nodes")

    triangles: list[list[int]] = []
    for elem in connect:
        nodes = [int(node) for node in elem if node >= 0]
        if len(nodes) == 3:
            triangles.append(nodes)
        elif len(nodes) == 4:
            triangles.append([nodes[0], nodes[1], nodes[2]])
            triangles.append([nodes[0], nodes[2], nodes[3]])
        elif len(nodes) > 4:
            for i in range(1, len(nodes) - 1):
                triangles.append([nodes[0], nodes[i], nodes[i + 1]])
        else:
            raise ValueError("Each element must have at least 3 valid nodes to plot a heat map")

    return np.asarray(triangles, dtype=int)


def nodal_heatmap(
    p: NDArray,
    connect: NDArray,
    z: NDArray,
    *,
    title: str = "Nodal Field",
    label: str | None = None,
    ax: Axes | None = None,
    cmap: str = "turbo",
    show_mesh: bool = True,
    mesh_color: str = "k",
    mesh_linewidth: float = 0.3,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot a nodal scalar field as a 2D heat map over the mesh."""

    fig: Figure | SubFigure
    created_axes = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    triangles = _triangles_from_connect(connect)
    triang = tri.Triangulation(p[:, 0], p[:, 1], triangles)
    contour = ax.tricontourf(triang, z, levels=50, cmap=cmap)
    if show_mesh:
        ax.triplot(triang, color=mesh_color, linewidth=mesh_linewidth)

    cbar = fig.colorbar(contour, ax=ax, label=label)
    cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    if created_axes:
        plt.tight_layout()
        plt.show()
    return fig, ax


def tplot3d(
    p: NDArray, t: NDArray, z: NDArray, label: str | None = None, title: str = "FE Solution"
) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(  # type: ignore
        triang, z, cmap="turbo", linewidth=0.2, antialiased=True
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label:
        ax.set_zlabel(label)  # type: ignore
    fig.colorbar(surf, ax=ax, shrink=0.6, label=label)

    plt.title(title)
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")
