import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.typing import NDArray


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


def tplot3d(
    p: NDArray, t: NDArray, z: NDArray, label: str | None = None, title: str = "FE Solution"
) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(triang, z, cmap="turbo", linewidth=0.2, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label:
        ax.set_zlabel(label)
    fig.colorbar(surf, ax=ax, shrink=0.6, label=label)

    plt.title(title)
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")
