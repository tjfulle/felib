import matplotlib.pyplot as plt
import matplotlib.tri as tri
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
