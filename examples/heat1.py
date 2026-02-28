import argparse
import sys
from typing import Sequence

import numpy as np

np.set_printoptions(precision=2)
import fem

X = fem.X
Y = fem.Y


def heat1(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1] with hole of radius .3 at its center.
    • Spatially varying heat source: 1000 / √(x^2 + y^2)
    • Fixed temperature on left edge: 200˚
    • Fixed temperature on right edge: 50˚
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    """

    class Everywhere(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class Top(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] > 0.999:
                return True
            return False

    class Bottom(fem.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] < -0.999:
                return True
            return False

    nodes, elements = fem.meshing.uniform_plate(esize=esize)
    mesh_builder = fem.builder.MeshBuilder(nodes=nodes, elements=elements)
    mesh_builder.block(name="Block-1", region=Everywhere(), cell_type=fem.cell.Tri3)
    mesh_builder.sideset("Top", region=Top())
    mesh_builder.sideset("Bottom", region=Bottom())
    mesh_builder.elemset("All", region=Everywhere())
    mesh = mesh_builder.build()

    m = fem.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
    builder = fem.builder.ModelBuilder(mesh, name="heat1")
    builder.assign_properties(block="Block-1", element=fem.element.DCP3(), material=m)
    step = builder.heat_transfer_step()
    step.film(sideset="Top", h=250.0, ambient_temp=25.0)
    step.dflux(sideset="Bottom", magnitude=2000.0, direction=[0.0, 1.0])
    model = builder.build()
    model.solve()
    u = model.u[1]
    fem.plotting.tplot(model.coords, model.connect, u)
    thi = u[np.where(np.abs(mesh.coords[:, 1] - 1.0) < 1e-6)[0]]
    assert np.allclose(thi, 33)
    tlo = u[np.where(np.abs(mesh.coords[:, 1] + 1.0) < 1e-6)[0]]
    assert np.allclose(tlo, 366.33333)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    heat1(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
