import argparse
import sys

import numpy as np

np.set_printoptions(precision=2)
import felib

X = felib.X
Y = felib.Y


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

    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, el) -> bool:
            return True

    class Top(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] > 0.999

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] < -0.999

    nodes, elements = felib.meshing.uniform_plate(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.sideset("Top", region=Top())
    mesh.sideset("Bottom", region=Bottom())
    mesh.elemset("All", region=Everywhere())

    m = felib.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
    model = felib.model.Model(mesh, name="heat1")
    model.assign_properties(block="Block-1", element=felib.element.DCP3(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.heat_transfer_step()
    step.film(sideset="Top", h=250.0, ambient_temp=25.0)
    step.dflux(sideset="Bottom", magnitude=2000.0, direction=[0.0, 1.0])
    simulation.run()
    u = simulation.ndata["T"]
    felib.plotting.tplot(model.coords, model.connect, u)
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
