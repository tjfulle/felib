import argparse
import sys
from typing import Sequence

import numpy as np

np.set_printoptions(precision=2)
import felib

X = felib.X
Y = felib.Y


def heat2(esize: float = 0.05):
    """
    Solve the 2D heat equation over a square domain

    • Bounds: x ∈ [-1, 1], y ∈ [-1, 1] with hole of radius .3 at its center.
    • Spatially varying heat source: 1000 / √(x^2 + y^2)
    • Fixed temperature on left edge: 200˚
    • Fixed temperature on right edge: 50˚
    • Heat flux along bottom edge: 2000
    • Convection along top edge: far field temperature 25˚ with convection coefficient 250

    """

    class Everywhere(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            return True

    class Top(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] > 0.999:
                return True
            return False

    class Left(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[0] < -0.999:
                return True
            return False

    class Right(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[0] > 0.999:
                return True
            return False

    class Bottom(felib.collections.RegionSelector):
        def __call__(self, x: Sequence[float], on_boundary: bool) -> bool:
            if on_boundary and x[1] < -0.999:
                return True
            return False

    class HeatSource(felib.collections.ScalarField):
        def __call__(self, x: Sequence[float], time: Sequence[float]) -> float:
            return 1000.0 / np.sqrt(x[0] ** 2 + x[1] ** 2)

    nodes, elements = felib.meshing.plate_with_hole(esize=esize)
    mesh = felib.mesh.Mesh(nodes=nodes, elements=elements)
    mesh.block(name="Block-1", region=Everywhere(), cell_type=felib.element.Tri3)
    mesh.nodeset("LHS", region=Left())
    mesh.nodeset("RHS", region=Right())
    mesh.sideset("Top", region=Top())
    mesh.sideset("Bottom", region=Bottom())
    mesh.elemset("All", region=Everywhere())

    m = felib.material.HeatConduction(conductivity=12.0, specific_heat=1.0)
    model = felib.model.Model(mesh, name="heat2")
    model.assign_properties(block="Block-1", element=felib.element.DCP3(), material=m)

    simulation = felib.simulation.Simulation(model)
    step = simulation.heat_transfer_step()
    step.temperature(nodes="LHS", value=200.0)
    step.temperature(nodes="RHS", value=50.0)
    step.film(sideset="Top", h=250.0, ambient_temp=25.0)
    step.dflux(sideset="Bottom", magnitude=2000.0, direction=[0.0, 1.0])
    step.source(elements="All", field=HeatSource())
    simulation.run()
    u = simulation.dofs[1]
    felib.plotting.tplot(model.coords, model.connect, u)
    felib.plotting.rplot1(model.coords, simulation.csteps[-1].solution.react)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-s", type=float, default=0.05, help="Element size [default: %(default)s]")
    args = p.parse_args()
    heat2(esize=args.s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
