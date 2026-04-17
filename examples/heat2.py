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

    class Everywhere(felib.collections.ElementSelector):
        def __call__(self, el) -> bool:
            return True

    class Top(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] > 0.999

    class Bottom(felib.collections.SideSelector):
        def __call__(self, side: felib.collections.Side) -> bool:
            return side.x[1] < -0.999

    class Left(felib.collections.NodeSelector):
        def __call__(self, node: felib.collections.Node) -> bool:
            return node.on_boundary and node.x[0] < -0.999

    class Right(felib.collections.NodeSelector):
        def __call__(self, node: felib.collections.Node) -> bool:
            return node.on_boundary and node.x[0] > 0.999

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
    u = simulation.ndata["T"]
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
